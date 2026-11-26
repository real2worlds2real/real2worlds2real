"""Acquire calibrated task-object observations; agent-authored Blender is the default.

    python -m calibrated_scene_reconstruction.scene.objects.object_fit SESSION --from-config --observations-only --apply

Scene requests use fit.model and require agent source before acquisition. An
unfitted model starts from its own geometry envelope and the reference photo's
instance mask projected onto the observed support plane. It needs no legacy
single-image mesh or textured scene layout. These are initialization hypotheses,
not measured object dimensions or approval. Blender fitting and review use the
cached observations through scene.objects.blender.pipeline.

Observation crops come from registered cameras; masks, windows, identity and
semantic exclusions are retained under splat/object_fit. Explicit observation
refreshes preserve the earlier evidence. --only limits acquisition to named
requests, and --reuse requires compatible settings.

The lower-level kit fitter remains an explicit template fallback, selected with
--object-route template and a declared fit.kit or --kit NAME=KIND. Its shape,
colour, rig and cavity heuristics do not run for agent-model requests. A model's
own opening and collision geometry require separate task-feasibility review.
"""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import minimize

from calibrated_scene_reconstruction.assets import kits
from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.objects import frame_color
from calibrated_scene_reconstruction.scene.objects.object_hull import CropSam3, photo_start, project

OUTLIER = 0.5  # a frame whose silhouette the first fit overlaps less is another thing's (left out, fitted again)
POOR = 0.85  # a last fit's IoU under it: not the next fit's start
LIFT = 0.012  # m: how far a fit may set a thing off the carved top
REACH = 0.10  # m round an object's start its silhouette is looked for
FIT_PX = 480  # px across a silhouette's window in the fit at most
CROP, SHIFT, TILE = 320, 4, 300  # SAM 3's crop, px; sub-pixel bits of the projected outlines; a QA tile, px
RIG_SCALE = 0.5  # a rig's silhouettes: the whole frame at half size


def reviewed_observations(names, frames, masks, windows, rejections):
    """Apply named, reasoned semantic rejections before any Blender fitting.

    The review concerns instance identity/occlusion, never a model's fit error.
    Original acquisitions are retained separately from the reviewed fit cache.
    """
    if not isinstance(rejections, dict) or any(not isinstance(reason, str) or not reason.strip()
                                               for reason in rejections.values()):
        raise ValueError('Observation rejections require a frame-name to nonempty reason map')
    unknown = set(rejections) - set(map(str, names))
    if unknown:
        raise ValueError(f'Unknown rejected observation frame names: {sorted(unknown)}')
    keep = [i for i, frame in enumerate(frames) if str(names[frame]) not in rejections]
    return tuple([[values[i] for i in keep] for values in (frames, masks, windows)])


def fit_shape(kind, params):
    """A model request's box is only a search envelope, never a fitted/exported shape."""
    if kind != "_model":
        return kits.build(kind, params)
    size = np.asarray(params["size"], float)
    return {"half": size[:2] / 2, "height": size[2]}


def kit_outlines(kind, params, x, y, yaw, z0):
    """The kit's parts, each as its convex point set in the world (placed at x, y on z0, turned by yaw)."""
    k = fit_shape(kind, params)
    z0 = z0 + params.get("lift", 0.0)  # the fit's own height over the carved top (a top a few mm off under it)
    c, s_ = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
    return [np.asarray(p["V"], float) @ R.T + [x, y, z0] for p in k["parts"]]


def pick_frames(z, c, n, reach=2.5, sectors=12, margin=40):
    """Up to n frames that see c, the nearest of each of `sectors` directions round it first, then the next nearest."""
    W, H = (int(v) for v in z["size"][0])
    by = {}
    for i in range(len(z["names"])):
        u, v, d = project(c[None], z["K"][i], z["R_world_cam"][i], z["centre"][i])
        dist = float(np.linalg.norm(c - z["centre"][i]))
        if d[0] > 0.2 and dist < reach and margin < u[0] < W - margin and margin < v[0] < H - margin:
            w = c - z["centre"][i]
            by.setdefault(int((np.degrees(np.arctan2(w[1], w[0])) + 180) // (360 / sectors)) % sectors, []).append(
                (dist, i))
    for k in by:
        by[k].sort()
    out, r = [], 0
    while len(out) < n and any(len(v) > r for v in by.values()):
        out += [v[r][1] for v in by.values() if len(v) > r]
        r += 1
    return out[:n]


class Fit:
    """The kit's silhouettes in the frames' windows against SAM 3's. A window (x0, y0, w, h, f): the frame's pixels
    from (x0, y0), w x h of them, at f times their size."""

    def __init__(self, z, frames, masks, windows, kind, fixed, names, lo, hi, z0):
        self.z, self.frames, self.masks, self.windows = z, frames, masks, windows
        self.kind, self.fixed, self.names, self.lo, self.hi, self.z0 = kind, fixed, names, lo, hi, z0

    def params(self, q):
        q = np.clip(q, 0, 1)
        v = self.lo + q * (self.hi - self.lo)
        p = kits.constrain(self.kind, {**self.fixed, **dict(zip(self.names, v[:-3]))})
        return p, v[-3], v[-2], v[-1]

    def render(self, k, parts):
        i, (x0, y0, w, h, f) = self.frames[k], self.windows[k]
        canvas = np.zeros((int(round(h * f)), int(round(w * f))), np.uint8)
        for P in parts:
            u, v, d = project(P, self.z["K"][i], self.z["R_world_cam"][i], self.z["centre"][i])
            if (d <= 0.05).any():
                continue
            pts = np.round(np.c_[(u - x0) * f, (v - y0) * f] * (1 << SHIFT)).astype(np.int32)
            cv2.fillConvexPoly(canvas, cv2.convexHull(pts), 1, lineType=cv2.LINE_8, shift=SHIFT)
        return canvas.astype(bool)

    def ious(self, q):
        p, x, y, yaw = self.params(q)
        parts = kit_outlines(self.kind, p, x, y, yaw, self.z0)
        out = []
        for k, m in enumerate(self.masks):
            r = self.render(k, parts)
            out.append((r & m).sum() / max((r | m).sum(), 1))
        return np.array(out)

    def cost(self, q):
        e = np.sort(1 - self.ious(q))
        return float(e[:max(1, int(0.8 * len(e)))].mean()) + 5 * float(np.sum(np.clip(q, None, 0) ** 2 +
                                                                              np.clip(q - 1, 0, None) ** 2))


def frame_bands(f, p, x, y, yaw, z0, min_px=150, present=0.35):
    """Estimate the lower bands of a frame by rendering each band into the views that see it."""
    cover, bands = {}, []
    for sd in ("front", "back", "left", "right"):
        band = kit_outlines("box_frame", {**p, "bands": [sd]}, x, y, yaw, z0)[-1]
        sh = []
        for k, m in enumerate(f.masks):
            r = f.render(k, [band])
            if r.sum() >= min_px:
                sh.append((r & m).sum() / r.sum())
        cover[sd] = round(float(np.median(sh)), 2) if len(sh) >= 3 else None
        if (cover[sd] or 0) >= present or (cover[sd] is None and sd in p.get("bands", [])):
            bands.append(sd)
    return {**p, "bands": bands}, cover


def _boxes_world(boxes, x, y, yaw, z0):
    """Boxes (lo, hi) in a kit's own frame as convex point sets in the world (placed at x, y on z0, turned by yaw)."""
    c, s_ = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
    out = []
    for lo, hi in boxes:
        V = np.array([[a, b, d] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for d in (lo[2], hi[2])], float)
        out.append(V @ R.T + [x, y, z0])
    return out


def _cover(f, parts, min_px=150, min_frames=3, slack=0):
    """The median share of the parts' pixels on the silhouettes (grown by `slack` px: a post 4 cm across sits a
    few pixels off where the fit put it), over the frames that show at least min_px of them (None: too few frames
    show them)."""
    sh = []
    ker = np.ones((2 * slack + 1, 2 * slack + 1), np.uint8) if slack else None
    for k, m in enumerate(f.masks):
        if ker is not None:
            m = cv2.dilate(m.astype(np.uint8), ker).astype(bool)
        r = f.render(k, parts)
        if r.sum() >= min_px:
            sh.append((r & m).sum() / r.sum())
    return round(float(np.median(sh)), 2) if len(sh) >= min_frames else None


def frame_top(f, p, x, y, yaw, z0, present=0.35, step=0.05, top=2.2):
    """A box frame's height from the frames: its top rails kept where the silhouettes hold them."""
    L, D, w = p["length"], p["depth"], p.get("profile", 0.04)
    x0, x1, y0, y1 = -L / 2, L / 2, -D / 2, D / 2
    corners = [(cx, cy) for cx in (x0, x1 - w) for cy in (y0, y1 - w)]

    def rails(H):
        return _boxes_world([([x0, y0, H - w], [x1, y0 + w, H]), ([x0, y1 - w, H - w], [x1, y1, H]),
                             ([x0, y0, H - w], [x0 + w, y1, H]), ([x1 - w, y0, H - w], [x1, y1, H])], x, y, yaw, z0)

    now = _cover(f, rails(p["height"]), slack=6)
    if now is not None and now >= present:
        return p, {"rails_at_fit": now}
    seen, trace = 0.0, []
    for h in np.arange(0.0, top, 0.1):  # the posts, 10 cm at a time, as long as the silhouettes hold them
        c = _cover(f, _boxes_world([([cx, cy, h], [cx + w, cy + w, h + 0.1]) for cx, cy in corners], x, y, yaw, z0),
                   min_px=60, slack=6)
        trace.append(f"{h:.1f}:{c}")
        if c is None or c < present:
            break
        seen = h + 0.1
    print(f"[object_fit] the frame's posts on the silhouettes by height (m: share): {' '.join(trace)}", flush=True)
    for H in np.arange(max(seen, 0.3), top, step):
        c = _cover(f, rails(H), slack=6)
        if c is not None and c >= present:
            return {**p, "height": round(float(H), 3), "rails": True}, {"rails_at_fit": now, "posts_seen_to": round(float(seen), 2),
                                                                      "rails_found_at": round(float(H), 3)}
    return {**p, "height": round(float(max(seen, p["height"])), 3), "rails": False},\
        {"rails_at_fit": now, "posts_seen_to": round(float(seen), 2), "rails_found_at": None}


def recon_box(s, name):
    """The world box (lo, hi) of an object's reconstruction from the photo (objects/<obj>/mesh_cam.glb)."""
    import trimesh

    T = np.asarray(s.manifest["photo_camera"]["T_world_cam_cv"], float) if (s / "scene" / "manifest.json").exists()\
        else None
    if T is None:
        raise SystemExit(f"{name}: no scene manifest yet (run layout first) and no measured box to start from")
    V = np.asarray(trimesh.load(s / "objects" / name / "mesh_cam.glb", force="mesh").vertices) @ T[:3, :3].T + T[:3, 3]
    return V.min(0), V.max(0)


def fit_requests(sg):
    """Default agent-model requests, or explicitly selected template fallbacks, with their prompts."""
    from calibrated_scene_reconstruction.scene.objects.routes import routed_graph
    sg, _ = routed_graph(sg)
    kits_, prompts = [], {}
    for n, e in sg.get("dynamic_objects", {}).items():
        f = e.get("fit")
        if f and f.get("kit") and f.get("model"):
            raise ValueError(f"{n}: fit.kit and fit.model are mutually exclusive")
        if f and f.get("model") and not f.get("accepted"):
            kits_.append(f"{n}=_model")
            prompts[n] = f.get("prompt", e.get("_label", n))
        if f and f.get("kit") and not f.get("accepted"):  # an accepted fit stays as the review took it
            kits_.append(f"{n}={f['kit']}")
            if f.get("prompt"):
                prompts[n] = f["prompt"]
    return kits_, prompts


def init_params(kind, size):
    if kind == "_model":
        return {"size": list(size)}
    sx, sy, sz = (float(v) for v in size)
    if kind == "cylinder":
        return {"diameter": min(sy, sz), "length": sx, "lying": True}
    if kind == "plate_post_hole":
        side = max(sx, sy)
        return {"plate_x": 1.1 * side, "plate_y": 1.1 * side, "plate_thickness": 0.008, "post_x": 0.6 * side,
                "post_y": 0.6 * side, "post_corner": 0.01, "post_height": max(sz - 0.008, 0.02), "post_dx": 0.0,
                "post_dy": 0.0, "hole_diameter": 0.02, "hole_depth": 0.03}
    if kind == "cup":
        top = max(sx, sy)
        return {"top_diameter": top, "bottom_diameter": 0.8 * top, "height": sz}
    if kind == "box_frame":
        return {"length": sx, "depth": sy, "height": sz, "profile": 0.04}
    raise SystemExit(f"no initial params for kit {kind}")


def rig_from_points(X, box, w=0.04):
    """A box_frame rig's start from COLMAP's sparse points: its posts are the densest columns over the support (the
    points between 15 cm and 1.3 m above its top, within 0.4 m round it, binned along x and along y: a peak near each
    end), its height the posts' 98th percentile -- the silhouettes alone, mostly from the front, do not tell its
    depth."""
    lo, hi = np.asarray(box[0], float), np.asarray(box[1], float)
    top = hi[2]
    near = (X[:, 2] > top + 0.15) & (X[:, 2] < top + 1.3) & np.all(X[:, :2] > lo[:2] - 0.4, 1) &\
        np.all(X[:, :2] < hi[:2] + 0.4, 1)
    P = X[near]
    if len(P) < 200:
        return None
    # a post is a column: a 3 cm cell whose points fill most 5 cm levels of its height over the top -- one per corner
    # quadrant round the support's middle
    cell, lv = 0.03, 0.05
    ij = np.floor((P[:, :2] - (lo[:2] - 0.4)) / cell).astype(int)
    k = np.floor((P[:, 2] - top - 0.15) / lv).astype(int)
    nlev = int(np.ceil(1.15 / lv))
    occ = {}
    for (i, j), kk in zip(map(tuple, ij), k):
        occ.setdefault((i, j), set()).add(int(kk))
    mid = (lo[:2] + hi[:2]) / 2
    best = {}
    for (i, j), ks in occ.items():
        xy = lo[:2] - 0.4 + (np.array([i, j]) + 0.5) * cell
        q = (int(xy[0] > mid[0]), int(xy[1] > mid[1]))
        score = len(ks) / nlev
        if score > best.get(q, (0, None))[0]:
            best[q] = (score, xy)
    if len(best) < 4 or min(v[0] for v in best.values()) < 0.3:
        return None
    xl = (best[(0, 0)][1][0] + best[(0, 1)][1][0]) / 2
    xr = (best[(1, 0)][1][0] + best[(1, 1)][1][0]) / 2
    yf = (best[(0, 0)][1][1] + best[(1, 0)][1][1]) / 2
    yb = (best[(0, 1)][1][1] + best[(1, 1)][1][1]) / 2
    cols = np.zeros(len(X), bool)
    for x_ in (xl, xr):
        for y_ in (yf, yb):
            cols |= (np.abs(X[:, 0] - x_) < 0.05) & (np.abs(X[:, 1] - y_) < 0.05) & (X[:, 2] > top)
    z_top = float(np.percentile(X[cols, 2], 98)) if cols.sum() > 50 else top + 1.0
    return {"x": float((xl + xr) / 2), "y": float((yf + yb) / 2), "length": float(xr - xl + w),
            "depth": float(yb - yf + w), "height": float(z_top - top), "posts": [float(xl), float(xr), float(yf), float(yb)]}


def rig_height_from_corners(X, box, r=0.35, min_score=0.3):
    """A frame's height over its support from the posts the points do hold: per corner of the support's top, the
    densest 3 cm column within `r` of it (filling at least `min_score` of the 5 cm levels from 15 cm to 1.3 m over
    the top), its 98th percentile; the median over the corners that have one, or None."""
    lo, hi = np.asarray(box[0], float), np.asarray(box[1], float)
    top = hi[2]
    cell, lv = 0.03, 0.05
    nlev = int(np.ceil(1.15 / lv))
    heights = []
    for cx in (lo[0], hi[0]):
        for cy in (lo[1], hi[1]):
            P = X[(np.hypot(X[:, 0] - cx, X[:, 1] - cy) < r) & (X[:, 2] > top + 0.15) & (X[:, 2] < top + 1.3)]
            occ = {}
            for p in P:
                occ.setdefault(tuple(np.floor(p[:2] / cell).astype(int)), set()).add(int((p[2] - top - 0.15) // lv))
            if not occ:
                continue
            score, key = max((len(v) / nlev, k) for k, v in occ.items())
            if score < min_score:
                continue
            c = (np.array(key) + 0.5) * cell
            col = (np.abs(X[:, 0] - c[0]) < 0.04) & (np.abs(X[:, 1] - c[1]) < 0.04) & (X[:, 2] > top)
            if col.sum() > 20:
                heights.append(float(np.percentile(X[col, 2], 98)) - top)
    return float(np.median(heights)) if heights else None


def frames_support_z(s):
    """The support's top as the walk's frames place it, or None: the lowest of recon.object_hull's hull bottoms -- a
    hull's bottom only ever rises above the top it stands on (a thin plate the 85 % agreement drops, a foot hidden
    behind a rail), never sinks below it. Fits and marks are measured on it, the
    twin keeps its own height."""
    cp = s / "splat" / "support_carve.json"  # Prefer the support surface carved from the frames.
    if cp.exists():
        return float(json.loads(cp.read_text())["top_z"])
    hp = s / "splat" / "object_hull.json"
    b = [v["bottom_z"] for v in (json.loads(hp.read_text()) if hp.exists() else {}).values() if v.get("bottom_z")]
    return float(np.min(b)) if b else None


def placement(s):
    """The frames' placement a fit was measured in (splat/align.json's similarity, rounded): a fitted place holds in
    it alone."""
    al = json.loads((s / "splat" / "align.json").read_text())["sim3"]
    return [round(float(al["scale"]), 5)] + [round(float(v), 4) for v in al["t"]]


def targets(s, sg, a, prompts, support, z=None):
    """What is fitted: (name, kind, start {p0, x, y, yaw, z0}, prompt, rig) per object of --kit and rig of `rigs`."""
    out = []
    here = placement(s)
    # the support's top as the frames see it: where recon.object_hull's hulls rest
    z_frames = frames_support_z(s)
    lp = s / "splat" / "object_fit.json"
    last = json.loads(lp.read_text()) if lp.exists() else {}
    for kv in a.kit:
        name, kind = kv.split("=", 1)
        spec = sg["dynamic_objects"][name]
        fix = spec.get("fix") or {}
        if "xy" in fix and fix.get("placement") != here:  # fitted in another placement of the frames (S1 rerun:
            print(f"[object_fit] {name}: its fitted place is of another placement of the frames: from the photo's",
                  flush=True)
            fix = {k: v for k, v in fix.items() if k not in ("xy", "yaw_deg", "size_yaw_deg")}
        if (last.get(name) or {}).get("iou_median", 1.0) < POOR:  # its last fit failed: not its start again
            print(f"[object_fit] {name}: its last fit (IoU {last[name]['iou_median']}) is no start: from the photo's",
                  flush=True)
            fix = {k: v for k, v in fix.items() if k not in ("xy", "yaw_deg", "size_yaw_deg", "params", "kit")}
        prior = (fix.get("params") or {}) if fix.get("kit") == kind else {}
        # the agent's parameters the frames show and the fit does not measure (a plate's mounting holes): kept
        prior = {**prior, **((spec.get("fit") or {}).get("params") or {})}
        # one height for all (an object's own hull bottom is high where its foot was hidden)
        z0 = float(z_frames or fix.get("support_z", support[1][2]))  # the carved top: the fit sets its own lift
        from_recon = False
        if kind == "_model":
            size = fix.get("size")
            if size is None:
                from calibrated_scene_reconstruction.scene.objects.blender.blender_run import Blender
                from calibrated_scene_reconstruction.scene.objects.blender.fit import model_path
                with Blender(model_path(s, name), s / "objects" / name / "cache") as builder:
                    _, built = builder.evaluate()
                size = built["geometry"]["extent_m"]
            if 'xy' in fix:
                x, y = fix['xy']
            else:
                # Agent models need their own photo/camera evidence to locate
                # observations, not a legacy SAM3D mesh or textured layout.
                start = photo_start(s,z,spec.get('ids',[]),z0) if z is not None else None
                if start is None or not np.isfinite(start).all():
                    raise ValueError(f'{name}: agent-model observations require a registered reference '
                                     'photo, its instance mask and observed support plane')
                x, y = start
                print(f'[object_fit] {name}: initialize from current posed photo and agent geometry envelope',
                      flush=True)
            yaw0 = np.radians(fix.get('yaw_deg',0.0))
        elif fix.get("kit") == kind and "xy" in fix:
            size, (x, y), yaw0 = None, fix["xy"], np.radians(fix.get("yaw_deg", 0.0))
        else:
            size = fix.get("size")
            from_recon = False
            x, y = fix.get("xy") or [0.0, 0.0]
            yaw0 = np.radians(fix.get("size_yaw_deg", 0.0))
            if size is None or "xy" not in fix:  # from the photo's reconstruction: its box in the world, level
                lo, hi = recon_box(s, name)
                size, (x, y), yaw0 = (hi - lo).tolist(), ((lo + hi) / 2)[:2].tolist(), 0.0
                if kind == "cylinder" and size[1] > size[0]:  # lying along y
                    size, yaw0 = [size[1], size[0], size[2]], np.pi / 2
                from_recon = True
                ph = photo_start(s, z, spec.get("ids", []), z0) if z is not None else None
                if ph is not None:
                    print(f"[object_fit] {name}: from the photo's posed camera at ({ph[0]:.3f}, {ph[1]:.3f}), "
                          f"{100 * np.hypot(ph[0] - x, ph[1] - y):.1f} cm from its reconstruction", flush=True)
                    x, y = ph
        p0 = {**init_params(kind, size), **prior} if size is not None else {**prior}
        out.append((name, kind, {"p0": p0, "x": x, "y": y, "yaw": yaw0, "z0": z0, "from_recon": from_recon},
                    prompts.get(name, spec.get("_label", name)), False))
    from calibrated_scene_reconstruction.scene.twin.gates import meshed_structures

    if a.from_config and meshed_structures(s):  # (background splat: the splat's; zone: left out)
        for name, r in (sg.get("rigs") or {}).items():
            if r.get("accepted"):
                continue
            on = sg["furniture"][r.get("on", sg["task"]["support"])]["box"]
            lo, hi = np.asarray(on[0], float), np.asarray(on[1], float)
            ext = hi - lo
            p0 = {**init_params(r["kit"], [0.9 * ext[0], 0.9 * ext[1], r.get("height", 1.0)]), **(r.get("params") or {})}
            st = {"p0": p0, "x": float((lo[0] + hi[0]) / 2), "y": float((lo[1] + hi[1]) / 2), "yaw": 0.0,
                  "z0": float(hi[2])}
            if r["kit"] == "box_frame" and (s / "splat" / "data" / "sparse" / "full" / "points3D.bin").exists():
                from calibrated_scene_reconstruction.scene.capture.colmap_points import world_points

                pts = rig_from_points(world_points(s)[0], (lo, hi), p0.get("profile", 0.04))
                xl, xr, yf, yb = pts["posts"] if pts else (0, 0, 0, 0)
                off_edge = max(abs(xl - lo[0]), abs(xr - hi[0]), abs(yf - lo[1]), abs(yb - hi[1])) if pts else 0
                if pts and off_edge > 0.2:
                    # a frame stands at its support's edges; columns far inside are something else's
                    print(f"[object_fit] {name}: posts in the points {pts['length']:.2f} x {pts['depth']:.2f} m, one "
                          f"{off_edge:.2f} m off the support's edges: not the frame's, from the top", flush=True)
                    pts = None
                    p0.update({"length": float(1.05 * ext[0]), "depth": float(1.05 * ext[1])})
                    h = rig_height_from_corners(world_points(s)[0], (lo, hi))
                    if h:
                        p0["height"] = h
                        st["bounds"] = {"height": 0.08}
                        print(f"[object_fit] {name}: its height {h:.2f} m over the top from the posts the points "
                              f"hold at the top's corners", flush=True)
                if pts:  # the posts in the points: the start, the silhouettes refine it within a few cm
                    p0.update({k: pts[k] for k in ("length", "depth", "height")})
                    st.update({"x": pts["x"], "y": pts["y"], "points": pts,
                               "bounds": {"length": 0.06, "depth": 0.06, "height": 0.06, "xy": 0.04}})
                    print(f"[object_fit] {name}: posts in the sparse points at x {pts['posts'][0]:.2f} / "
                          f"{pts['posts'][1]:.2f}, y {pts['posts'][2]:.2f} / {pts['posts'][3]:.2f}, top "
                          f"{float(hi[2]) + pts['height']:.2f} m", flush=True)
            out.append((name, r["kit"], st, r.get("prompt", name), True))
    return out


def local_top(XP, x, y, ext, ztop, band=0.02, min_pts=40):
    """The support's height in a ring round a thing (from its extent's half + 2 cm, 10 cm wide): the median of
    COLMAP's points within `band` of the carved top there, or None."""
    r0 = ext / 2 + 0.02
    d = np.hypot(XP[:, 0] - x, XP[:, 1] - y)
    m = (d > r0) & (d < r0 + 0.10) & (np.abs(XP[:, 2] - ztop) < band)
    return float(np.median(XP[m, 2])) if m.sum() >= min_pts else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--kit", action="append", default=[], help="explicit low-level template fallback: obj=kind (assets.kits.FIT)")
    ap.add_argument("--prompts", default="", help="obj=prompt,... (default: the object's label)")
    ap.add_argument("--frames", type=int, default=48)  # (32: a two-part prompt found whole in 7, under the 8 the report holds)
    ap.add_argument('--only', default='', help='comma-separated object names; leave other observation caches unchanged')
    ap.add_argument("--set", action="append", default=[], help="obj.param=value: a kit param not fitted")
    ap.add_argument("--clearance", type=float, default=0.002,
                    help="the goal's hole over the target's diameter, m (1 mm: the peg jammed at the rim in PhysX)")
    ap.add_argument("--depth", type=float, default=0.8, help="the hole's depth, a share of the post's height")
    ap.add_argument("--observations-only", action="store_true", help="acquire fit.model observations only")
    ap.add_argument("--reuse", action="store_true", help="the silhouettes of the last run (same prompt)")
    ap.add_argument('--refresh-observations', action='store_true',
                    help='acquire new silhouettes even with a same-prompt cache; preserve reviewed starts with --reuse')
    ap.add_argument("--object-route", choices=("agent_blender", "template"), default="agent_blender")
    ap.add_argument("--from-config", action="store_true",
                    help="the scene graph's model requests; templates require explicit fallback selection")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    s = Session(a.session)
    kits.use_session(s.config_dir)  # the agent's own kits for this session's objects
    sg = s.scene_graph
    if a.from_config or not a.kit:
        from calibrated_scene_reconstruction.scene.objects.routes import ensure
        sg = ensure(s, a.object_route, apply=a.apply)
        a.from_config = True
    z = np.load(s / "parsing" / "posed_frames" / "cameras.npz")
    img_dir = Path(str(z["source"])) / "splat" / "data" / "images"
    from .observation_history import color_inputs, check_color, semantic_policy
    color_inputs(s, z, [])  # Refuse legacy HDR source images before acquiring labels.
    names = [str(n) for n in z["names"]]
    W, H = (int(v) for v in z["size"][0])
    prompts = dict(p.split("=", 1) for p in a.prompts.split(",") if "=" in p)
    if a.from_config:
        req, req_prompts = fit_requests(sg)
        given = {q.split("=", 1)[0] for q in a.kit}
        a.kit = a.kit + [k for k in req if k.split("=", 1)[0] not in given]
        prompts = {**req_prompts, **prompts}
    sets = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        o, p = k.split(".", 1)
        sets.setdefault(o, {})[p] = float(v)
    support = sg["furniture"][sg["task"]["support"]]["box"]
    if a.observations_only:
        a.kit = [v for v in a.kit if v.endswith("=_model")]
    if a.only:
        # Do not build unrelated models before honoring a scoped acquisition.
        # A separate unready flexible asset must not block other observations.
        selected = set(a.only.split(','))
        known = {item.split('=', 1)[0] for item in a.kit} | set(sg.get('rigs', {}))
        if selected-known:
            raise ValueError(f'Unknown or inactive fit requests: {sorted(selected-known)}')
        a.kit = [item for item in a.kit if item.split('=', 1)[0] in selected]
    for item in a.kit:
        name = item.split('=',1)[0]
        if (sg['dynamic_objects'].get(name,{}).get('fit') or {}).get('collection_context'):
            raise ValueError('Use scene.objects.collection_observations for source-context instance collections')
    todo = targets(s, sg, a, prompts, support, z)
    if a.only:
        selected = set(a.only.split(','))
        unknown = selected - {item[0] for item in todo}
        if unknown:
            raise ValueError(f'Unknown or inactive fit requests: {sorted(unknown)}')
        todo = [item for item in todo if item[0] in selected]
    if not todo:
        print("done: no kit to fit (no --kit, no fit requests or rigs in the scene graph)")
        return
    if not a.observations_only:
        from .object_understanding import declaration as understanding
        for name, kind, _, _, rig in todo:
            if not rig and kind != '_model':
                understanding(s, name)  # Explicit template fallback retains intrinsic requirements.
    segs = {}
    out_qa = s / "qa" / "object_fit"
    out_qa.mkdir(parents=True, exist_ok=True)
    res_path = s / "splat" / "object_fit.json"
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    from calibrated_scene_reconstruction.scene.twin.gates import meshed_structures

    rigs_now = set(sg.get("rigs") or {}) if meshed_structures(s) else set()  # an earlier run's fit of a thing the
    res = {k: v for k, v in res.items() if k in sg.get("dynamic_objects", {}) or k in rigs_now}  # graph lost: gone
    imgs = {}

    def image(i):
        if names[i] not in imgs:
            imgs[names[i]] = np.asarray(Image.open(img_dir / names[i]).convert("RGB"))
        return imgs[names[i]]

    # each task object's neighbours among the others (starts within 25 cm): passed over in its frames
    def place_starts(starts, near=0.25, cell=0.04, min_frames=4):
        """Task objects whose starts lie within `near` of each other, placed from the frames: every instance of each
        object's prompt round them in up to 16 frames (a 60 cm crop), its silhouette's middle cast onto the plane
        at the objects' mid height, the casts gathered into places (`cell`); a place counts with `min_frames`
        frames. Each object to a different place, the assignment with the highest sum of SAM 3's mean scores. {name: (xy, why)}."""
        import itertools

        close = [n for n in starts if not (sg['dynamic_objects'][n].get('fit') or {}).get('identity_from_pose', False)
                 and any(m != n and np.hypot(*(starts[m][0][:2] - starts[n][0][:2])) < near
                                          for m in starts)]
        if len(close) < 2:
            return {}
        seg = segs[CROP] = segs.get(CROP) or CropSam3(CROP)
        cen = np.mean([starts[n][0] for n in close], 0)
        casts = {n: [] for n in close}
        for i in pick_frames(z, cen, 16, reach=2.5):
            u, v, _ = project(cen[None], z["K"][i], z["R_world_cam"][i], z["centre"][i])
            dist = float(np.linalg.norm(cen - z["centre"][i]))
            crop = int(np.clip(z["K"][i][0][0] * 0.6 / max(dist, 0.1), CROP, min(W, H)))
            for n in close:
                for m, sc in seg.instances(image(i), starts[n][1].split("|")[0], u[0], v[0], crop):
                    if sc < 0.3 or m.sum() < 50:
                        continue
                    ys_, xs_ = np.nonzero(m)
                    d_ = z["R_world_cam"][i] @ np.linalg.solve(z["K"][i], [xs_.mean(), ys_.mean(), 1.0])
                    if abs(d_[2]) < 1e-6:
                        continue
                    p_ = z["centre"][i] + (cen[2] - z["centre"][i][2]) / d_[2] * d_
                    if np.hypot(*(p_[:2] - cen[:2])) < near + 0.1:
                        casts[n].append((p_[:2], sc, i))
        places = []  # [xy, {name: [scores]}, {frames}]
        for n in close:
            for xy, sc, i in sorted(casts[n], key=lambda q: -q[1]):
                pl = next((p for p in places if np.hypot(*(p[0] - xy)) < cell), None)
                if pl is None:
                    places.append([xy, {}, set()])
                    pl = places[-1]
                pl[1].setdefault(n, []).append(sc)
                pl[2].add(i)
        places = [p for p in places if len(p[2]) >= min_frames]
        if len(places) < len(close):
            print(f"[object_fit] {', '.join(close)}: {len(places)} place(s) found for {len(close)} close objects -- "
                  "the starts kept", flush=True)
            return {}
        def own(n, j):  # how much more its prompt takes the place than the others' do ('white block' scored the peg
            sc = {m: float(np.mean(places[j][1].get(m, [0.0]))) for m in close}  # at 0.80, 'white peg' higher)
            return sc[n] - max(v for m, v in sc.items() if m != n)
        best = None
        for perm in itertools.permutations(range(len(places)), len(close)):
            tot = sum(own(n, j) for n, j in zip(close, perm))
            if best is None or tot > best[0]:
                best = (tot, perm)
        return {n: (places[j][0], f"its prompt's place among {len(places)} round the close objects (mean score "
                                  f"{np.mean(places[j][1].get(n, [0.0])):.2f} over {len(places[j][2])} frames)")
                for n, j in zip(close, best[1])}

    starts = {n_: (np.array([st_["x"], st_["y"], st_["z0"] + 0.03]), pr_) for n_, _, st_, pr_, rg_ in todo if not rg_}
    if not a.reuse and len(starts) > 1:  # two task objects close together: their starts from the frames' instances
        placed = place_starts(starts)
        for n_, (xy_, why_) in placed.items():
            starts[n_] = (np.array([xy_[0], xy_[1], starts[n_][0][2]]), starts[n_][1])
            print(f"[object_fit] {n_}: starts at ({xy_[0]:.3f}, {xy_[1]:.3f}) -- {why_}", flush=True)
        todo = [(n_, k_, ({**st_, "x": float(starts[n_][0][0]), "y": float(starts[n_][0][1])} if n_ in placed
                          else st_), pr_, rg_) for n_, k_, st_, pr_, rg_ in todo]
    others_near = {n_: ([] if (sg['dynamic_objects'][n_].get('fit') or {}).get('identity_from_pose', False) else
                       [(m_, c_, p_) for m_, (c_, p_) in starts.items()
                        if m_ != n_ and np.linalg.norm(c_[:2] - starts[n_][0][:2]) < 0.25]) for n_ in starts}
    def taken(i, name, prompt, u, v, seg, cs=None, by_score=True):
        """The silhouettes of the neighbours of `name` that hold the instance its own prompt takes at (u, v) in frame
        i with a higher SAM 3 score -- the thing is theirs -- or None."""
        own = [seg(image(i), part, u, v, with_score=True, crop=cs) for part in prompt.split("|")]
        ex = None
        for _, st2, pr2 in others_near[name]:
            u2, v2, _ = project(st2[None], z["K"][i], z["R_world_cam"][i], z["centre"][i])
            for part2 in pr2.split("|"):
                m2, s2 = seg(image(i), part2, u2[0], v2[0], with_score=True, crop=cs)
                over = [s1 for m1, s1 in own if m2 is not None and m1 is not None and
                        (m1 & m2).sum() > 0.5 * min(m1.sum(), m2.sum())]
                if over and (not by_score or s2 > max(s1 for _, s1 in own)):  # its best-scored part against theirs
                    ex = m2 if ex is None else ex | m2
        return ex

    def relocate(name, prompt, c, wide=960, min_frames=4):
        """Its place when its start holds a neighbour: in a wide crop, the instance of its prompt nearest the start
        that is not the neighbour's, in each frame; the rays through their middles, nearest point (None: the start
        is its own, or too few frames find it)."""
        seg = segs[CROP] = segs.get(CROP) or CropSam3(CROP)  # one model, the crop per call
        A_, b_, n_, held = np.zeros((3, 3)), np.zeros(3), 0, 0
        for i in pick_frames(z, c, 16, reach=2.5):
            u, v, _ = project(c[None], z["K"][i], z["R_world_cam"][i], z["centre"][i])
            # one instance under both starts, whichever SAM 3 scores higher: the later one in the graph looks round it
            ex = taken(i, name, prompt, u[0], v[0], seg, wide, by_score=False)
            if ex is None:
                continue
            held += 1
            m = seg(image(i), prompt.split("|")[0], u[0], v[0], exclude=ex, near=wide / 2, crop=wide)
            if m is None:
                continue
            ys, xs = np.nonzero(m)
            d_ = z["R_world_cam"][i] @ np.linalg.solve(z["K"][i], [xs.mean(), ys.mean(), 1.0])
            d_ /= np.linalg.norm(d_)
            M_ = np.eye(3) - np.outer(d_, d_)
            A_ += M_
            b_ += M_ @ z["centre"][i]
            n_ += 1
        print(f"[object_fit] {name}: its start's instance shared with a neighbour in {held} frames, another found "
              f"round it in {n_}", flush=True)
        if held < 4 or n_ < min_frames:
            return None
        return np.linalg.solve(A_, b_)

    board = (s.config_dir / "charuco_board.json").exists()  # a calibration board lies in the capture
    swaps = {}  # {name: (neighbour, the instance it kept, the other round it)}: two things that started as one
    moved_starts = set()

    from calibrated_scene_reconstruction.scene.capture.colmap_points import world_points

    XP = world_points(s)[0]

    def run(force):
        """Every fit, each object from its start (`force`: {name: [x, y]} instead), written into res and qa."""
        for n_, xy_ in force.items():
            starts[n_] = (np.array([xy_[0], xy_[1], starts[n_][0][2]]), starts[n_][1])
        for k_, lst in others_near.items():
            others_near[k_] = [(m_, starts[m_][0], p_) for m_, _, p_ in lst]
        for name, kind, st, prompt, rig in todo:
            p0, x, y, yaw0, z0 = st["p0"], st["x"], st["y"], st["yaw"], st["z0"]
            if name in starts and (force or name in moved_starts):  # set by a swap, or moved off a neighbour
                x, y = float(starts[name][0][0]), float(starts[name][0][1])
            h0 = fit_shape(kind, p0)["height"]
            c = np.array([x, y, z0 + h0 / 2])
            cache = s / "splat" / "object_fit" / f"{name}.npz"
            frames, masks, windows = [], [], []
            component_gap = (sg['dynamic_objects'][name].get('fit') or {}).get('mask_component_gap_pixels') if kind == '_model' else None
            if component_gap is not None and (type(component_gap) is not int or not 0 <= component_gap <= 8):
                raise ValueError('mask_component_gap_pixels must be an integer0..8 for an observed contact-connected assembly')
            if not a.refresh_observations and cache.exists():
                with np.load(cache) as previous:
                    previous_policy=json.loads(str(previous['mask_component_policy'])) if 'mask_component_policy' in previous else None
                    check_color(previous, color_inputs(s, z, previous["frames"]))
                    if kind == '_model':
                        from .observation_history import check_policy
                        check_policy(previous, sg['dynamic_objects'][name].get('fit') or {})
                if previous_policy != component_gap:
                    raise ValueError('Mask component policy changed; preserve and refresh observations explicitly')
            if not a.refresh_observations and (a.reuse or kind == "_model") and cache.exists() and str(np.load(cache)["prompt"]) == prompt:
                d = np.load(cache, allow_pickle=True)
                from .mask_cache import read_masks
                frames, masks = [int(i) for i in d["frames"]], read_masks(d)
                windows = [tuple(w) if len(w) == 5 else (w[0], w[1], CROP, CROP, 1.0) for w in d["windows"]]  # (x0, y0): a crop
            else:
                from .observation_history import archive
                archive(cache, {'prompt': prompt, 'refresh_requested': a.refresh_observations})
                key = None if rig else CROP
                segs[key] = segs.get(key) or CropSam3(key)
                if not rig and not force and others_near.get(name):  # a neighbour holding the instance at its
                    # start: two things, two instances -- the one at the start and the other found round it. This one
                    # keeps the first, the neighbour starts at the other; the kits' fits then try the swap (SAM 3's
                    # scores could not tell: 'white peg' scored the block above 'white block' did)
                    c2 = relocate(name, prompt, c)
                    if c2 is not None:
                        for n2, st2, pr2 in others_near[name]:
                            if np.hypot(*(st2[:2] - c[:2])) > 0.05:
                                continue
                            print(f"[object_fit] {name} and {n2} start on one thing; another round it at "
                                  f"({c2[0]:.3f}, {c2[1]:.3f}): {n2} starts there", flush=True)
                            starts[n2] = (np.array([c2[0], c2[1], starts[n2][0][2]]), pr2)
                            moved_starts.add(n2)
                            swaps[name] = (n2, c.copy(), c2.copy())
                            for k_, lst in others_near.items():  # the neighbours' view of the moved one
                                others_near[k_] = [(m_, starts[m_][0], p_) for m_, _, p_ in lst]
                            break
                ext = 2 * max(float(max(fit_shape(kind, p0)["half"])), h0 / 2)  # the thing's extent, m
                request = (sg.get('dynamic_objects', {}).get(name, {}).get('fit') or {})
                segs[key].threshold = semantic_policy(request).get('instance_confidence_min', .2)
                crop_min = request.get('crop_min', CROP)
                if type(crop_min) is not int or not 32 <= crop_min <= min(W, H):
                    raise ValueError('fit.crop_min must be integer pixels within32..min(image width,height)')
                component_review, raw_component_masks, raw_component_names, raw_part_masks = [], [], [], []
                collection_masks, collection_review = {}, []
                collection_policy = semantic_policy(request).get('part_instances', {})
                for i in pick_frames(z, c, a.frames, reach=4.5 if rig else 2.5):
                    u, v, _ = project(c[None], z["K"][i], z["R_world_cam"][i], z["centre"][i])
                    # the crop three times the thing's projected extent (at least CROP): a block filling a close-up
                    # overflowed 320 px, its mask cut at the crop's edge and its neighbour out of reach
                    dist = float(np.linalg.norm(c - z["centre"][i]))
                    cs = int(np.clip(3 * z["K"][i][0][0] * ext / max(dist, 0.1), crop_min, min(W, H)))
                    # the search reaches 10 cm round the start at its distance
                    ex, reach_px = None, max(40.0, z["K"][i][0][0] * REACH / max(dist, 0.1))  # another task object close by that took the same instance: the instance
                    # goes to the prompt SAM 3 scores higher on it, the other searches on without it
                    if not rig and others_near.get(name):
                        ex = taken(i, name, prompt, u[0], v[0], segs[key], cs)
                        reach_px = cs / 2 if ex is not None else reach_px
                    for exclude_prompt in request.get('exclude_prompts', []):
                        for excluded_mask, confidence in segs[key].instances(image(i), exclude_prompt, u[0], v[0], cs):
                            if confidence >= .4:
                                ex = excluded_mask if ex is None else ex | excluded_mask
                    if not rig and board:  # Exclude the calibration board from task-object masks.
                        mb = segs[key](image(i), "checkerboard", u[0], v[0], union=1e9, crop=cs)
                        if mb is not None:
                            ex = mb if ex is None else ex | mb
                    observed_parts=[]
                    m, found = None, 0  # a prompt of parts "a|b": their silhouettes together, in the frames where
                    for part in prompt.split("|"):
                        if observed_parts and not observed_parts[0].any():
                            break  # Never anchor a secondary query to a missing primary.
                        if part in collection_policy and not observed_parts:
                            continue  # Missing primary evidence rejects this view below.
                        part_u,part_v=float(u[0]),float(v[0])
                        anchor=(request.get('part_anchors') or {}).get(part)
                        if anchor is not None:
                            from .mask_components import query_anchor
                            if not observed_parts:
                                raise ValueError('A semantic part anchor requires an earlier observed part')
                            part_u,part_v=query_anchor(observed_parts[0],anchor)
                        if part in collection_policy:
                            from .mask_components import contact_instances
                            if not observed_parts:
                                raise ValueError('Part collection requires an independently observed primary part')
                            candidates = segs[key].instances(image(i), part, part_u, part_v, cs)
                            mp, decision = contact_instances(candidates, observed_parts[0],
                                                           collection_policy[part]['gap_pixels'], ex)
                            prefix = f'view_{i}_part_{len(observed_parts)}'
                            collection_masks[prefix+'_anchor'] = observed_parts[0].copy()
                            for proposal_index, (proposal, _) in enumerate(candidates):
                                collection_masks[prefix+f'_proposal_{proposal_index}'] = proposal
                            collection_review.append({'frame': names[i], 'prompt': part,
                                                      'array_prefix': prefix, **decision})
                        else:
                            mp = segs[key](image(i), part, part_u, part_v, union=1e9 if rig else None, exclude=ex,
                                           near=reach_px, crop=None if rig else cs)  # outlined the post
                        observed_parts.append(mp.copy() if mp is not None else np.zeros((H,W),bool))
                        if mp is not None:  # alone in the oblique close-ups, 'white base plate' the plate -- in half the
                            m, found = (mp if m is None else (m | mp)), found + 1  # frames alone: the fit shrank it)
                    optional=set(semantic_policy(request).get('optional_part_queries',[]))
                    missing=[part for part,mask in zip(prompt.split('|'),observed_parts) if not mask.any()]
                    if m is None or any(part not in optional for part in missing):
                        continue
                    # Keep each independent source query even when no connected-
                    # component filter is requested. Otherwise a thin functional
                    # part can be confused with a shadow inside the merged mask
                    # and its original evidence is lost to later inspection.
                    if component_gap is not None or len(observed_parts) > 1:
                        raw_component_masks.append(m.copy());raw_component_names.append(names[i]);raw_part_masks.append(np.stack(observed_parts))
                    if component_gap is not None:
                        from .mask_components import anchored
                        m, selected = anchored(observed_parts, component_gap)
                        component_review.append({'frame': names[i], **selected})
                        if m is None:continue
                    elif len(observed_parts) > 1:
                        component_review.append({'frame': names[i], 'component_filter_applied': False,
                                                 'source_part_prompts': prompt.split('|'),
                                                 'unobserved_optional_queries':missing,
                                                 'primary_query_scope':'Whole visible assembly; secondary masks are separate part evidence, never proof of absent material'})
                    if rig:  # the whole frame at half size
                        win = (0, 0, W, H, RIG_SCALE)
                        m = cv2.resize(m.astype(np.uint8), (int(W * RIG_SCALE), int(H * RIG_SCALE)),
                                       interpolation=cv2.INTER_NEAREST).astype(bool)
                    else:
                        # the fit's window round the silhouette found (1.6 times its extent, at most FIT_PX across
                        # once scaled): the search's wide crop held other white things and a 9 cm block in 100 px
                        ys_, xs_ = np.nonzero(m)
                        side = int(np.clip(1.6 * max(np.ptp(xs_), np.ptp(ys_)), 160, min(W, H)))
                        x0 = int(np.clip((xs_.min() + xs_.max()) / 2 - side / 2, 0, W - side))
                        y0 = int(np.clip((ys_.min() + ys_.max()) / 2 - side / 2, 0, H - side))
                        f_ = min(1.0, FIT_PX / side)
                        m = m[y0:y0 + side, x0:x0 + side]
                        if f_ < 1.0:
                            m = cv2.resize(m.astype(np.uint8), (int(round(side * f_)),) * 2,
                                           interpolation=cv2.INTER_NEAREST) > 0
                        win = (x0, y0, side, side, f_)
                    frames.append(i), masks.append(m), windows.append(win)
                cache.parent.mkdir(parents=True, exist_ok=True)
                if collection_policy:
                    np.savez_compressed(cache.with_suffix('.collections.acquired.npz'), **collection_masks)
                    cache.with_suffix('.collections.json').write_text(json.dumps(collection_review, indent=2)+'\n')
                if component_gap is not None or raw_component_names:
                    np.savez_compressed(cache.with_suffix('.components.acquired.npz'),
                                        names=raw_component_names, masks=np.asarray(raw_component_masks),
                                        part_masks=np.asarray(raw_part_masks), prompts=prompt.split('|'))
                    cache.with_suffix('.components.json').write_text(json.dumps(component_review, indent=2)+'\n')
                np.savez_compressed(cache, frames=np.array(frames, int), masks=np.array(masks, dtype=object),
                                    windows=np.array(windows, float), prompt=prompt,
                                    semantic_query_policy=json.dumps(semantic_policy(request),sort_keys=True),
                                    mask_component_policy=json.dumps(component_gap),
                                    source_color_inputs=json.dumps(color_inputs(s,z,frames),sort_keys=True))
                if kind == '_model':
                    shutil.copy2(cache, cache.with_suffix('.acquired.npz'))
            rejections = ((sg.get('dynamic_objects', {}).get(name, {}).get('fit') or {})
                          .get('observation_rejections', {}))
            if kind == '_model' and rejections:
                before = len(frames)
                frames, masks, windows = reviewed_observations(names, frames, masks, windows, rejections)
                print(f'[object_fit] {name}: {before-len(frames)} semantically rejected observations '
                      'excluded by the recorded instance review', flush=True)
                np.savez_compressed(cache, frames=np.array(frames, int), masks=np.array(masks, dtype=object),
                                    windows=np.array(windows, float), prompt=prompt,
                                    review_rejections=json.dumps(rejections, sort_keys=True),
                                    semantic_query_policy=json.dumps(semantic_policy(sg['dynamic_objects'][name].get('fit') or {}),sort_keys=True),
                                    mask_component_policy=json.dumps(component_gap),
                                    source_color_inputs=json.dumps(color_inputs(s,z,frames),sort_keys=True))
            if len(frames) < 6:
                print(f"[object_fit] {name}: silhouettes in {len(frames)} frames, not fitted", flush=True)
                continue
            if not rig:  # Initialize the rigid object's position by triangulating silhouette centroids.
                A_, b_ = np.zeros((3, 3)), np.zeros(3)
                for k, i in enumerate(frames):
                    x0_, y0_, _, _, f_ = windows[k]
                    ys_, xs_ = np.nonzero(masks[k])
                    if not len(xs_):
                        continue
                    uv1 = np.array([x0_ + xs_.mean() / f_, y0_ + ys_.mean() / f_, 1.0])
                    d_ = z["R_world_cam"][i] @ np.linalg.solve(z["K"][i], uv1)
                    d_ /= np.linalg.norm(d_)
                    M_ = np.eye(3) - np.outer(d_, d_)
                    A_ += M_
                    b_ += M_ @ z["centre"][i]
                p_ = np.linalg.solve(A_, b_)
                for _ in range(3):  # the frames whose ray misses the others' point took another thing: left out
                    res_, rows_ = [], []
                    for k, i in enumerate(frames):
                        x0_, y0_, _, _, f_ = windows[k]
                        ys_, xs_ = np.nonzero(masks[k])
                        if not len(xs_):
                            res_.append(np.inf)
                            continue
                        d_ = z["R_world_cam"][i] @ np.linalg.solve(z["K"][i], [x0_ + xs_.mean() / f_, y0_ + ys_.mean() / f_, 1.0])
                        d_ /= np.linalg.norm(d_)
                        w_ = p_ - z["centre"][i]
                        res_.append(float(np.linalg.norm(w_ - (w_ @ d_) * d_)))
                        rows_.append(k)
                    res_ = np.array(res_)
                    ok_ = res_ < max(0.03, 2.5 * np.median(res_[np.isfinite(res_)]))
                    if ok_.sum() < 6 or ok_.all():
                        break
                    if kind == '_model':
                        # A noncentral or articulated silhouette can move its
                        # image centroid without changing object identity.
                        # Retain every observation; only agent-reviewed semantic
                        # rejections may remove it from Blender evidence.
                        flags=[{'frame': names[frames[k]], 'ray_residual_m': float(res_[k]),
                                'action': 'Inspect raw RGB/identity; geometric disagreement alone is not rejection'}
                               for k in np.flatnonzero(~ok_)]
                        cache.with_suffix('.association_flags.json').write_text(json.dumps(flags, indent=2)+'\n')
                        print(f'[object_fit] {name}: {len(flags)} centroid-ray flags retained for semantic review', flush=True)
                        break
                    frames, masks, windows = ([v for v, kk in zip(lst, ok_) if kk] for lst in (frames, masks, windows))
                    A_, b_ = np.zeros((3, 3)), np.zeros(3)
                    for k, i in enumerate(frames):
                        x0_, y0_, _, _, f_ = windows[k]
                        ys_, xs_ = np.nonzero(masks[k])
                        d_ = z["R_world_cam"][i] @ np.linalg.solve(z["K"][i], [x0_ + xs_.mean() / f_, y0_ + ys_.mean() / f_, 1.0])
                        d_ /= np.linalg.norm(d_)
                        M_ = np.eye(3) - np.outer(d_, d_)
                        A_ += M_
                        b_ += M_ @ z["centre"][i]
                    p_ = np.linalg.solve(A_, b_)
                    print(f"[object_fit] {name}: {int((~ok_).sum())} frame(s) whose silhouette is another thing's left out",
                          flush=True)
                if np.hypot(p_[0] - x, p_[1] - y) < 0.3:
                    print(f"[object_fit] {name}: placed by its silhouettes at ({p_[0]:.3f}, {p_[1]:.3f}), "
                          f"{100 * np.hypot(p_[0] - x, p_[1] - y):.1f} cm from the start", flush=True)
                    x, y = float(p_[0]), float(p_[1])
                if st.get("from_recon") and kind != "_model":  # its size from the silhouettes too: SAM 3D's box of a 30 px thing is no start
                    # (the peg's 14 x 11 x 12 cm collapsed the fit to a disc); per frame the mask's extent at its distance
                    wide, tall = [], []
                    for k, i in enumerate(frames):
                        ys_, xs_ = np.nonzero(masks[k])
                        if len(xs_) < 20:
                            continue
                        f_, dist = windows[k][4], float(np.linalg.norm(np.array([x, y, c[2]]) - z["centre"][i]))
                        m_ = dist / (z["K"][i][0][0] * f_)
                        wide.append(np.ptp(xs_) * m_), tall.append(np.ptp(ys_) * m_)
                    if len(wide) >= 4:
                        sx_ = float(np.percentile(wide, 80))
                        sz_ = float(np.median(tall))
                        sy_ = float(np.percentile(wide, 20)) if kind == "cylinder" else sx_
                        if kind == "cylinder":  # lying: its length the widest view, its diameter the narrowest and its height
                            sy_ = sz_ = min(sy_, sz_)
                        p0 = {**init_params(kind, [sx_, sy_, sz_]), **{k_: v_ for k_, v_ in p0.items()
                                                                      if k_ not in kits.FIT[kind]}}
                        print(f"[object_fit] {name}: sized by its silhouettes {100 * sx_:.1f} x {100 * sy_:.1f} x "
                              f"{100 * sz_:.1f} cm", flush=True)
            if not rig:  # the top's own height round the thing
                zl = local_top(XP, x, y, 2 * max(float(max(fit_shape(kind, p0)["half"])), h0 / 2), z0)
                if zl is not None:
                    z0 = zl
            if kind == "_model":
                from calibrated_scene_reconstruction.scene.objects.blender.contract import write_json
                # Persist the final observation set after triangulation rejects
                # misidentified instances, rather than retaining its raw input.
                np.savez_compressed(cache, frames=np.array(frames, int), masks=np.array(masks, dtype=object),
                                    windows=np.array(windows, float), prompt=prompt,
                                    review_rejections=json.dumps(rejections, sort_keys=True),
                                    semantic_query_policy=json.dumps(semantic_policy(sg['dynamic_objects'][name].get('fit') or {}),sort_keys=True),
                                    mask_component_policy=json.dumps(component_gap),
                                    source_color_inputs=json.dumps(color_inputs(s,z,frames),sort_keys=True))
                write_json(cache.with_suffix(".start.json"), {"xy": [x, y], "yaw_deg": float(np.degrees(yaw0)),
                           "local_top": z0, "placement": placement(s), 'observation_rejections': rejections})
                print(f"[object_fit] {name}: Blender observations cached in {len(frames)} frames", flush=True)
                continue
            fit_names = list(kits.FIT[kind])
            fitb = dict(kits.FIT[kind])
            if not rig:
                fit_names.append("lift")  # mm off the carve's, the fit thinned it to 1.6 cm to keep off the gap)
                fitb["lift"] = (-LIFT, 0.0)  # down only: a cup let up stood 6 mm in the air, 5 mm short
                p0 = {**p0, "lift": 0.0}
            # a rig's yaw: the support's box is level with the world's axes, its frame need not be -- 30 deg either way, from three turns
            dxy, dyaw_max = (0.15, np.radians(30)) if rig else (0.05, np.pi / 2)
            bnd = st.get("bounds") or {}
            dxy = bnd.get("xy", dxy)
            lo = np.array([max(fitb[k][0], p0[k] - bnd[k]) if k in bnd else fitb[k][0]
                           for k in fit_names] + [x - dxy, y - dxy, yaw0 - dyaw_max])
            hi = np.array([min(fitb[k][1], p0[k] + bnd[k]) if k in bnd else fitb[k][1]
                           for k in fit_names] + [x + dxy, y + dxy, yaw0 + dyaw_max])
            fixed = {k: v for k, v in p0.items() if k not in fit_names}
            f = f_all = Fit(z, frames, masks, windows, kind, fixed, fit_names, lo, hi, z0)  # f_all: every frame (the pieces)
            best = None
            turns = (0.0, np.radians(15), -np.radians(15)) if rig else (0.0, np.pi / 4, -np.pi / 4) if kind != "cylinder" else\
                (0.0, np.pi / 6, -np.pi / 6, np.pi / 3, -np.pi / 3)
            for dyaw in turns:
                v0 = np.array([p0[k] for k in fit_names] + [x, y, yaw0 + dyaw])
                q0 = np.clip((v0 - lo) / (hi - lo), 0.02, 0.98)
                r = minimize(f.cost, q0, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-5})
                r = minimize(f.cost, r.x, method="Nelder-Mead", options={"maxiter": 3000, "xatol": 1e-5, "fatol": 1e-6})
                if best is None or r.fun < best.fun:
                    best = r
            ious = f.ious(best.x)
            keep = ious >= OUTLIER
            n_all = len(frames)
            if keep.sum() >= 6 and not keep.all():  # SAM 3 took another thing there (a hand over it): fit again without
                frames, masks, windows = ([v for v, k in zip(lst, keep) if k] for lst in (frames, masks, windows))
                f = Fit(z, frames, masks, windows, kind, fixed, fit_names, lo, hi, z0)
                best = minimize(f.cost, best.x, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-5, "fatol": 1e-6})
                ious = f.ious(best.x)
            p, bx, by_, byaw = f.params(best.x)
            z0 = z0 + float(p.pop("lift", 0.0))  # where it stands, measured
            band_cover = None
            if kind == "box_frame":  # which sides have a bottom band: each side's band against the silhouettes
                p, band_cover = frame_bands(f_all, p, bx, by_, byaw, z0)
                print(f"[object_fit] {name}: bottom bands {p['bands']} (share of each band's pixels on the frame's "
                      f"silhouette, median: {band_cover})", flush=True)
                p, top_cover = frame_top(f_all, p, bx, by_, byaw, z0)
                band_cover = {**band_cover, "top": top_cover}
                print(f"[object_fit] {name}: height {100 * p['height']:.1f} cm, top rails "
                      f"{'kept' if p.get('rails', True) else 'not seen: none'} ({top_cover})", flush=True)
            parts = kit_outlines(kind, p, bx, by_, byaw, z0)
            if kind == "box_frame":
                ious = np.array([(f.render(k, parts) & m).sum() / max((f.render(k, parts) | m).sum(), 1)
                                 for k, m in enumerate(f.masks)])
            px = []  # its colour: the best-fitting half of the frames, inside both silhouettes, away from their edges
            for k in np.argsort(-ious)[:max(3, len(ious) // 2)]:
                i, (x0, y0, w, h, fs) = frames[k], windows[k]
                both = cv2.erode((masks[k] & f.render(k, parts)).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
                win = image(i)[int(y0):int(y0 + h), int(x0):int(x0 + w)]
                px.append(cv2.resize(win, both.shape[::-1], interpolation=cv2.INTER_AREA)[both] if fs != 1 else win[both])
            alb, clipped = frame_color.albedo(np.concatenate(px))
            if "color" not in fixed:
                p["color"] = alb.round(3).tolist()
            p = {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v) for k, v in p.items()}
            res[name] = {"kit": kind, "params": p, "xy": [round(float(bx), 4), round(float(by_), 4)],
                         "yaw_deg": round(float(np.degrees(byaw)), 1), "support_z": round(z0, 4), "rig": rig,
                         "iou_median": round(float(np.median(ious)), 3), "iou_trimmed": round(1 - best.fun, 3),
                         **({"band_cover": band_cover} if band_cover else {}),
                         # the silhouette's share the kit leaves out (a missing band, handle or plate shows here, where
                         # the IoU barely moves)
                         "unexplained_median": round(float(np.median([(m & ~f.render(k, parts)).sum() / max(m.sum(), 1)
                                                                      for k, m in enumerate(f.masks)])), 3),
                         "frames": [names[i] for i in frames], "frames_left_out": n_all - len(frames), "prompt": prompt}
            print(f"[object_fit] {name} = {kind}: IoU median {np.median(ious):.2f} over {len(frames)} frames "
                  f"({n_all - len(frames)} left out); xy ({bx:.3f}, {by_:.3f}) yaw {np.degrees(byaw):.0f}; "
                  + ", ".join(f"{k} {100 * v:.1f}" for k, v in p.items() if k in fit_names) + " cm; colour (linear) "
                  f"{p.get('color')}" + (" (clipped: as bright as white)" if clipped.any() else ""), flush=True)
            order = np.argsort(-ious)  # the best four, the middle four, the worst four
            mid = len(order) // 2
            show = list(dict.fromkeys(list(order[:4]) + list(order[max(0, mid - 2):mid + 2]) + list(order[::-1][:4])))
            tiles = []
            for k in show:  # zoomed on the thing: 1.6 times its extent, at TILE px
                i, (x0, y0, w, h, fs) = frames[k], windows[k]
                ren = f.render(k, parts)
                win = image(i)[int(y0):int(y0 + h), int(x0):int(x0 + w)]
                win = cv2.resize(win, ren.shape[::-1], interpolation=cv2.INTER_AREA) if fs != 1 else win
                ys, xs = np.nonzero(ren | masks[k])
                cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
                hh = int(np.clip(0.8 * max(np.ptp(xs), np.ptp(ys)), 24, min(ren.shape) // 2))
                a0 = int(np.clip(cx - hh, 0, ren.shape[1] - 2 * hh))
                b0 = int(np.clip(cy - hh, 0, ren.shape[0] - 2 * hh))
                sl = np.s_[b0:b0 + 2 * hh, a0:a0 + 2 * hh]
                crop = cv2.resize(win[sl], (TILE, TILE), interpolation=cv2.INTER_CUBIC)
                for msk, col in ((masks[k], (255, 0, 0)), (ren, (0, 255, 0))):
                    big = cv2.resize(msk[sl].astype(np.uint8), (TILE, TILE), interpolation=cv2.INTER_NEAREST)
                    cnt, _ = cv2.findContours(big, cv2.RETR_EXTERNAL if not rig else cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(crop, cnt, -1, col, 1)
                cv2.putText(crop, f"{names[i]} IoU {ious[k]:.2f}", (4, 14), 0, 0.42, (255, 255, 0), 1)
                tiles.append(crop)
            while len(tiles) % 4:
                tiles.append(np.zeros_like(tiles[0]))
            sheet = np.vstack([np.hstack(tiles[j:j + 4]) for j in range(0, len(tiles), 4)])
            cv2.imwrite(str(out_qa / f"{name}.jpg"), sheet[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])

    run({})
    for n1, (n2, ca, cb) in list(swaps.items()):  # the two instances to the two objects: the kits decide
        if n1 not in res or n2 not in res:
            continue
        kept_ = res[n1]["iou_median"] + res[n2]["iou_median"]
        run({n1: cb[:2], n2: ca[:2]})
        swapped = res[n1]["iou_median"] + res[n2]["iou_median"]
        print(f"[object_fit] {n1} / {n2}: the kits fit {kept_:.2f} as found, {swapped:.2f} swapped", flush=True)
        if swapped < kept_:
            run({n1: ca[:2], n2: cb[:2]})
    # what no frame sees: the goal's hole fits the target (a peg-in-hole set's does), unless set
    goal, target = sg["task"].get("goal"), sg["task"].get("target")
    for name, r in res.items():
        if r.get("route") == "blender" or "hole_diameter" not in r["params"]:
            continue
        p = r["params"]
        t = res.get(target) or {}
        if name == goal and t.get("kit") == "cylinder":
            p["hole_diameter"] = round(t["params"]["diameter"] + a.clearance, 4)
            r["hole_from"] = f"the task's target {target}: its diameter + {1000 * a.clearance:.1f} mm"
        p["hole_depth"] = round(a.depth * p["post_height"], 4)
        p.update(sets.get(name, {}))
        print(f"[object_fit] {name}: hole {100 * p['hole_diameter']:.2f} cm across, {100 * p['hole_depth']:.1f} cm "
              f"deep ({r.get('hole_from', 'as given')})", flush=True)
    for name, kv in sets.items():
        if name in res:
            res[name]["params"].update(kv)
    res_path.parent.mkdir(parents=True, exist_ok=True)
    res_path.write_text(json.dumps(res, indent=1) + "\n")
    if a.apply:
        cfg = s.config_dir / "scene_graph.json"
        g = json.loads(cfg.read_text())
        for name, kind, _, _, rig in todo:
            if name not in res or kind == "_model":
                continue
            r = res[name]
            note = (f"recon.object_fit: {r['kit']} fitted to SAM 3 silhouettes in {len(r['frames'])} frames (IoU median "
                    f"{r['iou_median']})")
            if rig:
                apply_rig(g, name, r, note)
                continue
            old = g["dynamic_objects"][name].get("fix") or {}
            g["dynamic_objects"][name]["fix"] = {**{k: v for k, v in old.items() if k in ("support_z",)},
                                                 "kit": r["kit"], "params": r["params"], "xy": r["xy"],
                                                 "support_z": r["support_z"],
                                                 "yaw_deg": r["yaw_deg"], "placement": placement(s)}
            g["dynamic_objects"][name]["_object_fit"] = note + "; the reconstruction replaced" + (
                f"; hole from {r['hole_from']}" if r.get("hole_from") else "")
        cfg.write_text(json.dumps(g, indent=1, ensure_ascii=False) + "\n")
        print(f"applied -> {cfg}", flush=True)
    print(f"done: {len(res)} fitted -> {res_path}, {out_qa}", flush=True)


def apply_rig(g, name, r, note):
    """A fitted rig's pieces as furniture boxes <name>_<piece> (the room's axes: the corners turned by its yaw, their
    bounds), steel in its measured colour; the rig's earlier boxes go."""
    k = kits.build(r["kit"], r["params"])
    yaw = np.radians(r["yaw_deg"])
    c, s_ = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s_], [s_, c]])
    for key in [k_ for k_ in g["furniture"] if k_.startswith(f"{name}_")]:
        del g["furniture"][key]
    col = r["params"].get("color", [0.03, 0.03, 0.035])
    for piece, (lo, hi) in k["boxes"].items():
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        corners = np.array([[x, y] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])]) @ R.T + r["xy"]
        blo = [*corners.min(0).round(4).tolist(), round(float(r["support_z"] + lo[2]), 4)]
        bhi = [*corners.max(0).round(4).tolist(), round(float(r["support_z"] + hi[2]), 4)]
        g["furniture"][f"{name}_{piece}"] = {"role": "furniture", "box": [blo, bhi], "material": "steel",
                                             "look": {"all": {"color": col, "roughness": 0.45}}, "_object_fit": note}


if __name__ == "__main__":
    main()
