"""Draft scene graph from perception: geometry measured automatically, judgment calls left to review.

    python -m calibrated_scene_reconstruction.recon.autograph sessions/<name> [--keep-frame] [--out scene_graph.auto.json]

Reads parsing/{geometry.npz, masks.npz, instances.json}; writes configs/<name>/<out> (never scene_graph.json itself),
parsing/autograph.json (per-instance measurements) and parsing/qa_autograph.png (the draft projected on the photo).
Unless --keep-frame (or parsing/world_frame.json exists), the world frame is written with the room's yaw: walls end
up axis-aligned instead of following the camera's x axis.

What is automatic: floor / ceiling / wall planes (RANSAC on vertical planes, merged per side, missing sides closed at
the observed extent), windows on their wall, support slabs (horizontal top of table / counter masks, split into
connected pieces), furniture boxes (robust extents; floor-standing pieces down to z = 0; neighbouring same-label pieces
merged), the box faces each instance textures, sinks as slab cut-outs, objects nested in containers (fruit in a bowl)
or detected under two prompts (one mask as "cup" and "glass": one object, the surer label),
dynamic vs static objects (small, resting on a slab), movable furniture groups, and a task (graspable target on a
slab, near an edge with free floor).
What needs review (the overlay makes it quick): mislabelled instances (a counter's front masked as "wall"), which slab
is the task's support, templates for thin metal fixtures, masses, swap fallbacks.
"""
import argparse
import json
import re

import numpy as np

from calibrated_scene_reconstruction.assets.physprops import mass_from_box
from calibrated_scene_reconstruction.scene.session import Session

STRUCTURE = {"floor": "floor", "ceiling": "ceiling", "wall": "wall", "window": "window", "window blind": "window"}
SUPPORT = re.compile(r"\b(countertop|counter|(?:dining |coffee |side |cafe )?table|desk|island|workbench|lab bench|"
                     r"vanity top|work surface)\b(?! lamp)")
BOX = re.compile(r"cabinet|refrigerator|fridge|dresser|nightstand|drawer|vanity|bookcase|bookshelf|shelf|oven|"
                 r"dishwasher|wardrobe|sideboard|cupboard|bed|sofa|couch")
MOVABLE = {"bar stool": "stool", "stool": "stool", "chair": "chair", "armchair": "chair", "bench": "bench"}
SLAB_PARTS = re.compile(r"cooktop|stove|hob")
SINK = re.compile(r"sink|basin")
STATIC_HINT = re.compile(r"plant|picture|frame|painting|lamp|faucet|tap|mirror|blind|curtain|tv|monitor|laptop|towel|"
                         r"toilet|range hood")
IGNORE = re.compile(r"^(door|rug|carpet|curtain|ceiling light|pendant lamp|light)$")  # part of the room's texture
GRASP = re.compile(r"cup|mug|bottle|can\b|bowl|apple|orange|lemon|banana|pear|fruit|box|glass|jar|spray|soap|"
                   r"pitcher|vase|candle|clock")
CONTAINER = re.compile(r"bowl|plate|tray|basket")  # nested items join these, unless the container is too big
SOFT = re.compile(r"sofa|couch|bed")
ROBOT = re.compile(r"\brobot\b")  # the real robot in the photo: removed from the plate, the simulated one stands there


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def floor_plane(points, floor_mask):
    """The floor's plane (unit normal, a point on it): the mask eroded by 7 px (its rim takes flying pixels between
    the floor and what stands on it), fitted twice, the second time on the points within 3 cm of the first plane
."""
    import cv2

    m = cv2.erode(floor_mask.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    return plane_robust(points[m if m.sum() > 2000 else floor_mask])


def plane_robust(P, thr=0.03):
    """A plane fitted to points, then again to those within `thr` of it: (unit normal, a point on it)."""
    for _ in range(2):
        c = P.mean(0)
        _, V = np.linalg.eigh(np.cov((P - c).T))
        n = V[:, 0]
        near = np.abs((P - c) @ n) < thr
        P = P[near] if near.sum() > 500 else P
    return n, P.mean(0)


def floor_frame(points, floor_mask):
    """Session.world_frame's floor fit: z up, x = camera right on the floor, origin under the camera."""
    n, c = floor_plane(points, floor_mask)
    if n[1] > 0:
        n = -n
    d = -n @ c
    x = np.array([1.0, 0, 0]) - n[0] * n
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(n, x), n]), -d * n, float(d)


def vertical_planes(P, rng, max_planes=2, thr=0.03, iters=300, min_frac=0.25):
    """RANSAC vertical planes (normal in xy) on world points; returns [(normal_xy, offset, inlier mask)]."""
    out, rest = [], np.ones(len(P), bool)
    for _ in range(max_planes):
        idx = np.flatnonzero(rest)
        if len(idx) < 200:
            break
        best = None
        for _ in range(iters):
            a, b = P[rng.choice(idx, 2, replace=False), :2]
            t = b - a
            if np.linalg.norm(t) < 0.1:
                continue
            n = np.array([-t[1], t[0]]) / np.linalg.norm(t)
            inl = np.abs(P[idx, :2] @ n - n @ a) < thr
            if best is None or inl.sum() > best[2].sum():
                best = (n, float(n @ a), inl)
        if best is None or best[2].sum() < min_frac * len(P):
            break
        n, _, inl = best
        m = np.zeros(len(P), bool)
        m[idx[inl]] = True
        c = P[m, :2].mean(0)  # refit on inliers (3x3 / 2x2 covariance only)
        _, V = np.linalg.eigh(np.cov((P[m, :2] - c).T))
        n = V[:, 0]
        out.append((n, float(n @ c), m))
        rest &= ~m
    return out


def room_yaw(planes):
    """Dominant wall direction modulo 90 degrees (inlier-weighted circular mean of 4 theta)."""
    if not planes:
        return 0.0
    th = np.array([np.arctan2(n[1], n[0]) for n, _, _, w in planes])
    w = np.array([w for *_, w in planes], float)
    return float(np.arctan2((w * np.sin(4 * th)).sum(), (w * np.cos(4 * th)).sum()) / 4)


def pct_box(P, lo=2, hi=98):
    return np.percentile(P, lo, axis=0), np.percentile(P, hi, axis=0)


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def build(s: Session, keep_frame=False, seed=0, max_objects=None):
    rng = np.random.default_rng(seed)
    geo, masks = s.geometry(), s.masks()
    inst = json.loads((s / "parsing" / "instances.json").read_text())
    valid = geo["mask"]
    labels = [i["label"] for i in inst]
    clutter = set(s.config("perception.json", {}).get("clutter", []))  # labels of the capture rig: cameras, tripods,
    # cables -- neither reconstructed nor left in the plate (recon.inpaint removes them with the objects)
    floor_ids = [i for i, lab in enumerate(labels) if lab == "floor"]
    sup_ids = [i for i, lab in enumerate(labels) if SUPPORT.search(lab)]
    # a "floor" that is a table's or desk's mask is its top: a photo looking down at a desk that fills it named the
    # desk floor too, and the room's floor stood at the desk's height
    floor_ids = [i for i in floor_ids if not any(
        (masks[i] & masks[j]).sum() > 0.5 * min(masks[i].sum(), masks[j].sum()) for j in sup_ids)] or floor_ids
    if not floor_ids:
        from calibrated_scene_reconstruction.scene.capture.draft_floor import verify_active
        try:
            verify_active(s)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise SystemExit('No primary floor; prepare and review an observed auxiliary floor: '+str(exc)) from exc

    frame_path = s / "parsing" / "world_frame.json"
    if frame_path.exists() or keep_frame:
        R_wc, o, _ = s.world_frame()
        yaw_applied = None
    else:
        fm = np.zeros(masks.shape[1:], bool)
        for i in floor_ids:
            fm |= masks[i]
        R_wc, o, cam_h = floor_frame(geo["points"], fm & valid)
        yaw_applied = 0.0
        mv = s / "parsing" / "multiview.json"
        rp = json.loads(mv.read_text()).get("robot_pose") if mv.exists() else None
        if rp and rp.get("residual_m", 1.0) < 0.03:
            # the floor's height where the robot stands on it (recon.robot_scale: its wheels' lowest point), its tilt
            # from the floor's points: the floor the photo sees lies metres off, where the depth errs most
            b = np.asarray(rp["base_in_map"], float) * json.loads(mv.read_text())["scale"]["value"]
            h = float(R_wc[2] @ (b - o))
            o, cam_h = o + h * R_wc[2], cam_h - h
            print(f"[frame] the floor at the robot's wheels: {h:+.3f} m from the photo's floor plane", flush=True)

    import cv2

    vs = s / "parsing" / "multiview" / "support_points.npz"
    view_support = dict(np.load(vs)) if vs.exists() else {}

    def world(i, cap=40000):
        """World points of an instance without flying pixels: mask eroded by 2 px (depth edges interpolate between
        foreground and background), then depth within 3 MAD (at least 4 cm) of the instance's median."""
        m = masks[i] & valid
        if m.sum() > 400:
            m = cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & valid
        P = geo["points"][m]
        if len(P) > 20:
            z = P[:, 2]
            med = np.median(z)
            mad = max(1.4826 * np.median(np.abs(z - med)), 0.04 if STRUCTURE.get(labels[i]) or
                      SUPPORT.search(labels[i]) or BOX.search(labels[i]) else 0.02)
            P = P[np.abs(z - med) < 3 * mad] if not STRUCTURE.get(labels[i]) else P
        if str(i) in view_support:  # the support as more photos see it (recon.multiview): its far part too
            P = np.concatenate([P, view_support[str(i)]])
        if len(P) > cap:
            P = P[rng.choice(len(P), cap, replace=False)]
        return (P - o) @ R_wc.T

    pts = {i: world(i) for i in range(len(inst))}

    # --- walls: vertical planes -> room yaw
    wall_ids = [i for i, lab in enumerate(labels) if STRUCTURE.get(lab) == "wall"]
    raw_planes = []
    for i in wall_ids:
        for n, d, m in vertical_planes(pts[i], rng):
            raw_planes.append((n, d, (i, m), int(m.sum())))
    yaw = room_yaw(raw_planes)
    if yaw_applied is not None:
        Rz = _rz(-yaw)
        R_wc = Rz @ R_wc
        pts = {i: P @ Rz.T for i, P in pts.items()}
        yaw_applied = yaw
        T = np.eye(4)
        T[:3, :3] = R_wc
        T[:3, 3] = -R_wc @ o
        frame_path.write_text(json.dumps({"R_wc": R_wc.tolist(), "o": o.tolist(), "T_world_cam_cv": T.tolist(),
                                          "camera_height_m": cam_h, "room_yaw_deg": float(np.degrees(yaw))},
                                         indent=1))
    cam = -R_wc @ o  # camera centre in world (x, y ~ 0)

    allP = np.concatenate([pts[i] for i in range(len(inst)) if len(pts[i])])
    ceil_ids = [i for i, lab in enumerate(labels) if lab == "ceiling"]
    ceiling_z = float(np.median(np.concatenate([pts[i][:, 2] for i in ceil_ids]))) if ceil_ids else \
        float(max(2.4, np.percentile(allP[:, 2], 99.5) + 0.05))

    walls = []  # (axis, at, [lo, hi] other axis, ids)
    for i in wall_ids:
        for n, d, m in vertical_planes(pts[i], rng):
            P = pts[i][m]
            ax = int(np.argmax(np.abs(n)))
            at = float(np.median(P[:, ax]))
            lo, hi = np.percentile(P[:, 1 - ax], [1, 99])
            walls.append([ax, at, [float(lo), float(hi)], [i], len(P)])
    for w in walls:
        w.append(float(np.percentile(pts[w[3][0]][:, 2], 98)))  # top of the wall piece
    merged = []
    for w in sorted(walls, key=lambda w: -w[4]):
        for mw in merged:
            if mw[0] == w[0] and abs(mw[1] - w[1]) < 0.12:
                mw[2] = [min(mw[2][0], w[2][0]), max(mw[2][1], w[2][1])]
                mw[3] = sorted(set(mw[3] + w[3]))
                mw[1] = (mw[1] * mw[4] + w[1] * w[4]) / (mw[4] + w[4])
                mw[4] += w[4]
                break
        else:
            merged.append(list(w))
    merged = [w for w in merged if w[2][1] - w[2][0] >= 0.5]  # narrower pieces: pillar faces, wall fragments
    fronts = []  # (axis, at, ids): low vertical pieces with a real wall further out on the same side
    for w in list(merged):
        side = w[1] > cam[w[0]]
        behind = [v for v in merged if v is not w and v[0] == w[0] and (v[1] > cam[v[0]]) == side and
                  abs(v[1] - cam[v[0]]) > abs(w[1] - cam[w[0]]) + 0.3]
        if w[5] < 1.2 and behind:
            merged.remove(w)
            fronts.append((w[0], w[1], w[3]))

    # floor extent: walls bound it where present, observed points elsewhere
    ext = [[float(np.percentile(allP[:, a], 1)) - 0.3, float(np.percentile(allP[:, a], 99)) + 0.3] for a in (0, 1)]
    ext[1][0] = min(ext[1][0], cam[1] - 0.8)  # the room continues behind the camera
    for a_ in (0, 1):  # and the photo was taken inside it
        ext[a_] = [min(ext[a_][0], cam[a_] - 0.5), max(ext[a_][1], cam[a_] + 0.5)]
    for ax in (0, 1):
        for side in (0, 1):
            c = [w[1] for w in merged if w[0] == ax and (w[1] > cam[ax]) == bool(side)]
            if c:
                ext[ax][side] = max(c) if side else min(c)
    structure, names = {}, {(0, 0): "left", (0, 1): "right", (1, 0): "front", (1, 1): "back"}
    used = {}
    for ax, at, rng_o, ids, *_ in sorted(merged, key=lambda w: abs(w[1] - cam[w[0]])):
        side = int(at > cam[ax])
        base = f"wall_{names[(ax, side)]}"
        k = used.get(base, 0)
        used[base] = k + 1
        name = base if k == 0 else f"{base}_{k + 1}"
        lo, hi = max(rng_o[0] - 0.1, ext[1 - ax][0]), min(rng_o[1] + 0.1, ext[1 - ax][1])
        structure[name] = {"role": "wall", "plane": "xy"[ax], "at": round(at, 3), "range": [[round(lo, 3), round(hi, 3)],
                           [0.0, round(ceiling_z, 3)]], "ids": ids, "normal": "-" if side else "+"}
    for (ax, side), nm in names.items():  # close the room where no wall was seen
        if not any(v["plane"] == "xy"[ax] and (v["at"] > cam[ax]) == bool(side) for v in structure.values()):
            structure[f"wall_{nm}"] = {"role": "wall", "plane": "xy"[ax], "at": round(ext[ax][side], 3),
                                       "range": [[round(ext[1 - ax][0], 3), round(ext[1 - ax][1], 3)],
                                                 [0.0, round(ceiling_z, 3)]], "ids": [], "normal": "-" if side else "+"}
    structure["floor"] = {"role": "floor", "plane": "z", "at": 0.0, "range": [[round(v, 3) for v in ext[0]],
                                                                             [round(v, 3) for v in ext[1]]],
                          "ids": floor_ids, "collide": True}
    structure["ceiling"] = {"role": "ceiling", "plane": "z", "at": round(ceiling_z, 3), "range": structure["floor"]["range"],
                            "ids": ["*"] if ceil_ids else [], "normal": "-"}

    # windows: on the nearest wall plane, slightly in front of it
    glass = np.zeros(masks.shape[1:], bool)  # windows reflected in a picture's glass or a mirror are not windows
    for i, lab in enumerate(labels):
        if re.search("picture|frame|painting|mirror", lab):
            glass |= masks[i]
    win_ids = [i for i, lab in enumerate(labels) if STRUCTURE.get(lab) == "window" and len(pts[i]) > 50 and
               (masks[i] & glass).sum() < 0.5 * masks[i].sum()]
    wins = []
    for i in win_ids:
        P = pts[i]
        best = min((v for v in structure.values() if v.get("role") == "wall"),
                   key=lambda v: np.median(np.abs(P[:, "xy".index(v["plane"])] - v["at"])))
        ax = "xy".index(best["plane"])
        lo, hi = pct_box(P)
        sgn = 1 if best["normal"] == "+" else -1
        r = [[float(lo[1 - ax]), float(hi[1 - ax])], [float(lo[2]), float(hi[2])]]
        for w in wins:  # blinds / sashes of the same window
            if w["plane"] == best["plane"] and abs(w["at"] - (best["at"] + 0.008 * sgn)) < 0.01 and \
                    _overlap(w["range"][0][0] - 0.15, w["range"][0][1] + 0.15, *r[0]) > 0 and \
                    _overlap(w["range"][1][0] - 0.15, w["range"][1][1] + 0.15, *r[1]) > 0:
                w["range"] = [[min(w["range"][0][0], r[0][0]), max(w["range"][0][1], r[0][1])],
                              [min(w["range"][1][0], r[1][0]), max(w["range"][1][1], r[1][1])]]
                w["ids"].append(i)
                break
        else:
            wins.append({"role": "window", "plane": best["plane"], "at": best["at"] + 0.008 * sgn, "range": r,
                         "ids": [i], "normal": best["normal"], "emissive": 0.6})
    changed = True
    while changed:  # merging in id order can miss pieces a later window grew over
        changed = False
        for a_ in wins:
            for b_ in wins:
                if a_ is not b_ and a_["plane"] == b_["plane"] and abs(a_["at"] - b_["at"]) < 0.01 and \
                        all(_overlap(a_["range"][k][0] - 0.15, a_["range"][k][1] + 0.15, *b_["range"][k]) > 0
                            for k in (0, 1)):
                    a_["range"] = [[min(a_["range"][k][0], b_["range"][k][0]), max(a_["range"][k][1], b_["range"][k][1])]
                                   for k in (0, 1)]
                    a_["ids"] = sorted(set(a_["ids"] + b_["ids"]))
                    wins.remove(b_)
                    changed = True
                    break
            if changed:
                break
    wins = [w for w in wins if (w["range"][0][1] - w["range"][0][0]) * (w["range"][1][1] - w["range"][1][0]) > 0.05]
    for k, w in enumerate(wins):
        w["at"] = round(w["at"], 3)
        w["range"] = [[round(a, 3) for a in r] for r in w["range"]]
        structure[f"window_{k + 1}"] = w

    # --- support slabs: horizontal tops of counter / table masks, split into connected pieces
    furniture, slab_of = {}, {}
    for i, lab in enumerate(labels):
        if not SUPPORT.search(lab) or len(pts[i]) < 200:
            continue
        P = pts[i]
        # Search broadly before stage S1 anchors metric scale.
        hist, edges = np.histogram(P[:, 2], bins=np.arange(0.2, 2.5, 0.01))
        z_top = float(edges[np.argmax(hist)] + 0.01)
        top = P[np.abs(P[:, 2] - z_top) < 0.03]
        cells = {tuple(c) for c in np.floor(top[:, :2] / 0.04).astype(int)}
        comps, seen = [], set()
        for c in cells:  # 8-connected components on a 4 cm grid
            if c in seen:
                continue
            stack, comp = [c], []
            seen.add(c)
            while stack:
                u = stack.pop()
                comp.append(u)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        v = (u[0] + dx, u[1] + dy)
                        if v in cells and v not in seen:
                            seen.add(v)
                            stack.append(v)
            comps.append(np.array(comp))
        for comp in comps:
            if len(comp) * 0.0016 < 0.05:
                continue
            sel = np.isin(np.floor(top[:, :2] / 0.04).astype(int).view([("a", int), ("b", int)]).ravel(),
                          np.asarray(comp).view([("a", int), ("b", int)]).ravel())
            lo, hi = pct_box(top[sel], 1, 99)
            name = f"{re.sub(r'[^a-z]+', '_', lab).strip('_')}_{sum(k.startswith(lab[:4]) for k in furniture) + 1}"
            furniture[name] = {"role": "support_top", "box": [[float(lo[0]), float(lo[1]), z_top - 0.04],
                                                              [float(hi[0]), float(hi[1]), z_top]],
                               "faces": {"top": [i]}, "_label": lab}
            slab_of.setdefault(i, []).append(name)

    for name in list(furniture):  # pieces of one surface (overlapping footprints, same height) -> one slab
        if name not in furniture or furniture[name].get("role") != "support_top":
            continue
        a = np.asarray(furniture[name]["box"])
        for other in list(furniture):
            if other == name or furniture[other].get("role") != "support_top":
                continue
            b = np.asarray(furniture[other]["box"])
            ov = _overlap(a[0, 0], a[1, 0], b[0, 0], b[1, 0]) * _overlap(a[0, 1], a[1, 1], b[0, 1], b[1, 1])
            small = min(np.prod(a[1, :2] - a[0, :2]), np.prod(b[1, :2] - b[0, :2]))
            if ov > 0.6 * small and abs(a[1, 2] - b[1, 2]) < 0.03:
                a = np.stack([np.minimum(a[0], b[0]), np.maximum(a[1], b[1])])
                furniture[name]["box"] = a.tolist()
                furniture[name]["faces"]["top"] = sorted(set(furniture[name]["faces"]["top"] +
                                                             furniture[other]["faces"]["top"]))
                del furniture[other]
    slab_names = [n for n, f in furniture.items() if f.get("role") == "support_top"]
    for k, n in enumerate(slab_names):  # renumber
        f = furniture.pop(n)
        furniture[f"{re.sub(r'[^a-z]+', '_', f['_label']).strip('_')}_{k + 1}"] = f

    # --- furniture boxes (cabinets, fridges...), neighbouring same-label pieces merged
    pieces = []
    for i, lab in enumerate(labels):
        if BOX.search(lab) and len(pts[i]) > 200:
            lo, hi = pct_box(pts[i])
            if lo[2] < 0.35:
                lo[2] = 0.0
            pieces.append([lab, lo, hi, [i]])
    boxes = []
    for p in sorted(pieces, key=lambda p: p[1][2]):
        for b in boxes:
            near = all(_overlap(b[1][a] - 0.06, b[2][a] + 0.06, p[1][a], p[2][a]) > 0 for a in (0, 1))
            zov = _overlap(b[1][2], b[2][2], p[1][2], p[2][2]) / max(1e-3, min(b[2][2] - b[1][2], p[2][2] - p[1][2]))
            if b[0] == p[0] and near and zov > 0.5:
                b[1], b[2], b[3] = np.minimum(b[1], p[1]), np.maximum(b[2], p[2]), b[3] + p[3]
                break
        else:
            boxes.append(list(p))
    dedup = []  # one piece detected under several labels (cabinet / nightstand / dresser): keep the larger
    for p_ in sorted(boxes, key=lambda p_: -np.prod(np.maximum(p_[2] - p_[1], 1e-3))):
        for q in dedup:
            inter = np.prod([_overlap(p_[1][a], p_[2][a], q[1][a], q[2][a]) for a in range(3)])
            if inter > 0.6 * min(np.prod(np.maximum(p_[2] - p_[1], 1e-3)), np.prod(np.maximum(q[2] - q[1], 1e-3))):
                q[3] = q[3] + p_[3]
                break
        else:
            dedup.append(p_)
    boxes = dedup
    wall_planes = [(ax, at) for ax, at, *_ in merged]
    under_slab = {}
    for lab, lo, hi, ids in boxes:
        for name, slab in furniture.items():
            sl = np.asarray(slab["box"])
            if slab.get("role") == "support_top" and lo[2] < sl[0, 2] and sl[0, 2] - hi[2] < 0.12 and \
                    _overlap(lo[0], hi[0], sl[0, 0], sl[1, 0]) * _overlap(lo[1] - 0.1, hi[1] + 0.1, sl[0, 1], sl[1, 1]) > 0:
                under_slab.setdefault(name, []).extend(ids)
                break
    boxes = [b for b in boxes if not any(set(b[3]) & set(v) for v in under_slab.values())]
    for lab, lo, hi, ids in boxes:  # only the front of a cabinet is seen: its back is the wall behind it
        for ax in (0, 1):
            far = 1 if (lo[ax] + hi[ax]) / 2 > cam[ax] else 0
            cands = [at for a, at in wall_planes if a == ax and (at > cam[ax]) == bool(far) and
                     0 <= (at - hi[ax] if far else lo[ax] - at) < 0.8]
            if cands and (hi[1 - ax] - lo[1 - ax]) > (hi[ax] - lo[ax]):
                if far:
                    hi[ax] = min(cands)
                else:
                    lo[ax] = max(cands)
    for lab, lo, hi, ids in boxes:
        kind = "upper" if lo[2] > 1.1 else "box"
        base = re.sub(r"[^a-z]+", "_", lab).strip("_") + ("_upper" if kind == "upper" else "")
        k = sum(n.startswith(base) for n in furniture) + 1
        for name, slab in furniture.items():  # a floor box right under a slab grows up to it
            sl = np.asarray(slab["box"])
            if slab.get("role") == "support_top" and lo[2] == 0 and 0 < sl[0, 2] - hi[2] < 0.1 and \
                    _overlap(lo[0], hi[0], sl[0, 0], sl[1, 0]) > 0.5 * (hi[0] - lo[0]):
                hi[2] = sl[0, 2]
        furniture[f"{base}_{k}"] = {"role": "cabinet" if "cabinet" in lab or "drawer" in lab else "appliance"
                                    if re.search("fridge|refrigerator|oven|dishwasher", lab) else "furniture",
                                    "box": [[float(v) for v in lo], [float(v) for v in hi]], "faces": {"front": ids},
                                    "_label": lab}
        if re.search("refrigerator|fridge|oven|dishwasher", lab):
            furniture[f"{base}_{k}"]["material"] = "steel"

    # support bodies: the floor box under each slab, else (counters) a solid block, (tables) legs only
    for name, f in list(furniture.items()):
        if f.get("role") != "support_top":
            continue
        sl = np.asarray(f["box"])
        under = [n for n, g in furniture.items() if g.get("role") != "support_top" and g["box"][0][2] == 0 and
                 _overlap(sl[0, 0], sl[1, 0], g["box"][0][0], g["box"][1][0]) *
                 _overlap(sl[0, 1], sl[1, 1], g["box"][0][1], g["box"][1][1]) >
                 0.3 * (sl[1, 0] - sl[0, 0]) * (sl[1, 1] - sl[0, 1])]
        fronts_here = [i for ax, at, ids in fronts for i in ids  # a recessed front (knee room) under this slab
                       if sl[0, ax] - 0.1 < at < sl[1, ax] + 0.1] + under_slab.get(name, [])
        if under:
            f["body"] = under[0]
            furniture[under[0]]["faces"]["front"] = sorted(set(furniture[under[0]]["faces"].get("front", []) +
                                                               fronts_here))
        elif re.search("counter|island|vanity", f["_label"]):
            inset = 0.03
            body = f"{name}_body"
            furniture[body] = {"role": "support_body", "box": [[sl[0, 0] + inset, sl[0, 1] + inset, 0.0],
                                                               [sl[1, 0] - inset, sl[1, 1] - inset, sl[0, 2]]],
                               "faces": {"front": sorted(set(fronts_here))} if fronts_here else {},
                               "smooth_front": 25, "_label": "counter body (inferred)"}
            f["body"] = body

    for name, f in list(furniture.items()):  # pieces seen inside a support's body (cabinet doors of an island)
        if f.get("role") not in ("cabinet", "furniture"):
            continue
        b = np.asarray(f["box"])
        vol = np.prod(np.maximum(b[1] - b[0], 1e-3))
        for bname, g in furniture.items():
            if g.get("role") != "support_body" or bname == name:
                continue
            c = np.asarray(g["box"])
            inter = np.prod([_overlap(b[0, a], b[1, a], c[0, a], c[1, a]) for a in range(3)])
            if inter > 0.5 * vol:
                g["faces"]["front"] = sorted(set(g["faces"].get("front", []) + f["faces"].get("front", [])))
                del furniture[name]
                break

    # box faces: which instances texture which face (points within 4 cm of the face, inside the box footprint)
    FACES = {"left": (0, 0), "right": (0, 1), "front": (1, 0), "back": (1, 1), "bottom": (2, 0), "top": (2, 1)}
    for name, f in furniture.items():
        b = np.asarray(f["box"])
        faces = {}
        for i, P in pts.items():
            furn_like = SUPPORT.search(labels[i]) or BOX.search(labels[i]) or SINK.search(labels[i]) or \
                SLAB_PARTS.search(labels[i]) or any(i in ids for *_, ids in fronts)
            if not furn_like or len(P) < 50:
                continue
            inside = np.all((P > b[0] - 0.04) & (P < b[1] + 0.04), axis=1)
            if inside.mean() < 0.3:
                continue
            Q = P[inside]
            for face, (ax, side) in FACES.items():
                near = np.abs(Q[:, ax] - b[side, ax]) < 0.04
                if near.mean() > 0.25 and (face != "back" and face != "bottom" or labels[i] == "kitchen cabinet"):
                    faces.setdefault(face, []).append(i)
        visible = {"front", "top"} | ({"left"} if b[0, 0] > cam[0] else {"right"}) | \
                  ({"bottom"} if b[0, 2] > cam[2] else set())
        keep = {k: v for k, v in f.get("faces", {}).items() if v}
        for k, v in faces.items():
            if k in visible:
                keep[k] = sorted(set(keep.get(k, [])) | set(v))
        f["faces"] = keep

    # sinks and cooktops belong to the slab they sit in
    for i, lab in enumerate(labels):
        if not (SINK.search(lab) or SLAB_PARTS.search(lab)) or len(pts[i]) < 50:
            continue
        lo, hi = pct_box(pts[i])
        for name, f in furniture.items():
            sl = np.asarray(f["box"])
            if f.get("role") == "support_top" and sl[0, 0] < (lo[0] + hi[0]) / 2 < sl[1, 0] and \
                    sl[0, 1] < (lo[1] + hi[1]) / 2 < sl[1, 1]:
                f["faces"].setdefault("top", []).append(i)
                if SINK.search(lab):
                    f["sink"] = {"xy": [[float(lo[0]), float(lo[1])], [float(hi[0]), float(hi[1])]], "depth": 0.2}
                break

    # --- objects: everything else; nested ones join their container
    obj_ids = [i for i, lab in enumerate(labels) if not (STRUCTURE.get(lab) or IGNORE.search(lab) or SUPPORT.search(lab)
                                                           or BOX.search(lab) or ROBOT.search(lab) or lab in clutter
                                                           or MOVABLE.get(lab) or SINK.search(lab) or
                                                           SLAB_PARTS.search(lab)) and len(pts[i]) > 30]
    same = {}  # one thing detected under two prompts (a glass as "cup" and as "glass"): one object, the surer label
    for a in obj_ids:
        for b in obj_ids:
            if a < b and a not in same and b not in same and \
                    (masks[a] & masks[b]).sum() > 0.7 * (masks[a] | masks[b]).sum():
                keep, drop = (a, b) if inst[a].get("score", 0) >= inst[b].get("score", 0) else (b, a)
                same[drop] = keep
    obj_ids = [i for i in obj_ids if i not in same]

    def round_box(P):
        """Only an object's front is seen: assume it is at least as deep as it is wide (roughly round)."""
        lo, hi = pct_box(P)
        w = max(hi[0] - lo[0], hi[1] - lo[1])
        far = 1 if (lo[1] + hi[1]) / 2 > cam[1] else 0
        if far:
            hi[1] = max(hi[1], lo[1] + w)
        else:
            lo[1] = min(lo[1], hi[1] - w)
        return lo, hi

    bbox = {i: round_box(pts[i]) for i in obj_ids}
    parent = {}
    for i in obj_ids:
        ci = (bbox[i][0] + bbox[i][1]) / 2
        cands = [j for j in obj_ids if j != i and
                 np.prod(bbox[j][1][:2] - bbox[j][0][:2]) > 1.5 * np.prod(bbox[i][1][:2] - bbox[i][0][:2]) and
                 np.all(ci[:2] > bbox[j][0][:2] - 0.01) and np.all(ci[:2] < bbox[j][1][:2] + 0.01) and
                 bbox[i][0][2] < bbox[j][1][2] + 0.05 and CONTAINER.search(labels[j]) is not None and
                 max(bbox[j][1][:2] - bbox[j][0][:2]) < 0.28]
        if cands:
            parent[i] = min(cands, key=lambda j: np.prod(bbox[j][1] - bbox[j][0]))
    groups = {}
    for i in obj_ids:
        r = i
        while r in parent:
            r = parent[r]
        groups.setdefault(r, []).append(i)
    for drop, keep in same.items():
        while keep in same:  # a chain of duplicates: to the one kept at its end
            keep = same[keep]
        g = next((g for g in groups.values() if keep in g), None)
        if g is not None:
            g.append(drop)
        else:  # the one it duplicates is no object here: it stands alone
            groups.setdefault(drop, []).append(drop)
    slabs = {n: np.asarray(f["box"]) for n, f in furniture.items() if f.get("role") == "support_top"}
    # tops of floor-standing furniture at table height (nightstands, dressers, vanities) are supports too, once
    # something rests on them
    box_tops = {f"{n}_top": np.asarray(f["box"]) for n, f in furniture.items()
                if f.get("role") in ("furniture", "cabinet") and f["box"][0][2] < 0.05 and 0.35 <= f["box"][1][2] <= 1.15
                and not SOFT.search(f.get("_label", ""))
                and not any(g.get("body") == n for g in furniture.values())}
    for n, b in box_tops.items():
        slabs[n] = np.stack([[b[0, 0], b[0, 1], b[1, 2] - 0.03], b[1]])
    static, dynamic, name_count = {}, {}, {}
    for r, ids in groups.items():
        P = np.concatenate([pts[i] for i in ids])
        lo, hi = round_box(P)
        size = hi - lo
        c = (lo + hi) / 2  # resting on a slab: bottom at its top (or a little above: in a basket, on a coaster)
        on = [n for n, b in slabs.items() if -0.03 < lo[2] - b[1, 2] < 0.12 and b[0, 0] - 0.05 < c[0] < b[1, 0] + 0.05
              and b[0, 1] - 0.05 < c[1] < b[1, 1] + 0.05]
        base = re.sub(r"[^a-z]+", "_", labels[r]).strip("_")
        name_count[base] = name_count.get(base, 0) + 1
        name = base if name_count[base] == 1 else f"{base}_{name_count[base]}"
        entry = {"ids": sorted(ids), "_size": [round(float(v), 3) for v in size], "_on": on[0] if on else None,
                 "_label": labels[r]}
        if on and size.max() < 0.25 and not STATIC_HINT.search(labels[r]):
            entry["mass"] = round(mass_from_box(labels[r], size), 2)  # by class (assets.physprops)
            dynamic[name] = entry
        else:
            static[name] = entry

    used_tops = {d["_on"] for d in list(dynamic.values()) + list(static.values())} & set(box_tops)
    for n in used_tops:  # materialize: a 3 cm slab on top of the piece, which becomes its body
        body = n[:-len("_top")]
        sl = slabs[n]
        furniture[body]["box"][1][2] = float(sl[0, 2])
        furniture[body]["role"] = "support_body"
        furniture[n] = {"role": "support_top", "box": sl.tolist(), "faces": {"top": furniture[body]["faces"].get("top", [])},
                        "body": body, "_label": furniture[body]["_label"] + " top"}

    movable = {}
    for i, lab in enumerate(labels):
        cat = MOVABLE.get(lab)
        if cat and len(pts[i]) > 100:
            movable.setdefault(cat, []).append(i)
    movable_fur = {}
    for cat, ids in movable.items():
        h = {i: float(pct_box(pts[i])[1][2]) for i in ids}
        best = max(([j for j in ids if abs(h[j] - h[i]) < 0.1 * h[i]] for i in ids), key=len)
        movable_fur["stools" if cat == "stool" else f"{cat}s"] = {"ids": best, "shared": len(best) > 1,
                                                                  "category": cat, "min_iou": 0.5}
        for i in ids:
            if i not in best:
                static[f"{cat}_{i}"] = {"ids": [i], "fallback": {"category": cat, "min_iou": 0.6},
                                        "_label": labels[i], "_note": "height differs from the set: another piece"}

    # --- task: graspable target on a slab, near an edge with free floor
    from calibrated_scene_reconstruction.robots.taskframe import approach_edges

    sg = {"structure": structure, "furniture": {k: {kk: vv for kk, vv in v.items()} for k, v in furniture.items()},
          "static_objects": static, "dynamic_objects": dynamic, "movable_furniture": movable_fur}
    robot_ids = [i for i, lab in enumerate(labels) if ROBOT.search(lab)]
    if robot_ids:
        sg["photo_robot"] = {"ids": robot_ids, "_note": "the real robot in the photo (SAM 3 prompts naming a robot): "
                             "not reconstructed, removed from the plate by recon.inpaint; recon.multiview measures the "
                             "scale on its parts and records where it stands"}
        mv = s / "parsing" / "multiview.json"
        rb = json.loads(mv.read_text()).get("robot", {}).get("base") if mv.exists() else None
        if rb:  # where the real robot stands: apps/placements.py tries this base first
            org = R_wc @ (np.asarray(rb["origin_cam"]) - o)
            fw = R_wc @ np.asarray(rb["forward_cam"])
            sg["photo_robot"]["base"] = {"xy": [round(float(org[0]), 3), round(float(org[1]), 3)],
                                         "yaw_deg": round(float(np.degrees(np.arctan2(fw[1], fw[0]))), 1)}
    clutter_ids = [i for i, lab in enumerate(labels) if lab in clutter]
    if clutter_ids:
        sg["clutter"] = {"ids": clutter_ids, "labels": sorted(clutter),
                         "_note": "the capture rig in the photo (perception.json `clutter` labels): not reconstructed, "
                                  "removed from the plate by recon.inpaint"}
    best = None
    for name, d in dynamic.items():
        if not d["_on"]:
            continue
        sz = np.asarray(d["_size"])
        P = np.concatenate([pts[i] for i in d["ids"]])
        c = (np.percentile(P, 2, 0) + np.percentile(P, 98, 0)) / 2
        sg["task"] = {"support": d["_on"], "support_body": furniture[d["_on"]].get("body"), "target": name}
        e = approach_edges(sg, c)[0]
        if sz[2] < 0.04:  # coasters, plates: nothing to close a hand around
            continue
        score = e["score"] + (0.5 if GRASP.search(d["_label"]) else 0) - 2 * max(0, max(sz[:2]) - 0.12)
        if best is None or score > best[0]:
            best = (score, dict(sg["task"]))
    sg["task"] = best[1] if best else {}

    dropped = []
    if max_objects is not None and sg["task"]:  # reconstruct what matters for the task, the rest stays photo texture
        t = sg["task"]
        tc = np.median(np.concatenate([pts[i] for i in sg["dynamic_objects"][t["target"]]["ids"]]), 0)

        def prio(sec, name, d):
            c = np.median(np.concatenate([pts[i] for i in d["ids"]]), 0)
            return (name != t["target"], d.get("_on") != t["support"], sec != "dynamic_objects",
                    float(np.linalg.norm(c[:2] - tc[:2])))

        ranked = sorted(((prio(sec, k, v), sec, k) for sec in ("dynamic_objects", "static_objects")
                         for k, v in sg[sec].items()))
        for _, sec, k in ranked[max_objects:]:
            dropped.append(k)
            del sg[sec][k]
    report = {"room_yaw_deg": float(np.degrees(yaw)), "yaw_applied": yaw_applied is not None, "dropped_objects": dropped,
              "ceiling_z": ceiling_z, "walls_raw": len(walls), "walls": len(merged), "windows": len(wins),
              "slabs": list(slabs), "n_dynamic": len(dynamic), "n_static": len(static), "task": sg["task"]}
    from calibrated_scene_reconstruction.scene.objects.routes import routed_graph
    sg, _ = routed_graph(sg)
    return sg, report, pts


def qa_image(s, sg, out):
    """Draft projected on the photo: slabs green, boxes orange, wall / floor footprints blue, target red."""
    import cv2
    from PIL import Image

    img = np.asarray(Image.open(s / "capture" / "rgb.png").convert("RGB")).copy()
    K = s.geometry()["K"]
    R_wc, o, _ = s.world_frame()

    def proj(Pw):
        Pc = Pw @ R_wc + o
        ok = Pc[:, 2] > 0.05
        uv = (Pc @ K.T)[:, :2] / np.maximum(Pc[:, 2:3], 1e-6)
        return uv, ok

    def box_lines(b, color, label):
        lo, hi = np.asarray(b[0]), np.asarray(b[1])
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        uv, ok = proj(corners)
        for a, c in [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]:
            if ok[a] and ok[c]:
                cv2.line(img, tuple(int(v) for v in uv[a]), tuple(int(v) for v in uv[c]), color, 2)
        if ok.any():
            p = uv[ok].min(0)
            cv2.putText(img, label, (int(p[0]) + 3, int(p[1]) + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    for name, v in sg["structure"].items():
        if v["plane"] in "xy" and v.get("role") in ("wall", "window"):
            ax = "xy".index(v["plane"])
            lo, hi = [0, 0, v["range"][1][0]], [0, 0, v["range"][1][1]]
            lo[ax] = hi[ax] = v["at"]
            lo[1 - ax], hi[1 - ax] = v["range"][0]
            box_lines([lo, hi], (60, 120, 255) if v["role"] == "wall" else (120, 220, 255), name)
    for name, f in sg["furniture"].items():
        box_lines(f["box"], (60, 220, 60) if f.get("role") == "support_top" else (255, 150, 40), name)
    Image.fromarray(img).save(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--keep-frame", action="store_true")
    ap.add_argument("--out", default="scene_graph.auto.json")
    ap.add_argument("--max-objects", type=int, default=None,
                    help="keep the target, objects on its support, then the nearest; the rest stays photo texture")
    a = ap.parse_args()
    s = Session(a.session)
    sg, report, _ = build(s, a.keep_frame, max_objects=a.max_objects)
    s.config_dir.mkdir(parents=True, exist_ok=True)
    clean = json.loads(json.dumps(sg, default=lambda x: x.tolist() if hasattr(x, "tolist") else float(x)))
    (s.config_dir / a.out).write_text(json.dumps(clean, indent=1, ensure_ascii=False) + "\n")
    s.write_json("parsing/autograph.json", report)
    qa_image(s, clean, s / "parsing" / "qa_autograph.png")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
