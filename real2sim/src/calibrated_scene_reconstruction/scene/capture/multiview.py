"""S1a more photos of the scene: one metric frame for all of them, the scale from a calibration board or
the robot in the picture, object sizes seen from several sides.

    python -m calibrated_scene_reconstruction.recon.multiview sessions/<name> [--rerun-da3]

Runs after `perceive` and before `graph` when capture/views/ holds more photos of the scene (calibrated_scene_reconstruction.capture --views),
configs/<name>/multiview.json exists (the robot scale on the photo alone) or configs/<name>/charuco_board.json does
(a board lying in the scene, recon.board); otherwise it leaves the session as it is. The scale's sources, first
found first: a fixed `scale`, the board, the robot at its joint readout (configs/<name>/
robot_state.json, recon.robot_scale), the robot's parts; the others are reported as checks.

1. Depth Anything 3 (envs/da3, calibrated_scene_reconstruction.recon.da3_views) on the photo and the views: their camera poses and depth in one
   frame. Its depth of the photo is matched to MoGe-2's (the median ratio over confident pixels), so the views land in
   the photo camera's frame in the point map's units.
2. Scale: MoGe-2's metric depth is a guess from one image (a 10-20 % error moves a table top by centimetres and makes
   a cup a size too big). Parts of the real robot whose width is known (configs/robots/scale_refs/<robot>.json: the
   widths, left-right as the robot faces the camera, of its URDF link groups by forward kinematics -- a pitch joint
   does not change them) are segmented with SAM 3 in the photo (the first of a part's prompts that finds it) and
   measured: the mask's horizontal span per image row (90th percentile over the rows, so neither the mask's ragged
   rim nor a narrower part below it decides) at the part's median depth. The scale is the median of reference /
   measured. `configs/<name>/multiview.json` may name the robot (`robot`), its own parts (`parts`) or fix the scale
   (`scale`). parsing/geometry.npz is rescaled (points, depth); MoGe-2's own is kept as geometry.moge.npz and every
   run starts from it, so the stage can be re-run. DA3's own metric depth is reported beside it, as a check.
3. Objects: every graspable / container instance of the photo (autograph's GRASP / CONTAINER labels) is looked for in
   each view (SAM 3 with its label, the mask whose 3D centroid lies nearest) and its points from all views are fused:
   height above its support plane and footprint (horizontal principal extents), e.g. a cup's height from the front and
   its diameter from above.
4. Where the real robot stands: the plane of its base's front face (normal towards the camera, levelled: the robot's
   forward; `front_to_origin` back to the base frame), for autograph's photo_robot.base.

Writes parsing/multiview.json, parsing/multiview/{da3.npz, fused.ply, qa_multiview.jpg}.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image

from calibrated_scene_reconstruction.core.gpu import device
from calibrated_scene_reconstruction.core.paths import CONFIGS, ROOT
from calibrated_scene_reconstruction.scene.session import Session

from calibrated_scene_reconstruction.core.geometry import mask_iou as _mask_iou

POOR_FIT = 0.3  # the robot pose fit's share of the model within 2 cm under which the photo cannot take its scale

DA3_PY = ROOT / "envs" / "da3" / "bin" / "python"


def run_da3(s, rerun=False):
    out = s / "parsing" / "multiview" / "da3.npz"
    inputs = [s / "capture" / "rgb.png"] + sorted((s / "capture" / "views").glob("*.png"))
    if rerun or not out.exists() or out.stat().st_mtime < max(p.stat().st_mtime for p in inputs) or \
            "prep" not in np.load(out).files:
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        subprocess.run([str(DA3_PY), "-m", "calibrated_scene_reconstruction.scene.capture.da3_views", str(s.dir)], check=True, env=env)
    return dict(np.load(out))


def da3_geometry(da, W, H, fx=None):
    """The photo's DA3 depth as the session's geometry (DA3's units; perceive's layout: points, depth, mask, K at the
    capture size): the photo's depth as DA3 solved it jointly with the views, so the views agree with it to a few
    centimetres, where MoGe-2's one-image map differs from them by a drift that no one scalar removes. The prepared image was the capture scaled
    by `s` and cropped at (x0, y0) (never turned: the photo sets the orientation), so a capture pixel samples the
    depth at (u s - x0, v s - y0); the crop's few outer rows are left unmasked."""
    rot, sc, x0, y0 = da["prep"][0]
    assert rot == 0
    d, conf = da["depth"][0], da["conf"][0]
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    mu, mv_ = (u * sc - x0 - 0.5).astype(np.float32), (v * sc - y0 - 0.5).astype(np.float32)
    depth = cv2.remap(d, mu, mv_, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    c = cv2.remap(conf, mu, mv_, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    mask = (depth > 0) & np.isfinite(depth) & (c >= np.percentile(conf, 10))
    K = da["K"][0].astype(np.float64).copy()
    K[0, 0] /= sc
    K[1, 1] /= sc
    K[0, 2] = (K[0, 2] + x0) / sc
    K[1, 2] = (K[1, 2] + y0) / sc
    if fx:  # the calibrated focal length (configs/<name>/camera.json) in place of DA3's guess, its depth kept
        K[0, 0] = K[1, 1] = fx
        K[0, 2], K[1, 2] = W / 2, H / 2
    else:  # square pixels, as the camera has and the renderer draws
        K[0, 0] = K[1, 1] = float(np.sqrt(K[0, 0] * K[1, 1]))
    rays = np.linalg.inv(K) @ np.stack([u.ravel(), v.ravel(), np.ones(u.size)])
    points = (rays.T * depth.ravel()[:, None]).reshape(H, W, 3).astype(np.float32)
    return {"points": points, "depth": depth.astype(np.float32), "mask": mask, "K": K}


def mask_iou(a, b):
    return float(_mask_iou(a, b))


def robust_z(P, k=3.0):
    """Points of one surface: depth within k MADs of the median (flying pixels at mask edges dropped)."""
    if len(P) < 20:
        return P
    z = P[:, 2]
    med = np.median(z)
    mad = max(1.4826 * np.median(np.abs(z - med)), 0.01 * med)
    return P[np.abs(z - med) < k * mad]


def extent(v, lo=1, hi=99):
    return float(np.percentile(v, hi) - np.percentile(v, lo)) if len(v) else 0.0


def erode(m, px):
    return cv2.erode(m.astype(np.uint8), np.ones((px, px), np.uint8)) > 0 if px > 1 else m


def row_span(m, q=90):
    """The mask's horizontal span in pixels: per image row (rows with any pixel), the q-th percentile over rows."""
    rows = np.nonzero(m.any(1))[0]
    if len(rows) < 5:
        return 0.0
    sub = m[rows]
    first = sub.argmax(1)
    last = sub.shape[1] - 1 - sub[:, ::-1].argmax(1)
    return float(np.percentile(last - first + 1, q))


def fit_plane(P):
    c = np.median(P, 0)
    for _ in range(3):  # trimmed PCA
        _, V = np.linalg.eigh(np.cov((P - c).T))
        n = V[:, 0]
        d = (P - c) @ n
        keep = np.abs(d) < max(3 * 1.4826 * np.median(np.abs(d)), 1e-3)
        P = P[keep]
        c = P.mean(0)
    return n, c


class Views:
    """DA3's views in the photo camera's frame and the point map's units. DA3 saw each image turned, scaled and
    cropped to one size (da3_views `prep`); `to_da3` takes an image-sized array the same way."""

    def __init__(self, s, da, geo):
        self.names = [str(n) for n in da["names"]]
        self.depth, self.conf, self.K, self.prep = da["depth"], da["conf"], da["K"], da["prep"]
        self.imgs = [Image.open(s / "capture" / "rgb.png").convert("RGB")] + \
            [Image.open(s / "capture" / "views" / f"{n}.png").convert("RGB") for n in self.names[1:]]
        d0 = self.to_da3(0, np.where(geo["mask"], geo["depth"], 0).astype(np.float32), cv2.INTER_NEAREST)
        ok = (d0 > 0) & np.isfinite(d0) & (self.conf[0] >= np.percentile(self.conf[0], 50))
        self.to_photo = float(np.median(d0[ok] / self.depth[0][ok]))  # DA3 units -> point map units
        E = np.tile(np.eye(4), (len(self.names), 1, 1))
        E[:, :3, :4] = da["w2c"]
        self.T = np.einsum("ij,njk->nik", E[0], np.linalg.inv(E))  # view camera -> photo camera
        self.T[:, :3, 3] *= self.to_photo
        self.T0 = self.T.copy()  # DA3's own poses, before the per-view corrections (recon.board triangulates with them)

    def to_da3(self, i, a, interp=cv2.INTER_NEAREST):
        rot, s, x0, y0 = self.prep[i]
        a = np.ascontiguousarray(np.rot90(a, int(rot)))
        h, w = self.depth[i].shape
        a = cv2.resize(a, (round(a.shape[1] * s), round(a.shape[0] * s)), interpolation=interp)
        return a[int(y0):int(y0) + h, int(x0):int(x0) + w]

    def lift(self, i, mask, erode_px=3, own=False):
        """Points of a mask (at the view's capture size) in the photo camera's frame (`own`: the view's), point map
        units."""
        m = self.to_da3(i, mask.astype(np.uint8)) > 0
        m = erode(m, erode_px) & (self.conf[i] >= np.percentile(self.conf[i], 30))
        v, u = np.nonzero(m)
        z = self.depth[i][v, u] * self.to_photo
        X = (np.linalg.inv(self.K[i]) @ np.stack([u + 0.5, v + 0.5, np.ones_like(u, float)])).T * z[:, None]
        X = robust_z(X)
        return X if own or not len(X) else X @ self.T[i][:3, :3].T + self.T[i][:3, 3]

    def correct(self, i, pairs):
        """A view's own scale and offset against the photo, from objects both see: DA3 keeps two very different views
        (a front and a top view) apart by tens of percent in scale. Each object's width along the photo camera's x
        axis (across both lines of sight when one view looks from the front and the other from above) is measured in
        the photo (`width`: the mask's row span at its depth, as the robot parts) and in the view (its unshrunk
        extent: no erosion, 0.5-99.5 %); the scale is their ratio's median weighted by the width squared -- a large
        flat object on the support (a plate) rather than a cup, whose rim DA3 draws too close to the table seen from
        above, which shrinks it. The offset is the weighted median of the centroids' differences. `pairs`: (photo
        width, photo centroid, the view's own points)."""
        R = self.T[i][:3, :3]
        r = np.array([w / max(extent((X @ R.T)[:, 0], 0.5, 99.5), 1e-6) for w, _, X in pairs])
        wt = np.array([w * w for w, _, _ in pairs])

        def wmedian(v):
            o = np.argsort(v)
            return v[o][np.searchsorted(np.cumsum(wt[o]), 0.5 * wt.sum())]

        k = float(wmedian(r))
        d = np.array([c - k * np.median(X, 0) @ R.T for _, c, X in pairs])
        t = np.array([wmedian(d[:, a]) for a in range(3)])
        self.T[i][:3, :3] = k * R  # view points -> photo frame: k R X + t
        self.T[i][:3, 3] = t
        return k, t, r

    def px_to_da3(self, i, px):
        """Pixel positions of a view's capture image (OpenCV's: centres at integers) in DA3's image of it."""
        rot, s, x0, y0 = self.prep[i]
        w, h = self.imgs[i].size
        c = np.asarray(px, float) + 0.5
        for _ in range(int(rot)):  # turned 90 degrees counter-clockwise, as prepare() and to_da3() do
            c = np.stack([c[:, 1], w - c[:, 0]], 1)
            w, h = h, w
        return c * s - np.array([x0, y0]) - 0.5

    def points(self, i):
        """A view's points at DA3's resolution, in the photo camera's frame and the point map's units, and the
        confident ones."""
        h, w = self.depth[i].shape
        v, u = np.mgrid[0:h, 0:w]
        z = self.depth[i] * self.to_photo
        X = np.einsum("ij,hwj->hwi", np.linalg.inv(self.K[i]), np.stack([u + 0.5, v + 0.5, np.ones((h, w))], -1))
        X = (X * z[..., None]) @ self.T[i][:3, :3].T + self.T[i][:3, 3]
        return X, (z > 0) & (self.conf[i] >= np.percentile(self.conf[i], 30))

    def cloud(self, i, step=4):
        h, w = self.depth[i].shape
        m = np.zeros((h, w), bool)
        m[::step, ::step] = True
        m &= self.conf[i] >= np.percentile(self.conf[i], 40)
        v, u = np.nonzero(m)
        z = self.depth[i][v, u] * self.to_photo
        X = (np.linalg.inv(self.K[i]) @ np.stack([u + 0.5, v + 0.5, np.ones_like(u, float)])).T * z[:, None]
        X = X @ self.T[i][:3, :3].T + self.T[i][:3, 3]
        img = self.to_da3(i, np.asarray(self.imgs[i]), cv2.INTER_AREA)
        return X, img[v, u]


SAM3_CACHE = None  # parsing/multiview/sam3_cache.npz: re-runs (another view, a changed rule) need no GPU


def sam3_masks(pil, prompts, tag="photo"):
    """{prompt: [(mask, score), ...]} on one image (perceive's SAM 3 call), cached per image tag and prompt."""
    from calibrated_scene_reconstruction.scene.capture.perceive import run_sam3

    cache = dict(np.load(SAM3_CACHE)) if SAM3_CACHE is not None and SAM3_CACHE.exists() else {}
    todo = [p for p in prompts if f"{tag}|{p}|scores" not in cache]
    if todo:
        got = {p: [] for p in todo}
        for inst in run_sam3(pil, todo, device()):
            got[inst["label"]].append((inst["mask"], inst["score"]))
        for p, v in got.items():
            cache[f"{tag}|{p}|scores"] = np.array([sc for _, sc in v], float)
            cache[f"{tag}|{p}|masks"] = np.packbits(np.stack([m for m, _ in v]), axis=-1) if v else np.zeros((0, 0))
        if SAM3_CACHE is not None:
            np.savez_compressed(SAM3_CACHE, **cache)
    out = {}
    for p in prompts:
        sc, mk = cache[f"{tag}|{p}|scores"], cache[f"{tag}|{p}|masks"]
        out[p] = [(np.unpackbits(mk[k], axis=-1)[:, :pil.size[0]].astype(bool), float(sc[k])) for k in range(len(sc))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--rerun-da3", action="store_true")
    a = ap.parse_args()
    s = Session(a.session)
    from calibrated_scene_reconstruction.scene.capture.inputs import validate_session

    validate_session(s)
    par = s / "parsing"
    cfg = s.config("multiview.json", {})
    robot_scale_route = s.config('twin.json', {}).get('scale')
    visual_robot = robot_scale_route in ('robot_visual','robot_joints')
    views = sorted((s / "capture" / "views").glob("*.png"))
    board_path = s.config_dir / cfg.get("board", "charuco_board.json") if cfg.get("board", True) else None
    board_path = board_path if board_path and board_path.exists() else None
    if visual_robot:
        board_path = None  # visual robot calibration must not consume a supplied board
    if not views and not (s.config_dir / "multiview.json").exists() and not board_path:
        print("done: no views, no configs/<name>/multiview.json, no board (nothing to do)")
        return
    robot = cfg.get("robot", "vega_1p_sharpa")
    ref_file = CONFIGS / "robots" / "scale_refs" / f"{robot}.json"
    parts = cfg.get("parts") or (json.loads(ref_file.read_text())["parts"] if ref_file.exists() else [])
    moge = par / "geometry.moge.npz"
    if not moge.exists():
        shutil.copy(par / "geometry.npz", moge)
    geo = dict(np.load(moge))
    pil = Image.open(s / "capture" / "rgb.png").convert("RGB")
    # A video's joint depth uses its own intrinsics; calibrated_fx is an explicit option.
    video = (s / "capture" / "keyframes.json").exists()
    geometry = cfg.get("geometry", "da3" if video and views else "moge")
    da = run_da3(s, a.rerun_da3) if views else None
    if geometry == "da3":
        cam = s.config("camera.json", {})
        fx_cal = cam["K"][0][0] * pil.size[0] / cam["size"][0] if cam and cam.get("size") and \
            (cam["size"][0] > cam["size"][1]) == (pil.size[0] > pil.size[1]) else None
        geo = da3_geometry(da, *pil.size, fx=fx_cal if cfg.get("calibrated_fx", False) else None)
    fx = float(geo["K"][0, 0])
    inst = json.loads((par / "instances.json").read_text())
    masks = s.masks()
    labels = [i["label"] for i in inst]
    out = par / "multiview"
    out.mkdir(parents=True, exist_ok=True)
    global SAM3_CACHE
    SAM3_CACHE = out / "sam3_cache.npz"

    V = Views(s, da, geo) if views else None
    if V:
        print(f"[multiview] geometry {geometry}; DA3 -> point map units x{V.to_photo:.3f}; views {V.names[1:]}")

    # gravity: the support (table, counter) under the task, else the floor
    from calibrated_scene_reconstruction.scene.twin.graph import CONTAINER, GRASP, SUPPORT

    sup = [i for i, lab in enumerate(labels) if SUPPORT.search(lab)]
    sup = [max(sup, key=lambda i: masks[i].sum())] if sup else [i for i, lab in enumerate(labels) if lab == "floor"]
    sup_pts = np.concatenate([geo["points"][erode(masks[i], 7) & geo["mask"]] for i in sup]) if sup else None
    up = None
    if sup_pts is not None and len(sup_pts) > 100:
        n, c = fit_plane(sup_pts[np.random.default_rng(0).permutation(len(sup_pts))[:200000]])
        up = n if n @ (-c) > 0 else -n  # the camera is above the support / floor
    floor_ids = [i for i, lab in enumerate(labels) if lab == "floor"]
    floor_pts = np.concatenate([geo["points"][erode(masks[i], 7) & geo["mask"]] for i in floor_ids]) if floor_ids else None
    top_views = [i for i in range(1, len(V.names)) if up is not None and float(-V.T[i][:3, 2] @ up) >
                 np.cos(np.radians(35))] if V else []

    # 2. scale from the robot: each part's first prompt that SAM 3 finds (perceive's instances first)
    def find(prompts):
        for p in prompts:
            c = [(masks[i], inst[i]["score"]) for i, lab in enumerate(labels) if lab == p]
            if c:
                return p, max(c, key=lambda x: x[1])
        extra = sam3_masks(pil, prompts)
        for p in prompts:
            if extra[p]:
                return p, max(extra[p], key=lambda x: x[1])
        return None, None

    meas, part_masks, found = [], {}, []
    for ref in parts:
        prompts = ref.get("prompts") or [ref["prompt"]]
        p, hit = find(prompts)
        if hit is None:
            print(f"[scale] {prompts[0]}: not found in the photo")
            continue
        part_masks[prompts[0]] = hit[0]
        found.append((ref, prompts[0], p, hit))

    def forward(name):  # the robot's forward: its base's front face, levelled
        P = robust_z(geo["points"][erode(part_masks[name], 7) & geo["mask"]])
        n, c = fit_plane(P)
        n = n if n @ (-c) > 0 else -n
        fw = n - (n @ up) * up
        return fw / np.linalg.norm(fw)

    fw = next((forward(name) for ref in parts if "front_to_origin" in ref and up is not None
               for name in [(ref.get("prompts") or [ref["prompt"]])[0]] if name in part_masks), None)
    lat = np.cross(up, fw) if fw is not None else np.array([1.0, 0.0, 0.0])  # the widths' axis
    for ref, name, p, (m, score) in found:
        P = robust_z(geo["points"][erode(m, 3) & geo["mask"]])
        if len(P) < 50:
            continue
        z, px = float(np.median(P[:, 2])), row_span(m)
        w_rows, w_3d = px * z / fx, extent(P @ lat, 1, 99)
        # a large part: its points' extent along the lateral axis; a small one (a 60 px
        # head): the rows, which 3 px of erosion would narrow by a tenth
        e, how = (w_3d, "3d") if px >= 150 else (w_rows, "rows")
        meas.append({"part": name, "prompt": p, "view": "photo", "dim": "width", "measured": e, "method": how,
                     "width_rows": w_rows, "width_3d": w_3d, "px": float(px), "reference": ref["width"],
                     "ratio": ref["width"] / e, "score": score, "centroid": np.median(P, 0).tolist()})
    for m_ in meas:
        print(f"[scale] {m_['part']} ('{m_['prompt']}', {m_['score']:.2f}, {m_['px']:.0f} px) width: {m_['measured']:.3f} "
              f"in the point map by {m_['method']} (rows {m_['width_rows']:.3f}, 3d {m_['width_3d']:.3f}), "
              f"{m_['reference']:.3f} m -> x{m_['ratio']:.3f}")
    ratios = np.array([m_["ratio"] for m_ in meas])

    # 3. the views against the photo: matched objects, each view's scale and offset (in the point map's units,
    # before the metric scale)
    objects, corrections = [], {}
    if V:
        obj_ids = [i for i, lab in enumerate(labels) if GRASP.search(lab) or CONTAINER.search(lab)]
        obj_ids.sort(key=lambda j: -inst[j]["score"])  # one thing under two prompts (a paper cup as "cup" too): once
        obj_ids = [j for k, j in enumerate(obj_ids) if not any(mask_iou(masks[j], masks[q]) > 0.5 for q in obj_ids[:k])]
        view_sam = {i: sam3_masks(V.imgs[i], sorted({labels[j] for j in obj_ids}), V.names[i])
                    for i in range(1, len(V.names))}
        photo_pts, matched = {}, {}
        for j in obj_ids:
            P0 = robust_z(geo["points"][erode(masks[j], 5) & geo["mask"]])
            if len(P0) >= 30:
                photo_pts[j] = P0
        for i in range(1, len(V.names)):
            # each view's masks of a label to the photo's objects of it, nearest pairs first, each mask and each
            # object once (two cups 15 cm apart would otherwise both take the nearer one): DA3 alone puts a view
            # 5-20 cm off the point map, so the gate is wide
            lifted = {}
            for lab in {labels[j] for j in photo_pts}:
                for k, (m, score) in enumerate(view_sam[i][lab]):
                    X = V.lift(i, m, erode_px=1, own=True)
                    if len(X) >= 30:
                        lifted[(lab, k)] = (X, np.median(X, 0) @ V.T[i][:3, :3].T + V.T[i][:3, 3])
            cands = []
            for j, P0 in photo_pts.items():
                c0, r0 = np.median(P0, 0), max(extent(P0[:, 0]), 0.03)
                for (lab, k), (X, c) in lifted.items():
                    if lab == labels[j] and np.linalg.norm(c - c0) < max(3 * r0, 0.25):
                        cands.append((np.linalg.norm(c - c0), j, (lab, k)))
            used = set()
            for d, j, key in sorted(cands, key=lambda t: t[0]):
                if j in matched.get(i, {}) or key in used:
                    continue
                matched.setdefault(i, {})[j] = lifted[key][0]
                used.add(key)
        corrections = {}
        sup_h = float(np.median(sup_pts @ up)) if up is not None and sup_pts is not None else None

        def on_support(j):  # the objects standing on the support: background clutter matches badly across views
            return sup_h is None or -0.03 < float(np.median(photo_pts[j], 0) @ up) - sup_h < 0.4

        for i in list(matched):
            objs = {j: X for j, X in matched[i].items() if on_support(j)}
            if objs:
                matched[i] = objs
            else:
                del matched[i]
        mode = cfg.get("view_corrections", "planes" if video else "objects")
        if mode == "planes":
            # a video's views agree among themselves (one chain of overlapping frames) but as a whole sit off the
            # photo's map by a scale and an offset: one similarity from the floor-to-support height every view and
            # the photo see (two textured planes: a cup 25 px wide would add +-10 % noise per view), the offset
            # along up from the support plane and across it from the objects' centroids
            sup_lab = labels[sup[0]] if sup and SUPPORT.search(labels[sup[0]]) else None
            h_ratios, hs = [], {}
            if up is not None and sup_lab and floor_pts is not None and len(floor_pts) > 100:
                h_sup, h_floor = sup_h, float(np.median(floor_pts @ up))
                for i in range(1, len(V.names)):
                    got = sam3_masks(V.imgs[i], ["floor", sup_lab], V.names[i])
                    if got["floor"] and got[sup_lab]:
                        Fi = V.lift(i, max(got["floor"], key=lambda c: c[1])[0], erode_px=5)
                        Si = V.lift(i, max(got[sup_lab], key=lambda c: c[1])[0], erode_px=5)
                        r = (h_sup - h_floor) / (float(np.median(Si @ up)) - float(np.median(Fi @ up)))
                        if len(Fi) > 200 and len(Si) > 200 and 0.5 < r < 2.0:  # else a mask of the wrong thing
                            hs[i] = float(np.median(Si @ up))
                            h_ratios.append(r)
            if len(h_ratios) >= 2:
                # per view: DA3's scale drifts along a walk, so
                # each view takes its own ratio, and one without the planes its nearest neighbour's along the walk
                ks = dict(zip(hs, h_ratios))
                for i in range(1, len(V.names)):
                    if i not in ks:
                        ks[i] = ks[min(hs, key=lambda j: abs(j - i))]
                        hs[i] = None
                across = {i: [np.median(photo_pts[j], 0) - ks[i] * (np.median(X, 0) @ V.T[i][:3, :3].T + V.T[i][:3, 3])
                              for j, X in objs.items()] for i, objs in matched.items()}
                flat = [a_ for v in across.values() for a_ in v]
                a_all = np.median(flat, 0) if flat else np.zeros(3)
                ds = {i: h_sup - ks[i] * hs[i] for i in hs if hs[i] is not None}
                for i in range(1, len(V.names)):
                    d = ds.get(i, ds[min(ds, key=lambda j: abs(j - i))])
                    a_ = np.median(across[i], 0) if across.get(i) else a_all
                    t = d * up + (a_ - (a_ @ up) * up)
                    V.T[i][:3, :3] *= ks[i]
                    V.T[i][:3, 3] = ks[i] * V.T[i][:3, 3] + t
                corrections = {"mode": "planes", "scale": float(np.median(h_ratios)),
                               "per_view": {V.names[i]: round(float(ks[i]), 3) for i in sorted(ks)},
                               "measured": [V.names[i] for i in sorted(ds)]}
                print(f"[multiview] the views against the photo: x{np.median(h_ratios):.3f} from the floor-to-{sup_lab} "
                      f"height, per view {corrections['per_view']} ({len(h_ratios)} measured, {len(flat)} object "
                      f"sightings across)")
            else:
                print("[multiview] planes: floor and support not seen together in two views -- views left as DA3 put them")
        elif mode == "objects":
            for i in list(matched):  # each view's own scale and offset from the objects it shares with the photo
                objs = matched[i]
                pairs = [(row_span(masks[j]) * float(np.median(photo_pts[j][:, 2])) / fx, np.median(photo_pts[j], 0), X)
                         for j, X in objs.items()]
                k, t, r = V.correct(i, pairs)
                corrections[V.names[i]] = {"scale": k, "per_object": {labels[j]: round(float(x), 3)
                                                                      for j, x in zip(objs, r)}}
                print(f"[multiview] {V.names[i]}: x{k:.3f} against the photo (per object "
                      f"{corrections[V.names[i]]['per_object']})")

    # 4. scale from a calibration board in the scene (recon.board): the photo's own board, else its corners
    # triangulated over the views with DA3's poses, else one view's depth (after the views' corrections)
    board = None
    if board_path:
        from calibrated_scene_reconstruction.scene.capture import board as BD

        imgs = {"photo": np.asarray(pil)} | ({n: np.asarray(V.imgs[i]) for i, n in enumerate(V.names) if i} if V else {})
        plane = fit_plane(sup_pts[np.random.default_rng(0).permutation(len(sup_pts))[:200000]]) \
            if up is not None and sup_pts is not None else None
        rows, sightings = [], {}
        for name, (px, B) in BD.find(imgs, board_path, min_corners=6).items():
            if name == "photo":
                k = geo["depth"].shape[1] / pil.size[0]  # the map at the photo's size, or scaled
                r = BD.measure(name, geo["points"], geo["mask"] & (geo["depth"] > 0), (px * k, B))
            else:
                i = V.names.index(name)
                q = V.px_to_da3(i, px)
                P = V.K[i] @ np.linalg.inv(V.T0[i])[:3]
                for b, uv in zip(B, q + 0.5):  # K takes pixel centres at +0.5
                    sightings.setdefault(tuple(np.round(b, 5)), (b, []))[1].append((P, uv))
                r = BD.measure(name, *V.points(i), (q, B))
            if r is None:
                print(f"[board] {name}: {len(px)} corners, too few on the depth map")
                continue
            if plane is not None:  # the board lies on the support: its tilt and height there
                r["tilt_deg"] = float(np.degrees(np.arccos(min(1.0, abs(float(np.dot(r["normal"], up)))))))
                r["above_support_m"] = float((np.asarray(r["centre"]) - plane[1]) @ up) * r["scale"]
            rows.append(r)
            print(f"[board] {name}: {r['used']}/{r['corners']} corners, its depth x{r['scale']:.3f}, rms "
                  f"{1000 * r['rms_m']:.1f} mm" + (f", tilt to the support {r['tilt_deg']:.1f} deg, "
                                                   f"{100 * r['above_support_m']:+.1f} cm above it" if "tilt_deg" in r else ""))
        Xt, Bt = BD.triangulate(sightings)
        tri = None
        if len(Xt) >= 8:
            sc, rms, keep, _, _ = BD.fit_board(Xt, Bt)
            tri = {"scale": sc, "rms_m": rms, "corners": int(len(Xt)), "used": int(keep.sum())}
            print(f"[board] triangulated over the views: {tri['used']}/{tri['corners']} corners, x{sc:.3f}, "
                  f"rms {1000 * rms:.1f} mm")
        own = [r for r in rows if r["view"] == "photo"]
        if own:
            board = {"scale": own[0]["scale"], "from": "photo"}
        elif tri:
            board = {"scale": tri["scale"], "from": f"{tri['used']} corners triangulated"}
        elif rows:
            board = {"scale": float(np.median([r["scale"] for r in rows])), "from": f"{len(rows)} views' depth"}
            print("[board] warning: the board in one view only: its depth, which drifts along a walk, sets the scale")
        if board:
            board |= {"triangulated": tri, "rows": [{k: v for k, v in r.items() if k != "px"} for r in rows]}
            first = (own or sorted(rows, key=lambda r: -r["used"]))[:1]
            if first:
                Image.fromarray(BD.draw(imgs[first[0]["view"]], first)).save(out / "qa_board.jpg", quality=90)
        else:
            print(f"[board] {board_path.name}: the board is not found in the photo or the views")

    # 4b. scale from the robot's whole body at the capture's joint readout (recon.robot_scale): configs/<name>/
    # robot_state.json and parsing/robot_cloud.npz (apps/robot_cloud.py) -- no board, no tape
    robot_pose = None
    rc_path = par / "robot_cloud.npz"
    if not visual_robot and (s.config_dir / "robot_state.json").exists() and rc_path.exists() and up is not None:
        from calibrated_scene_reconstruction.scene.capture import robot_scale as RS

        rob_lab = re.compile(r"robot|vega")
        pts = [geo["points"][erode(masks[j], 5) & geo["mask"]] for j, lab in enumerate(labels) if rob_lab.search(lab)]
        n_photo = sum(len(p) for p in pts)
        if V:
            for i in range(1, len(V.names)):
                got = sam3_masks(V.imgs[i], ["humanoid robot"], V.names[i])["humanoid robot"]
                for m_, sc_ in got:
                    if sc_ >= 0.5:
                        X = V.lift(i, m_, erode_px=5)
                        if len(X) > 200:
                            pts.append(X[np.random.default_rng(i).permutation(len(X))[:20000]])
        if pts and sum(len(p) for p in pts) > 2000:
            robot_pose = RS.fit(np.load(rc_path)["points"].astype(np.float64), np.concatenate(pts), up)
            robot_pose["photo_points"] = n_photo
            print(f"[robot pose] the robot at the readout fitted to {robot_pose['scene_points']} points ({n_photo} from "
                  f"the photo): x{robot_pose['scale']:.3f}, the closest 60 % at {1000 * robot_pose['residual_m']:.1f} mm, "
                  f"{100 * robot_pose['within_2cm']:.0f} % of the model within 2 cm", flush=True)
        else:
            print("[robot pose] too few robot points in the photo and the views")

    # 5. the scale: twin.json's choice (docs/twin_pipeline.md S1), the others its checks; else the first found of
    # a fixed one, the board, the robot's pose, the robot's parts
    from calibrated_scene_reconstruction.scene.twin.gates import options as twin_options

    parts_scale = None
    if len(ratios):
        wts_ = np.array([m_["px"] for m_ in meas]) if ratios.max() / ratios.min() > 1.10 else np.ones(len(ratios))
        o_ = np.argsort(ratios)
        parts_scale = float(np.median(ratios)) if wts_.min() == 1 else \
            float(ratios[o_][np.searchsorted(np.cumsum(wts_[o_]), 0.5 * wts_.sum())])
    sources = {"board": board["scale"] if board else None,
               "robot_joints": ((1.0 / V.to_photo if V else 1.0) if robot_scale_route == 'robot_joints'
                                else robot_pose["scale"] if robot_pose else None), "robot_parts": parts_scale,
               # A generic depth prior supplies only a coordinate scale for
               # drafting. Part-width heuristics can be badly foreshortened;
               # they remain a diagnostic, never the visual CAD anchor.
               "robot_visual": (1.0 / V.to_photo if V else 1.0), "prior": 1.0}
    topt = twin_options(s)
    if topt:
        want = topt["scale"]
        if sources[want] is None:
            raise SystemExit(f"[scale] twin.json asks for {want}, which this capture does not give")
        checks = {k: round(v / sources[want], 4) for k, v in sources.items() if v is not None and k not in (want, "prior")}
        print(f"[scale] {want} x{sources[want]:.3f} (twin.json); the others against it: "
              + (", ".join(f"{k} x{v:.3f} ({100 * (v - 1):+.1f} %)" for k, v in checks.items()) or "none"), flush=True)
    if topt and topt["scale"] == "robot_joints" and robot_pose and robot_pose["within_2cm"] < POOR_FIT \
            and sources.get("robot_parts") and abs(sources["robot_parts"] / sources["robot_joints"] - 1) > 0.25:
        # Occlusion can invalidate the joint fit. Use visible parts for the photo scale
        # and anchor the frames to the recorded joint state at S1.
        scale, source = sources["robot_parts"], "robot_parts (the joints' fit too poor for the photo; S1 anchors it)"
        print(f"[scale] robot_joints' fit holds {100 * robot_pose['within_2cm']:.0f} % within 2 cm: robot_parts "
              f"x{scale:.3f} for the photo, the robot at its readout anchors the frames in S1", flush=True)
    elif topt:
        scale, source = sources[topt["scale"]], topt["scale"]
        if visual_robot:
            source = 'provisional depth placement; '+robot_scale_route+' image/CAD anchor required before S1'
    elif "scale" in cfg:
        scale, source = float(cfg["scale"]), "config"
    elif board:  # the robot parts are the check
        scale, source = board["scale"], f"board ({board['from']})"
    elif robot_pose:  # the robot parts are the check
        scale, source = robot_pose["scale"], "robot pose (the readout's URDF fitted, recon.robot_scale)"
    elif len(ratios):
        # the parts' median while they agree; when they do not, the median weighted
        # by pixel width, i.e. the largest part
        wts = np.array([m_["px"] for m_ in meas]) if ratios.max() / ratios.min() > 1.10 else np.ones(len(ratios))
        o = np.argsort(ratios)
        scale, source = float(np.median(ratios)) if wts.min() == 1 else \
            float(ratios[o][np.searchsorted(np.cumsum(wts[o]), 0.5 * wts.sum())]), "robot"
    else:
        scale, source = 1.0, "none (MoGe-2 as is)"
    spread = float(ratios.max() / ratios.min()) if len(ratios) > 1 else None
    da3_metric = 1.0 / V.to_photo if V else None  # DA3's metric depth in point map units
    if board and not source.startswith("board"):
        print(f"[scale] the board would say x{board['scale']:.3f}")
    print(f"[scale] x{scale:.3f} ({source}); robot measurements {len(ratios)}, spread "
          f"{'-' if spread is None else f'{spread:.2f}'}" + (f"; DA3's metric depth would say x{da3_metric:.3f}" if V else ""))
    np.savez_compressed(par / "geometry.npz", points=geo["points"] * scale, depth=geo["depth"] * scale,
                        mask=geo["mask"], K=geo["K"])
    print(f"[geometry] {geometry} x{scale:.3f} -> parsing/geometry.npz")
    (par / "world_frame.json").unlink(missing_ok=True)  # autograph's frame of an earlier geometry: fitted again

    # 6. the supports and the objects seen from several sides (positions in the rescaled photo frame)
    if V:
        # the supports (table, counter tops) as the corrected views see them: a top seen at a grazing angle from the
        # front loses its far part in the photo's depth; autograph adds these points to the photo's
        sup_ids = [j for j in sup if SUPPORT.search(labels[j])]
        sup_c = {j: np.median(geo["points"][erode(masks[j], 7) & geo["mask"]], 0) for j in sup_ids}
        support_pts = {}
        for i in (range(1, len(V.names)) if corrections.get("mode") == "planes" else matched):
            found_s = sam3_masks(V.imgs[i], sorted({labels[j] for j in sup_ids}), V.names[i])
            for lab in sorted({labels[j] for j in sup_ids}):
                if found_s[lab]:  # the view's surest mask of the label, to the nearest photo instance of it
                    X = V.lift(i, max(found_s[lab], key=lambda c: c[1])[0], erode_px=5)
                    if len(X) > 200:
                        X = X[np.random.default_rng(i).permutation(len(X))[:30000]]
                        j = min((j for j in sup_ids if labels[j] == lab),
                                key=lambda j: np.linalg.norm(np.median(X, 0) - sup_c[j]))
                        support_pts.setdefault(str(j), []).append(X * scale)
        if support_pts:
            np.savez_compressed(out / "support_points.npz", **{j: np.concatenate(v) for j, v in support_pts.items()})
            print(f"[supports] {', '.join(f'{labels[int(j)]} {j}: {sum(len(x) for x in v)} points' for j, v in support_pts.items())} "
                  f"from the views")
        elif (out / "support_points.npz").exists():
            (out / "support_points.npz").unlink()
        for j, P0 in photo_pts.items():
            c0 = np.median(P0, 0)
            per_view, seen, gaps = [(0, P0)], ["photo"], []
            for i, objs in matched.items():
                if j in objs:
                    P = objs[j] @ V.T[i][:3, :3].T + V.T[i][:3, 3]
                    per_view.append((i, P))
                    seen.append(V.names[i])
                    gaps.append(round(float(np.linalg.norm(np.median(P, 0) - c0) * scale), 3))
            rec = {"id": j, "label": labels[j], "views": seen, "view_offsets": gaps,
                   "centroid_cam": (c0 * scale).tolist()}
            if up is not None:
                # measured view by view (a fused cloud would widen by what registration leaves): the footprint's shape
                # from a view from above (the whole rim; seen from the front a flat object's depth is the depth
                # noise), its size set by the photo's width (the mask's row span, as the robot parts: DA3 draws a tall
                # object's rim too close to the table seen from above, which inflates it), the height from the side
                width = row_span(masks[j]) * float(np.median(P0[:, 2])) / fx * scale
                rec["width_photo"] = width
                fps, hts = {}, []
                near = np.linalg.norm(sup_pts - c0, axis=1) < 0.3 / scale if sup_pts is not None else None
                base_h = float(np.median(sup_pts[near] @ up)) * scale if near is not None and near.sum() > 50 else None
                for i, P in per_view:
                    P = P * scale
                    h = P @ up
                    Q = P - np.outer(h, up)
                    _, E = np.linalg.eigh(np.cov(Q.T))
                    fp = sorted([extent(Q @ E[:, 2], 0.5, 99.5), extent(Q @ E[:, 1], 0.5, 99.5)], reverse=True)
                    if i in top_views:  # the shape from above, the size along the photo's x from the photo
                        f = width / max(extent(P[:, 0], 0.5, 99.5), 1e-6)
                        fp = [fp[0] * f, fp[1] * f]
                    fps[i] = fp
                    if i == 0:  # the height above the support from the photo (a view's residual scale error stays out)
                        hts.append(float(np.percentile(h, 99) - (base_h if base_h is not None else np.percentile(h, 1))))
                tops = [i for i in fps if i in top_views]
                rec |= {"height": max(hts) if hts else None,  # no view from above: the median over the views (a
                        # mask that took the neighbour too, or a sliver, is one vote)
                        "footprint": fps[tops[0]] if tops else np.median(np.array(list(fps.values())), 0).tolist(),
                        "footprint_from": V.names[tops[0]] if tops else "median"}
            objects.append(rec)
            print(f"[objects] {j} {labels[j]}: views {seen} (offsets {gaps} m)" + (
                f", height {rec['height']:.3f} m, footprint {rec['footprint'][0]:.3f} x {rec['footprint'][1]:.3f} m"
                if rec.get("height") is not None else ""))

    # 7. where the real robot stands: its base's front face
    robot_parts = {m_["part"]: (np.array(m_["centroid"]) * scale).tolist() for m_ in meas}
    base = None
    for ref in parts:
        name = (ref.get("prompts") or [ref["prompt"]])[0]
        if "front_to_origin" in ref and up is not None and name in part_masks:
            P = robust_z(geo["points"][erode(part_masks[name], 7) & geo["mask"]]) * scale
            n, c = fit_plane(P)
            n = n if n @ (-c) > 0 else -n
            fw = n - (n @ up) * up
            fw /= np.linalg.norm(fw)
            base = {"part": name, "origin_cam": (np.median(P, 0) - ref["front_to_origin"] * fw).tolist(),
                    "forward_cam": fw.tolist()}
            print(f"[robot] base from the {name}'s front: {np.round(base['origin_cam'], 3)}, forward "
                  f"{np.round(fw, 2)} (camera frame)")
            break
    s.write_json("parsing/multiview.json", {
        "views": [{"name": n, "T_photo_view": V.T[i].tolist(), "top": i in top_views}
                  for i, n in enumerate(V.names)] if V else [],
        "da3": {"to_point_map": V.to_photo, "metric_scale": da3_metric,
                "view_corrections": corrections if V else {}} if V else None,
        "geometry": geometry,
        "scale": {"value": scale, "source": source, "spread": spread, "measurements": meas,
                  "against": {k: round(v / scale, 4) for k, v in sources.items()
                              if v is not None and k != "prior" and v != scale}},
        "board": board,
        "robot_pose": robot_pose,
        "up_cam": up.tolist() if up is not None else None,
        "objects": objects,
        "robot": {"spec": robot, "parts_centroid_cam": robot_parts, "base": base},
    })

    # QA: the photo with the robot parts measured, each view; a fused cloud of all views
    vis = np.asarray(pil).copy()
    for name, m in part_masks.items():
        vis[m] = (0.5 * vis[m] + 0.5 * np.array([255, 60, 60])).astype(np.uint8)
    for m_ in meas:
        c = np.array(m_["centroid"])
        u, v = (geo["K"] @ c)[:2] / c[2]
        cv2.putText(vis, f"{m_['part']} {m_['measured'] * scale:.3f} m (ref {m_['reference']:.3f})",
                    (int(u) - 200, int(v)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)
    tiles = [vis]
    if V:
        tiles += [np.asarray(V.imgs[i]) for i in range(1, len(V.names))]
        Xs, Cs = zip(*[V.cloud(i) for i in range(len(V.names))])
        X, C = np.concatenate(Xs) * scale, np.concatenate(Cs)
        with open(out / "fused.ply", "w") as f:
            f.write(f"ply\nformat ascii 1.0\nelement vertex {len(X)}\nproperty float x\nproperty float y\n"
                    "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            for p, c in zip(X, C):
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")
    hh = 900
    tiles = [cv2.resize(t, (round(t.shape[1] * hh / t.shape[0]), hh)) for t in tiles]
    Image.fromarray(np.concatenate(tiles, 1)).save(out / "qa_multiview.jpg", quality=88)
    print(f"done: scale x{scale:.3f} ({source}), {len(objects)} objects across {len(V.names) if V else 1} images")


if __name__ == "__main__":
    sys.exit(main())
