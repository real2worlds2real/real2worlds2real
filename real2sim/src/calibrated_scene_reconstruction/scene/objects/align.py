"""S3 gravity-aware alignment of SAM 3D meshes to the photo.

    python -m calibrated_scene_reconstruction.recon.align sessions/<name> [--only cup]

1. SAM 3D's GLB local frame vs. its pose output is ambiguous up to an axis permutation: enumerate the 24 proper
   axis rotations and score the silhouette IoU against the instance mask.
2. Silhouettes cannot separate a squat cylinder standing from lying, or an open vessel upright from upside down:
   among near-best IoUs keep the stable resting poses (largest support polygon under gravity) and pick the one with
   the lowest surface-area centroid. An open vessel (cup, mug, glass, bowl, ...) whose photo mask is clearly wider
   at one end keeps that end where the photo has it: a cup widening upwards has its lowest centroid on its rim.
3. ICP against the observed MoGe points in the world frame, restricted to yaw + translation + isotropic scale, so
   alignment can never tilt an object off its support.
Writes objects/<obj>/{mesh_cam.glb (OpenCV photo-camera frame), align.json, qa_align.png}.
"""
from calibrated_scene_reconstruction.assets.mesh_alignment import axis_rotations as axis_rotations, local_to_cam_cv as local_to_cam_cv, support_area as support_area, centroid_height_ratio as centroid_height_ratio  # noqa: F401
import argparse
import itertools
import json

import cv2
import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.transform import Rotation

from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.objects.reconstruct import object_specs


from calibrated_scene_reconstruction.core.geometry import mask_iou





def project_mask(v, faces, K, shape):
    uv = v[:, :2] / v[:, 2:3] * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
    canvas = np.zeros(shape, np.uint8)
    cv2.fillPoly(canvas, list(np.round(uv[faces]).astype(np.int32)), 1)
    return canvas.astype(bool)


def iou(a, b):
    return mask_iou(a, b)


def visible_vertices(v, K, shape):
    """Vertices that are front-most at their pixel (vertex z-buffer)."""
    uv = np.round(v[:, :2] / v[:, 2:3] * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])).astype(int)
    H, W = shape
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    zbuf = np.full(shape, np.inf)
    np.minimum.at(zbuf, (uv[ok, 1], uv[ok, 0]), v[ok, 2])
    vis = np.zeros(len(v), bool)
    vis[ok] = v[ok, 2] <= zbuf[uv[ok, 1], uv[ok, 0]] + 0.01
    return vis






VESSEL = ("cup", "mug", "glass", "tumbler", "bowl", "vase", "jar", "bucket", "pot")


def band_widths(rows_or_z, widths_fn, frac=0.2):
    """(width of the lowest `frac`, width of the highest) of an object, given a per-band width function."""
    lo, hi = np.min(rows_or_z), np.max(rows_or_z)
    span = max(hi - lo, 1e-9)
    return widths_fn(lo, lo + frac * span), widths_fn(hi - frac * span, hi)


def mask_taper(mask):
    """The photo's top / bottom width ratio of a mask (>1: wider at the top, as a cup's rim over its foot)."""
    ys = np.nonzero(mask.any(1))[0]
    if len(ys) < 10:
        return 1.0

    def width(a, b):
        rows = [np.ptp(np.nonzero(mask[y])[0]) + 1 for y in range(int(a), int(b) + 1) if mask[y].any()]
        return float(np.median(rows)) if rows else 0.0
    bottom, top = band_widths(ys, width)  # image rows grow downwards: the "lowest" rows are the top
    return bottom / max(top, 1e-9)


def mesh_taper(Vw):
    """The same ratio of a mesh standing in the world frame (z up)."""
    z = Vw[:, 2]

    def width(a, b):
        P = Vw[(z >= a) & (z <= b), :2]
        return float(np.linalg.norm(np.ptp(P, 0))) if len(P) > 3 else 0.0
    bottom, top = band_widths(z, width)
    return top / max(bottom, 1e-9)


def icp_yaw(Vw, target_w, T_wc, K, shape, iters=40, scale_clip=(0.97, 1.03), fix_z=False, scale_xy=False):
    """World-frame ICP: rotation about +z only, translation, isotropic scale (clamped per step; None = rigid).
    fix_z keeps the height (pieces standing on the floor); scale_xy scales the footprint only."""
    v = Vw.copy()
    s_total = 1.0
    for _ in range(iters):
        vis = visible_vertices((v - T_wc[:3, 3]) @ T_wc[:3, :3], K, shape)
        if vis.sum() < 10:
            break
        src = v[vis]
        d, idx = cKDTree(src).query(target_w)
        keep = d < np.percentile(d, 85)
        a, b = src[idx[keep]], target_w[keep]
        A, B = a - a.mean(0), b - b.mean(0)
        U, _, Vt = np.linalg.svd(A[:, :2].T @ B[:, :2])
        R2 = Vt.T @ U.T
        if np.linalg.det(R2) < 0:
            Vt[1] *= -1
            R2 = Vt.T @ U.T
        R = np.eye(3)
        R[:2, :2] = R2
        d = slice(0, 2) if scale_xy else slice(0, 3)
        s = 1.0 if scale_clip is None else float(np.clip((B[:, d] * (A @ R.T)[:, d]).sum() / max((A[:, d] ** 2).sum(), 1e-9),
                                                       *scale_clip))
        S = np.diag([s, s, 1.0 if scale_xy else s])
        t = b.mean(0) - R @ S @ a.mean(0)
        if fix_z:
            t[2] = 0.0
        v = v @ (R @ S).T + t
        s_total *= s
    return v, s_total


def align(s: Session, name):
    img = np.asarray(Image.open(s / "capture" / "rgb.png").convert("RGB"))
    geo, masks = s.geometry(), s.masks()
    K, shape = geo["K"], img.shape[:2]
    _, _, T_wc = s.world_frame()
    d = s / "objects" / name
    pose = json.loads((d / "pose.json").read_text())
    glb = trimesh.load(d / "raw.glb", force="mesh")
    faces = np.asarray(glb.faces)
    mask = np.zeros(shape, bool)
    for i in pose["ids"]:
        mask |= masks[i]
    to_w = lambda v: v @ T_wc[:3, :3].T + T_wc[:3, 3]  # noqa: E731

    cands = []
    for M in axis_rotations():
        v = local_to_cam_cv(np.asarray(glb.vertices) @ M.T, pose)
        Vw = to_w(v)
        cands.append((iou(project_mask(v, faces, K, shape), mask), support_area(Vw), centroid_height_ratio(Vw, faces), M, v))
    top = max(c[0] for c in cands)
    near = [c for c in cands if c[0] >= top - 0.08]
    spec = {**s.scene_graph.get("dynamic_objects", {}), **s.scene_graph.get("static_objects", {})}.get(name, {})
    label = spec.get("_label", name)
    taper = mask_taper(mask)
    if any(w in label for w in VESSEL) and abs(np.log(taper)) > np.log(1.08):
        # Use the observed taper to keep the rim of an open vessel at the correct end.
        agree = [c for c in near if (mesh_taper(to_w(c[4])) > 1) == (taper > 1)]
        near = agree or near
    amax = max(c[1] for c in near)
    stable = [c for c in near if c[1] >= 0.5 * amax]
    iou0, area0, com0, M, v0 = min(stable, key=lambda c: c[2])

    er = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & geo["mask"]
    Vw1, scale = icp_yaw(to_w(v0), to_w(geo["points"][er]), T_wc, K, shape)
    v1 = (Vw1 - T_wc[:3, 3]) @ T_wc[:3, :3]
    proj = project_mask(v1, faces, K, shape)
    iou1 = iou(proj, mask)
    mesh = glb.copy()
    mesh.vertices = v1
    mesh.export(d / "mesh_cam.glb")
    for stale in ("mesh_cam.recon.glb", "albedo.recon.png"):  # a substitute of recon.audit's for the old alignment
        (d / stale).unlink(missing_ok=True)
    img_tex = getattr(mesh.visual.material, "baseColorTexture", None)
    if img_tex is not None:
        img_tex.convert("RGB").save(d / "albedo.png")
    (d / "align.json").write_text(json.dumps({
        "local_fix": M.tolist(), "support_area_cm2": area0 * 1e4, "centroid_height_ratio": com0,
        "iou_convention": float(iou0), "iou_final": float(iou1), "icp_scale": scale,
        "extent_m": (Vw1.max(0) - Vw1.min(0)).tolist(),
    }, indent=1))
    qa = img.astype(np.float32).copy()
    qa[proj] = 0.5 * qa[proj] + 0.5 * np.array([255, 0, 0])
    qa[mask & ~proj] = 0.5 * qa[mask & ~proj] + 0.5 * np.array([0, 0, 255])
    Image.fromarray(qa.astype(np.uint8)).save(d / "qa_align.png")
    print(f"[{name}] {len(near)} near-best / {len(stable)} stable conventions, support {area0 * 1e4:.0f} cm2, "
          f"IoU {iou0:.3f} -> {iou1:.3f} after yaw-ICP")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    s = Session(args.session)
    for name in object_specs(s.scene_graph):
        if not args.only or name in args.only:
            align(s, name)


if __name__ == "__main__":
    main()
