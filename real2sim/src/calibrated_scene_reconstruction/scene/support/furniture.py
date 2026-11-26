"""S3b movable furniture: one shared reconstruction per set of identical pieces (e.g. bar stools).

    python -m calibrated_scene_reconstruction.recon.furniture sessions/<name> [--only stools]

scene graph: "movable_furniture": {"<group>": {"ids": [..], "shared": true, "category": "stool", "min_iou": 0.5}}

1. Reference = the member least cut by the frame (largest mask not touching the left / right border).
2. SAM 3D on the reference (recon.objects worker). Its shape prior completes what the frame cuts off (legs below
   the lower edge), but for a truncated piece neither its pose nor its proportions are reliable.
3. Floor-standing fit instead of recon.align: the mesh (y-up, as SAM 3D emits it) stands upright on the floor, its
   height set to the observed top (seat), then yaw x 24 + planar ICP with a footprint scale, scored by silhouette IoU.
4. Every member gets a rigid copy, moved to its own observed points and refined by planar ICP (yaw + xy), scored
   by visible IoU (other instances' pixels ignored); below member_min_iou (0.35) after a full yaw search the member
   is dropped from the scene (recon.audit shows it).
QA: in-frame silhouette IoU of the reference; below min_iou the group is left to recon.retrieve (library model).
Writes objects/<group>/{mesh_local.glb, albedo.png, furniture.json, qa_furniture.png}.
"""
import argparse

import cv2
import numpy as np
import trimesh
from PIL import Image

from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.objects.align import icp_yaw, iou, project_mask
from calibrated_scene_reconstruction.scene.objects.reconstruct import run_worker


def pick_reference(masks, ids):
    """Least truncated member: not touching the left/right border first, then the largest mask."""
    return max(ids, key=lambda i: (not (masks[i][:, :3].any() or masks[i][:, -3:].any()), int(masks[i].sum())))


class PhotoFrame:
    """Photo camera, world frame and observed world points per instance."""

    def __init__(self, s: Session):
        self.s = s
        self.img = np.asarray(Image.open(s / "capture" / "rgb.png").convert("RGB"))
        geo = s.geometry()
        self.K, self.points, self.valid = geo["K"], geo["points"], geo["mask"]
        self.masks = s.masks()
        self.shape = self.img.shape[:2]
        _, _, self.T_wc = s.world_frame()

    def to_world(self, v_cam):
        return v_cam @ self.T_wc[:3, :3].T + self.T_wc[:3, 3]

    def to_cam(self, v_w):
        return (v_w - self.T_wc[:3, 3]) @ self.T_wc[:3, :3]

    def mask(self, m):
        """Instance id or boolean mask."""
        return self.masks[m] if np.ndim(m) == 0 else m

    def observed(self, m, erode=5):
        er = cv2.erode(self.mask(m).astype(np.uint8), np.ones((erode, erode), np.uint8)).astype(bool) & self.valid
        return self.to_world(self.points[er])

    def silhouette_iou(self, Vw, faces, m):
        return float(iou(project_mask(self.to_cam(Vw), faces, self.K, self.shape), self.mask(m)))

    def visible_iou(self, Vw, faces, i):
        """Silhouette IoU of instance i that ignores pixels of the other instances: a piece hidden behind a table is
        not penalised for the part the table covers (the plain IoU is, which makes occluded members look misfit)."""
        other = self.masks.any(0) & ~self.masks[i]
        return float(iou(project_mask(self.to_cam(Vw), faces, self.K, self.shape) & ~other, self.masks[i]))


def planar_transform(A, B):
    """Rigid (yaw, t) with B ~ Rz(yaw) A + t from corresponding world points."""
    a, b = A[:, :2] - A[:, :2].mean(0), B[:, :2] - B[:, :2].mean(0)
    U, _, Vt = np.linalg.svd(a.T @ b)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1] *= -1
        R = Vt.T @ U.T
    return float(np.arctan2(R[1, 0], R[0, 0])), B[:, :2].mean(0) - R @ A[:, :2].mean(0), R


def place_members(pf: PhotoFrame, Vw_ref, faces, ref_id, ids, min_iou=0.35, n_yaw=12):
    """Copy the reference piece onto every member: shift by the observed-point offset, then planar ICP (kept only
    when it improves the member's visible IoU). A member still below min_iou gets a full yaw search (n_yaw starts +
    ICP); if that fails too, it is marked drop (another style, or too little of it is seen to place it), so the
    scene leaves it out instead of standing a misplaced copy there. Returns [{id, center, yaw (relative to the
    reference), iou (visible IoU), drop?}]."""
    c_ref = (Vw_ref.min(0) + Vw_ref.max(0))[:2] / 2
    m_ref = np.median(pf.observed(ref_id)[:, :2], 0)
    sub = np.linspace(0, len(Vw_ref) - 1, min(len(Vw_ref), 20000)).astype(int)

    def icp(V0, tgt):
        V1s, _ = icp_yaw(V0[sub], tgt, pf.T_wc, pf.K, pf.shape, iters=30, scale_clip=None, fix_z=True)
        _, t, R = planar_transform(V0[sub], V1s)
        V1 = V0.copy()
        V1[:, :2] = V0[:, :2] @ R.T + t
        return V1

    out = []
    for i in ids:
        if i == ref_id:
            out.append({"id": int(i), "center": c_ref.tolist(), "yaw": 0.0, "iou": pf.visible_iou(Vw_ref, faces, i)})
            continue
        tgt = pf.observed(i)
        V0 = Vw_ref + np.r_[np.median(tgt[:, :2], 0) - m_ref, 0.0]
        cands = [V0, icp(V0, tgt)]
        if max(pf.visible_iou(V, faces, i) for V in cands) < min_iou:
            c0 = np.median(tgt[:, :2], 0)
            for a in np.linspace(0, 2 * np.pi, n_yaw, endpoint=False)[1:]:
                R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
                Va = Vw_ref.copy()
                Va[:, :2] = (Vw_ref[:, :2] - c_ref) @ R.T + c0
                cands.append(icp(Va, tgt))
        scores = [pf.visible_iou(V, faces, i) for V in cands]
        V = cands[int(np.argmax(scores))]
        yaw, t, R = planar_transform(Vw_ref, V)
        m = {"id": int(i), "center": (R @ c_ref + t).tolist(), "yaw": yaw, "iou": max(scores)}
        if m["iou"] < min_iou:
            m["drop"] = True
        out.append(m)
    return out


def load_zup(file):
    """glTF mesh (y-up: library models and SAM 3D output alike) as z-up vertices / faces."""
    m = trimesh.load(file, force="mesh")
    V = np.asarray(m.vertices, float)
    return np.stack([V[:, 0], -V[:, 2], V[:, 1]], 1), np.asarray(m.faces), m


def place_upright(Vz, yaw, scale, xy, support_z, scale_xy=1.0):
    """Same convention as sim.assets.add_library_model: yaw about +z, bbox centred on xy, resting on support_z.
    scale_xy (footprint / height ratio correction) is only used for reconstructed pieces."""
    c, s = np.cos(yaw), np.sin(yaw)
    V = (Vz * np.array([scale * scale_xy, scale * scale_xy, scale])) @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T
    lo, hi = V.min(0), V.max(0)
    return V + np.r_[np.asarray(xy) - (lo[:2] + hi[:2]) / 2, support_z + 0.002 - lo[2]]


def fit_upright(pf, Vz, F, m, height, support_z, n_yaw=24, fit_footprint=False, coarse=False):
    """Stand a z-up mesh on its support at the observed height; search yaw with planar ICP, keep the best silhouette IoU
    against the instance (id or mask). fit_footprint then sweeps a footprint scale (SAM 3D proportions of a truncated
    piece are unreliable): jointly with yaw, or (coarse) at the three best yaws of a search at footprint 1 -- a fifth
    of the ICP runs. Returns {iou, yaw, xy, scale, scale_xy}."""
    tgt = pf.observed(m)
    scale = height / max(np.ptp(Vz[:, 2]), 1e-6)
    sub = np.linspace(0, len(Vz) - 1, min(len(Vz), 15000)).astype(int)

    def place_icp(yaw, xy, sxy):
        V0 = place_upright(Vz, yaw, scale, xy, support_z, sxy)
        V1s, _ = icp_yaw(V0[sub], tgt, pf.T_wc, pf.K, pf.shape, iters=12, scale_clip=None, fix_z=True)
        dyaw, t, R = planar_transform(V0[sub], V1s)
        V1 = V0.copy()
        V1[:, :2] = V0[:, :2] @ R.T + t
        return {"iou": pf.silhouette_iou(V1, F, m), "yaw": float((yaw + dyaw) % (2 * np.pi)),
                "xy": ((V1.min(0) + V1.max(0))[:2] / 2).tolist(), "scale": float(scale), "scale_xy": float(sxy)}

    xy0 = np.median(tgt[:, :2], 0)
    yaws = np.linspace(0, 2 * np.pi, n_yaw, endpoint=False)
    scales = np.arange(0.4, 1.31, 0.1) if fit_footprint else [1.0]  # yaw and footprint are coupled: joint search
    if fit_footprint and coarse:
        first = sorted((place_icp(yaw, xy0, 1.0) for yaw in yaws), key=lambda r: -r["iou"])
        seeds = [r["yaw"] for r in first[:3]]
        best = max([first[0]] + [place_icp(yaw, xy0, sxy) for yaw in seeds for sxy in scales if sxy != 1.0],
                   key=lambda r: r["iou"])
    else:
        best = max((place_icp(yaw, xy0, sxy) for yaw in yaws for sxy in scales), key=lambda r: r["iou"])
    if fit_footprint:
        best = max([best] + [place_icp(best["yaw"], best["xy"], best["scale_xy"] + ds) for ds in (-0.05, -0.025, 0.025, 0.05)],
                   key=lambda r: r["iou"])
    return best


def qa_overlay(pf: PhotoFrame, meshes, ids, path):
    qa = pf.img.astype(np.float32).copy()
    proj = np.zeros(pf.shape, bool)
    for Vw, F in meshes:
        proj |= project_mask(pf.to_cam(Vw), F, pf.K, pf.shape)
    obs = np.zeros(pf.shape, bool)
    for i in ids:
        obs |= pf.masks[i]
    qa[proj] = 0.5 * qa[proj] + 0.5 * np.array([255, 0, 0])
    qa[obs & ~proj] = 0.5 * qa[obs & ~proj] + 0.5 * np.array([0, 0, 255])
    Image.fromarray(qa.astype(np.uint8)).save(path)


def member_mesh(Vw_ref, c_ref, m):
    R = np.array([[np.cos(m["yaw"]), -np.sin(m["yaw"])], [np.sin(m["yaw"]), np.cos(m["yaw"])]])
    V = Vw_ref.copy()
    V[:, :2] = (Vw_ref[:, :2] - c_ref) @ R.T + np.asarray(m["center"])
    return V


def furnish(s: Session, group, spec, refit=False):
    pf = PhotoFrame(s)
    ids = spec["ids"]
    ref = pick_reference(pf.masks, ids)
    d = s / "objects" / group
    d.mkdir(parents=True, exist_ok=True)
    report = {"group": group, "reference": int(ref), "ok": False}
    if not (refit and (d / "raw.glb").exists()) and run_worker(s.dir, group, [ref]) != 0:
        report["reason"] = "SAM 3D failed"
        s.write_json(f"objects/{group}/furniture.json", report)
        print(f"[{group}] SAM 3D failed -> left to recon.retrieve")
        return report
    Vz, faces, mesh = load_zup(d / "raw.glb")
    height = float(np.percentile(pf.observed(ref)[:, 2], 98))
    f = fit_upright(pf, Vz, faces, ref, height, 0.0, fit_footprint=True)
    Vw = place_upright(Vz, f["yaw"], f["scale"], f["xy"], 0.0, f["scale_xy"])
    members = place_members(pf, Vw, faces, ref, ids, spec.get("member_min_iou", 0.35))
    c_ref = (Vw.min(0) + Vw.max(0))[:2] / 2
    local = mesh.copy()
    local.vertices = Vw - np.r_[c_ref, 0.0]  # origin: footprint centre on the floor, axes: world
    local.export(d / "mesh_local.glb")
    tex = getattr(mesh.visual.material, "baseColorTexture", None)
    if tex is not None:
        tex.convert("RGB").save(d / "albedo.png")
    report.update({"ok": f["iou"] >= spec.get("min_iou", 0.5), "iou_ref": f["iou"], "height_m": height,
                   "scale_xy": f["scale_xy"], "extent_m": (Vw.max(0) - Vw.min(0)).tolist(), "members": members})
    s.write_json(f"objects/{group}/furniture.json", report)
    qa_overlay(pf, [(member_mesh(Vw, c_ref, m), faces) for m in members if not m.get("drop")], ids, d / "qa_furniture.png")
    print(f"[{group}] reference {ref}: IoU {f['iou']:.3f}, height {height:.2f} m, footprint x{f['scale_xy']:.2f}, "
          f"extent {np.round(report['extent_m'], 2).tolist()}; members "
          + ", ".join(f"{m['id']}: IoU {m['iou']:.2f} yaw {np.degrees(m['yaw']):+.0f}" + (" DROP" if m.get("drop") else "")
                    for m in members)
          + ("" if report["ok"] else " -> below min_iou, left to recon.retrieve"))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--refit", action="store_true", help="reuse objects/<group>/raw.glb (skip SAM 3D)")
    args = ap.parse_args()
    s = Session(args.session)
    for group, spec in s.scene_graph.get("movable_furniture", {}).items():
        if spec.get("shared") and (not args.only or group in args.only):
            furnish(s, group, spec, args.refit)


if __name__ == "__main__":
    main()
