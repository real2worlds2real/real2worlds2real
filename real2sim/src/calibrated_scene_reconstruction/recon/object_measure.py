"""The task objects' size and place measured in the splat: where hundreds of frames saw them, not one photo.

    tools/run_splat.sh -m calibrated_scene_reconstruction.recon.object_measure sessions/<name> [--apply]   (after splat_export, robot_fit)

A small thing on the table is a few dozen pixels in the photo, and SAM 3D's reconstruction from it guesses its size. The splat, trained on every frame the walk took
round the table, holds it at the scale the scene has. Per dynamic object: the splat's opaque, small Gaussians
(export/full.ply, before the crop) above the support's top (from `--lift` up to 0.4 m), not within `--robot_clear` of
the robot (its surface at the readout where recon.robot_fit placed it: a hand rests beside the objects), on a 5 mm
grid; the cluster connected to the object's place (the centre of its reconstruction in the twin, or the nearest cell
within `--search`) is the object, its box its size (2nd-98th percentile) and its centre its place. Writes
splat/object_measure.json and qa/splat/object_measure.png (top view per object: the cells, the cluster, the twin's
box); `--apply` writes each into the scene graph's dynamic_objects.<name>.fix (size, xy) with `_object_measure`,
which sim.scene builds from (the mesh scaled to the box axis by axis)."""
import argparse
import json

import cv2
import numpy as np
import trimesh
from scipy import ndimage
from scipy.spatial import cKDTree

from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.capture.robot_fit import rot_z
from calibrated_scene_reconstruction.scene.background.splat.config import folder
from calibrated_scene_reconstruction.scene.review.zone_check import world_gaussians

CELL = 0.005


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--lift", type=float, default=0.004, help="above the support's top, m")
    ap.add_argument("--robot_clear", type=float, default=0.03)
    ap.add_argument("--search", type=float, default=0.06, help="how far from the twin's centre the cluster may start")
    a = ap.parse_args()
    s = Session(a.session)
    out = folder(s)
    X, op, size = world_gaussians(s, out)
    sg = s.scene_graph
    sup = np.asarray(sg["furniture"][sg["task"]["support"]]["box"], float)
    z_top = float(sup[1, 2])
    keep = (op > 0.3) & (size < 0.03) & (X[:, 2] > z_top - 0.08) & (X[:, 2] < z_top + 0.4) & \
        np.all(X[:, :2] > sup[0, :2] - 0.1, 1) & np.all(X[:, :2] < sup[1, :2] + 0.1, 1)
    P = X[keep]
    fit_p = out / "robot_fit.json"
    rc_p = next((f for f in (s / "parsing" / "robot_cloud.npz", out / "robot_cloud.npz") if f.exists()), None)
    if fit_p.exists() and rc_p:
        fit = json.loads(fit_p.read_text())
        M = np.load(rc_p)["points"].astype(np.float64)
        R = M @ rot_z(np.radians(fit["base"]["yaw_deg"])).T + [*fit["base"]["xy"], fit["height_offset_m"]]
        d, _ = cKDTree(R).query(P)
        P = P[d > a.robot_clear]
    T_wc = np.asarray(s.manifest["photo_camera"]["T_world_cam_cv"], float)
    res, tiles = {}, []
    for name, spec in sg.get("dynamic_objects", {}).items():
        mesh = trimesh.load(s / "objects" / name / "mesh_cam.glb", force="mesh")
        V = np.asarray(mesh.vertices) @ T_wc[:3, :3].T + T_wc[:3, 3]
        c = (V.min(0) + V.max(0)) / 2
        half = max(0.15, float(np.ptp(V[:, :2], axis=0).max()))
        # the top's own height round the object in the splat (1 cm bins, the densest within 8 cm of the twin's): the
        # twin's top came from the photo's depth
        R_ = P[np.all(np.abs(P[:, :2] - c[:2]) < half, 1) & (np.abs(P[:, 2] - z_top) < 0.08)]
        zt = z_top
        if len(R_) > 200:
            h_, e_ = np.histogram(R_[:, 2], bins=np.arange(z_top - 0.08, z_top + 0.0801, 0.01))
            pk = e_[h_.argmax()] + 0.005
            zt = float(np.median(R_[np.abs(R_[:, 2] - pk) < 0.01, 2]))
        lo = np.r_[c[:2] - half, zt + a.lift]
        Q = P[np.all(P[:, :2] > lo[:2], 1) & np.all(P[:, :2] < c[:2] + half, 1) & (P[:, 2] > zt + a.lift)]
        if len(Q) < 30:
            res[name] = {"why": f"{len(Q)} points round it"}
            continue
        ijk = np.floor((Q - lo) / CELL).astype(int)
        n = ijk.max(0) + 1
        occ = np.zeros(n, bool)
        occ[tuple(ijk.T)] = True
        lab, _ = ndimage.label(ndimage.binary_closing(occ, np.ones((3, 3, 3))), structure=np.ones((3, 3, 3)))
        ci = np.floor((np.r_[c[:2], zt + 0.01] - lo) / CELL).astype(int)
        cells = np.argwhere(lab > 0)
        dist = np.linalg.norm((cells[:, :2] - ci[:2]) * CELL, axis=1)
        if not len(cells) or dist.min() > a.search:
            res[name] = {"why": f"no cluster within {a.search} m of the twin's place"}
            continue
        li = lab[tuple(cells[np.argmin(dist)])]
        member = lab[tuple(ijk.T)] == li
        O = Q[member]
        p_lo, p_hi = np.percentile(O, 2, 0), np.percentile(O, 98, 0)
        sz = p_hi - p_lo
        sz[2] = p_hi[2] - zt  # standing on the top
        res[name] = {"size": sz.round(4).tolist(), "xy": ((p_lo[:2] + p_hi[:2]) / 2).round(4).tolist(), "top_z": round(zt, 4),
                     "points": int(member.sum()), "twin_size": np.ptp(V, axis=0).round(4).tolist(),
                     "twin_xy": c[:2].round(4).tolist()}
        print(f"[measure] {name}: {sz[0] * 100:.1f} x {sz[1] * 100:.1f} x {sz[2] * 100:.1f} cm at "
              f"({res[name]['xy'][0]:.3f}, {res[name]['xy'][1]:.3f}) from {member.sum()} Gaussians; the twin had "
              f"{' x '.join(f'{v * 100:.1f}' for v in np.ptp(V, axis=0))} cm at ({c[0]:.3f}, {c[1]:.3f})", flush=True)
        # QA tile: top view, 2 mm px, grey the points round it, red the cluster, green the twin's box
        px = 0.002
        Wd = int(2 * half / px)
        img = np.full((Wd, Wd, 3), 255, np.uint8)
        for pts, col in ((Q, (170, 170, 170)), (O, (40, 40, 220))):
            ii, jj = ((pts[:, 0] - lo[0]) / px).astype(int), ((pts[:, 1] - lo[1]) / px).astype(int)
            k = (ii >= 0) & (ii < Wd) & (jj >= 0) & (jj < Wd)
            img[jj[k], ii[k]] = col
        b0, b1 = ((V.min(0)[:2] - lo[:2]) / px).astype(int), ((V.max(0)[:2] - lo[:2]) / px).astype(int)
        cv2.rectangle(img, tuple(b0), tuple(b1), (40, 170, 40), 1)
        img = cv2.flip(img, 0)
        cv2.putText(img, name, (6, 18), 0, 0.5, (0, 0, 0), 1)
        tiles.append(cv2.resize(img, (300, 300)))
    (out / "object_measure.json").write_text(json.dumps(res, indent=1) + "\n")
    if tiles:
        qa = s / "qa" / "splat"
        qa.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(qa / "object_measure.png"), np.hstack(tiles))
    if a.apply:
        g = s.config("scene_graph.json")
        for name, r in res.items():
            if "size" in r:
                g["dynamic_objects"][name]["fix"] = {**(g["dynamic_objects"][name].get("fix") or {}),
                                                     "size": r["size"], "xy": r["xy"]}
        g["_object_measure"] = {"note": "the task objects' boxes and places measured in the splat (recon.object_measure)",
                                "measured": res}
        (s.config_dir / "scene_graph.json").write_text(json.dumps(g, indent=1, ensure_ascii=False) + "\n")
        print(f"applied to {s.config_dir / 'scene_graph.json'}")


if __name__ == "__main__":
    main()
