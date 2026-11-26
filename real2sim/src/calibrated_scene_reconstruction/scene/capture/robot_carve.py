"""The real robot's place from the posed frames when there is no joint readout to fit: its torso carved as a column.

    tools/run_splat.sh -m calibrated_scene_reconstruction.recon.robot_carve sessions/<name> [--apply]
    (after recon.splat_masks --prompts "humanoid robot,robot head,robot hand,mobile robot base" --suffix _robot)

A 2 cm grid on the floor round the support; over each cell a column of points from `--z0` to `--z1` (the torso's
height above the support's top, where the arms' reach over the top is thin and the desk hides nothing). In every
posed frame that sees the whole column, the share of its points inside the robot's silhouette (<image>_robot.png);
a cell is the robot's when that share is over 0.8 in `--agree` of the frames that see it (at least `--min-frames`).
The base is the largest such region's middle; it faces the support's nearest point (a robot stands at the table it
works at). Writes splat/robot_carve.json and qa/splat/robot_carve.png (top view: the support's box, the carved
region, the base and its facing); `--apply` writes the scene graph's photo_robot.base with `_base_from`.

With a readout, recon.robot_fit fits the robot's surface instead (millimetres); this places it to a few centimetres,
which the delivery reports.
"""
import argparse
import json

import cv2
import numpy as np
import pycolmap

from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.background.splat.config import folder


def overlay(s, rec, img_dir, c, R, t, bx, by, yaw, suffix, n=6):
    """qa/splat/robot_carve_frames.jpg: the home pose's base and torso placed at the carved base, drawn into n frames
    that show the robot (its silhouette outlined) -- S1's check that the robot lands on itself."""
    home = s / "parsing" / "robot_cloud_home.npz"
    if not home.exists():
        return
    hz = np.load(home)
    keep = np.isin(hz["link"], [i for i, nm in enumerate(hz["links"]) if any(k in str(nm) for k in ("base", "wheel", "torso"))])
    P = hz["points"][keep][::20].astype(np.float64)
    cy_, sy_ = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    W = np.c_[bx + cy_ * P[:, 0] - sy_ * P[:, 1], by + sy_ * P[:, 0] + cy_ * P[:, 1], P[:, 2]]
    Pc = (W - t) @ R / c
    shots = []
    for img in sorted(rec.images.values(), key=lambda i: i.name):
        mp = img_dir / (img.name[:-4] + suffix + ".png")
        if mp.exists() and (cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) < 128).mean() > 0.05:
            shots.append(img)
    if not shots:
        return
    tiles = []
    for img in [shots[int(k)] for k in np.linspace(0, len(shots) - 1, min(n, len(shots)))]:
        cam = rec.cameras[img.camera_id]
        fx, fy, cx, cy = (float(x) for x in cam.params[:4])
        M = np.asarray(img.cam_from_world().matrix())
        X = Pc @ M[:, :3].T + M[:, 3]
        ok = X[:, 2] > 0.1
        u, v = fx * X[ok, 0] / X[ok, 2] + cx, fy * X[ok, 1] / X[ok, 2] + cy
        im = cv2.imread(str(img_dir / img.name))
        m = (cv2.imread(str(img_dir / (img.name[:-4] + suffix + ".png")), cv2.IMREAD_GRAYSCALE) < 128).astype(np.uint8)
        cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, cnt, -1, (0, 0, 255), 3)
        for x_, y_ in zip(u.astype(int), v.astype(int)):
            if 0 <= x_ < im.shape[1] and 0 <= y_ < im.shape[0]:
                cv2.circle(im, (x_, y_), 2, (255, 160, 30), -1)
        cv2.putText(im, img.name, (20, 50), 0, 1.4, (0, 255, 255), 3)
        tiles.append(cv2.resize(im, (640, 360)))
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(s / "qa" / "splat" / "robot_carve_frames.jpg"),
                np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--suffix", default="_robot")
    ap.add_argument("--reach", type=float, default=2.0, help="m round the support searched")
    ap.add_argument("--cell", type=float, default=0.02)
    ap.add_argument("--z0", type=float, default=0.15, help="m above the support's top: the column's foot")
    ap.add_argument("--z1", type=float, default=0.45, help="m above the support's top: the column's head")
    ap.add_argument("--agree", type=float, default=0.4,
                    help="Minimum fraction of visible frames in which the robot silhouette covers the column")
    ap.add_argument("--min-frames", type=int, default=6)
    a = ap.parse_args()
    s = Session(a.session)
    out = folder(s)
    al = json.loads((out / "align.json").read_text())
    c, R, t = al["sim3"]["scale"], np.asarray(al["sim3"]["R"]), np.asarray(al["sim3"]["t"])
    sg = s.scene_graph
    lo, hi = (np.asarray(b, float) for b in sg["furniture"][sg["task"]["support"]]["box"])
    top = float(hi[2])
    xs = np.arange(lo[0] - a.reach, hi[0] + a.reach, a.cell)
    ys = np.arange(lo[1] - a.reach, hi[1] + a.reach, a.cell)
    gx, gy = np.meshgrid(xs, ys)
    cells = np.stack([gx.ravel(), gy.ravel()], 1)
    zs = top + np.linspace(a.z0, a.z1, 5)
    rec = pycolmap.Reconstruction(str(out / "data" / "sparse" / "0"))
    img_dir = out / "data" / "images"
    seen = np.zeros(len(cells), np.int32)
    hit = np.zeros(len(cells), np.int32)
    n_img = 0
    for img in rec.images.values():
        mp = img_dir / (img.name[:-4] + a.suffix + ".png")
        if not mp.exists():
            continue
        m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) < 128  # (splat_masks: 0 where the prompts are, 255 elsewhere)
        if m.mean() < 0.002:
            continue
        cam = rec.cameras[img.camera_id]
        fx, fy, cx, cy = (float(x) for x in cam.params[:4])
        M = np.asarray(img.cam_from_world().matrix())
        inside = np.ones(len(cells), bool)
        frac = np.zeros(len(cells))
        for z in zs:
            P = np.c_[cells, np.full(len(cells), z)]
            X = ((P - t) @ R / c) @ M[:, :3].T + M[:, 3]
            ok = X[:, 2] > 0.1
            u = np.where(ok, fx * X[:, 0] / np.where(ok, X[:, 2], 1) + cx, -1)
            v = np.where(ok, fy * X[:, 1] / np.where(ok, X[:, 2], 1) + cy, -1)
            ok &= (u >= 0) & (u < m.shape[1] - 1) & (v >= 0) & (v < m.shape[0] - 1)
            inside &= ok
            idx = np.flatnonzero(ok)
            frac[idx] += m[v[idx].astype(int), u[idx].astype(int)]
        frac /= len(zs)
        seen[inside] += 1
        hit[inside & (frac > 0.8)] += 1
        n_img += 1
    share = hit / np.maximum(seen, 1)
    print(f"[robot_carve] columns seen by >= {a.min_frames} frames: {(seen >= a.min_frames).sum()}; the robot's share in them: "
          f"max {share[seen >= a.min_frames].max() if (seen >= a.min_frames).any() else 0:.2f}, cells over 0.3: "
          f"{((share > 0.3) & (seen >= a.min_frames)).sum()}", flush=True)
    robot = ((hit >= a.agree * np.maximum(seen, 1)) & (seen >= a.min_frames)).reshape(gy.shape)
    K = cv2.morphologyEx(robot.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(K)
    if n < 2:
        raise SystemExit(f"robot_carve: no column is the robot's in {n_img} frames (masks {a.suffix})")
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    jj, ii = np.nonzero(lab == big)
    bx, by = float(xs[ii].mean()), float(ys[jj].mean())
    # it faces the support's nearest point
    nx, ny = float(np.clip(bx, lo[0], hi[0])), float(np.clip(by, lo[1], hi[1]))
    yaw = float(np.degrees(np.arctan2(ny - by, nx - bx)))
    # the column is the torso, not the base's origin: the robot's own body at those heights (its home pose,
    # parsing/robot_cloud_home.npz) lies off its origin -- moved back by it, turned with the facing
    home = s / "parsing" / "robot_cloud_home.npz"
    if home.exists():
        hz = np.load(home)
        torso = np.isin(hz["link"], [i for i, n in enumerate(hz["links"]) if "torso" in str(n)])  # (the arms reach
        H = hz["points"][torso]  # forward in the home pose: the torso's own links alone)
        band = H[(H[:, 2] > top + a.z0) & (H[:, 2] < top + a.z1)]
        if len(band) > 100:
            d = np.median(band[:, :2], 0)
            cy_, sy_ = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
            bx, by = bx - (cy_ * d[0] - sy_ * d[1]), by - (sy_ * d[0] + cy_ * d[1])
            print(f"[robot_carve] the torso at those heights {100 * np.hypot(*d):.0f} cm off the base's origin "
                  f"(home pose): the base moved back by it", flush=True)
    res = {"base": {"xy": [round(bx, 4), round(by, 4)], "yaw_deg": round(yaw, 1)},
           "region_m2": round(float(len(ii)) * a.cell ** 2, 4), "region_extent_m": [round(float(np.ptp(xs[ii])), 3),
                                                                                    round(float(np.ptp(ys[jj])), 3)],
           "frames": n_img, "column_m_over_top": [a.z0, a.z1], "facing": "the support's nearest point"}
    (out / "robot_carve.json").write_text(json.dumps(res, indent=1) + "\n")
    vis = np.full((*gy.shape, 3), 255, np.uint8)
    vis[seen.reshape(gy.shape) < a.min_frames] = (225, 225, 225)
    vis[lab == big] = (30, 140, 255)
    px = lambda x, y: (int((x - xs[0]) / a.cell), int((y - ys[0]) / a.cell))  # noqa: E731
    cv2.rectangle(vis, px(lo[0], lo[1]), px(hi[0], hi[1]), (60, 60, 60), 2)
    cv2.arrowedLine(vis, px(bx, by), px(bx + 0.4 * np.cos(np.radians(yaw)), by + 0.4 * np.sin(np.radians(yaw))),
                    (220, 60, 30), 3)
    (s / "qa" / "splat").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(s / "qa" / "splat" / "robot_carve.png"), cv2.flip(vis, 0))
    print(f"[robot_carve] {n_img} frames: the robot's column over {res['region_m2']:.3f} m2 "
          f"({res['region_extent_m'][0]:.2f} x {res['region_extent_m'][1]:.2f} m), base ({bx:.3f}, {by:.3f}) facing "
          f"{yaw:.1f} deg (the support's nearest point)", flush=True)
    overlay(s, rec, img_dir, c, R, t, bx, by, yaw, a.suffix)
    if a.apply:
        sgc = s.config("scene_graph.json")
        sgc.setdefault("photo_robot", {})["base"] = res["base"]
        sgc["photo_robot"]["_base_from"] = ("recon.robot_carve: the robot's torso carved as a column from its "
                                            "silhouettes in the posed frames (no joint readout)")
        (s.config_dir / "scene_graph.json").write_text(json.dumps(sgc, indent=1, ensure_ascii=False) + "\n")
        print(f"applied to {s.config_dir / 'scene_graph.json'}", flush=True)


if __name__ == "__main__":
    main()
