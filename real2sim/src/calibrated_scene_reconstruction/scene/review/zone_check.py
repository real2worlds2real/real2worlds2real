"""The twin's task zone against the trained splat: does the zone's geometry stand where the frames saw it?

    tools/run_splat.sh -m calibrated_scene_reconstruction.recon.zone_check sessions/<name>        (after recon.splat_export: export/full.ply)

The splat is trained on the posed frames at the board's scale (recon.splat_capture), and before the export's crop
it holds the task zone too: the support, the rig on it, what stands round it. Its opaque, small Gaussians (opacity >
0.3, largest axis < 10 cm) on a 4 cm grid are the frames' geometry; the twin's collision boxes (scene/manifest.json)
and the robot's column are what the twin puts there. Writes splat/zone_check.json and qa/splat/zone_check.png
(top view and front view: grey the splat's geometry, green the twin's boxes, red the unexplained clusters)."""
import argparse
import json

import cv2
import numpy as np
from plyfile import PlyData
from scipy import ndimage

from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.background.splat.config import folder, settings, sim3
from calibrated_scene_reconstruction.scene.background.splat.export import unnormalize

CELL = 0.04


def world_gaussians(s, out):
    """(X world, opacity, largest axis m) of the trained model before the crop (export/full.ply)."""
    v = PlyData.read(str(out / "export" / "full.ply"))["vertex"].data
    P = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)
    if json.loads((out / "train.json").read_text())["settings"].get("normalize", True):
        P, v = unnormalize(out, P, v)
    c, R, t = sim3(s)
    op = 1 / (1 + np.exp(-v["opacity"].astype(np.float64)))
    size = c * np.exp(np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], 1).astype(np.float64)).max(1)
    return c * P @ R.T + t, op, size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--reach", type=float, default=0.5, help="how far round the support's box to look, m")
    ap.add_argument("--min-cluster", type=int, default=15, help="cells an unexplained cluster needs")
    a = ap.parse_args()
    s = Session(a.session)
    out = folder(s)
    crop = settings(s)["crop"]
    X, op, size = world_gaussians(s, out)
    X = X[(op > 0.3) & (size < 0.1)]
    sg = s.scene_graph
    key = sg["task"]["support"]
    boxes = {k: np.asarray(b, float) for k, b in s.manifest["collision_boxes"].items()}
    slo, shi = boxes[key]
    base = (sg.get("photo_robot") or {}).get("base")
    lo = np.r_[slo[:2] - a.reach, crop["keep_floor"] + 0.02]
    hi = np.r_[shi[:2] + a.reach, crop["robot_h"] + 0.3]
    n = np.ceil((hi - lo) / CELL).astype(int)
    occ = np.zeros(n, bool)
    ijk = np.floor((X - lo) / CELL).astype(int)
    ok = np.all((ijk >= 0) & (ijk < n), 1)
    occ[tuple(ijk[ok].T)] = True
    centres = np.stack(np.meshgrid(*[lo[d] + (np.arange(n[d]) + 0.5) * CELL for d in range(3)], indexing="ij"), -1)

    def near(b, pad):
        return np.all((centres >= b[0] - pad) & (centres <= b[1] + pad), -1)

    twin = np.zeros(n, bool)
    for b in boxes.values():
        twin |= near(b, 0.05)
    if base:
        twin |= ((np.hypot(centres[..., 0] - base["xy"][0], centres[..., 1] - base["xy"][1]) < crop["robot_r"])
                 & (centres[..., 2] < crop["robot_h"]))
    for o in s.config("scene_graph.json").get("dynamic_objects", {}).values():
        pass  # the objects on the support are a few cm: inside the support's 5 cm pad or too small for a cluster
    # boxes: the share of each box's cells with the splat's geometry within 3 cm
    occ_d = ndimage.binary_dilation(occ, iterations=1)
    rep = {"boxes": {}, "unexplained": []}
    for k, b in boxes.items():
        if not (np.all(b[0][:2] < hi[:2]) and np.all(b[1][:2] > lo[:2])):
            continue
        m = near(b, 0.0)
        if not m.any():
            continue
        rep["boxes"][k] = {"box": b.round(3).tolist(), "seen_share": round(float(occ_d[m].mean()), 3)}
    lab, nl = ndimage.label(occ & ~twin, structure=np.ones((3, 3, 3)))
    for i in range(1, nl + 1):
        m = lab == i
        if m.sum() < a.min_cluster:
            continue
        c = centres[m]
        rep["unexplained"].append({"cells": int(m.sum()), "box": [c.min(0).round(2).tolist(), c.max(0).round(2).tolist()]})
    rep["unexplained"].sort(key=lambda u: -u["cells"])
    (out / "zone_check.json").write_text(json.dumps(rep, indent=1) + "\n")
    for k, r in rep["boxes"].items():
        if r["seen_share"] < 0.5:
            print(f"[zone] {k}: {100 * r['seen_share']:.0f} % of its cells where the frames saw geometry", flush=True)
    for u in rep["unexplained"][:12]:
        print(f"[zone] unexplained {u['cells']} cells at x {u['box'][0][0]}..{u['box'][1][0]}, "
              f"y {u['box'][0][1]}..{u['box'][1][1]}, z {u['box'][0][2]}..{u['box'][1][2]}", flush=True)

    # QA: top view (max over z) and front view (max over y)
    def view(axis):
        g = occ.any(axis)
        t = twin_boxes = np.zeros(n, bool)
        for b in boxes.values():
            t |= near(b, 0.0)
        t = twin_boxes.any(axis)
        u = (lab > 0).any(axis) & np.isin(lab, [i for i in range(1, nl + 1) if (lab == i).sum() >= a.min_cluster]).any(axis)
        img = np.full((*g.shape, 3), 255, np.uint8)
        img[g] = (150, 150, 150)
        img[t] = (60, 170, 60)
        img[u & ~t] = (40, 40, 220)
        img = np.flipud(np.swapaxes(img, 0, 1))
        return cv2.resize(img, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST)
    top, front = view(2), view(1)
    h = max(top.shape[0], front.shape[0])
    pad = lambda im: np.pad(im, ((0, h - im.shape[0]), (0, 10), (0, 0)), constant_values=255)  # noqa: E731
    qa = s / "qa" / "splat"
    qa.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(qa / "zone_check.png"), np.hstack([pad(top), pad(front)]))
    print(f"done: {len(rep['boxes'])} boxes checked, {len(rep['unexplained'])} unexplained clusters -> "
          f"{out / 'zone_check.json'}, {qa / 'zone_check.png'}")


if __name__ == "__main__":
    main()
