"""Photo measurements for the set pieces the agent put in place of reconstructions (scene graph static_objects.<name>
.fix {"kit": ...}, sim.setpieces): where the piece stands, how long and tall it is, and the photo's colours along it.

    python -m calibrated_scene_reconstruction.recon.setpieces sessions/<name> [--only books]

The instance masks often hold part of a row only (seven of twenty spines in a rack): with a `prompt` in the fix
("row of books in a book rack"), SAM 3 finds the whole thing and its detections that overlap the instance masks are
added. 3D comes from the photo's point map (parsing/geometry.npz, the world frame of the twin), not from rays on the
support plane (the camera looks at a table top almost edge-on: a pixel of error is decimetres there).
  row     (book_row) the points' principal horizontal axis; the front faces the camera; length = 1st..99th
          percentile along it, height above the support = 98th percentile; the colours every 1.5 cm along the row
          (median of the mask's middle band, linear RGB, brightened out of the rack's shadow: spine_albedo)
  round   (tray_set) centre and radius of the footprint
  box     (cooktop and other flat inlays) the oriented footprint
Writes objects/<name>/kit.json, which sim.setpieces reads (fix params override it).
"""
import argparse
import json

import numpy as np
from PIL import Image

from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.assets.support_geometry import TOP_TOL, support_top  # noqa: F401 -- legacy API

BIN = 0.015


def spine_albedo(cols, white=0.7, max_gain=5.0):
    """Linear photo colours of a row's spines -> albedo: a rack against a window is in its own shadow, so the
    brightest spines (the 90th percentile of luminance: white and cream paper) are scaled to `white`."""
    lum = cols @ np.array([0.2126, 0.7152, 0.0722])
    gain = np.clip(white / max(float(np.percentile(lum, 90)), 1e-3), 1.0, max_gain)
    return np.clip(gain * cols, 0, 0.9)


def world_points(s, M):
    g = np.load(s / "parsing" / "geometry.npz")
    T = np.asarray(s.manifest["photo_camera"]["T_world_cam_cv"], float)
    P = g["points"][M]
    ok = np.isfinite(P).all(1)
    return P @ T[:3, :3].T + T[:3, 3], ok


def support_z(s, lo, hi, bottom):
    return support_top(s.manifest["collision_boxes"], lo, hi, bottom)


def measure(s, name, spec, fix, sam=None):
    masks = np.load(s / "parsing" / "masks.npz")["masks"]
    M = np.any(masks[list(spec["ids"])], 0)
    photo = np.asarray(Image.open(s / "capture" / "rgb.png").convert("RGB"), np.float32) / 255.0
    if fix.get("prompt") and sam is not None:
        added = 0
        for inst in sam(Image.fromarray((photo * 255).astype(np.uint8)), [fix["prompt"]]):
            m = inst["mask"]
            if (m & M).sum() > 0.3 * min(m.sum(), M.sum()):
                M |= m
                added += 1
        print(f"[setpieces] {name}: SAM 3 '{fix['prompt']}' added {added} region(s)", flush=True)
    W, ok = world_points(s, M)
    uv = np.argwhere(M)[ok]
    W = W[ok]
    lo, hi = np.percentile(W, 2, 0), np.percentile(W, 98, 0)
    keep = ((W >= lo - 0.02) & (W <= hi + 0.02)).all(1)
    W, uv = W[keep], uv[keep]
    z0 = support_z(s, W[:, :2].min(0), W[:, :2].max(0), float(np.percentile(W[:, 2], 1)))
    cam = np.asarray(s.manifest["photo_camera"]["T_world_cam_cv"], float)[:2, 3]
    c = np.median(W[:, :2], 0)
    kind = fix["kit"]
    out = {"kit": kind, "support_z": z0, "pixels": int(M.sum())}
    if kind == "tray_set":
        r = float(np.percentile(np.linalg.norm(W[:, :2] - c, axis=1), 97))
        out.update(center=c.round(4).tolist(), radius=round(r, 3), height=float(np.percentile(W[:, 2], 98) - z0))
    else:
        _, _, Vt = np.linalg.svd(W[:, :2] - c, full_matrices=False)
        u = Vt[0]
        n = np.array([-u[1], u[0]])
        if (cam - c) @ n < 0:  # n: the front, toward the camera
            n = -n
        if u[0] * -n[1] - u[1] * -n[0] < 0:  # x along the row, y = -front: right-handed
            u = -u
        t, d = (W[:, :2] - c) @ u, (W[:, :2] - c) @ n
        t0, t1 = np.percentile(t, [1, 99])
        front = float(np.percentile(d, 90))
        out.update(length=round(float(t1 - t0), 3), height=round(float(np.percentile(W[:, 2], 98) - z0), 3),
                   front=(c + 0.5 * (t0 + t1) * u + front * n).round(4).tolist(),
                   yaw=float(np.arctan2(u[1], u[0])), seen_depth=round(float(front - np.percentile(d, 10)), 3))
        if kind == "book_row":
            zc = W[:, 2] - z0
            band = (zc > 0.35 * out["height"]) & (zc < 0.75 * out["height"])
            cols = []
            for b0 in np.arange(t0, t1, BIN):
                sel = band & (t >= b0) & (t < b0 + BIN)
                px = photo[uv[sel, 0], uv[sel, 1]]
                if len(px) > 5:
                    cols.append(np.median(px, 0) ** 2.2)
            out["colours"] = spine_albedo(np.array(cols)).round(3).tolist() if cols else []
        elif kind == "cooktop":
            out["depth"] = round(float(np.percentile(d, 99) - np.percentile(d, 1)), 3)
    (s / "objects" / name).mkdir(parents=True, exist_ok=True)
    (s / "objects" / name / "kit.json").write_text(json.dumps(out, indent=1))
    print(f"[setpieces] {name}: {kind} " + ", ".join(f"{k} {v}" for k, v in out.items()
                                                     if k in ("length", "height", "radius", "depth", "center")),
          flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    s = Session(args.session)
    todo = {n: sp for n, sp in s.scene_graph.get("static_objects", {}).items()
            if (sp.get("fix") or {}).get("kit") and sp.get("ids") and (not args.only or n in args.only)}
    if not todo:
        print("[setpieces] no kit fixes in the scene graph")
        return
    sam = None
    if any(sp["fix"].get("prompt") for sp in todo.values()):
        from calibrated_scene_reconstruction.scene.capture.perceive import run_sam3

        def sam(pil, prompts):
            return run_sam3(pil, prompts, "cuda", threshold=0.35)
    for name, spec in todo.items():
        measure(s, name, spec, spec["fix"], sam)


if __name__ == "__main__":
    main()
