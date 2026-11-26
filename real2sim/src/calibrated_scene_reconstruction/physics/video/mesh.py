"""Apply a reviewed agent Blender mesh and track it; SAM 3D is an explicit fallback.

    tools/run_capped.sh envs/recon/bin/python -m calibrated_scene_reconstruction.physics.video.mesh real/<name>

Default: verify objects/object/approved.json from calibrated_scene_reconstruction.agent.physics_models before
writing any output. Missing or stale evidence stops without fallback. The rest of
this description documents --object-route sam3d.

After perceive and track. Reconstructs the object from the reference frame (recon.objects: SAM 3D conditioned on the
MoGe-2 point map), stands it on the support plane and tracks it through the video by its silhouette, replacing the
shape-free or primitive track. Writes

  objects/object/{raw.glb, pose.json}   SAM 3D's output (recon.objects)
  mesh.npz        V (N, 3) in the object's frame (origin: the middle of its footprint on the plane, z up), F (M, 3),
                  UV (N, 2); Vs, Fs a coarse mesh with the same outline (raster.simplify); albedo.png its texture
  plate.png       the reference frame with everything that moves taken out (plate). The backdrop of the rendered
                  twin.
  twin.json       "object" becomes {"shape": "mesh", "mesh": "mesh.npz", "pose": [x, y, 0, yaw_deg] on the reference
                  frame, "dims" of its bounding cylinder, "mesh_iou": silhouette IoU on the reference frame}
  track.npz       xy, yaw of the mesh per frame (track.run's layout), iou its silhouette's IoU with the mask

The mesh's size is SAM 3D's at MoGe-2's scale, times the twin's scale factor (track.metric_scale); position, yaw and
a size correction of at most 15 % are then fitted to the reference mask.
"""
from calibrated_scene_reconstruction.physics.common.schema import read_document, write_document
import argparse
import json
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.physics.common.observations import cam_to_world, iou, mask_stats, project
from .raster import fill, simplify
from .scene import ray_plane, read_frames
from .track import on_border, touches


def plate(frames, moving, ref=-1, max_frames=60):
    """(H, W, 3) uint8: the reference frame with everything that moves taken out of it. Where the object or a
    manipulator covers the reference frame, the pixel is the median of the frames in which nothing covers it,
    brought to the reference frame's exposure (a phone re-exposes when a hand enters: the plain median of other
    frames leaves a ghost of a different brightness), and what no frame ever shows is inpainted."""
    import cv2
    from scipy.ndimage import binary_dilation

    ref = ref % len(frames)
    idx = np.unique(np.linspace(0, len(frames) - 1, max_frames).round().astype(int))
    stack = frames[idx].astype(np.float32)
    for k, i in enumerate(idx):
        stack[k][binary_dilation(moving[i], iterations=6)] = np.nan
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(stack, 0)
    base = frames[ref].astype(np.float32)
    # what else stands in the reference frame and not in most others (someone walking through, whom no prompt
    # segments) goes the same way: where the frame is far from the median of all
    from scipy.ndimage import binary_opening

    other = binary_opening(np.nan_to_num(np.abs(base - med).mean(-1), nan=0.0) > 35.0, iterations=3)
    mov_ref = moving[ref] | binary_dilation(other, iterations=4)
    # the object's soft shadow lies around its mask on the reference frame: take a wide margin out with it, and
    # feather the seam
    grow = int(np.clip(0.2 * np.sqrt(max(moving[ref].sum(), 1)), 12, 40))
    region = binary_dilation(mov_ref, iterations=grow)
    # the fill comes from the frames that show the whole region uncovered when there are any (their median is one
    # picture, taken at one exposure and white balance: its ratio to the reference frame around the region is the
    # ratio inside it); the per-pixel median over whichever frames uncover a pixel has no such single exposure
    clear = [k for k, i in enumerate(idx) if not (binary_dilation(moving[i], iterations=6) & region).any()]
    if clear:
        med = np.median(frames[idx[clear]].astype(np.float32), 0)
    # exposure: the smooth ratio of the reference frame to that picture around the region, carried into it
    seen = (~region & np.isfinite(med[..., 0])).astype(np.float32)[..., None]
    sig = 1.5 * grow
    num = cv2.GaussianBlur(base * seen, (0, 0), sig)
    den = cv2.GaussianBlur(np.nan_to_num(med) * seen, (0, 0), sig)
    gain = np.clip(num / np.maximum(den, 1e-3), 0.6, 1.6)
    gain[cv2.GaussianBlur(seen, (0, 0), sig)[..., None].repeat(3, -1) < 0.02] = 1.0
    fill = med * gain
    hole = region & ~np.isfinite(fill[..., 0])
    fill[~np.isfinite(fill)] = base[~np.isfinite(fill)]
    w = cv2.GaussianBlur(region.astype(np.float32), (0, 0), grow / 3)[..., None]
    w = np.where(region[..., None], np.maximum(w, 0.999 * binary_dilation(mov_ref, iterations=6)[..., None]), w)
    out = np.clip(base * (1 - w) + fill * w, 0, 255).astype(np.uint8)
    if hole.any():
        out = cv2.inpaint(out, hole.astype(np.uint8), 5, cv2.INPAINT_TELEA)
    return out


def silhouette(V, F, K, T, pose, shape):
    """Filled silhouette (H, W) of the mesh standing at pose (x, y, yaw_deg[, scale])."""
    c, s = np.cos(np.radians(pose[2])), np.sin(np.radians(pose[2]))
    k = pose[3] if len(pose) > 3 else 1.0
    W = (k * V) @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1.0]]) + [pose[0], pose[1], 0.0]
    return fill(project(K, T, W)[0][F], shape)


def fit_pose(mask, V, F, K, T, x0, free=(True, True, True, False), steps=(0.02, 0.02, 12.0, 0.05)):
    """(x, y, yaw_deg, scale) maximising the silhouette's IoU with `mask`, Nelder-Mead from x0 over the free ones."""
    from scipy.optimize import minimize

    x0 = np.asarray(x0, float)
    idx = [i for i in range(4) if free[i]]

    def cost(v):
        p = x0.copy()
        p[idx] = v
        if not 0.5 < p[3] < 2.0:
            return 2.0
        return 1.0 - iou(silhouette(V, F, K, T, p, mask.shape), mask)

    sim = np.vstack([x0[idx]] + [x0[idx] + np.eye(len(idx))[j] * steps[i] for j, i in enumerate(idx)])
    r = minimize(cost, x0[idx], method="Nelder-Mead", options={"initial_simplex": sim, "xatol": 2e-4, "fatol": 1e-4,
                                                               "maxiter": 250})
    p = x0.copy()
    p[idx] = r.x
    return p, 1.0 - float(r.fun)


def stand(path, twin, per):
    """SAM 3D's mesh stood on the plane on the reference frame. Returns (V local, F, UV, pose (4,), IoU, texture)."""
    import trimesh

    from calibrated_scene_reconstruction.assets.mesh_alignment import axis_rotations, centroid_height_ratio, local_to_cam_cv, support_area

    d = Path(path) / "objects" / "object"
    pose = read_document(d / "pose.json")
    glb = trimesh.load(d / "raw.glb", force="mesh")
    Vr, F = np.asarray(glb.vertices, float), np.asarray(glb.faces)
    K, ref = np.asarray(twin["K"], float), twin["ref_frame"]
    f = float(twin["scale"]["factor"])
    T = np.asarray(twin["T_world_cam"], float)
    T0 = T.copy()
    T0[2, 3] /= f  # the camera at MoGe-2's scale, where SAM 3D placed the mesh
    mask = per["object"][ref]
    cands = []
    for M in axis_rotations():
        v = local_to_cam_cv(Vr @ M.T, pose)
        Vw = cam_to_world(T0, v)
        vs, Fs = simplify(v, F)
        sil = fill(project(K, np.eye(4), vs)[0][Fs], mask.shape)
        cands.append((iou(sil, mask), support_area(Vw), centroid_height_ratio(Vw, F), Vw))
    top = max(c[0] for c in cands)
    near = [c for c in cands if c[0] >= top - 0.08]
    stable = [c for c in near if c[1] >= 0.5 * max(c[1] for c in near)]
    Vw = min(stable, key=lambda c: c[2])[3] * f  # scaled about the camera's foot, like the rest of the twin
    Vw[:, 2] -= np.percentile(Vw[:, 2], 1)  # on the plane
    foot = Vw[Vw[:, 2] < 0.01][:, :2]
    o = np.r_[(foot.max(0) + foot.min(0)) / 2 if len(foot) >= 3 else Vw[:, :2].mean(0), 0.0]
    V = Vw - o
    Vs, Fs = simplify(V, F)
    p, q = fit_pose(mask, Vs, Fs, K, T, [o[0], o[1], 0.0, 1.0], free=(True, True, True, True))
    p[3] = float(np.clip(p[3], 0.85, 1.15))
    p, q = fit_pose(mask, Vs, Fs, K, T, p, free=(True, True, True, False))
    V = V * p[3]
    uv = getattr(glb.visual, "uv", None)
    tex = getattr(getattr(glb.visual, "material", None), "baseColorTexture", None)
    return V, F, (np.asarray(uv, float) if uv is not None else None), p[:3], q, tex


def track_mesh(per, twin, V, F, pose_ref):
    """track.run's track dict with the mesh: from the reference frame outwards, each frame from the last one's pose
    moved onto its mask's centroid."""
    obj_m, score = per["object"], per["score"]
    Fr = len(obj_m)
    K, T, ref = np.asarray(twin["K"], float), np.asarray(twin["T_world_cam"], float), twin["ref_frame"]
    hz = float(V[:, 2].max()) / 2
    xy, yaw, q = np.full((Fr, 2), np.nan), np.full(Fr, np.nan), np.zeros(Fr)
    for order in (range(ref, Fr), range(ref, -1, -1)):
        p = np.r_[pose_ref, 1.0]
        for i in order:
            if score[i] <= 0:
                continue
            c = ray_plane(K, T, mask_stats(obj_m[i])[0], z=hz)
            if np.all(np.isfinite(c)):
                # the mesh's own centroid is not its origin: shift by the difference, not onto the ray point
                s0 = silhouette(V, F, K, T, p, obj_m[i].shape)
                c0 = ray_plane(K, T, mask_stats(s0)[0], z=hz) if s0.any() else np.full(3, np.nan)
                if np.all(np.isfinite(c0)):
                    p[:2] += (c - c0)[:2]
            p, q[i] = fit_pose(obj_m[i], V, F, K, T, p)
            xy[i], yaw[i] = p[:2], p[2]
    contact, border = touches(obj_m, per["manipulator"]), on_border(obj_m)
    return {"t": np.asarray(per["t"], float), "xy": xy, "yaw": yaw, "iou": q, "valid": (score > 0) & (q > 0.5),
            "contact": contact, "border": border}


def sam3d_run(path, device="cuda", reuse=False):
    import torch
    from PIL import Image

    from calibrated_scene_reconstruction.core.perception.sam3d import run_worker
    from .perceive import load_moge

    path = Path(path)
    frames, _fps = read_frames(path)
    per = np.load(path / "perception.npz")
    twin = read_document(path / "twin.json")
    K, ref = np.asarray(twin["K"], float), twin["ref_frame"]
    Image.fromarray(plate(frames, per["object"] | per["manipulator"], ref)).save(path / "plate.png")
    d = path / "objects" / "object"
    if not (reuse and (d / "raw.glb").exists()):
        model = load_moge(device)
        x = torch.tensor(frames[ref] / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
        with torch.no_grad():
            out = model.infer(x, fov_x=float(np.degrees(2 * np.arctan(frames.shape[2] / 2 / K[0, 0]))))
        pts = out["points"].cpu().numpy()
        del model, out
        torch.cuda.empty_cache()
        inputs = path / "sam3d_inputs.npz"
        np.savez_compressed(inputs, rgb=frames[ref], mask=per["object"][ref], points=pts)
        if run_worker(path, "object", [0], inputs) != 0:
            raise RuntimeError("SAM 3D failed")
    V, F, UV, pose, q, tex = stand(path, twin, per)
    Vs, Fs = simplify(V, F)  # the outline's mesh: silhouettes (tracking, the replay's loss)
    np.savez_compressed(path / "mesh.npz", V=V, F=F, Vs=Vs, Fs=Fs, **({"UV": UV} if UV is not None else {}))
    if tex is not None:
        tex.convert("RGB").save(path / "albedo.png")
    r = float(np.linalg.norm(V[:, :2], axis=1).max())
    twin["object"] = {"shape": "mesh", "mesh": "mesh.npz", "pose": [float(pose[0]), float(pose[1]), 0.0, float(pose[2])],
                      "dims": {"radius": r, "height": float(V[:, 2].max())}, "mesh_iou": q,
                      "first_frame_iou": q, "albedo": "albedo.png" if tex is not None else None}
    write_document(path / "twin.json", twin, indent=1)
    if read_document(path / "meta.json").get("motion") == "drop":
        return twin, dict(np.load(path / "track.npz"))  # in the air the mesh is not on the plane: the track stays
    track = track_mesh(per, twin, Vs, Fs, pose)
    np.savez_compressed(path / "track.npz", **track)
    return twin, track


def run(path, device="cuda", reuse=False, object_route="agent_blender"):
    """Use the reviewed agent model by default; SAM 3D is an explicit fallback."""
    if object_route == 'sam3d':
        return sam3d_run(path, device, reuse)
    if object_route != 'agent_blender':
        raise ValueError('Unknown object route: ' + object_route)
    from PIL import Image
    from .model_inputs import require_model

    path = Path(path)
    proof, V, F = require_model(path)
    with np.load(path / 'perception.npz') as data:
        per = dict(data)
    twin = read_document(path / 'twin.json')
    frames, _ = read_frames(path)
    Image.fromarray(plate(frames, per['object'] | per['manipulator'], twin['ref_frame'])).save(path / 'plate.png')
    Vs, Fs = simplify(V, F)
    np.savez_compressed(path / 'mesh.npz', V=V, F=F, Vs=Vs, Fs=Fs)
    pose = proof['world_pose']
    twin['object'] = {'shape': 'mesh', 'mesh': 'mesh.npz', 'pose': pose,
                      'dims': {'radius': float(np.linalg.norm(V[:, :2], axis=1).max()),
                               'height': float(V[:, 2].max())},
                      'mesh_iou': proof['iou_median'], 'first_frame_iou': proof['iou_median'],
                      'albedo': None, 'route': 'agent_blender', 'model_round': proof['round']}
    write_document(path / 'twin.json', twin, indent=1)
    if read_document(path / 'meta.json').get('motion') == 'drop':
        return twin, dict(np.load(path / 'track.npz'))
    track = track_mesh(per, twin, Vs, Fs, [pose[0], pose[1], pose[3]])
    np.savez_compressed(path / 'track.npz', **track)
    return twin, track


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording", nargs="+")
    ap.add_argument("--reuse", action="store_true", help="keep an existing objects/object/raw.glb")
    ap.add_argument('--object-route', choices=('agent_blender', 'sam3d'), default='agent_blender',
                    help='default: require a current reviewed Blender model; sam3d selects the fallback explicitly')
    args = ap.parse_args()
    failed = False
    for r in args.recording:
        try:
            twin, tr = run(r, reuse=args.reuse, object_route=args.object_route)
        except Exception as e:  # noqa: BLE001
            print(f"{r}: FAILED {type(e).__name__}: {e}", flush=True)
            failed = True
            continue
        o = twin["object"]
        print(f"{r}: mesh {o['dims']['height']:.3f} m tall, radius {o['dims']['radius']:.3f} m, IoU "
              f"{o['mesh_iou']:.2f} on frame {twin['ref_frame']}; tracked {int(tr['valid'].sum())}/{len(tr['t'])} "
              f"frames, median IoU {np.median(tr['iou'][tr['valid']]) if tr['valid'].any() else 0:.2f}", flush=True)
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
