"""Minimal tabletop twin of a recording from its first video frame, in the recording's world frame (camera.json), for
recordings whose scene was never reconstructed by the pipeline (public robot datasets).

    python -m calibrated_scene_reconstruction.physics.replay.tabletop real/<name>      # GPU: SAM 3 support mask + MoGe-2 on the first frame; CPU fit

Needs track.npz (calibrated_scene_reconstruction.physics.replay.observe: the object's first mask) and meta.json "twin": {"support": SAM 3 prompt of the
surface the object stands on, "shape": "cylinder" | "box"}. Writes real/<name>/{twin.json, twin_qa.png}.

support  SAM 3 mask of the support minus the dilated object mask; MoGe-2 point map with the camera's horizontal field of
         view from K (so depth and K agree) -> world points through camera.json -> the plane: the median height of the
         inliers (the robot base, hence world z, is level; the tilt of a free plane fit is reported as a check) and
         the 2nd-98th percentile xy extent -> a static 5 cm box whose top is the plane.
object   an upright primitive standing on the plane: cylinder (radius, height) or box (size x, size y, height, yaw).
         Initialised from the MoGe-2 points of the mask, then fitted by Nelder-Mead on 1 - IoU of its projected
         silhouette (exact: the convex hull of the projected vertices) and the first mask. That IoU is the static
         first-frame check: replaying a twin that does not match the first frame only calibrates its errors.
Metric depth carries uncertainty. Scaling the whole twin about the camera
centre (`scaled`) leaves every silhouette in this camera unchanged, so one view cannot tell those twins apart, but the
robot (metric through its kinematics and the calibrated extrinsics) touches the object earlier or later, on another
side: the replay fits that scale as a nuisance parameter (params `twin.depth_scale`, and `twin.support_dz` for an
error that is not a scale) next to the physical ones. A non-convex object (mug handle, bowl) is approximated by its
hull.
"""
from calibrated_scene_reconstruction.physics.common.schema import write_document
from calibrated_scene_reconstruction.assets.primitive_mesh import primitive_mesh as primitive_mesh  # noqa: F401
import argparse
import copy
import json

import numpy as np

from .data import load
from .observe import cam_to_world, convex_silhouette, iou, read_video

SUPPORT_THICKNESS = 0.05




def pose_matrix(pose):
    """[x, y, z, yaw_deg] -> 4x4."""
    x, y, z, yaw = pose
    c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    T[:3, 3] = x, y, z
    return T


def object_points(obj):
    """World vertices of the twin object (twin.json "object")."""
    if obj['shape'] == 'mesh':
        from calibrated_scene_reconstruction.assets.model_artifact import read_mesh
        from calibrated_scene_reconstruction.core.paths import resolve
        V, _F = read_mesh(resolve(obj['mesh']))
    else:
        V, _F = primitive_mesh(obj["shape"], obj["dims"])
    T = pose_matrix(obj["pose"])
    return V @ T[:3, :3].T + T[:3, 3]


def scaled(twin, T_world_cam, s, dz=0.0):
    """The twin scaled by s about the camera centre (support plane, object pose and size): the same first-frame
    silhouettes for every s, the depth ambiguity of a monocular twin. dz then lifts support and object together: not
    silhouette-preserving (the IoU of the frames before contact prices it), for depth errors that are not a scale."""
    c = np.asarray(T_world_cam, float)[:3, 3]
    t = copy.deepcopy(twin)
    o = t["object"]
    if o.get('route') == 'agent_blender':
        from .params import require_fixed_geometry
        require_fixed_geometry(calibrations=[{'twin': {'depth_scale': s, 'support_dz': dz}}])
        return t
    o["pose"][:3] = (c + s * (np.asarray(o["pose"][:3], float) - c)).tolist()
    o["dims"] = {k: float(v) * s for k, v in o["dims"].items()}
    lo, hi = (c + s * (np.asarray(v, float) - c) for v in t["support"]["box"])
    hi[2] += dz
    lo[2] = hi[2] - SUPPORT_THICKNESS
    o["pose"][2] += dz
    t["support"].update(box=[lo.tolist(), hi.tolist()], z=float(hi[2]))
    return t


def fit_plane(P, tol=0.02, iters=5):
    """Horizontal plane z of world points (N, 3) by trimmed median; returns (z, tilt_deg of a free fit, inliers)."""
    z = np.median(P[:, 2])
    for _ in range(iters):
        inl = np.abs(P[:, 2] - z) < tol
        z = np.median(P[inl, 2])
    Q = P[inl] - P[inl].mean(0)
    n = np.linalg.eigh(Q.T @ Q)[1][:, 0]
    return float(z), float(np.degrees(np.arccos(abs(n[2])))), inl


def _dims(shape, v):
    if shape == "cylinder":
        return {"radius": float(v[2]), "height": float(v[3])}
    return {"size_x": float(v[2]), "size_y": float(v[3]), "height": float(v[4])}


def fit_object(mask, K, T_world_cam, z, shape, init):
    """Primitive on the plane z whose projection best overlaps `mask`; init = (x, y, *dims[, yaw_deg]).
    Returns the twin.json object dict (without name / prompt) and the IoU."""
    from scipy.optimize import minimize

    def obj_of(v):
        v = np.asarray(v, float)
        return {"shape": shape, "dims": _dims(shape, np.abs(v)), "pose": [float(v[0]), float(v[1]), z,
                                                                             float(v[5]) if shape == "box" else 0.0]}

    nd = 3 if shape == "box" else 2  # size parameters after x, y

    def cost(v):
        if np.any(np.asarray(v[2:2 + nd]) <= 0.005):
            return 2.0
        return 1.0 - iou(convex_silhouette(K, T_world_cam, object_points(obj_of(v)), mask.shape), mask)

    starts = [np.r_[init[:5], yaw] for yaw in (0.0, 30.0, 60.0)] if shape == "box" else [np.asarray(init, float)]
    best = None
    for x0 in starts:
        step = np.r_[0.02, 0.02, 0.3 * x0[2:2 + nd], [15.0] * (shape == "box")]  # m, m, relative, deg
        sim = np.vstack([x0] + [x0 + np.eye(len(x0))[i] * step[i] for i in range(len(x0))])
        r = minimize(cost, x0, method="Nelder-Mead", options={"initial_simplex": sim, "xatol": 1e-4, "fatol": 1e-4,
                                                               "maxiter": 600})
        if best is None or r.fun < best.fun:
            best = r
    return obj_of(best.x), 1.0 - float(best.fun)


def build(rec, rgb, support_mask, points, obj_mask, name=None):
    """twin.json dict from the first frame: support mask, MoGe-2 camera-frame point map (H, W, 3), object mask."""
    from scipy.ndimage import binary_dilation, binary_erosion

    tw = rec.meta.get("twin", {})
    shape = tw.get("shape", "cylinder")
    T, K = rec.T_world_cam, rec.K
    sup = support_mask & ~binary_dilation(obj_mask, iterations=3) & np.all(np.isfinite(points), -1)
    Pw = cam_to_world(T, points[sup])
    z, tilt, inl = fit_plane(Pw)
    lo, hi = np.percentile(Pw[inl, :2], 2, 0), np.percentile(Pw[inl, :2], 98, 0)
    core = binary_erosion(obj_mask, iterations=1) & np.all(np.isfinite(points), -1)
    Po = cam_to_world(T, points[core if core.sum() >= 10 else obj_mask])
    xy = np.median(Po[:, :2], 0)
    height = max(float(np.percentile(Po[:, 2], 95) - z), 0.02)
    _ys, xs = np.nonzero(obj_mask)
    depth = float(np.median((points[obj_mask])[:, 2]))
    width = (xs.max() - xs.min() + 1) * depth / K[0, 0]
    ray = xy - T[:2, 3]
    xy = xy + width / 2 * ray / np.linalg.norm(ray)  # the mask shows the front surface: the centre is behind it
    init = [xy[0], xy[1], width / 2, height] if shape == "cylinder" else [xy[0], xy[1], width, width, height]
    obj, score = fit_object(obj_mask, K, T, z, shape, init)
    obj.update({"name": name or "".join(c if c.isalnum() else "_" for c in rec.meta["object"]),
                "prompt": rec.meta["object"], "mass": tw.get("mass", 0.3), "first_frame_iou": score,
                "init": {"xy": [float(v) for v in xy], "width": float(width), "height": height}})
    return {"frame": "world of camera.json (DROID: the robot base)",
            "support": {"prompt": tw.get("support", "table"), "z": z, "tilt_deg": tilt, "points": int(inl.sum()),
                        "box": [[float(lo[0]), float(lo[1]), z - SUPPORT_THICKNESS], [float(hi[0]), float(hi[1]), z]]},
            "object": obj, "camera_height_above_support": float(T[2, 3] - z)}


def qa_image(rgb, twin, rec, obj_mask):
    """First frame: object mask in green, the fitted primitive's silhouette in red, the support box top edges in blue."""
    from scipy.ndimage import binary_erosion

    from .observe import project

    vis = rgb.astype(float).copy()
    sil = convex_silhouette(rec.K, rec.T_world_cam, object_points(twin["object"]), obj_mask.shape)
    for m, col in ((obj_mask, (0, 255, 0)), (sil, (255, 0, 0))):
        edge = m & ~binary_erosion(m)
        vis[edge] = col
    (x0, y0, _), (x1, y1, z) = twin["support"]["box"]
    for a, b in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)), ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
        s = np.linspace(0, 1, 200)[:, None]
        P = np.c_[(1 - s) * np.array(a) + s * np.array(b), np.full(len(s), z)]
        uv = project(rec.K, rec.T_world_cam, P)[0]
        for u, v in uv[np.all(np.isfinite(uv), 1)].astype(int):
            if 0 <= u < vis.shape[1] and 0 <= v < vis.shape[0]:
                vis[v, u] = (0, 128, 255)
    return vis.astype(np.uint8)


def perceive(rec, device="cuda"):
    """GPU half: first frame, SAM 3 mask of the support (largest instance) and the MoGe-2 point map (camera frame)."""
    import torch
    from moge.model.v2 import MoGeModel

    from .observe import Sam3

    rgb = read_video(rec.video)[0][0]
    sam = Sam3(device)
    c = sam(rgb[None], rec.meta.get("twin", {}).get("support", "table"))[0]
    sam.close()
    if not c:
        raise RuntimeError("SAM 3 found no support surface; set meta.twin.support")
    support = max(c, key=lambda d: d["mask"].sum())["mask"]
    fov_x = float(np.degrees(2 * np.arctan(rgb.shape[1] / 2 / rec.K[0, 0])))
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device).eval()
    x = torch.tensor(rgb / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
    with torch.no_grad():
        out = model.infer(x, fov_x=fov_x)
    pts = out["points"].cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return rgb, support, pts


def main():
    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument("recording")
    ap.add_argument("--name", help="twin object name (default from the prompt)")
    ap.add_argument('--object-route', choices=('agent_blender', 'template'), default='agent_blender',
                    help='default: reviewed Blender iteration; template explicitly selects primitive fitting')
    args = ap.parse_args()
    rec = load(args.recording)
    tr = np.load(rec.path / "track.npz")
    s = int(tr["mask_stride"])
    first = int(np.argmax(tr["valid"]))
    obj_mask = np.kron(tr["masks"][first], np.ones((s, s), bool))[:int(rec.camera["height"]), :int(rec.camera["width"])]
    ff = rec.path / "first_frame.npz"
    if ff.exists():
        d = np.load(ff)
        rgb, support, pts = d["rgb"], d["support"], d["points"]
    else:
        rgb, support, pts = perceive(rec)
        np.savez_compressed(ff, rgb=rgb, support=support, points=pts)
    twin = build(rec, rgb, support, pts, obj_mask, args.name)
    if args.object_route == 'agent_blender':
        from calibrated_scene_reconstruction.assets.model_artifact import require_reviewed
        from calibrated_scene_reconstruction.core.paths import ROOT, rel
        write_document(rec.path / 'model_initialization.json', twin, indent=1)
        proof, V, F = require_reviewed(rec.path, ROOT / 'configs' / rec.path.name / 'agent/objects/object/model.py')
        camera = {'K': rec.camera['K'], 'T_world_cam': rec.camera['T_world_cam'], 'support_z': twin['support']['z']}
        if proof.get('mode') != 'replay' or proof.get('calibration') != camera:
            raise ValueError('Approved model belongs to different camera calibration or physics route')
        o = twin['object']
        o.update(shape='mesh', mesh=rel(rec.path / proof['mesh']), pose=proof['world_pose'],
                 dims={'height': float(V[:, 2].max()), 'radius': float(np.linalg.norm(V[:, :2], axis=1).max())},
                 route='agent_blender', model_round=proof['round'], first_frame_iou=proof['iou_median'])
    write_document(rec.path / "twin.json", twin, indent=1)
    Image.fromarray(qa_image(rgb, twin, rec, obj_mask)).save(rec.path / "twin_qa.png")
    o = twin["object"]
    print(f"support z {twin['support']['z']:.3f} m (tilt {twin['support']['tilt_deg']:.1f} deg), object {o['shape']} "
          f"{o['dims']} at {np.round(o['pose'], 3).tolist()}, first-frame IoU {o['first_frame_iou']:.2f}")


if __name__ == "__main__":
    main()
