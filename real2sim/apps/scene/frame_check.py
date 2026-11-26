"""The twin seen from where the walks were filmed: rendered at posed frames' cameras and held against each frame's
own depth, object by object -- what stands in the wrong place, at the wrong size, or not at all.

    tools/run_isaac.sh apps/scene/frame_check.py sessions/<name> [--robot NAME] [--frames 24]

The frames are recon.posed_frames' (parsing/posed_frames/frames.npz: COLMAP at the board's scale in the twin's world,
a MoGe-2 depth scaled to the frame's sparse points). Each is rendered with its own intrinsics and pose (the robot at
the real base in its home pose: the real one's arms differ, so its pixels are left out), with the depth along the
optical axis and the prims' instance ids, and compared where both have a depth: per prim, the median signed error
(render - frame: < 0 the twin's surface stands in front of the real one -- too big, too far forward, or something
the room does not have there; > 0 behind it -- too small, too far back, or missing in front of it) over every frame
that sees it, and per frame the share of pixels within 10 cm. The idea follows agentic_rsr's depth check against
frozen, calibrated cameras: geometry is judged where the real cameras were, not from viewpoints chosen to flatter it.
Writes qa/frame_check/<frame>.jpg (the frame | the render | the depth error, red: the twin in front, blue: behind)
and qa/frame_check/frame_check.json (per prim and per frame), for the agent's review.
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("session")
ap.add_argument("--robot", default="vega_1p_sharpa_arm")
ap.add_argument("--frames", type=int, default=24)
ap.add_argument("--near", type=float, default=0.10, help="a pixel agrees within this, m")
args = ap.parse_args()

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402

W, H = 960, 540
app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "width": W, "height": H})

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402

from calibrated_scene_reconstruction.sim.robot_articulation import Robot, add_robot  # noqa: E402
from calibrated_scene_reconstruction.robots.spec import load_spec  # noqa: E402
from calibrated_scene_reconstruction.scene.session import Session  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.variants import photo_goal, prepare_variant  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.runtime_inputs import prepare_scene  # noqa: E402
from calibrated_scene_reconstruction.sim.scene_stage import build_scene, configure_render  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import add_camera  # noqa: E402

configure_render(mode="rt")
s = Session(args.session)
sg = s.scene_graph
z = np.load(s / "parsing" / "posed_frames" / "frames.npz")
pick = np.unique(np.linspace(0, len(z["names"]) - 1, min(args.frames, len(z["names"]))).astype(int))

world = World(stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=1 / 30)
stage = omni.usd.get_context().get_stage()
build_scene(stage, prepare_scene(s.dir), photo_goal(s.dir, prepare_variant(s.dir, {})))
spec = load_spec(args.robot)
base = (sg.get("photo_robot") or {}).get("base")
support = sg["furniture"][sg["task"]["support"]]["box"]
if base:
    add_robot(stage, spec, tuple(base["xy"]), float(np.radians(base["yaw_deg"])), support_top=float(support[1][2]))
h0, w0 = z["depth"].shape[1:]
cam = add_camera(stage, "/World/Cams/check", z["K"][0], w0, h0, T_world_cam_cv=np.eye(4), far=60.0)
rp = rep.create.render_product(str(cam.GetPath()), (w0, h0))
ann = {k: rep.AnnotatorRegistry.get_annotator(k, init_params=p) for k, p in
       (("rgb", None), ("distance_to_image_plane", None), ("instance_id_segmentation", {"colorize": False}))}
for a in ann.values():
    a.attach(rp)
world.reset()
if base:
    rob = Robot(world, spec)
    rob.teleport(rob.home())

from calibrated_scene_reconstruction.sim.usd import set_pose, quat_from_matrix  # noqa: E402

out = s / "qa" / "frame_check"
out.mkdir(parents=True, exist_ok=True)
per_prim, per_frame = {}, []
for i in pick:
    K = z["K"][i]
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = z["R_world_cam"][i], z["centre"][i]
    cam.GetHorizontalApertureAttr().Set(float(w0 / K[0, 0] * 18.0))
    cam.GetVerticalApertureAttr().Set(float(h0 / K[1, 1] * 18.0))
    set_pose(cam.GetPrim(), T[:3, 3], quat_from_matrix(T[:3, :3] @ np.diag([1, -1, -1])))
    for _ in range(6):
        world.step(render=True)
    rgb = np.asarray(ann["rgb"].get_data())[..., :3]
    dr = np.asarray(ann["distance_to_image_plane"].get_data(), np.float32)
    seg = ann["instance_id_segmentation"].get_data()
    ids, labels = np.asarray(seg["data"]), {int(k): v for k, v in seg["info"]["idToLabels"].items()}
    D = z["depth"][i].astype(np.float32)
    ok = z["ok"][i] & (D > 0.1) & np.isfinite(dr) & (dr > 0.05) & (dr < 50)
    robot = np.isin(ids, [k for k, v in labels.items() if "/World/Robot" in str(v)])
    ok &= ~robot
    err = np.where(ok, dr - D, 0.0)
    within = float((np.abs(err[ok]) < args.near).mean()) if ok.any() else 0.0
    name = str(z["names"][i]).rsplit(".", 1)[0]
    per_frame.append({"frame": name, "within_10cm": round(within, 3), "pixels": int(ok.sum())})
    for k, v in labels.items():
        m = ok & (ids == k)
        if m.sum() < 200:
            continue
        key = str(v)
        for root in ("/World/Synthetic/", "/World/Static/", "/World/Objects/", "/World/Support/", "/World/Room/"):
            if root in key:  # the prim's own name: its first path element under the root
                key = root + key.split(root, 1)[1].split("/")[0]
                break
        e = err[m]
        d = per_prim.setdefault(key, {"pixels": 0, "frames": 0, "errors": []})
        d["pixels"] += int(m.sum())
        d["frames"] += 1
        d["errors"].append(np.random.default_rng(0).choice(e, min(2000, len(e)), replace=False))
    real = (z["images"][i][:h0, :w0]).astype(np.uint8)
    ec = np.clip(err / 0.5, -1, 1)
    vis = np.full((h0, w0, 3), 40, np.uint8)
    vis[ok] = np.stack([(255 * np.clip(-ec, 0, 1)), 255 * (1 - np.abs(ec)) * 0.6, 255 * np.clip(ec, 0, 1)], -1)[ok].astype(np.uint8)
    vis = vis[..., ::-1]  # RGB: red where the twin stands in front (ec < 0), blue behind
    # the support's top outline projected with the frame's camera on both: where the pose puts it
    sup_box = np.asarray(sg["furniture"][sg["task"]["support"]]["box"], float)
    (x0, y0, _), (x1, y1, zt) = sup_box
    corners = np.array([[x0, y0, zt], [x1, y0, zt], [x1, y1, zt], [x0, y1, zt]])
    Xc = (corners - T[:3, 3]) @ T[:3, :3]
    if np.all(Xc[:, 2] > 0.05):
        uv = (Xc[:, :2] / Xc[:, 2:3] * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]).astype(np.int32).reshape(-1, 1, 2)
        real, rgb = real.copy(), np.ascontiguousarray(rgb).copy()
        for im in (real, rgb):
            cv2.polylines(im, [uv], True, (0, 255, 0), 2)
    sheet = np.hstack([real, rgb, vis[..., ::-1]])
    cv2.putText(sheet, f"{name}: {100 * within:.0f} % within {args.near:.2f} m", (12, 30), 0, 0.8, (255, 255, 255), 2)
    cv2.imwrite(str(out / f"{name}.jpg"), sheet[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"[check] {name}: {100 * within:.0f} % within {args.near} m over {ok.sum()} px", flush=True)
prims = {}
for k, d in per_prim.items():
    e = np.concatenate(d["errors"])
    prims[k] = {"frames": d["frames"], "pixels": d["pixels"], "median_m": round(float(np.median(e)), 3),
                "abs_median_m": round(float(np.median(np.abs(e))), 3),
                "within_share": round(float((np.abs(e) < args.near).mean()), 3)}
worst = sorted(prims.items(), key=lambda kv: -kv[1]["abs_median_m"])
(out / "frame_check.json").write_text(json.dumps({"frames": per_frame, "prims": dict(worst)}, indent=1) + "\n")
print(f"done: {len(pick)} frames, {np.median([f['within_10cm'] for f in per_frame]) * 100:.0f} % of pixels within "
      f"{args.near} m (median frame); worst prims:", flush=True)
for k, d in worst[:12]:
    print(f"  {k}: median {d['median_m']:+.2f} m (|{d['abs_median_m']:.2f}|), {d['frames']} frames, "
          f"{100 * d['within_share']:.0f} % within", flush=True)
app.close()
