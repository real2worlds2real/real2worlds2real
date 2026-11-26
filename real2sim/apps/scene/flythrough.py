"""Camera fly-through of the twin (or a cousin: --tag, --variant) with a robot (--robot, configs/robots) standing at its best accepted placement.

    tools/run_isaac.sh apps/scene/flythrough.py sessions/<name> [--seconds 14] [--tag cousin_00 --variant sessions/<name>/cousins/cousin_00.json]
Key frames come from configs/<name>/views.json ("photo" = the photo camera). Writes renders/flythrough/flythrough.mp4 (a cousin: renders/flythrough/<tag>/flythrough.mp4); the last key frames pass over
the robot's shoulder to about its head camera.
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("session")
ap.add_argument("--robot", default="vega_1p_sharpa")
ap.add_argument("--seconds", type=float, default=14.0)
ap.add_argument("--tag", default="twin", help="the scene's name in the output (renders/flythrough/<tag>/ for a cousin)")
ap.add_argument("--variant", default=None, help="a cousin's json (default: the twin)")
args = ap.parse_args()

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402

W, H = 1280, 720
app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "width": W, "height": H})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from PIL import Image  # noqa: E402

from calibrated_scene_reconstruction.scene.session import Session  # noqa: E402
from calibrated_scene_reconstruction.robots.viewpoints import is_photo_base, view_bases  # noqa: E402
from calibrated_scene_reconstruction.sim.robot_articulation import Robot, add_robot  # noqa: E402
from calibrated_scene_reconstruction.robots.spec import load_spec  # noqa: E402
from calibrated_scene_reconstruction.sim.runtime import encode_video, grab, rgb_annotator  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.variants import photo_goal, prepare_variant  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.runtime_inputs import prepare_scene  # noqa: E402
from calibrated_scene_reconstruction.sim.scene_stage import build_scene, configure_render  # noqa: E402
from calibrated_scene_reconstruction.sim.sweep import RobotSweep, clear_robot, summarize, support_collisions  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import add_camera, look_at_quat, set_pose  # noqa: E402

configure_render()
s = Session(args.session)
out = s / "renders" / "flythrough" / ("" if args.tag == "twin" else args.tag)
(out / "frames").mkdir(parents=True, exist_ok=True)
for f in (out / "frames").glob("*"):  # an earlier, longer run's frames would be encoded after this run's
    f.unlink()
m = s.manifest
spec = load_spec(args.robot)
pl = s.read_json("placements/placements.json" if args.robot == "vega_1p_sharpa" else f"placements/{args.robot}.json")
photo_base = (s.scene_graph.get("photo_robot") or {}).get("base")
cands, warn = view_bases(pl["candidates"], photo_base)
if warn:
    print(f"[warn] {warn}")
bx, by, byaw = (cands[0] if is_photo_base(cands[0], photo_base) else max(cands, key=lambda c: c["coverage"]))["base"]

world = World(stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=1 / 30)
stage = omni.usd.get_context().get_stage()
variant = json.loads(Path(args.variant).read_text()) if args.variant else {}
ts = build_scene(stage, prepare_scene(s.dir), photo_goal(s.dir, prepare_variant(s.dir, variant)))
add_robot(stage, spec, (bx, by), np.radians(byaw),
          support_top=float(s.scene_graph["furniture"][s.scene_graph["task"]["support"]]["box"][1][2]))
support_collisions(stage, False)
T = np.asarray(m["photo_camera"]["T_world_cam_cv"])
photo_key = (T[:3, 3], T[:3, 3] + 3.0 * T[:3, 2])
keys = [photo_key if k == ["photo"] else (np.asarray(k[0], float), np.asarray(k[1], float))
        for k in s.config("views.json")["flythrough"]]
# then round to the robot: over its shoulder and down to about its head camera, where the policy sees the scene from
# (the photo side is where a photo texture looks right; the robot often works from across the table)
zt = float(s.scene_graph["furniture"][s.scene_graph["task"]["support"]]["box"][1][2])
fwd = np.array([np.cos(np.radians(byaw)), np.sin(np.radians(byaw)), 0.0])
left = np.array([-fwd[1], fwd[0], 0.0])
look = np.array([bx, by, zt]) + 0.55 * fwd
keys += [(np.array([bx, by, zt + 1.15]) - 0.35 * fwd + 0.9 * left, look),  # beside and above the robot (a camera
         (np.array([bx, by, zt + 0.95]) - 0.25 * fwd + 0.45 * left, look),  # further back met the wall behind it)
         (np.array([bx, by, zt + 0.72]) + 0.05 * fwd, look + 0.1 * fwd)]
K = np.asarray(m["photo_camera"]["K"]) * np.array([[W / m["photo_camera"]["width"]], [H / m["photo_camera"]["height"]], [1]])
K[0, 0] = K[1, 1] = min(K[0, 0], K[1, 1])
cam = add_camera(stage, "/World/Cams/fly", K, W, H, T_world_cam_cv=T)
world.reset()
vega = Robot(world, spec)
vega.teleport(vega.home())
for _ in range(90):
    world.step(render=False)
sweep = RobotSweep(stage, vega)
sweep.sample()
print(summarize(clear_robot(ts, sweep)), flush=True)  # stools out of the parked robot
support_collisions(stage, True)
ann = rgb_annotator(cam.GetPath(), W, H)


def catmull(p0, p1, p2, p3, t):
    return 0.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3)


n, segs = int(args.seconds * 30), len(keys) - 1
for f in range(n):
    u = (0.5 - 0.5 * np.cos(np.pi * f / (n - 1))) * segs
    i = min(int(u), segs - 1)
    idx = [max(i - 1, 0), i, i + 1, min(i + 2, segs)]
    eye = catmull(*[keys[j][0] for j in idx], u - i)
    tgt = catmull(*[keys[j][1] for j in idx], u - i)
    set_pose(cam.GetPrim(), eye, look_at_quat(eye, tgt))
    vega.command(vega.home())
    world.step(render=True)
    if f == 0:
        for _ in range(8):
            world.step(render=True)
    Image.fromarray(grab(ann)).save(out / "frames" / f"{f:05d}.jpg", quality=92)
encode_video(out / "frames", out / "flythrough.mp4")
print("saved", out / "flythrough.mp4")
app.close()
