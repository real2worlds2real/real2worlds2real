"""The whole room, for the agent's review of the twin and of every cousin (README: after `robot_views`): a ring of six
views (60 deg apart, 90 deg wide: focal 0.5 x width) at head height from where the robot stands and from the middle of the room.

    tools/run_isaac.sh apps/scene/room_views.py sessions/<name> [--tag twin | --tag cousin_00 --variant sessions/<name>/cousins/cousin_00.json]

The robot views (apps/robot_views.py) look at the table; these look everywhere else, where the room's objects
(recon.room_objects: reconstructed from the completion) and its structure stand: is every object a solid 3D thing at
a believable size, standing on the floor or on what holds it, is any wall, floor or ceiling still showing paint of
furniture, a blurred or stretched texture, a seam, a hole to the void? The agent writes its verdict per tag in the
scene graph: "_room_review": {"<tag>": {"ok": true | false, "note": ...}} (fixes go to the stage at fault: the
completion review, room_objects' pieces, the surface review).

Writes qa/room_views/<tag>.jpg (one row per standpoint).
"""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("session")
ap.add_argument("--robot", default="vega_1p_sharpa")
ap.add_argument("--tag", default="twin")
ap.add_argument("--variant", default=None, help="a cousin's json (default: the twin)")
ap.add_argument("--bases", type=int, default=2, help="robot standpoints (the best-covering accepted bases)")
args = ap.parse_args()

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402

W, H = 640, 640
app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "width": W, "height": H})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from calibrated_scene_reconstruction.scene.graph import structure_role  # noqa: E402
from calibrated_scene_reconstruction.scene.session import Session  # noqa: E402
from calibrated_scene_reconstruction.robots.viewpoints import coverage_score, is_photo_base, view_bases  # noqa: E402
from calibrated_scene_reconstruction.sim.runtime import grab, rgb_annotator  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.variants import photo_goal, prepare_variant  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.runtime_inputs import prepare_scene  # noqa: E402
from calibrated_scene_reconstruction.sim.scene_stage import build_scene, configure_render  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import add_camera, pinhole  # noqa: E402

configure_render()
s = Session(args.session)
sg = s.scene_graph
top = float(sg["furniture"][sg["task"]["support"]]["box"][1][2])
pf = s / "placements" / ("placements.json" if args.robot == "vega_1p_sharpa" else f"{args.robot}.json")
photo_base = (sg.get("photo_robot") or {}).get("base")
bases, warn = view_bases(json.loads(pf.read_text())["candidates"], photo_base)
if warn:
    print(f"[warn] {warn}")
bases = sorted(bases, key=lambda c: (not is_photo_base(c, photo_base), -coverage_score(c)))
# standpoints: the bases farthest apart first (the photo robot's and one across the table), then the room's middle
picked = bases[:1]
for c in bases[1:]:
    if len(picked) >= args.bases:
        break
    if min(np.hypot(c["base"][0] - p["base"][0], c["base"][1] - p["base"][1]) for p in picked) > 1.0:
        picked.append(c)
floor = np.asarray(next(v for k, v in sg["structure"].items() if structure_role(k, v) == "floor")["range"], float)
stands = [(f"base {c['id']}", np.array([c["base"][0], c["base"][1], top + 0.55]), np.radians(c["base"][2]))
          for c in picked]
stands.append(("room middle", np.array([floor[0].mean(), floor[1].mean(), 1.6]), 0.0))

world = World(stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=1 / 30)
stage = omni.usd.get_context().get_stage()
variant = json.loads(Path(args.variant).read_text()) if args.variant else {}
build_scene(stage, prepare_scene(s.dir), photo_goal(s.dir, prepare_variant(s.dir, variant)))
world.reset()
cams = []
for i, (_, eye, yaw0) in enumerate(stands):
    for j in range(6):
        a = yaw0 + np.radians(60 * j)
        at = eye + np.array([np.cos(a), np.sin(a), -0.12])
        cam = add_camera(stage, f"/World/Cams/ring_{i}_{j}", pinhole(W, H, 0.5), W, H,
                         [float(v) for v in eye], [float(v) for v in at])
        cams.append(rgb_annotator(cam.GetPath(), W, H))
for _ in range(12):
    world.step(render=True)
t = W // 2
sheet = Image.new("RGB", (6 * t, len(stands) * (t + 18)), (20, 20, 20))
d = ImageDraw.Draw(sheet)
for i, (label, eye, _) in enumerate(stands):
    d.text((4, i * (t + 18) + 3), f"{args.tag}: {label} at {np.round(eye, 2).tolist()}, every 60 deg (first: facing "
                                  "the table / +x)", fill=(255, 255, 0))
    for j in range(6):
        sheet.paste(Image.fromarray(grab(cams[i * 6 + j])).resize((t, t)), (j * t, i * (t + 18) + 18))
out = s / "qa" / "room_views"
out.mkdir(parents=True, exist_ok=True)
sheet.save(out / f"{args.tag}.jpg", quality=88)
print(f"[room_views] {args.tag}: {len(stands)} standpoints -> {out / (args.tag + '.jpg')}", flush=True)
app.close()
