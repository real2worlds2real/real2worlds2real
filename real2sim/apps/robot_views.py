"""What the robot sees: its head camera and a camera over its shoulder at every accepted base (apps/placements.py), in
the twin or a cousin -- the views the policy is trained on, for the agent's review (README: after `placements`). The
photo-camera comparison (apps/scene/render_views.py) shows the scene from where the photo was taken; the robot works from
elsewhere (often across the table, looking down), where a surface textured from the photo shows its baked light,
shadows and occluders, a completed wall its seams, a cousin's theme its fit to the room.

    tools/run_isaac.sh apps/robot_views.py sessions/<name> [--robot NAME] [--tag twin | --tag cousin_00 --variant sessions/<name>/cousins/cousin_00.json]

Writes qa/robot_views/<tag>_c<k>_head.jpg, <tag>_c<k>_shoulder.jpg and <tag>_sheet.jpg (one row per base).
"""
import argparse
import json
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("session")
ap.add_argument("--robot", default="vega_1p_sharpa")
ap.add_argument("--tag", default="twin")
ap.add_argument("--variant", default=None, help="a cousin's json (default: the twin)")
ap.add_argument("--renderer", default="rt", choices=["rt", "pt"], help="real-time RTX (default) or path tracing")
ap.add_argument("--spp", type=int, default=64, help="path tracing: samples per frame")
ap.add_argument("--no-fill", action="store_true", help="without the stand-in bounce lights (lights.json fill: false)")
ap.add_argument("--suffix", default="", help="appended to the output names (a renderer comparison)")
ap.add_argument("--bases", default="", help="comma-separated base ids (default: every accepted base)")
ap.add_argument("--background", default="mesh", choices=["mesh", "splat", "zone"],
                help="splat: the twin's Gaussian-splat background (recon.splat_export, sim.splat)")
args = ap.parse_args()

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402

W, H = 768, 480
PT = "--renderer" in sys.argv and sys.argv[sys.argv.index("--renderer") + 1] == "pt"  # before the app starts
app = SimulationApp({"headless": True, "renderer": "PathTracing" if PT else "RaytracedLighting", "width": W, "height": H})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from calibrated_scene_reconstruction.sim.robot_articulation import Robot, add_cameras, add_robot, place_robot  # noqa: E402
from calibrated_scene_reconstruction.robots.spec import load_spec  # noqa: E402
from calibrated_scene_reconstruction.scene.session import Session  # noqa: E402
from calibrated_scene_reconstruction.scene.capture.render_robot import capture_spec  # noqa: E402
from calibrated_scene_reconstruction.robots.viewpoints import view_bases  # noqa: E402
from calibrated_scene_reconstruction.sim.runtime import grab, rgb_annotator  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.variants import photo_goal, prepare_variant  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.runtime_inputs import prepare_background, prepare_scene  # noqa: E402
from calibrated_scene_reconstruction.sim.scene_stage import build_scene, configure_render  # noqa: E402
from calibrated_scene_reconstruction.sim.splat_stage import add_background  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import add_camera, pinhole  # noqa: E402

if args.background == "splat" and not args.suffix:
    args.suffix = "_splat"
configure_render(mode=args.renderer, spp=args.spp)
s = Session(args.session)
spec = capture_spec(s, load_spec(args.robot))
sg = s.scene_graph
pf = s / "placements" / ("placements.json" if args.robot == "vega_1p_sharpa" else f"{args.robot}.json")
bases, warn = view_bases(json.loads(pf.read_text())["candidates"], (sg.get("photo_robot") or {}).get("base"))
if warn:
    print(f"[warn] {warn}")
bases = [c for c in bases if not args.bases or str(c["id"]) in args.bases.split(",")]
if not bases:
    sys.exit(f"no base in {pf}")
out = s / "qa" / "robot_views"
out.mkdir(parents=True, exist_ok=True)
support = sg["furniture"][sg["task"]["support"]]["box"]
support_top = float(support[1][2])

world = World(stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=1 / 30)
stage = omni.usd.get_context().get_stage()
variant = json.loads(Path(args.variant).read_text()) if args.variant else {}
variant = photo_goal(s.dir, prepare_variant(s.dir, variant))
if args.no_fill or args.renderer == "pt":
    variant["lights"] = {**variant.get("lights", {}), "fill": False}
build_scene(stage, prepare_scene(s.dir), variant)
add_background(stage, prepare_background(s.dir), args.background)
b0 = bases[0]["base"]
add_robot(stage, spec, tuple(b0[:2]), float(np.radians(b0[2])), support_top=support_top)
cams = add_cameras(stage, spec, size=(W, H))
head_name = next((c["name"] for c in spec["cameras"] if c.get("role") == "head"), None)
if head_name is None:
    sys.exit(f"{spec['name']} has no head camera")
head_rgb = rgb_annotator(cams[head_name][0].GetPath(), W, H)
world.reset()


def look_at(eye, at):
    return [float(v) for v in eye], [float(v) for v in at]


def render(n):
    for _ in range(n):
        world.step(render=True)


rows = []
for k, c in enumerate(bases):
    bx, by, yaw_deg = c["base"]
    yaw = float(np.radians(yaw_deg))
    place_robot(stage, spec, (bx, by), yaw, support_top=support_top)
    world.reset()
    rob = Robot(world, spec)
    rob.teleport(rob.home())
    fwd, left = np.array([np.cos(yaw), np.sin(yaw), 0.0]), np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    at = np.array([bx, by, support_top]) + 0.55 * fwd
    # just behind and above the head, to one side: a camera further back stood behind a wall or a screen when the
    # robot's back is near one (the robot's own body keeps ~0.3 m round its base clear)
    eye = np.array([bx, by, 0.0]) - 0.28 * fwd + 0.45 * left + np.array([0, 0, support_top + 0.95])
    sh = add_camera(stage, f"/World/Cams/shoulder{k}", pinhole(W, H, 0.9), W, H, *look_at(eye, at))
    sh_rgb = rgb_annotator(sh.GetPath(), W, H)
    import time

    render(10 if args.renderer == "rt" else 15)  # warm up (path tracing's first frames come back empty)
    t0 = time.time()
    render(5)
    dt = (time.time() - t0) / 5
    ims = {"head": grab(head_rgb), "shoulder": grab(sh_rgb)}
    for name, im in ims.items():
        Image.fromarray(im).save(out / f"{args.tag}_c{c['id']:02d}_{name}{args.suffix}.jpg", quality=90)
    print(f"[robot_views] render {dt:.3f} s per frame ({args.renderer}{', %d spp' % args.spp if args.renderer == 'pt' else ''})",
          flush=True)
    rows.append((c, ims))
    print(f"[robot_views] {args.tag} base {c['id']} ({bx:.2f}, {by:.2f}, {yaw_deg:.0f} deg) -> {out}", flush=True)

sheet = Image.new("RGB", (2 * W // 2 + 12, len(rows) * (H // 2 + 22) + 6), (24, 24, 24))
d = ImageDraw.Draw(sheet)
for r, (c, ims) in enumerate(rows):
    y = 4 + r * (H // 2 + 22)
    d.text((6, y), f"{args.tag} base {c['id']} {np.round(c['base'], 2).tolist()}: head camera | over the shoulder",
           fill=(230, 230, 230))
    for j, name in enumerate(("head", "shoulder")):
        sheet.paste(Image.fromarray(ims[name]).resize((W // 2, H // 2)), (6 + j * (W // 2), y + 16))
sheet.save(out / f"{args.tag}_sheet{args.suffix}.jpg", quality=88)
print(f"[robot_views] {args.tag}: {len(rows)} bases -> {out / (args.tag + '_sheet.jpg')}", flush=True)
app.close()
