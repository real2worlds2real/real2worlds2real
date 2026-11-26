"""Render a twin / cousin from the photo camera and presentation views (configs/<name>/views.json). No robot.

    tools/run_isaac.sh apps/scene/render_views.py sessions/<name> --tag twin [--variant cousins/cousin_00.json] [--novel]
Writes renders/views/<tag>_<view>.jpg (+ <tag>_photo_vs_render.jpg).
"""
import argparse
import sys

ap = argparse.ArgumentParser()
ap.add_argument("session")
ap.add_argument("--tag", default="twin")
ap.add_argument("--variant")
ap.add_argument("--novel", action="store_true", help="also render the novel views of views.json")
ap.add_argument("--unseen", action="store_true", help="also render views of what the photo did not see (unseen_*)")
args = ap.parse_args()

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402

from calibrated_scene_reconstruction.scene.review.resolution import photo_resolution  # noqa: E402

W, H = photo_resolution(sys.argv[1])  # the photo's aspect: Isaac's pixels are square (calibrated_scene_reconstruction.resolution)
app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "width": W, "height": H})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from PIL import Image  # noqa: E402

from calibrated_scene_reconstruction.scene.session import Session  # noqa: E402
from calibrated_scene_reconstruction.sim.runtime import grab, load_variant, rgb_annotator  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.variants import prepare_variant  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.runtime_inputs import prepare_scene  # noqa: E402
from calibrated_scene_reconstruction.sim.scene_stage import build_scene, configure_render  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import add_camera, pinhole  # noqa: E402

configure_render()
s = Session(args.session)
out = s / "renders" / "views"
out.mkdir(parents=True, exist_ok=True)
views = s.config("views.json")
world = World(stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=1 / 30)
stage = omni.usd.get_context().get_stage()
build_scene(stage, prepare_scene(s.dir), prepare_variant(s.dir, load_variant(args.variant)))
cam = s.manifest["photo_camera"]
K = np.asarray(cam["K"]) * np.array([[W / cam["width"]], [H / cam["height"]], [1]])
cams = {"photo": add_camera(stage, "/World/Cams/photo", K, W, H, T_world_cam_cv=cam["T_world_cam_cv"]),
        "oblique": add_camera(stage, "/World/Cams/oblique", pinhole(W, H), W, H, *views["oblique"])}
if args.novel:
    for k, (eye, tgt) in views["novel"].items():
        cams[k] = add_camera(stage, f"/World/Cams/{k}", pinhole(W, H), W, H, eye, tgt)
if args.unseen:  # from the photo camera's spot (clamped in the room): left, right, behind; and from the far side back
    sg = s.scene_graph
    floor = np.asarray(sg["structure"]["floor"]["range"], float)
    lo, hi = floor[:, 0] + 0.3, floor[:, 1] - 0.3
    Tw = np.asarray(cam["T_world_cam_cv"], float)
    eye = np.r_[np.clip(Tw[:2, 3], lo, hi), 1.5]
    f = Tw[:2, 2] / max(np.linalg.norm(Tw[:2, 2]), 1e-6)
    rot = lambda a: np.array([np.cos(a) * f[0] - np.sin(a) * f[1], np.sin(a) * f[0] + np.cos(a) * f[1]])
    for k, a in (("unseen_left", 90), ("unseen_right", -90), ("unseen_back", 180)):
        cams[k] = add_camera(stage, f"/World/Cams/{k}", pinhole(W, H), W, H, eye, eye + np.r_[rot(np.radians(a)), -0.15])
    # the far side: walk along the photo direction until 0.3 m short of furniture (a camera inside a cabinet sees black)
    boxes = [np.asarray(b["box"], float) for sec in ("furniture", "synthetic_furniture")
             for b in sg.get(sec, {}).values()]
    far = eye
    for t in np.arange(0.0, 30.0, 0.05):
        p = np.r_[np.clip(Tw[:2, 3] + f * t, lo, hi), 1.6]
        if any(np.all(p > b[0] - 0.3) and np.all(p < b[1] + 0.3) for b in boxes):
            break
        far = p
    cams["unseen_far"] = add_camera(stage, "/World/Cams/unseen_far", pinhole(W, H), W, H, far, eye - np.r_[0, 0, 0.4])
world.reset()
for _ in range(90):  # let dynamic objects settle
    world.step(render=False)
annots = {k: rgb_annotator(c.GetPath(), W, H) for k, c in cams.items()}
for _ in range(12):
    world.step(render=True)
for k, a in annots.items():
    Image.fromarray(grab(a)).save(out / f"{args.tag}_{k}.jpg", quality=93)
side = Image.new("RGB", (2 * W, H))
side.paste(Image.open(s / "capture" / "rgb.png").convert("RGB").resize((W, H)), (0, 0))
side.paste(Image.open(out / f"{args.tag}_photo.jpg"), (W, 0))
side.save(out / f"{args.tag}_photo_vs_render.jpg", quality=93)
print("rendered", args.tag, sorted(cams))
app.close()
