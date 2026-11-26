"""The room's light as the splat background holds it: a panorama rendered from the splat alone at the task zone, for
the dome light that lights the twin's meshes (the support, the task objects) the way the real room lit them.

    tools/run_isaac.sh apps/scene/splat_envmap.py sessions/<name> [--height 0.35] [--face 512]

Six 90-degree faces from a point `height` m over the support's middle (every mesh hidden: the splat alone), joined into
a lat-long panorama (+Z up, as RTX maps a dome's pole). Writes scene/splat/envmap.png (sRGB, as rendered: auto
exposure off, so faces agree) and qa/splat/envmap.jpg.
"""
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("session")
ap.add_argument("--height", type=float, default=0.35, help="m over the support's top")
ap.add_argument("--face", type=int, default=512)
args = ap.parse_args()

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402

F = args.face
app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "width": F, "height": F})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

from calibrated_scene_reconstruction.scene.session import Session  # noqa: E402
from calibrated_scene_reconstruction.sim.runtime import grab, rgb_annotator  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.variants import prepare_variant  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.runtime_inputs import prepare_background, prepare_scene  # noqa: E402
from calibrated_scene_reconstruction.sim.scene_stage import build_scene, configure_render  # noqa: E402
from calibrated_scene_reconstruction.sim.splat_stage import PATH, add_background  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import add_camera, pinhole, quat_from_matrix, set_pose  # noqa: E402

configure_render(auto_exposure=False)
s = Session(args.session)
sg = s.scene_graph
world = World(stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=1 / 30)
stage = omni.usd.get_context().get_stage()
build_scene(stage, prepare_scene(s.dir), prepare_variant(s.dir, {}))
add_background(stage, prepare_background(s.dir), "splat")
for q in Usd.PrimRange(stage.GetPrimAtPath("/World")):  # the splat alone
    if q.IsA(UsdGeom.Mesh) and not str(q.GetPath()).startswith(PATH):
        UsdGeom.Imageable(q).MakeInvisible()
lo, hi = (np.asarray(b, float) for b in sg["furniture"][sg["task"]["support"]]["box"])
eye = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2] + args.height])
# faces: (look direction, up) -- +X, -X, +Y, -Y, +Z, -Z
DIRS = [((1, 0, 0), (0, 0, 1)), ((-1, 0, 0), (0, 0, 1)), ((0, 1, 0), (0, 0, 1)), ((0, -1, 0), (0, 0, 1)),
        ((0, 0, 1), (0, -1, 0)), ((0, 0, -1), (0, 1, 0))]
K = pinhole(F, F, 0.5)  # 90 degrees
anns = []
for k, (d, up) in enumerate(DIRS):
    cam = add_camera(stage, f"/World/Cams/env{k}", K, F, F)
    f = np.asarray(d, float)
    r = np.cross(f, up)
    set_pose(cam.GetPrim(), eye, quat_from_matrix(np.stack([r, np.cross(r, f), -f], 1)))
    anns.append(rgb_annotator(cam.GetPath(), F, F))
world.reset()
for _ in range(40):
    world.render()
faces = [grab(a).astype(np.float32) / 255.0 for a in anns]

# lat-long: column -> azimuth (+X at the middle, +Y a quarter to the left), row -> elevation (+Z at the top);
# rolled half a turn at the end for RTX
Wd, Hd = 4 * F, 2 * F
az = (0.5 - (np.arange(Wd) + 0.5) / Wd) * 2 * np.pi
el = (0.5 - (np.arange(Hd) + 0.5) / Hd) * np.pi
A, E = np.meshgrid(az, el)
D = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1)
pano = np.zeros((Hd, Wd, 3), np.float32)
face = np.argmax(D @ np.array([d for d, _ in DIRS], float).T, -1)  # the face each direction falls in
for k, (d, up) in enumerate(DIRS):
    f, up = np.asarray(d, float), np.asarray(up, float)
    r = np.cross(f, up)
    u_ = np.cross(r, f)
    sel = face == k
    z = D[sel] @ f
    x, y = (D[sel] @ r) / z, (D[sel] @ u_) / z
    px = np.clip(((x + 1) / 2 * F).astype(int), 0, F - 1)
    py = np.clip(((1 - y) / 2 * F).astype(int), 0, F - 1)
    pano[sel] = faces[k][py, px]
pano = np.roll(pano, Wd // 2, axis=1)  # RTX centres a lat-long dome on -X (seen along +X: the image's edge)
out = s / "scene" / "splat" / "envmap.png"
Image.fromarray((pano * 255).astype(np.uint8)).save(out)
qa = s / "qa" / "splat"
qa.mkdir(parents=True, exist_ok=True)
Image.fromarray((pano * 255).astype(np.uint8)).resize((Wd // 2, Hd // 2)).save(qa / "envmap.jpg", quality=88)
print(f"done: {out} from {np.round(eye, 2).tolist()}", flush=True)
app.close()
