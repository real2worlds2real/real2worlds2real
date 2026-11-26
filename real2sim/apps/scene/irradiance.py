"""Irradiance probe: white diffuse room under the twin lights, rendered linear (HDR) from the photo camera.
recon.layout then bakes albedo = photo / irradiance, so the relit twin reproduces the photo.

    tools/run_isaac.sh apps/scene/irradiance.py sessions/<name>
"""
import argparse
import sys

ap = argparse.ArgumentParser()
ap.add_argument("session")
args = ap.parse_args()

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402

from calibrated_scene_reconstruction.scene.review.resolution import photo_resolution  # noqa: E402

W, H = photo_resolution(sys.argv[1])  # the photo's aspect: Isaac's pixels are square (calibrated_scene_reconstruction.resolution)
app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "width": W, "height": H})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402

from calibrated_scene_reconstruction.scene.session import Session  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.runtime_inputs import prepare_scene  # noqa: E402
from calibrated_scene_reconstruction.sim.scene_stage import build_scene, configure_render  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import add_camera  # noqa: E402

configure_render(auto_exposure=False)
s = Session(args.session)
m = s.manifest
variant = {"room_overrides": {"": {"white": True}}, "skip_static": tuple(m["static_objects"]), "objects": [],
           "stool_models": [], "lights": s.config("lights.json", {"lights": {}})["lights"]}
stage = omni.usd.get_context().get_stage()
build_scene(stage, prepare_scene(s.dir), variant)
cam = m["photo_camera"]
K = np.asarray(cam["K"]) * np.array([[W / cam["width"]], [H / cam["height"]], [1]])
c = add_camera(stage, "/World/Cams/photo", K, W, H, T_world_cam_cv=cam["T_world_cam_cv"])
rp = rep.create.render_product(str(c.GetPath()), (W, H))
hdr = rep.AnnotatorRegistry.get_annotator("HdrColor")
hdr.attach(rp)
for _ in range(20):
    app.update()
rep.orchestrator.step(rt_subframes=24)
E = np.asarray(hdr.get_data())[..., :3].astype(np.float32)
np.save(s / "scene" / "irradiance_photo.npy", E / 0.8)  # per unit albedo
print(f"irradiance saved, mean {E.mean():.3f}")
app.close()
