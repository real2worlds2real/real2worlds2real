"""A USD / USDZ export opened on its own in Isaac Sim and rendered (no physics, as a viewer shows it): from the session's
photo camera, from above behind it and from the far side of the support, into <file>_preview_*.jpg beside it -- does
the one file hold the whole scene, its textures and the splat?

    tools/run_isaac.sh apps/scene/view_usd.py sessions/<name> <file.usd | file.usdz>
"""
import argparse
import hashlib
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("session")
ap.add_argument("usd")
ap.add_argument('--configuration', help='Configuration snapshot used to author this asset')
args = ap.parse_args()

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402

W, H = 1280, 720
app = SimulationApp({"headless": True, "renderer": "RaytracedLighting", "width": W, "height": H})

import numpy as np  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402

from calibrated_scene_reconstruction.scene.session import Session  # noqa: E402
from calibrated_scene_reconstruction.sim.runtime import grab, rgb_annotator  # noqa: E402
from calibrated_scene_reconstruction.sim.scene_stage import configure_render  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import add_camera  # noqa: E402
from calibrated_scene_reconstruction.scene.objects.mechanisms import declaration, fingerprint, sampling_tolerances  # noqa: E402
from calibrated_scene_reconstruction.sim.mechanism_stage import passive_audit  # noqa: E402
from pxr import Sdf, UsdGeom, UsdPhysics, UsdUtils  # noqa: E402
from calibrated_scene_reconstruction.scene.simulation import settings as simulation_settings  # noqa: E402
from calibrated_scene_reconstruction.sim.simulation import audit as audit_simulation  # noqa: E402

s = Session(args.session,args.configuration)
f = Path(args.usd).resolve()
omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate("omni.usd.schema.omni_nurec_types", True)
omni.usd.get_context().open_stage(str(f))
stage = omni.usd.get_context().get_stage()
mechanisms = declaration(s)
if mechanisms:
    if s.config('task.json', {}).get('mechanism_contract') and simulation_settings(s) is None:
        raise ValueError('Mechanism delivery requires a declared simulation.json profile')
    # Inspect the reopened delivery, not only the pre-export in-memory stage.
    audit = passive_audit(stage, mechanisms,sampling_tolerances=sampling_tolerances(s))
    simulation_profile = audit_simulation(stage, simulation_settings(s))
    from calibrated_scene_reconstruction.sim.collider_delivery import audit as audit_colliders
    collider_precision = audit_colliders(stage,s,audit['links'])
    _, _, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(f)))
    kit_modules = {'OmniPBR.mdl', 'OmniGlass.mdl', 'OmniSurface.mdl'}
    missing = [path for path in unresolved if path not in kit_modules]
    units, up = UsdGeom.GetStageMetersPerUnit(stage), str(UsdGeom.GetStageUpAxis(stage))
    if missing or units != 1. or up != 'Z':
        raise ValueError(f'Delivery dependencies/units invalid: {missing}, {units}, {up}')
    from calibrated_scene_reconstruction.scene.twin.reviews import package_startup  # noqa: E402
    startup=package_startup(s,f,simulation_profile,fingerprint(s))
    s.write_json(f'qa/package_{f.stem}.json', {**audit, 'asset': str(f),
        'asset_sha256': hashlib.sha256(f.read_bytes()).hexdigest(),
        'source_input_fingerprint': fingerprint(s),
        'collider_precision': collider_precision,
        'native_startup':startup,
        'simulation_profile': simulation_profile,
        'meters_per_unit': units, 'up_axis': up, 'unresolved_assets': missing,
        'renderer_modules': list(unresolved)})
stage.SetEditTarget(stage.GetSessionLayer())  # the cameras stay out of the file
# Match the captured-light splat or plain task-zone render after reopening.
fixed = (stage.GetPrimAtPath('/World/Zone/dome').IsValid()
         or any(p.GetTypeName() == "Volume" for p in stage.Traverse()))
configure_render(auto_exposure=not fixed)
if stage.GetRootLayer().customLayerData.get('r2s2r_environment_render'):
    from calibrated_scene_reconstruction.sim.environment_stage import apply_environment_render  # noqa: E402
    apply_environment_render(stage)
if fixed:
    from calibrated_scene_reconstruction.sim.splat_stage import fixed_exposure  # noqa: E402

    fixed_exposure(dict(stage.GetRootLayer().customLayerData).get('r2s2r_background_render'))
pc = s.manifest["photo_camera"]
T = np.asarray(pc["T_world_cam_cv"])
K = np.asarray(pc["K"]) * np.array([[W / pc["width"]], [H / pc["height"]], [1]])
K[0, 0] = K[1, 1] = min(K[0, 0], K[1, 1])
sup = s.scene_graph["furniture"][s.scene_graph["task"]["support"]]["box"]
c = (np.asarray(sup[0]) + np.asarray(sup[1])) / 2
fwd = T[:3, 2] * np.array([1, 1, 0]) / max(np.linalg.norm(T[:3, 2][:2]), 1e-6)
views = {"photo": dict(T_world_cam_cv=T),
         "above": dict(eye=T[:3, 3] - 0.8 * fwd + np.array([0, 0, 0.5]), target=c),
         "far": dict(eye=c + 2.6 * fwd + np.array([0, 0, 0.6]), target=c - 1.0 * fwd)}
anns = {k: rgb_annotator(add_camera(stage, f"/Preview/{k}", K, W, H, **v).GetPath(), W, H) for k, v in views.items()}
for _ in range(200):
    app.update()
rep.orchestrator.step(rt_subframes=24)
for k, a in anns.items():
    img = grab(a)
    out = f.parent / f"{f.stem}_preview_{k}.jpg"
    Image.fromarray(img).save(out, quality=90)
    print(f"[view] {out.name}: mean {img.mean():.0f}", flush=True)
splat = [p.GetPath() for p in stage.Traverse() if p.GetTypeName() == "Volume"]
print(f"done: {f.name} opened, {sum(1 for _ in stage.Traverse())} prims, NuRec volumes {splat}", flush=True)
app.close()
