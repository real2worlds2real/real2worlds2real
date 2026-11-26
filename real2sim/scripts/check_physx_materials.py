"""Opt-in native PhysX checks for granular solids and capsule-chain ropes.

Run with Isaac Sim Python. No robot policy, calibration or source-photo twin
quality is implied by these small physical interaction checks.
"""
# ruff: noqa: E402 -- Kit must start before runtime imports.
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--case", choices=["granular", "rope"], required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--dt", type=float, default=1 / 240)
parser.add_argument("--seconds", type=float, default=4)
parser.add_argument("--no-images", action="store_true")
args = parser.parse_args()
if not (0 < args.dt <= 1 / 60 and 2 <= args.seconds <= 20):
    parser.error("Choose dt <=1/60 s and duration 2..20 s")
args.output.mkdir(parents=True, exist_ok=False)

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "multi_gpu": False,
                     "active_gpu": 0, "physics_gpu": 0})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.version import get_version
from pxr import Gf, PhysxSchema, UsdGeom, UsdLux, UsdPhysics, UsdShade

from real2sim_harness.physx_granular import (
    GranularSettings,
    audit_grains,
    author_granular,
    read_granular_positions,
    sample_lattice,
)
from real2sim_harness.physx_rope import author_rope, read_rope_state


def box(stage, path, size, position, *, kinematic=False):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.)
    cube.AddTranslateOp().Set(Gf.Vec3d(*map(float, position)))
    cube.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
    cube.CreateDisplayColorPrimvar(UsdGeom.Tokens.constant).Set([Gf.Vec3f(.48, .58, .65)])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    col = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
    col.CreateContactOffsetAttr(.001)
    col.CreateRestOffsetAttr(0.)
    if kinematic:
        rb = UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        rb.CreateKinematicEnabledAttr(True)
    return cube


def snapshot(world, name):
    if args.no_images:
        return
    import omni.replicator.core as rep
    from PIL import Image
    focus = np.array([0., 0., .1])
    camera = rep.create.camera(position=tuple(focus + [.6, -.7, .6]), look_at=tuple(focus))
    product = rep.create.render_product(camera, (960, 640))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([product.path])
    try:
        for _ in range(12):
            world.render()
        data = rgb.get_data()
        if not isinstance(data, np.ndarray) or data.size == 0:
            raise RuntimeError("Renderer returned no native scene image")
        Image.fromarray(data).save(args.output / f"{name}.png")
    finally:
        rgb.detach([product.path])
        product.destroy()


report = {"case": args.case, "backend": "physx", "isaac_sim": get_version()[0],
          "dt_s": args.dt, "duration_s": args.seconds, "checks": {}}
source_root = Path(__file__).resolve().parents[1] / "src/real2sim_harness"
source_files = [source_root / name for name in ["physx_granular.py", "physx_rope.py"]]
source_hashes = {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
world = None
try:
    world = World(physics_dt=args.dt, rendering_dt=1 / 30, stage_units_in_meters=1,
                  backend="numpy", device="cpu")
    stage = world.stage
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    px = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    px.CreateSolverTypeAttr("TGS")
    px.CreateEnableGPUDynamicsAttr(args.case == "granular")
    px.CreateBroadphaseTypeAttr("GPU" if args.case == "granular" else "MBP")
    px.CreateMinPositionIterationCountAttr(16)
    px.CreateMinVelocityIterationCountAttr(4)
    px.CreateEnableStabilizationAttr(False)
    box(stage, "/World/Support", [1, 1, .04], [0, 0, -.02])
    dome = UsdLux.DomeLight.Define(stage, "/World/Light")
    dome.CreateIntensityAttr(700.)
    samples, times, rope_gaps = [], [], []
    if args.case == "granular":
        for i, (size, center) in enumerate([
            ([.02, .34, .08], [-.17, 0, .04]), ([.02, .34, .08], [.17, 0, .04]),
            ([.34, .02, .08], [0, -.17, .04]), ([.34, .02, .08], [0, .17, .04]),
        ]):
            box(stage, f"/World/Wall_{i}", size, center)
        seed = sample_lattice([-.12, -.12, .015], [.12, .12, .04], .01)
        asset = author_granular(stage, "/World/Grains", seed,
                                GranularSettings(.004, 4e-5, cohesion=.1),
                                simulation_owner="/physicsScene")
        pusher = box(stage, "/World/Scraper", [.02, .12, .035], [-.12, 0, .025], kinematic=True)
        report["configuration"] = asset
    else:
        seed = np.column_stack([np.linspace(-.25, .25, 21), np.zeros(21), np.full(21, .12)])
        asset = author_rope(stage, "/World/Rope", seed, radius_m=.004, density_kg_m3=1000)
        pusher = box(stage, "/World/Pusher", [.025, .025, .04], [0, -.12, .02], kinematic=True)
        report["configuration"] = asset
    world.reset()
    # Bind a physical material for deterministic contact response on unbound shapes.
    material = UsdShade.Material.Define(stage, "/World/TestContact")
    physical = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physical.CreateStaticFrictionAttr(.6)
    physical.CreateDynamicFrictionAttr(.5)
    physical.CreateRestitutionAttr(0.)
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/World/Support")).Bind(material, materialPurpose="physics")
    snapshot(world, "initial")
    pre_interaction = None
    for step in range(round(args.seconds / args.dt)):
        t = step * args.dt
        if .7 <= t <= 1.7:
            if pre_interaction is None:
                pre_interaction = read_granular_positions(stage, asset) if args.case == "granular" else read_rope_state(stage, asset)["centers_m"]
            progress = np.clip((t - .7), 0, 1)
            smooth = progress**2 * (3 - 2 * progress)
            xyz = [-.12 + .24 * smooth, 0, .025] if args.case == "granular" else [0, -.12 + .24 * smooth, .02]
            pusher.GetOrderedXformOps()[0].Set(Gf.Vec3d(*xyz))
        world.step(render=False)
        if step % 8 == 0:
            if args.case == "granular":
                state = read_granular_positions(stage, asset)
            else:
                reading = read_rope_state(stage, asset)
                state = reading["centers_m"]
                rope_gaps.append(reading["joint_gaps_m"])
            samples.append(state)
            times.append(t + args.dt)
        if step == round(1.8 / args.dt):
            snapshot(world, "interaction")
    checks = report["checks"]
    if args.case == "granular":
        final = read_granular_positions(stage, asset)
        report["particle_audit"] = audit_grains(final, expected_count=len(seed), radius_m=.004,
            floor_z_m=0, bounds_xy_m=[[-.16, -.16], [.16, .16]], tolerance_m=.003)
        displacement = np.linalg.norm(final - pre_interaction, axis=1)
        checks.update(grain_integrity=report["particle_audit"]["ok"],
                      tool_redistributed=bool(np.sum(displacement > .01) > 20))
        report["moved_grains"] = int(np.sum(displacement > .01))
    else:
        final = read_rope_state(stage, asset)
        report["max_joint_gap_m"] = float(max(np.max(rope_gaps), final["joint_gaps_m"].max()))
        checks.update(finite=final["finite"], continuous=report["max_joint_gap_m"] < .003,
                      support=final["minimum_surface_z_m"] >= -.001,
                      tool_redistributed=bool(np.max(np.linalg.norm(final["centers_m"] - pre_interaction, axis=1)) > .025))
    snapshot(world, "final")
    stage.GetRootLayer().Export(str(args.output / "scene.usda"))
    np.savez_compressed(args.output / "motion.npz", time_s=times, positions_m=samples, joint_gaps_m=rope_gaps)
    report["ok"] = bool(checks and all(checks.values()))
    assert all(hashlib.sha256((source_root / name).read_bytes()).hexdigest() == digest
               for name, digest in source_hashes.items()), "Source changed during native validation"
    report["implementation_sha256"] = source_hashes
except Exception as exc:
    import traceback
    report.update(ok=False, error=str(exc), traceback=traceback.format_exc())
    print(report["traceback"], flush=True)
finally:
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    app.close()
if not report.get("ok"):
    raise SystemExit(1)
