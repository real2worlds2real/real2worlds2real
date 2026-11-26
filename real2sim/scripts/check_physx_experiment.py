"""Opt-in real PhysX adapter smoke test; synthetic contact, not video calibration.

Run inside an operator-installed Isaac Sim Python environment. This script starts
Kit and may require graphics hardware even though dynamics use the CPU.
"""
# ruff: noqa: E402 -- simulator startup precedes runtime-only imports.
import argparse
from contextlib import contextmanager
import inspect
import json
from pathlib import Path
import traceback
from types import SimpleNamespace

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--articulation", action="store_true", help="Check driven hinge stiffness/damping instead of sliding friction")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)

from isaacsim import SimulationApp
app = SimulationApp({"headless": True, "disable_viewport_updates": True,
                     "multi_gpu": False, "active_gpu": 0, "physics_gpu": 0})

import numpy as np
import omni.kit.app
import omni.physics.tensors as tensors
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.version import get_version
from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

from real2sim_harness.artifacts import atomic_json
from real2sim_harness.physics_inference import run_trial
from real2sim_harness.physx_experiments import PhysXExperimentRunner


def author_hinge(stage):
    """Local fixed-base fixture with an explicit angular force drive."""
    UsdGeom.Xform.Define(stage, "/World/Hinge")
    for name, z in (("Base", 1.), ("Link", 1.2)):
        body = UsdGeom.Cube.Define(stage, "/World/Hinge/" + name)
        body.CreateSizeAttr(.1)
        body.AddTranslateOp().Set(Gf.Vec3d(0., 0., z))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(1.)
    fixed = UsdPhysics.FixedJoint.Define(stage, "/World/Hinge/Fixed")
    fixed.CreateBody1Rel().SetTargets([Sdf.Path("/World/Hinge/Base")])
    fixed.CreateLocalPos0Attr(Gf.Vec3f(0., 0., 1.))
    UsdPhysics.ArticulationRootAPI.Apply(fixed.GetPrim())
    joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Hinge/Joint")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/World/Hinge/Base")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/Hinge/Link")])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0., 0., .2))
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(-90.)
    joint.CreateUpperLimitAttr(90.)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(.1)
    drive.CreateDampingAttr(.02)
    drive.CreateMaxForceAttr(100.)
    drive.CreateTargetPositionAttr(0.)


@contextmanager
def session_factory(experiment):
    omni.usd.get_context().new_stage()
    dt = experiment["solver"]["sample_dt_s"] / experiment["solver"]["substeps"]
    world = World(physics_dt=dt, rendering_dt=dt, stage_units_in_meters=1., backend="numpy", device="cpu")
    stage = omni.usd.get_context().get_stage()
    articulated = experiment.get("fixture") == "hinge"
    if articulated:
        author_hinge(stage)
    for name, position, scale, dynamic in ([] if articulated else [
        ("Floor", [0., 0., -.05], [5., 5., .1], False),
        ("Probe", [0., 0., .026], [.12, .08, .05], True),
    ]):
        box = UsdGeom.Cube.Define(stage, "/World/" + name)
        box.CreateSizeAttr(1.)
        box.AddTranslateOp().Set(Gf.Vec3d(*position))
        box.AddScaleOp().Set(Gf.Vec3f(*scale))
        UsdPhysics.CollisionAPI.Apply(box.GetPrim())
        material = UsdShade.Material.Define(stage, "/World/" + name + "Material")
        props = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        props.CreateStaticFrictionAttr(0. if not dynamic else .5)
        props.CreateDynamicFrictionAttr(0. if not dynamic else .5)
        props.CreateRestitutionAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(box.GetPrim()).Bind(material, materialPurpose="physics")
        if dynamic:
            UsdPhysics.RigidBodyAPI.Apply(box.GetPrim())
            UsdPhysics.MassAPI.Apply(box.GetPrim()).CreateMassAttr(.1)
    version = omni.kit.app.get_app().get_extension_manager().get_enabled_extension_id("omni.physx")
    runtime = SimpleNamespace(views={}, metadata={"backend": "physx", "version": version,
                              "isaac_sim_version": get_version()[0], "device": "cpu"})
    simulation_view = None

    def reset(exp):
        nonlocal simulation_view
        world.reset()
        options = {}
        if "stage_id" in inspect.signature(tensors.create_simulation_view).parameters:
            options["stage_id"] = omni.usd.get_context().get_stage_id()
        simulation_view = tensors.create_simulation_view("numpy", **options)
        simulation_view.set_subspace_roots("/")
        indices = np.array([0], np.int32)
        if articulated:
            # Tensor views select the prim with ArticulationRootAPI, unlike
            # Lab's asset config which can resolve it below a parent Xform.
            view = simulation_view.create_articulation_view("/World/Hinge/Fixed")
            position = view.get_dof_positions().copy()
            position[:] = exp["initial_state"]["angle_rad"]
            view.set_dof_positions(position, indices)
            view.set_dof_velocities(np.zeros_like(position), indices)
            view.set_dof_position_targets(np.zeros_like(position), indices)
        else:
            view = simulation_view.create_rigid_body_view("/World/Probe")
            velocity = view.get_velocities().copy()
            velocity[0, :3] = exp["initial_state"]["velocity_m_s"]
            view.set_velocities(velocity, indices)
        runtime.views["asset"] = view
        return True

    def step(requested_dt):
        if not np.isclose(requested_dt, dt):
            raise ValueError("Requested step differs from configured PhysX timestep")
        world.step(render=False)

    runtime.reset, runtime.step = reset, step
    try:
        yield runtime
    finally:
        world.stop()
        runtime.views.clear()
        simulation_view = None
        World.clear_instance()


def check_articulation():
    template = {"schema_version": 1, "clock_source": "synthetic hinge timing", "tracks": [
        {"id": "hinge", "units": "rad", "frame": "bounded-Z-joint", "observable": "joint_angle",
         "source": "synthetic hinge test; angle never wraps", "time_s": [0., 1.],
         "values": [[.5], [.5]], "residual_scale": .01}]}
    experiment = {"target_backend": "physx", "fixture": "hinge", "observations": template,
                  "sources": {"geometry": "synthetic fixed-base hinge with angular force drive"},
                  "initial_state": {"angle_rad": .5}, "controls": {"position_target_rad": 0.},
                  "solver": {"frames": 121, "sample_dt_s": 1/120, "substeps": 2},
                  "comparison": {"max_gap_s": .02}}
    params = [{"name": name, "units": units, "bounds": [0., 2.], "target": "hinge drive",
               "rationale": "synthetic sensitivity check, not measured material physics"}
              for name, units in (("stiffness", "N*m/rad"), ("damping", "N*m*s/rad"))]
    bindings = {name: [{"view": "asset", "property": "joint_" + name, "indices": [[0, 0]]}]
                for name in ("stiffness", "damping")}
    runner = PhysXExperimentRunner(session_factory,
        lambda runtime, t, exp: {"hinge": runtime.views["asset"].get_dof_positions()[0]}, bindings)
    candidate = {"stiffness": .1, "damping": .02}
    run_trial(experiment, params, candidate, runner, args.output / "reference")
    reference = json.loads((args.output / "reference/rollout.json").read_text())
    reference["tracks"][0]["residual_scale"] = .01
    experiment["observations"] = reference
    repeated = run_trial(experiment, params, candidate, runner, args.output / "repeat")
    scores = {}
    for name, value in (("stiffness", 1.), ("damping", .2)):
        result = run_trial(experiment, params, {**candidate, name: value}, runner, args.output / name)
        scores[name] = result["score"]
    angles = np.asarray(reference["tracks"][0]["values"])
    if (repeated["score"] >= 1e-5 or min(scores.values()) <= .1
            or np.max(np.abs(angles)) >= np.pi/2 or np.ptp(angles) <= .1):
        raise RuntimeError("PhysX hinge sensitivity/reset check failed; inspect retained trials")
    atomic_json(args.output / "check.json", {"ok": True, "scope": "synthetic driven hinge only",
                "repeat_score": repeated["score"], "changed_scores": scores,
                "angle_range_rad": [float(angles.min()), float(angles.max())], "acceptance": False})


def main():
    if args.articulation:
        return check_articulation()
    template = {"schema_version": 1, "clock_source": "synthetic time template", "tracks": [
        {"id": "probe", "units": "m", "frame": "world", "observable": "body_origin",
         "source": "synthetic backend test", "time_s": [0., 1.],
         "values": [[0., 0., .026], [0., 0., .026]], "residual_scale": .01}]}
    experiment = {"target_backend": "physx", "observations": template,
                  "sources": {"geometry": "synthetic box and support"},
                  "initial_state": {"velocity_m_s": [1., 0., 0.]}, "controls": {"kind": "free release"},
                  "solver": {"frames": 31, "sample_dt_s": 1/30, "substeps": 8},
                  "comparison": {"max_gap_s": .04}}
    params = [{"name": "mu", "units": "1", "bounds": [0., 1.], "target": "probe material",
               "rationale": "synthetic parameter sensitivity, not measured real friction"}]
    bindings = {"mu": [{"view": "asset", "property": prop, "indices": [[0, 0]]}
                       for prop in ("static_friction", "dynamic_friction")]}
    runner = PhysXExperimentRunner(session_factory,
        lambda runtime, t, exp: {"probe": runtime.views["asset"].get_transforms()[0, :3]}, bindings)
    run_trial(experiment, params, {"mu": .02}, runner, args.output / "low")
    reference = json.loads((args.output / "low/rollout.json").read_text())
    reference["tracks"][0]["residual_scale"] = .01
    experiment["observations"] = reference
    repeated = run_trial(experiment, params, {"mu": .02}, runner, args.output / "repeat")
    changed = run_trial(experiment, params, {"mu": .8}, runner, args.output / "high")
    low_x = reference["tracks"][0]["values"][-1][0]
    high_x = json.loads((args.output / "high/rollout.json").read_text())["tracks"][0]["values"][-1][0]
    if low_x <= high_x + .1 or repeated["score"] >= 1e-5 or changed["score"] <= 1:
        raise RuntimeError("PhysX sensitivity/reset check failed; inspect retained trials")
    atomic_json(args.output / "check.json", {"ok": True, "scope": "synthetic PhysX adapter smoke only",
                "low_friction_end_x_m": low_x, "high_friction_end_x_m": high_x,
                "repeat_score": repeated["score"], "changed_score": changed["score"], "acceptance": False})


try:
    main()
except BaseException:
    # Kit shutdown can terminate Python before an exception reaches stderr.
    # Retain the traceback and a nonzero process result before closing Kit.
    traceback.print_exc()
    app._app.post_quit(1)
    raise
finally:
    app.close()
