"""Offline rigid fall and driven hinge smoke check for both PhysX profiles."""
# ruff: noqa: E402 -- start Kit before importing simulator APIs.
import argparse

from isaac_physx_compat import runtime

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=240)
runtime.add_launcher_args(parser)
args = parser.parse_args()
if args.steps < 120:
    parser.error("use at least 120 steps to check fall and settling")
runtime.prepare(args)

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sim.utils import get_current_stage
from pxr import Gf, Sdf, UsdGeom, UsdPhysics


def main():
    with runtime.launch(args) as physics:
        sim = sim_utils.SimulationContext(runtime.simulation_cfg(physics, dt=1 / 120, device=args.device))
        floor = sim_utils.CuboidCfg(
            size=(10, 10, 0.1), collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
        )
        floor.func("/World/Floor", floor, translation=(0, 0, -0.05))
        cube = RigidObject(RigidObjectCfg(
            prim_path="/World/Cube",
            spawn=sim_utils.CuboidCfg(
                size=(0.1, 0.1, 0.1), rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, 0.6)),
        ))
        stage = get_current_stage()
        UsdGeom.Xform.Define(stage, "/World/Hinge")
        base = UsdGeom.Cube.Define(stage, "/World/Hinge/Base")
        base.CreateSizeAttr(0.1)
        base.AddTranslateOp().Set(Gf.Vec3d(1, 0, 0.6))
        UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
        UsdPhysics.MassAPI.Apply(base.GetPrim()).CreateMassAttr(1.0)
        fixed = UsdPhysics.FixedJoint.Define(stage, "/World/Hinge/Fixed")
        fixed.CreateBody1Rel().SetTargets([Sdf.Path("/World/Hinge/Base")])
        fixed.CreateLocalPos0Attr(Gf.Vec3f(1, 0, 0.6))
        UsdPhysics.ArticulationRootAPI.Apply(fixed.GetPrim())
        body = UsdGeom.Cube.Define(stage, "/World/Hinge/Link")
        body.CreateSizeAttr(0.1)
        body.AddTranslateOp().Set(Gf.Vec3d(1, 0, 0.8))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        UsdPhysics.CollisionAPI.Apply(body.GetPrim())
        UsdPhysics.MassAPI.Apply(body.GetPrim()).CreateMassAttr(1.0)
        joint = UsdPhysics.RevoluteJoint.Define(stage, "/World/Hinge/Joint")
        joint.CreateBody0Rel().SetTargets([Sdf.Path("/World/Hinge/Base")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path("/World/Hinge/Link")])
        joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0.2))
        joint.CreateAxisAttr("Z")
        joint.CreateLowerLimitAttr(-90)
        joint.CreateUpperLimitAttr(90)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(100)
        drive.CreateDampingAttr(5)
        drive.CreateMaxForceAttr(1000)
        hinge = Articulation(ArticulationCfg(
            prim_path="/World/Hinge", spawn=None,
            actuators={"hinge": runtime.actuator_cfg(
                joint_names_expr=[".*"], stiffness=100.0, damping=5.0, joint_effort_limit=1000.0)},
        ))
        sim.reset()
        cube.update(1 / 120)
        hinge.update(1 / 120)
        initial_z = float(runtime.pose_xyzw(cube.data.root_link_pose_w)[0, 2])
        target = torch.full_like(runtime.tensor(hinge.data.joint_pos), 0.35)
        for _ in range(args.steps):
            runtime.set_position(hinge, target)
            hinge.write_data_to_sim()
            cube.write_data_to_sim()
            sim.step(render=False)
            hinge.update(1 / 120)
            cube.update(1 / 120)
        pose = runtime.pose_xyzw(cube.data.root_link_pose_w)
        angle = runtime.tensor(hinge.data.joint_pos)
        speed = runtime.tensor(cube.data.root_link_vel_w)[..., :3].norm()
        z = float(pose[0, 2])
        if not bool(torch.isfinite(pose).all() and torch.isfinite(angle).all()):
            raise RuntimeError("nonfinite PhysX state")
        if not (initial_z - z > 0.3 and 0.025 < z < 0.08 and float(speed) < 0.1):
            raise RuntimeError(f"rigid fall/contact failed: z={z}, speed={float(speed)}")
        if float((angle - target).abs().max()) > 0.05:
            raise RuntimeError(f"hinge control failed: {angle}")
        print(f"PHYSX_SMOKE_OK lab={'2.3' if runtime.legacy else '3'} steps={args.steps} z={z} joints={angle.tolist()}", flush=True)
        runtime.stop(sim)


if __name__ == "__main__":
    main()
