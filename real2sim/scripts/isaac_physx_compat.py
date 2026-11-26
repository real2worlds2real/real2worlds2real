"""PhysX API boundary for Lab 2.3 / Sim 5.1 and Lab 3 / Sim 6.2.

Importing this module does not import Kit-dependent simulation modules. Call
``prepare`` after parsing arguments and before importing those modules: Lab 2
requires a running SimulationApp, while Lab 3 owns startup in its context manager.
Saved bundle/replay poses always use XYZW regardless of the runtime convention.
"""

import os
import sys
import traceback
from contextlib import contextmanager


class PhysxRuntime:
    def __init__(self):
        import isaaclab.app as app

        self.app = app
        self.legacy = not hasattr(app, "launch_simulation")
        self.launcher = None

    def add_launcher_args(self, parser):
        if self.legacy:
            self.app.AppLauncher.add_app_launcher_args(parser)
            parser.add_argument("--viz", choices=("none", "gui"), default=None)
        else:
            self.app.add_launcher_args(parser)

    def prepare(self, args):
        args.physics = "isaacsim_physx"
        if os.environ.get("REAL2SIM_ISAAC_PROFILE") == "5.1" and not self.legacy:
            raise RuntimeError("The Sim 5.1 profile requires Isaac Lab 2.3, not Lab 3")
        if self.legacy:
            if args.viz == "none":
                args.headless = True
            # Lab 2 pops launcher fields from its input dictionary. Keep script
            # arguments (especially device) available after application startup.
            launch_args = {**vars(args), "fast_shutdown": True}
            launch_args.setdefault("multi_gpu", False)
            self.launcher = self.app.AppLauncher(launch_args)
            if os.environ.get("REAL2SIM_ISAAC_PROFILE") == "5.1":
                from isaacsim.core.version import get_version

                version = get_version()[0]
                if not version.startswith("5.1."):
                    message = f"Expected Isaac Sim 5.1, got {version}"
                    print(message, file=sys.stderr, flush=True)
                    self.launcher.app._app.post_quit(1)
                    self.launcher.app.close(skip_cleanup=True)
                    raise RuntimeError(message)

    @contextmanager
    def launch(self, args):
        if self.legacy:
            if self.launcher is None:
                raise RuntimeError("Call runtime.prepare(args) before simulation imports")
            try:
                yield None
            except BaseException as exc:
                # Kit's fast shutdown exits the process. Set its return code
                # before closing so failed validation cannot become success.
                code = exc.code if isinstance(exc, SystemExit) else 1
                code = code if isinstance(code, int) else 1
                if code:
                    traceback.print_exc()
                self.launcher.app._app.post_quit(code)
                raise
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                # Standalone workers have already persisted their evidence.
                # Lab 2 can enter a keep-rendering loop on timeline STOP;
                # let process shutdown release native resources directly.
                self.launcher.app.close(skip_cleanup=True)
        else:
            from isaaclab.physics import PhysicsCfg

            with self.app.launch_simulation(cfg=PhysicsCfg(), launcher_args=args) as physics:
                yield physics

    def simulation_cfg(self, physics, **kwargs):
        from isaaclab.sim import SimulationCfg

        if not self.legacy:
            kwargs["physics"] = physics
        return SimulationCfg(**kwargs)

    def ground(self, path, z=0.0):
        """Create a local floor without downloading the stock ground USD."""
        import isaaclab.sim as sim_utils

        cfg = sim_utils.CuboidCfg(
            size=(200.0, 200.0, 0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        )
        cfg.func(path, cfg, translation=(0.0, 0.0, z - 0.05))

    def stop(self, sim):
        if self.legacy:
            # Lab 2's standalone STOP callback otherwise renders forever,
            # preventing post-stop evidence writing. Its reset uses this guard.
            sim._disable_app_control_on_stop_handle = True
        sim.stop()

    @staticmethod
    def tensor(value):
        return getattr(value, "torch", value)

    def pose_xyzw(self, value):
        pose = self.tensor(value)
        if not self.legacy:
            return pose
        result = pose.clone()
        result[..., 3:6] = pose[..., 4:7]
        result[..., 6] = pose[..., 3]
        return result

    def camera_quaternion(self, xyzw):
        if not self.legacy:
            return xyzw
        if not hasattr(xyzw, "clone"):
            return (xyzw[3], *xyzw[:3])
        result = xyzw.clone()
        result[..., 0] = xyzw[..., 3]
        result[..., 1:] = xyzw[..., :3]
        return result

    def actuator_cfg(self, *, joint_effort_limit, **kwargs):
        from isaaclab.actuators import ImplicitActuatorCfg

        key = "effort_limit_sim" if self.legacy else "joint_effort_limit"
        return ImplicitActuatorCfg(**kwargs, **{key: joint_effort_limit})

    def set_position(self, asset, target):
        if self.legacy:
            asset.set_joint_position_target(target)
        else:
            asset.actuators.target_command.set_position_index(value=target)

    def set_effort(self, asset, target):
        if self.legacy:
            asset.set_joint_effort_target(target)
        else:
            asset.actuators.target_command.set_effort_index(value=target)

    def write_joint_state(self, asset, position, velocity):
        if self.legacy:
            asset.write_joint_state_to_sim(position, velocity)
        else:
            asset.write_joint_state_to_sim_index(position=position, velocity=velocity)

    def body_paths(self, asset):
        view = asset.root_physx_view if self.legacy else asset.root_view
        return view.link_paths[0]

    def rigid_body_path(self, asset):
        """Path of the single measured body, which may be below the asset root."""
        view = asset.root_physx_view if self.legacy else asset.root_view
        paths = view.prim_paths
        if len(paths) != 1:
            raise ValueError("Scene support review expects one rigid body per participant")
        return paths[0]

    def prepare_articulation(self, asset):
        """Decouple nested rigid transforms after spawning into the final world.

        Sim 5.1 requires a reset stack on nested rigid bodies. Apply it only to
        the live stage: portable source assets must still inherit scene placement.
        Preserve both the visual hierarchy and every authored world transform.
        """
        if not self.legacy:
            return
        from isaaclab.sim.utils import get_current_stage
        from pxr import Usd, UsdGeom, UsdPhysics

        stage = get_current_stage()
        root = stage.GetPrimAtPath(asset.cfg.prim_path)
        bodies = [prim for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
        paths = {prim.GetPath() for prim in bodies}
        cache = UsdGeom.XformCache()
        changes = []
        for prim in bodies:
            parent = prim.GetParent()
            while parent and parent != root.GetParent():
                if parent.GetPath() in paths:
                    changes.append((prim, cache.GetLocalToWorldTransform(prim)))
                    break
                parent = parent.GetParent()
        for prim, world in changes:
            xform = UsdGeom.Xformable(prim)
            xform.MakeMatrixXform().Set(world)
            xform.SetResetXformStack(True)

    def camera(self, *, carb_settings, **kwargs):
        from isaaclab.sensors import CameraCfg

        if self.legacy:
            import carb
            from isaaclab.sensors import Camera

            for key, value in carb_settings.items():
                carb.settings.get_settings().set(key, value)
            kwargs.setdefault("update_latest_camera_pose", True)
            return Camera(CameraCfg(**kwargs))
        from isaaclab.utils import instantiate
        from isaaclab_physx.renderers import IsaacRtxRendererCfg, IsaacRtxRendererGlobalSettingsCfg

        return instantiate(CameraCfg(
            **kwargs,
            renderer_cfg=IsaacRtxRendererCfg(
                global_settings=IsaacRtxRendererGlobalSettingsCfg(carb_settings=carb_settings)),
        ))

    def numpy_physics_view(self):
        import omni.physics.tensors as tensors

        if self.legacy:
            return tensors.create_simulation_view("numpy")
        from isaaclab.sim.utils.stage import get_current_stage_id

        return tensors.create_simulation_view("numpy", stage_id=get_current_stage_id())

    def contact_impulses(self, view):
        # dt=1 returns impulse units (N s) for both tensor API generations.
        if self.legacy:
            return view.get_net_contact_forces(dt=1.0)
        return view.get_net_normal_contact_forces(dt=1.0)


runtime = PhysxRuntime()
