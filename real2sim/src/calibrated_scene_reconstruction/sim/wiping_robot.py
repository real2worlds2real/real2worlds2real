"""Numpy planning interface with native articulation tensor-backend adaptation."""
import numpy as np
from calibrated_scene_reconstruction.sim.robot_articulation import Robot


class WipingRobot(Robot):
    def _ensure_views(self):
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.utils.stage import get_current_stage_id
        import omni.physics.tensors
        simulation = self.world.physics_sim_view
        if (simulation is None or getattr(simulation, '_backend', None) is None or not simulation.is_valid):
            # Retiring a PhysX actor invalidates the *simulation* tensor view,
            # even if Python wrappers remain non-None. Rebind to the live scene;
            # do not initialize/reset physics, advance time, or write any pose.
            simulation = omni.physics.tensors.create_simulation_view(
                SimulationManager.get_backend(), stage_id=get_current_stage_id())
            simulation.set_subspace_roots('/')
            if not simulation.is_valid:
                raise RuntimeError('Cannot rebind native view to the live physics scene')
            SimulationManager._physics_sim_view = simulation
            warp = omni.physics.tensors.create_simulation_view('warp', stage_id=get_current_stage_id())
            warp.set_subspace_roots('/')
            SimulationManager._physics_sim_view__warp = warp
            self.native_scene_rebinds = getattr(self, 'native_scene_rebinds', 0) + 1
            print('[rgb_wipe] rebound native tensor scene without resetting physics', flush=True)
        previous = getattr(self, '_bound_simulation_view', None)
        changed = previous is not None and previous is not simulation
        for view in (self.view, self.robot._articulation_view):
            native = getattr(view, '_physics_view', None)
            if changed or native is None or getattr(native, '_backend', None) is None:
                # The SDK resolves an assembly path to its articulation root.
                # Preserve those resolved patterns instead of the assembly Xform.
                patterns = [p.replace('.*', '*') for p in view._regex_prim_paths]
                view._physics_view = simulation.create_articulation_view(patterns)
                if getattr(view._physics_view, '_backend', None) is None:
                    raise RuntimeError('Cannot rebind live articulation view')
                self.native_articulation_rebinds = getattr(self, 'native_articulation_rebinds', 0) + 1
        self._bound_simulation_view = simulation
        return simulation

    def link_transforms(self):
        self._ensure_views()
        return super().link_transforms()

    def _values(self, q):
        return self.robot._backend_utils.convert(np.asarray(q, np.float32), device=self.robot._device)

    def teleport(self, q):
        from isaacsim.core.utils.types import ArticulationAction
        self._ensure_views()
        values = self._values(q)
        self.robot.set_joint_positions(values)
        self.robot.set_joint_velocities(self._values(np.zeros(len(q))))
        self.robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=values))

    def command(self, q, qd=None):
        from isaacsim.core.utils.types import ArticulationAction
        self._ensure_views()
        self.robot.get_articulation_controller().apply_action(ArticulationAction(
            joint_positions=self._values(q), joint_velocities=None if qd is None else self._values(qd)))

    def q(self):
        self._ensure_views()
        values = self.robot.get_joint_positions()
        return values.detach().cpu().numpy().astype(float) if hasattr(values, 'detach') else np.asarray(values, float)

    def sync(self):
        # Planning must not advance deformable physics while testing FK states.
        self._ensure_views().update_articulations_kinematic()
