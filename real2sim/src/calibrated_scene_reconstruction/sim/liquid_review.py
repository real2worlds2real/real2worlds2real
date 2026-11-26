"""Native liquid surface preparation and immutable camera-review checks."""
from pathlib import Path
import numpy as np
from calibrated_scene_reconstruction.sim.fluid_reference import snapshot as fluid_snapshot, verify_surfaces, verify_initialization


class Review:
    def __init__(self, stage, args, app, rep, carb):
        self.stage, self.args, self.app, self.rep, self.carb = stage, args, app, rep, carb
        before_fluids = fluid_snapshot(stage)
        fluid_initialization = None
        if args.initialize_fluid_surfaces and not before_fluids:
            raise ValueError('Fluid initialization requested but this stage has no enabled native fluid surfaces')
        self.before, self.initialization = before_fluids, fluid_initialization

    def start(self):
        import omni.timeline
        stage, args, app, rep, carb = self.stage, self.args, self.app, self.rep, self.carb
        before_fluids, fluid_initialization = self.before, self.initialization
        if before_fluids:
            # Load native particle post-processing without starting/stepping simulation.
            # A stopped headless stage does not initialize it from render updates alone.
            import omni.physx  # noqa: E402
            from omni.physx.bindings import _physx as physx_bindings  # noqa: E402
            from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402
            SimulationManager.enable_fabric(False)
            settings = carb.settings.get_settings()
            settings.set(physx_bindings.SETTING_UPDATE_TO_USD, True)
            settings.set(physx_bindings.SETTING_UPDATE_PARTICLES_TO_USD, True)
            settings.set('/physics/suppressReadback', False)
            omni.physx.get_physx_interface().force_load_physics_from_usd()
            if args.initialize_fluid_surfaces:
                owners = {p for system in before_fluids for p in
                          stage.GetPrimAtPath(system).GetRelationship('simulationOwner').GetTargets()}
                rates = [stage.GetPrimAtPath(p).GetAttribute('physxScene:timeStepsPerSecond').Get() for p in owners]
                if any(rate is None or not np.isfinite(rate) or rate <= 0 for rate in rates):
                    raise ValueError('Fluid preview needs explicit native scene timesteps')
                dt = 1. / max(rates)
                # Keep render-product capture running while preventing automatic physics
                # advances. Replicator.step on a stopped timeline resets the fluid mesh.
                settings.set('/app/player/playSimulations', False)
                rep.orchestrator.set_capture_on_play(True)
                omni.timeline.get_timeline_interface().play()
                app.update()
                # Starting capture must not silently advance physics before our one
                # explicit step. Compare positions/velocities, not generated surfaces.
                capture_fluids = fluid_snapshot(stage)
                for system, particles in before_fluids.items():
                    for path, attributes in particles.items():
                        for key, value in attributes.items():
                            if not np.array_equal(capture_fluids[system][path].get(key), value):
                                raise ValueError('Capture startup advanced fluid state before the explicit step: '+path+'.'+key)
                omni.physx.get_physx_interface().start_simulation()
                omni.physx.get_physx_interface().update_simulation(dt, 0.)
                omni.physx.get_physx_interface().update_transformations(False, True, True, False)
                fluid_initialization = {'steps': 1, 'duration_s': dt, 'task_physics_approved': False,
                                        'displacement': verify_initialization(stage, before_fluids)}
                before_fluids = fluid_snapshot(stage)
        self.before, self.initialization = before_fluids, fluid_initialization

    @property
    def capture_running(self):
        return bool(self.initialization)

    def verify(self):
        return verify_surfaces(self.stage, self.before)

    def finish(self, out):
        render = verify_surfaces(self.stage, self.before, output=out/'native_surfaces' if self.before else None)
        result = {'native_fluid_surfaces': render, 'fluid_initialization': self.initialization}
        if self.initialization:
            result['scope'] = 'Existing USD, source files unchanged; explicit single fluid startup step, then physics updates disabled; novel views unpaired.'
        return result


def prepare(stage, args, app, rep, carb):
    return Review(stage, args, app, rep, carb)
