"""Native two-fluid isolation, gravity, contact/capacity and mass-transfer control.

Synthetic numerical control only. No task or source appearance acceptance.
"""
import argparse
import json
from pathlib import Path
import traceback

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402

app = SimulationApp({'headless': True})
try:
    import carb
    import numpy as np
    import omni.usd
    from isaacsim.core.api import World
    from omni.physx.bindings import _physx as bindings
    from pxr import UsdPhysics
    from calibrated_scene_reconstruction.sim.native_fluid import NativeFluid
    from calibrated_scene_reconstruction.sim.cloth2d_wipe import Cloth2DWipe
    from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors
    from calibrated_scene_reconstruction.sim.simulation import apply

    hz = 480
    world = World(stage_units_in_meters=1., physics_dt=1/hz, rendering_dt=1/30)
    context = world.get_physics_context()
    context.enable_gpu_dynamics(True)
    context.set_broadphase_type('GPU')
    context.enable_fabric(False)
    settings = carb.settings.get_settings()
    settings.set('/physics/suppressReadback', False)
    for key in (bindings.SETTING_UPDATE_TO_USD, bindings.SETTING_UPDATE_VELOCITIES_TO_USD,
                bindings.SETTING_UPDATE_PARTICLES_TO_USD):
        settings.set(key, True)
    stage = omni.usd.get_context().get_stage()
    apply(stage, dict(physics_hz=hz, solver='TGS', friction_type='patch', gpu_dynamics=True,
                      rigid_iterations=[64, 16], basis='Explicit shared fluid/mechanism numerical control'))
    scene = next(p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene))
    errors = PhysicsErrors(args.output/'native_errors.json')
    errors.guard_steps(world)
    cloth = Cloth2DWipe(stage, [0, 0, .0675], 0., absorption_mass_in_dynamics=True)
    spacing = .0016
    xx, yy = np.meshgrid(np.arange(4)*spacing-.0024, np.arange(4)*spacing+.017)
    points = np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, .05185)]
    spec = dict(basis='Independent native freefall/contact controls; effective PBD coefficients, no material calibration',
                spacing_m=spacing, density_kg_m3=1000., iterations=32, isosurface=False,
                pbd=dict(friction=.1, cohesion=0., viscosity=0., surface_tension=0., adhesion=0., damping=0.))
    near = NativeFluid(stage, '/World/FluidNear', scene.GetPath(), points, spec)
    far = NativeFluid(stage, '/World/FluidFar', scene.GetPath(), points+[.20, 0, 0], spec)
    world.reset()
    settings.set('/physics/suppressReadback', False)
    cloth.initialize()
    cap = 8*near.particle_mass
    for fluid in (near, far):
        fluid.bind_absorber(cloth, capacity_kg=cap, mass_rate_kg_s=.005,
                            contact_distance_m=.0022, minimum_dwell_s=.008)
    p0 = far.state()[0]
    mass0 = float(cloth.masses().sum())
    rows = []
    for step in range(96):
        world.step(render=False)
        near.absorb(1/hz)
        far.absorb(1/hz)
        if step % 12 == 0:
            rows.append(dict(time_s=(step+1)/hz, near=near.ledger(), far=far.ledger(),
                             cloth_mass_kg=float(cloth.masses().sum())))
    pf, vf = far.state()
    near_mass = near.ledger()['absorbed_mass_kg']
    mass_error = float(cloth.masses().sum()) - mass0 - near_mass
    checks = dict(native_gravity=float(pf[:, 2].mean()-p0[:, 2].mean()) < -.03,
                  contact_absorption=near.absorbed_count > 0,
                  noncontact_control=far.absorbed_count == 0,
                  capacity_limit=near_mass <= cap+1e-10,
                  native_mass_transfer=abs(mass_error) < 1e-7,
                  mass_ledger=abs(near.ledger()['mass_error_kg']) < 1e-10,
                  independent_systems=near.system_path != far.system_path)
    errors.check()
    result = dict(pass_control=all(checks.values()), task_approved=False,
                  checks=checks, near=near.ledger(), far=far.ledger(),
                  native_absorber_mass_error_kg=mass_error,
                  far_mean_velocity_m_s=vf.mean(0).tolist(),
                  near_events=near.events, history=rows, native_physics=errors.report(),
                  scope='Synthetic solver control; real source fluid geometry and task validation remain required')
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print('NATIVE_FLUID_CONTROL', checks, flush=True)
    errors.close()
except Exception:
    (args.output/'error.txt').write_text(traceback.format_exc())
    raise
finally:
    app.close()
