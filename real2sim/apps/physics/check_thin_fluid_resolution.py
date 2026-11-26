"""Native thin-film resolution control, independent of a captured scene.

A stable bulk fluid does not establish a resolved thin film. Keep the physical
footprint and thickness fixed while varying the number of layers. Measure
coverage and particle clustering as well as motion; no task approval is issued.
"""
import argparse
import hashlib
import json
from pathlib import Path
import traceback

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--hz', type=int, default=3840)
parser.add_argument('--seconds', type=float, default=1.5)
args = parser.parse_args()
if not 960 <= args.hz <= 15360 or not .5 <= args.seconds <= 5:
    parser.error('Explicit bounded native timestep and observation duration required')
args.output.mkdir(parents=True, exist_ok=False)
files = [Path(__file__), Path('src/calibrated_scene_reconstruction/sim/native_fluid.py'),
         Path('src/calibrated_scene_reconstruction/sim/simulation.py')]
inputs = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(args.output/'inputs.json').write_text(json.dumps(dict(files=inputs, arguments=vars(args)),
                                                 default=str, indent=2)+'\n')
from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app = SimulationApp({'headless': True})
try:
    import carb
    import numpy as np
    import omni.usd
    from isaacsim.core.api import World
    from omni.physx.bindings import _physx as bindings
    from pxr import UsdGeom, UsdPhysics, Gf
    from scipy.spatial import cKDTree
    from calibrated_scene_reconstruction.sim.native_fluid import NativeFluid
    from calibrated_scene_reconstruction.sim.simulation import apply
    from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors

    world = World(stage_units_in_meters=1., physics_dt=1/args.hz, rendering_dt=1/30)
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
    apply(stage, dict(physics_hz=args.hz, solver='TGS', friction_type='patch',
                     gpu_dynamics=True, rigid_iterations=[64, 16],
                     basis='Independent analytic thin-film resolution control'))
    scene = next(p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene))
    errors = PhysicsErrors(args.output/'native_errors.json')
    errors.guard_steps(world)
    width, thickness = .024, .0012
    sample_axis = np.arange(-width/2+.00015, width/2, .0003)
    sample_xy = np.stack(np.meshgrid(sample_axis, sample_axis), -1).reshape(-1, 2)
    cases = []
    for index, layers in enumerate([1, 2, 3, 4]):
        spacing = thickness/layers
        origin = np.array([index*.06, 0., 0.])
        root = f'/World/Case{index}'
        UsdGeom.Xform.Define(stage, root)
        floor = UsdGeom.Cube.Define(stage, root+'/Floor')
        floor.CreateSizeAttr(1.)
        floor.AddTranslateOp().Set(Gf.Vec3d(*(origin+[0, 0, -.002])))
        floor.AddScaleOp().Set(Gf.Vec3d(.05, .05, .004))
        UsdPhysics.CollisionAPI.Apply(floor.GetPrim())
        axis = np.arange(-width/2+spacing/2, width/2, spacing)
        height = (np.arange(layers)+.5)*spacing
        points = np.stack(np.meshgrid(axis, axis, height, indexing='ij'), -1).reshape(-1, 3)+origin
        specification = dict(basis='Same analytic footprint, thickness and effective water preset at each resolution',
                             spacing_m=spacing, density_kg_m3=1000., iterations=64,
                             isosurface=True, pbd=dict(friction=.12, cohesion=.01, surface_tension=.0074,
                                                       viscosity=.000085, adhesion=0., damping=0.))
        fluid = NativeFluid(stage, root+'/Fluid', scene.GetPath(), points, specification)
        cases.append(dict(fluid=fluid, initial=points, origin=origin, spacing=spacing,
                          layers=layers, history=[]))
    world.reset()
    settings.set('/physics/suppressReadback', False)
    steps = round(args.seconds*args.hz)
    for step in range(steps):
        world.step(render=False)
        if step % max(1, args.hz//4) == 0 or step == steps-1:
            for case in cases:
                points, velocity = case['fluid'].state()
                xy = points[:, :2]-case['origin'][:2]
                distances = cKDTree(xy).query(sample_xy)[0]
                nearest = cKDTree(points).query(points, k=2)[0][:, 1]
                case['history'].append(dict(time_s=(step+1)/args.hz,
                    coverage_distance_percentiles_m=np.percentile(distances, [50, 95, 99, 100]).tolist(),
                    uncovered_fraction=float(np.mean(distances>case['spacing'])),
                    nearest_neighbour_percentiles_m=np.percentile(nearest, [1, 50, 99]).tolist(),
                    near_coincident_fraction=float(np.mean(nearest<case['spacing']*.125)),
                    rms_speed_m_s=float(np.sqrt(np.mean(np.sum(velocity**2, axis=1)))),
                    maximum_speed_m_s=float(np.linalg.norm(velocity, axis=1).max()),
                    z_percentiles_m=np.percentile(points[:, 2], [0, 50, 95, 100]).tolist(),
                    ledger=case['fluid'].ledger()))
            (args.output/'progress.json').write_text(json.dumps([c['history'] for c in cases], indent=2)+'\n')
    result = []
    for index, case in enumerate(cases):
        points, velocity = case['fluid'].state()
        np.savez_compressed(args.output/f'state_{index}.npz', initial=case['initial'],
                            points=points, velocity=velocity)
        result.append(dict(layers=case['layers'], spacing_m=case['spacing'],
                           initial=case['fluid'].initial, history=case['history']))
    errors.check()
    (args.output/'result.json').write_text(json.dumps(dict(accepted=False,
        inputs=inputs, inputs_unchanged=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in inputs.items()),
        physics_hz=args.hz, analytic_thickness_m=thickness, analytic_footprint_m2=width**2,
        cases=result, native_physics=errors.report(), scope=__doc__), indent=2)+'\n')
    print('THIN_FILM', [(c['layers'], c['history'][-1]['uncovered_fraction'],
                       c['history'][-1]['near_coincident_fraction']) for c in result], flush=True)
    errors.close()
except Exception:
    (args.output/'error.txt').write_text(traceback.format_exc())
    raise
finally:
    app.close()
