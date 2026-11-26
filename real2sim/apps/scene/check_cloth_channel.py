"""Synthetic channel-friction control on explicit source rigid geometry.

This is a local mechanism control, never whole-sheet or source/task approval.
The fixed source body/contents and artificial tangential gravity load are
declared fixture approximations, not a reconstructed initial state.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import traceback

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--protocol', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--friction', type=float, required=True)
parser.add_argument('--diagnostic-material', action='store_true',
                    help='Continue a finite local strain-limit excess as a failed diagnostic; never grant acceptance')
args = parser.parse_args()
protocol_path = Path(args.protocol).resolve()
protocol = json.loads(protocol_path.read_text())
if not 0 <= args.friction <= 2:
    parser.error('Bounded explicit fixture friction required')
for name in ['mesh', 'profile']:
    if not Path(protocol[name]).is_file():
        parser.error('Missing protocol input: ' + name)
out = Path(args.output).resolve()
out.mkdir(parents=True, exist_ok=False)
shutil.copy2(__file__, out/'source.py')
shutil.copy2(protocol_path, out/'protocol.json')
shutil.copy2(protocol['profile'], out/'profile.json')

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app = SimulationApp({'headless': True})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade, Vt  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_channel import bowed  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_metric import MaterialMetric  # noqa: E402
from calibrated_scene_reconstruction.sim.surface_cloth import author, SurfaceView  # noqa: E402
from calibrated_scene_reconstruction.sim.cloth_contact import self_intersections  # noqa: E402
from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors  # noqa: E402
from calibrated_scene_reconstruction.sim.simulation import apply as apply_simulation  # noqa: E402
from calibrated_scene_reconstruction.sim.surface_distance import SignedSurface, triangle_samples  # noqa: E402
from calibrated_scene_reconstruction.tasks.rigid.task_checks import as_numpy  # noqa: E402

errors = PhysicsErrors(out/'native_physics_errors.json')
try:
    profile = json.loads(Path(protocol['profile']).read_text())
    settings = {**profile['settings'], 'friction': args.friction}
    hz = profile['physics_hz']
    world = World(stage_units_in_meters=1., physics_dt=1/hz, device='cuda:0', backend='torch')
    errors.guard_steps(world)
    pc = world.get_physics_context()
    pc.enable_gpu_dynamics(True)
    pc.set_broadphase_type('GPU')
    stage = omni.usd.get_context().get_stage()
    apply_simulation(stage, {'physics_hz': hz, 'solver': 'PGS', 'friction_type': 'patch',
                            'basis': 'Declared synthetic channel control'}, flexible=True)
    scene = UsdPhysics.Scene(next(p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)))
    scene.GetGravityMagnitudeAttr().Set(0.)
    scene.GetGravityDirectionAttr().Set(Gf.Vec3f(0., 0., 1.))
    with np.load(protocol['mesh']) as data:
        links = json.loads(str(data['links_json']))
        link = next(row for row in links if row['name'] == protocol['part'])
        a, n, b, m = [link[k] for k in ['vertex_start','vertex_count','face_start','triangle_count']]
        solid_v = data['V'][a:a+n].copy()
        solid_f = data['F'][b:b+m].copy()-a
    def mesh(path, points, faces):
        value = UsdGeom.Mesh.Define(stage, path)
        value.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype('float32')))
        value.CreateFaceVertexCountsAttr([3]*len(faces))
        value.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
        value.CreateSubdivisionSchemeAttr('none')
        return value
    rigid = mesh('/World/FixedSourceSolid', solid_v, solid_f)
    UsdPhysics.CollisionAPI.Apply(rigid.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(rigid.GetPrim()).CreateApproximationAttr('sdf')
    PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(rigid.GetPrim()).CreateSdfResolutionAttr(protocol['sdf']['resolution'])
    collision = PhysxSchema.PhysxCollisionAPI.Apply(rigid.GetPrim())
    collision.CreateContactOffsetAttr(protocol['sdf']['contact_offset'])
    collision.CreateRestOffsetAttr(0.)
    material = UsdShade.Material.Define(stage, '/World/FixtureMaterial')
    physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_material.CreateStaticFrictionAttr(args.friction)
    physics_material.CreateDynamicFrictionAttr(args.friction)
    physics_material.CreateRestitutionAttr(0.)
    UsdShade.MaterialBindingAPI.Apply(rigid.GetPrim()).Bind(material, UsdShade.Tokens.weakerThanDescendants, 'physics')
    points, faces, chart, geometry = bowed(**protocol['coupon'])
    # Local X spans radial gap; local Y is tangential bow, Z is extraction.
    origin = np.asarray(protocol['origin_m'])
    radial = np.asarray(protocol['radial_xy'])
    tangent = np.array([-radial[1], radial[0]])
    points[:, :2] = points[:, :1]*radial + points[:, 1:2]*tangent
    points += origin
    part = {'V': points, 'F': faces, 'cloth_rest': chart, 'kind': 'cloth', 'backend': 'surface_fem'}
    sheet = mesh('/World/ChannelSheet', points, faces)
    author(stage, sheet, part, settings)
    stage.GetRootLayer().Export(str(out/'initial.usda'))
    world.reset()
    errors.check()
    view = SurfaceView(stage, '/World/ChannelSheet', part)
    solid_distance = SignedSurface(solid_v, solid_f)
    initial = as_numpy(view.get_world_positions())[0].copy()
    masses = as_numpy(view.get_particle_masses())[0]
    metric = MaterialMetric(initial, faces, chart, max_edge_strain=.05, max_area_change=.03)
    states, velocities, rows = [], [], []
    mass = float(masses.sum())
    centre0 = np.average(initial, axis=0, weights=masses)
    def record(step, load):
        actual = as_numpy(view.get_world_positions())[0].copy()
        velocity = as_numpy(view.get_velocities())[0].copy()
        centre = np.average(actual, axis=0, weights=masses)
        material_result = metric.check(actual)
        cross = self_intersections(actual, faces, example_limit=2)
        minimum_clearance = float(solid_distance(triangle_samples(actual, faces)).min())
        row = {'time_s':step/hz, 'load_N':load, 'upward_displacement_m':float(centre[2]-centre0[2]),
               'maximum_speed_m_s':float(np.linalg.norm(velocity,axis=1).max()),
               'minimum_sampled_solid_clearance_m':minimum_clearance,
               'solid_thickness_pass':minimum_clearance >= settings['thickness']/2-2e-6,
               'material':material_result, 'self_intersection':cross}
        states.append(actual); velocities.append(velocity); rows.append(row)
        (out/'progress.json').write_text(json.dumps(row, indent=2)+'\n')
        return row
    record(0, 0.)
    step = 0
    abort = None
    for phase in protocol['phases']:
        load = phase['force_N']
        scene.GetGravityMagnitudeAttr().Set(load/mass)
        # Sync Isaac's cached runtime gravity as well as USD.
        pc.set_gravity(load/mass)
        if load and (tuple(scene.GetGravityDirectionAttr().Get()) != (0.,0.,1.)
                     or not np.isclose(scene.GetGravityMagnitudeAttr().Get(),load/mass)):
            raise ValueError('Fixture load gravity readback differs from protocol')
        for local in range(round(phase['seconds']*hz)):
            world.step(render=False); step += 1
            if local % max(1,round(.005*hz)) == 0:
                row = record(step, load)
                material_abort = (not row['material']['geometry_valid']
                    or row['material']['maximum_principal_material_strain'] > .25
                    or (not row['material']['pass'] and not args.diagnostic_material))
                if (material_abort or not row['self_intersection']['pass']
                        or not row['solid_thickness_pass']):
                    abort = 'Invalid physical sheet state'; break
                if abs(row['upward_displacement_m']) > protocol['stop_displacement_m']:
                    abort = 'Declared fixture travel limit reached'; break
        if abort:break
    errors.check()
    record(step, load)
    np.savez_compressed(out/'states.npz', points=states, velocities=velocities, faces=faces,
                        rest=np.asarray(chart['points']), masses=masses)
    result = {'synthetic_control':True, 'task_approved':False, 'source_approved':False,
              'fixed_source_body':True, 'rigid_contents_approximation':True,
              'attachments':False, 'positions_written_after_reset':False,
              'loading':'Artificial +Z gravity gives explicit total tangential force = mass * acceleration; not source gravity',
              'friction':args.friction, 'mass_kg':mass, 'settings':settings, 'geometry':geometry,
              'diagnostic_material_continuation':args.diagnostic_material,
              'raw_material_limits_pass':all(row['material']['pass'] for row in rows),
              'input_sha256':{key:hashlib.sha256(Path(protocol[key]).read_bytes()).hexdigest() for key in ['mesh','profile']},
              'native_physics_errors':errors.report(), 'precision_audit':errors.precision_audit(),
              'abort':abort, 'rows':rows,
              'scope':'Local channel mechanism only. Contents compressibility, whole sheet, free bottle, closed lid and Sharpa grasp remain untested.'}
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print('CHANNEL_CONTROL_RESULT', json.dumps({'friction':args.friction,'abort':abort,
          'upward_displacement_m':rows[-1]['upward_displacement_m'],'mass_kg':mass}), flush=True)
except Exception as error:
    (out/'error.json').write_text(json.dumps({'error':str(error),'traceback':traceback.format_exc()},indent=2)+'\n')
    raise
finally:
    errors.close(); app.close()
