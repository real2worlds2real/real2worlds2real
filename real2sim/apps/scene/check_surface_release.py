"""Whole-source-sheet free release, with explicit contact isolation.

No source state, material or task approval. No rigid bodies, gravity, guides,
attachments or native state writes; preserve authored geometry and rest chart.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import traceback

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--mesh', required=True)
p.add_argument('--profile', required=True)
p.add_argument('--cloth-link', default='wipe')
p.add_argument('--self-contact', choices=['on', 'off'], required=True)
p.add_argument('--seconds', type=float, default=.4)
p.add_argument('--record-fps', type=int, default=80)
p.add_argument('--output', required=True)
args = p.parse_args()
if not 0 < args.seconds <= 2:
    p.error('Bounded positive isolation duration required')
profile = json.loads(Path(args.profile).read_text())
hz = profile['physics_hz']
if not 1 <= args.record_fps <= hz or hz % args.record_fps:
    p.error('Record frequency must divide the native clock')
for path in [args.mesh, args.profile]:
    if not Path(path).is_file(): p.error('Missing input: '+path)
out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=False)
shutil.copy2(__file__, out/'source.py')
shutil.copy2(args.profile, out/'profile.json')
(out/'protocol.json').write_text(json.dumps(vars(args), indent=2)+'\n')

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app = SimulationApp({'headless': True})
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from pxr import UsdGeom, UsdPhysics, Vt  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_metric import MaterialMetric  # noqa: E402
from calibrated_scene_reconstruction.assets.membrane_energy import estimate  # noqa: E402
from calibrated_scene_reconstruction.sim.surface_cloth import author, SurfaceView  # noqa: E402
from calibrated_scene_reconstruction.sim.cloth_contact import self_intersections  # noqa: E402
from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors  # noqa: E402
from calibrated_scene_reconstruction.sim.simulation import apply as apply_simulation  # noqa: E402
from calibrated_scene_reconstruction.tasks.rigid.task_checks import as_numpy  # noqa: E402

errors = PhysicsErrors(out/'native_physics_errors.json')
try:
    with np.load(args.mesh) as data:
        link = next(row for row in json.loads(str(data['links_json'])) if row['name']==args.cloth_link)
        a,n,b,m = [link[k] for k in ['vertex_start','vertex_count','face_start','triangle_count']]
        points = data['V'][a:a+n].astype(float).copy()
        faces = data['F'][b:b+m].copy()-a
        chart = json.loads(str(data['cloth_rest_json']))[args.cloth_link]
    if link.get('backend')!='surface_fem' or len(points)>500000:
        raise ValueError('Explicit surface FEM source within node budget required')
    settings = profile['settings']
    rest = np.asarray(chart['points'])
    world = World(stage_units_in_meters=1., physics_dt=1/hz, device='cuda:0', backend='torch')
    errors.guard_steps(world)
    pc = world.get_physics_context(); pc.enable_gpu_dynamics(True); pc.set_broadphase_type('GPU')
    stage = omni.usd.get_context().get_stage()
    apply_simulation(stage, {'physics_hz':hz, 'solver':'PGS', 'friction_type':'patch',
                            'basis':'Unforced whole-source-sheet contact isolation'}, flexible=True)
    scene = UsdPhysics.Scene(next(q for q in stage.Traverse() if q.IsA(UsdPhysics.Scene)))
    scene.GetGravityMagnitudeAttr().Set(0.)
    mesh = UsdGeom.Mesh.Define(stage, '/World/SourceSheet')
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype('float32')))
    mesh.CreateFaceVertexCountsAttr([3]*len(faces)); mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr('none')
    part = {'V':points, 'F':faces, 'cloth_rest':chart, 'kind':'cloth', 'backend':'surface_fem'}
    author(stage, mesh, part, settings)
    mesh.GetPrim().GetAttribute('physxDeformableBody:selfCollision').Set(args.self_contact=='on')
    if any((q.HasAPI(UsdPhysics.CollisionAPI) and q.GetPath()!=mesh.GetPath()) or q.HasAPI(UsdPhysics.RigidBodyAPI)
           or 'Attachment' in q.GetTypeName() for q in stage.Traverse()):
        raise ValueError('Isolation unexpectedly contains a rigid partner or attachment')
    stage.GetRootLayer().Export(str(out/'initial.usda'))
    world.reset(); errors.check()
    if mesh.GetPrim().GetAttribute('physxDeformableBody:selfCollision').Get() != (args.self_contact=='on'):
        raise ValueError('Self-contact flag readback mismatch')
    if scene.GetGravityMagnitudeAttr().Get()!=0.: raise ValueError('Isolation gravity changed')
    view = SurfaceView(stage, '/World/SourceSheet', part)
    initial = as_numpy(view.get_world_positions())[0].copy()
    masses = as_numpy(view.get_particle_masses())[0]
    metric = MaterialMetric(points, faces, chart, max_edge_strain=.05, max_area_change=.03)
    energy_settings = {key:settings[key] for key in ['youngs_modulus','poissons_ratio','thickness','areal_density']}
    states, speeds, rows = [], [], []
    centre0 = np.average(initial, axis=0, weights=masses)
    def record(step):
        actual = as_numpy(view.get_world_positions())[0].copy()
        velocity = as_numpy(view.get_velocities())[0].copy()
        if not np.isfinite(actual).all() or not np.isfinite(velocity).all():
            raise ValueError('Nonfinite native release')
        row = {'time_s':step/hz, 'material':metric.check(actual),
               'self_intersection':self_intersections(actual,faces,example_limit=2),
               'maximum_displacement_from_native_t0_m':float(np.linalg.norm(actual-initial,axis=1).max()),
               'maximum_displacement_from_authored_m':float(np.linalg.norm(actual-points,axis=1).max()),
               'centre_of_mass_drift_m':float(np.linalg.norm(np.average(actual,axis=0,weights=masses)-centre0)),
               'maximum_speed_m_s':float(np.linalg.norm(velocity,axis=1).max()),
               'net_momentum_Ns':np.sum(masses[:,None]*velocity,axis=0).tolist(),
               'kinetic_energy_J':float(.5*np.sum(masses[:,None]*velocity**2)),
               'membrane_energy':estimate(rest,actual,faces,**energy_settings)}
        states.append(actual); speeds.append(velocity); rows.append(row)
        (out/'progress.json').write_text(json.dumps(row,indent=2)+'\n')
    record(0)
    steps=round(args.seconds*hz); stride=hz//args.record_fps
    for step in range(1, steps+1):
        world.step(render=False)
        if step%stride==0 or step==steps: record(step)
    errors.check()
    np.savez_compressed(out/'states.npz', points=states, velocities=speeds, authored=points,
                        faces=faces, rest=rest, masses=masses)
    sources=[Path(__file__),Path(args.mesh),Path(args.profile),
             Path('src/calibrated_scene_reconstruction/sim/surface_cloth.py'),Path('src/calibrated_scene_reconstruction/assets/membrane_energy.py')]
    result={'task_approved':False, 'source_approved':False, 'diagnostic_only':True,
            'complete':True,'rigid_partners':0,'attachments':0,'gravity':0.,'state_writes_after_reset':False,
            'self_contact':args.self_contact,'profile':profile,
            'authored_material':metric.check(points),
            'authored_membrane_energy':estimate(rest,points,faces,**energy_settings),
            'initial_native_displacement_m':float(np.linalg.norm(initial-points,axis=1).max()),
            'maximum_displacement_m':max(row['maximum_displacement_from_native_t0_m'] for row in rows),
            'maximum_com_drift_m':max(row['centre_of_mass_drift_m'] for row in rows),
            'native_physics_errors':errors.report(),'rows':rows,
            'input_sha256':{str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
            'scope':'Compare matched full-source-shape trajectories. Self-contact-off crossings remain recorded; this isolation cannot approve a physical initial state.'}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('SURFACE_RELEASE_RESULT',json.dumps({k:result[k] for k in ['self_contact','initial_native_displacement_m','maximum_displacement_m','maximum_com_drift_m']}),flush=True)
except Exception as error:
    (out/'error.json').write_text(json.dumps({'error':str(error),'traceback':traceback.format_exc()},indent=2)+'\n')
    raise
finally:
    errors.close(); app.close()
