"""Synthetic legacy FEM material-reference serialization control; no task approval.

Two identical compressed shapes: one retains its uncompressed material
reference, the other is a mass-matched stress-free counterpart. Reload the USD
before starting physics, inspect native initial positions, then release both.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import traceback

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',required=True)
parser.add_argument('--backend',choices=['legacy','volume_fem'],default='legacy')
parser.add_argument('--diagnostic-elasticity-damping',type=float,
                    help='Independent native volume material-rate intervention; does not qualify a reconstruction')
parser.add_argument('--seconds',type=float,default=.15)
args=parser.parse_args()
if args.diagnostic_elasticity_damping is not None and (args.backend!='volume_fem' or not 0<args.diagnostic_elasticity_damping<=100):
    parser.error('Positive bounded material-rate diagnostic requires volume_fem')
if not .15<=args.seconds<=2:parser.error('Bounded control duration required')
out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
shutil.copy2(__file__,out/'source.py')
from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True})
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import omni.physics.tensors as tensors  # noqa: E402
from pxr import UsdGeom,UsdPhysics,PhysxSchema,Vt  # noqa: E402
from scipy.spatial import Delaunay  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from calibrated_scene_reconstruction.assets.tetra_boundary import surface,determinants  # noqa: E402
from calibrated_scene_reconstruction.sim.volume_initial_state import author  # noqa: E402
from calibrated_scene_reconstruction.sim.reset_timestep import reset  # noqa: E402
from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors  # noqa: E402
from calibrated_scene_reconstruction.tasks.rigid.task_checks import as_numpy  # noqa: E402
errors=PhysicsErrors(out/'native_physics_errors.json')
try:
    stage=omni.usd.get_context().get_stage()
    scene=UsdPhysics.Scene.Define(stage,'/physicsScene');scene.CreateGravityMagnitudeAttr(0.)
    api=PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim());api.CreateEnableGPUDynamicsAttr(True)
    api.CreateBroadphaseTypeAttr('GPU');api.CreateSolverTypeAttr('PGS')
    base=np.array([[x,y,z] for x in [-.01,.01] for y in [-.01,.01] for z in [-.01,.01]])
    cells=Delaunay(base).simplices.copy();negative=determinants(base,cells)<0
    cells[negative]=cells[negative][:,[1,0,2,3]]
    values={};authoring={}
    cases=[('preserved',[-.1,0,0],True),('baked_control',[.1,0,0],False)]
    if args.diagnostic_elasticity_damping is not None:cases.append(('damped',[0,.1,0],True))
    for name,offset,prestrained in cases:
        original=base+offset;current=base*[.85,1.,1.]+offset
        reference=original if prestrained else current
        state={'version':1,'basis':'Synthetic15percent compression and mass-matched stress-free counterpart',
               'skin_rest_points':reference,'simulation_rest_points':reference,'simulation_points':current,
               'simulation_indices':cells,'collision_rest_points':reference,'collision_points':current,'collision_indices':cells}
        f=surface(reference,cells);mesh=UsdGeom.Mesh.Define(stage,'/World/'+name)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy((current if args.backend=='volume_fem' else reference).astype('float32')))
        mesh.CreateFaceVertexCountsAttr([3]*len(f));mesh.CreateFaceVertexIndicesAttr(f.ravel().tolist())
        mesh.CreateSubdivisionSchemeAttr('none')
        settings={'youngs_modulus':20000.,'poissons_ratio':.2,'density':500. if prestrained else 500./.85,
                  'dynamic_friction':.55,**({'linear_damping':0.} if args.backend=='volume_fem' else {'damping_scale':.05}),'solver_position_iterations':128,
                  'contact_offset':.0003,'rest_offset':0.}
        authoring[name]=author(stage,mesh,state,settings,backend=args.backend);values[name]=state
        if name=='damped':
            mat=stage.GetPrimAtPath('/World/'+name+'_material')
            if not mat.ApplyAPI('PhysxDeformableMaterialAPI'):raise ValueError('Native volume material damping unavailable')
            mat.GetAttribute('physxDeformableMaterial:elasticityDamping').Set(args.diagnostic_elasticity_damping)
    serialized=out/'serialized.usda';stage.GetRootLayer().Export(str(serialized))
    loaded=omni.usd.get_context().open_stage(str(serialized))
    if not (loaded[0] if isinstance(loaded,tuple) else loaded):raise ValueError('Serialized initial state did not reload')
    stage=omni.usd.get_context().get_stage()
    world=World(stage_units_in_meters=1.,physics_dt=1/3840,device='cuda:0',backend='torch')
    errors.guard_steps(world)
    pc=world.get_physics_context();pc.enable_gpu_dynamics(True);pc.set_broadphase_type('GPU');pc.set_solver_type('PGS')
    scene=UsdPhysics.Scene(stage.GetPrimAtPath('/physicsScene'));scene.GetGravityMagnitudeAttr().Set(0.)
    bootstrap=reset(world,bootstrap_hz=1000000)
    owner=tensors.create_simulation_view('torch');owner.set_subspace_roots('/')
    views={name:(owner.create_volume_deformable_body_view('/World/'+name) if args.backend=='volume_fem'
                 else owner.create_soft_body_view('/World/'+name)) for name in values}
    if any(view is None or view.count!=1 for view in views.values()):raise ValueError('Reloaded native volume missing')
    references={};initial={};metrics={}
    def positions(view):
        return as_numpy(view.get_simulation_nodal_positions() if args.backend=='volume_fem' else view.get_sim_nodal_positions())[0].copy()
    def tetrahedra(view):
        return as_numpy(view.get_simulation_element_indices() if args.backend=='volume_fem' else view.get_sim_element_indices())[0].astype(int)
    for name,view in views.items():
        body=PhysxSchema.PhysxDeformableBodyAPI(stage.GetPrimAtPath('/World/'+name))
        rest=(as_numpy(view.get_rest_nodal_positions())[0].copy() if args.backend=='volume_fem'
              else np.asarray(body.GetSimulationRestPointsAttr().Get(),float))
        indices=tetrahedra(view);current=positions(view)
        expected=values[name]['simulation_points']
        np.savez_compressed(out/(name+'_native_reset.npz'),rest=rest,indices=indices,current=current,
                            expected_rest=values[name]['simulation_rest_points'],expected_current=expected,expected_indices=cells)
        same_cells=(indices.shape==cells.shape and set(map(tuple,np.sort(indices,axis=1)))==set(map(tuple,np.sort(cells,axis=1))))
        if current.shape!=expected.shape or not same_cells:
            raise ValueError('Explicit material indexing changed on reload: '+str((name,current.shape,expected.shape,indices.shape,cells.shape)))
        references[name]=determinants(rest,indices);initial[name]=current
        metrics[name]={'reference_error_m':float(np.linalg.norm(rest-values[name]['simulation_rest_points'],axis=1).max()),
                       'initial_position_error_m':float(np.linalg.norm(current-expected,axis=1).max())}
    rows=[];recorded={name:[] for name in views}
    def record(time):
        row={'time_s':time}
        for name,view in views.items():
            points=positions(view);recorded[name].append(points)
            indices=tetrahedra(view)
            jacobian=determinants(points,indices)/references[name]
            row[name]={'maximum_motion_m':float(np.linalg.norm(points-initial[name],axis=1).max()),
                       'minimum_volume_jacobian':float(jacobian.min()),'maximum_volume_jacobian':float(jacobian.max())}
        rows.append(row)
    record(0.)
    for step in range(round(args.seconds*3840)):
        world.step(render=False)
        if (step+1)%192==0:record((step+1)/3840)
    errors.check();peak={name:max(row[name]['maximum_motion_m'] for row in rows) for name in views}
    checks={'same_compressed_initial_shapes':all(v['initial_position_error_m']<2e-5 for v in metrics.values()),
            'separate_original_references':all(v['reference_error_m']<1e-7 for v in metrics.values()),
            'reference_preserving_state_recoils':peak['preserved']>.0001,
            'baked_reference_counterpart_stays_quiet':peak['baked_control']<1e-5,
            'positive_material_cells':all(row[name]['minimum_volume_jacobian']>0 for row in rows for name in views),
            'native_error_check':errors.report()['pass']}
    damping_control=None
    if args.diagnostic_elasticity_damping is not None:
        def oscillation(name):
            states=np.asarray(recorded[name]);states-=states.mean(axis=1,keepdims=True)
            late=states[len(states)//2:]
            return float(np.linalg.norm(late-late.mean(axis=0),axis=2).max())
        undamped=oscillation('preserved');damped=oscillation('damped')
        mat=stage.GetPrimAtPath('/World/damped_material')
        readback=float(mat.GetAttribute('physxDeformableMaterial:elasticityDamping').Get())
        damping_control={'rate_requested':args.diagnostic_elasticity_damping,'rate_usd_readback':readback,
                         'undamped_late_oscillation_m':undamped,'damped_late_oscillation_m':damped,
                         'native_effect_observed':bool(damped<.5*undamped),
                         'scope':'Internal prestrain recoil with no contact or external load. This does not identify a real material rate or qualify load/contact behavior.'}
        checks['independent_native_damping_effect']=damping_control['native_effect_observed']
    np.savez_compressed(out/'states.npz',**recorded,**{name+'_rest':value['simulation_rest_points'] for name,value in values.items()},tetrahedra=cells)
    dependencies=[__file__,'src/calibrated_scene_reconstruction/sim/volume_initial_state.py','src/calibrated_scene_reconstruction/assets/volume_initial_state.py',
                  'src/calibrated_scene_reconstruction/assets/volume_contact.py','src/calibrated_scene_reconstruction/assets/tetra_boundary.py','src/calibrated_scene_reconstruction/sim/reset_timestep.py']
    result={'pass':all(checks.values()),'checks':checks,'synthetic_control':True,'task_approved':False,'backend':args.backend,
            'metrics':metrics,'peak_motion_m':peak,'rows':rows,'authoring':authoring,'bootstrap':bootstrap,
            'diagnostic_damping_control':damping_control,
            'native_physics_errors':errors.report(),'sources_sha256':{str(Path(p).resolve()):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in dependencies},
            'serialized_sha256':hashlib.sha256(serialized.read_bytes()).hexdigest(),
            'scope':'Native explicit legacy material-reference serialization capability only. No captured full-asset state, source fit, contact preload, task or delivered USDZ qualification.'}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('VOLUME_INITIAL_STATE',json.dumps({'pass':result['pass'],'checks':checks,'peak_motion_m':peak}),flush=True)
except Exception as error:
    (out/'error.json').write_text(json.dumps({'pass':False,'error':str(error),'traceback':traceback.format_exc()},indent=2)+'\n');print(traceback.format_exc(),flush=True)
finally:
    errors.close();app.close()
