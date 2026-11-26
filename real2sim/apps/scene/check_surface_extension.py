"""Independent free extensional vibration of a native thin strip.

The initial small strain excites a known free/free bar mode. There are no
attachments, applied forces, velocity writes, or prescribed points after reset.
This tests membrane elasticity without a solver-specific force adapter.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import traceback

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',required=True)
p.add_argument('--physical-priors',required=True)
p.add_argument('--physical-link',required=True)
p.add_argument('--case',choices=['low','nominal','high'],default='nominal')
p.add_argument('--solver',choices=['PGS','TGS'],default='TGS')
p.add_argument('--physics-hz',type=int,default=1920)
p.add_argument('--position-iterations',type=int,default=128)
p.add_argument('--length-m',type=float,default=.16)
p.add_argument('--width-m',type=float,default=.001)
p.add_argument('--spacing-m',type=float,default=.00025)
p.add_argument('--strain',type=float,default=.002)
p.add_argument('--seconds',type=float,default=.1)
a=p.parse_args()
if (not 240<=a.physics_hz<=30720 or not 4<=a.position_iterations<=255
        or not .00005<=a.spacing_m<=.002 or not 2*a.spacing_m<=a.width_m<=a.length_m/40
        or not .02<=a.length_m<=.5 or not .0005<=a.strain<=.005 or not .05<=a.seconds<=2):
    p.error('Bounded resolved slender coupon and small initial strain required')
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
shutil.copy2(__file__,out/'source.py');shutil.copy2(a.physical_priors,out/'physics_priors_input.json')
from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True})
import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from pxr import UsdGeom,UsdPhysics,PhysxSchema,Vt  # noqa: E402
from calibrated_scene_reconstruction.assets.physical_priors import sample  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_rest import triangulate_grid  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_metric import MaterialMetric  # noqa: E402
from calibrated_scene_reconstruction.assets.flexure_reference import oscillation  # noqa: E402
from calibrated_scene_reconstruction.sim.surface_cloth import author,SurfaceView  # noqa: E402
from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors  # noqa: E402
from calibrated_scene_reconstruction.tasks.rigid.task_checks import as_numpy  # noqa: E402
errors=PhysicsErrors(out/'native_physics_errors.json')
result={'pass':False,'synthetic_control':True,'task_approved':False,
        'external_forces':False,'positions_written_after_reset':False,
        'velocity_writes':False,'fixed_or_kinematic_nodes':0,'parameters':vars(a)}
try:
    estimates=sample(json.loads((out/'physics_priors_input.json').read_text()),a.case)
    material=estimates['bodies'][a.physical_link]['parameters']
    settings={k:material[k] for k in ['youngs_modulus','poissons_ratio','thickness','flexural_rigidity','areal_density']}
    h=settings['thickness'];rho=settings['areal_density']/h
    expected=np.sqrt(settings['youngs_modulus']/rho)/(2*a.length_m)
    if a.physics_hz/expected<8 or a.seconds*expected<8:
        raise ValueError('At least eight samples per reference cycle and eight reference cycles required')
    settings.update(elasticity_damping=0.,bend_damping=0.,linear_damping=0.,friction=0.,
                    contact_offset=.75*h,rest_offset=h/2,self_collision_filter_distance=h,
                    solver_position_iterations=a.position_iterations)
    world=World(stage_units_in_meters=1.,physics_dt=1/a.physics_hz,device='cuda:0',backend='torch')
    errors.guard_steps(world);pc=world.get_physics_context()
    pc.enable_gpu_dynamics(True);pc.set_broadphase_type('GPU');pc.set_solver_type(a.solver)
    stage=omni.usd.get_context().get_stage();scene=next(q for q in stage.Traverse() if q.IsA(UsdPhysics.Scene))
    UsdPhysics.Scene(scene).CreateGravityMagnitudeAttr(0.)
    PhysxSchema.PhysxSceneAPI.Apply(scene).CreateEnableExternalForcesEveryIterationAttr(a.solver=='TGS')
    x=np.linspace(-a.width_m/2,a.width_m/2,round(a.width_m/a.spacing_m)+1)
    y=np.linspace(-a.length_m/2,a.length_m/2,round(a.length_m/a.spacing_m)+1)
    xx,yy=np.meshgrid(x,y);rest=np.c_[xx.ravel(),yy.ravel(),np.zeros(xx.size)]
    wave=np.sin(np.pi*rest[:,1]/a.length_m)
    points=rest.copy();points[:,1]+=a.strain*a.length_m/np.pi*wave
    points[:,0]*=1-settings['poissons_ratio']*a.strain*np.cos(np.pi*rest[:,1]/a.length_m)
    faces,chart,_=triangulate_grid(points,rest,len(y),len(x));part={'V':points,'F':faces,'cloth_rest':chart}
    mesh=UsdGeom.Mesh.Define(stage,'/World/FreeStrip')
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype('float32')))
    mesh.CreateFaceVertexCountsAttr([3]*len(faces));mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr('none');author(stage,mesh,part,settings)
    stage.GetRootLayer().Export(str(out/'fixture.usda'));world.reset()
    view=SurfaceView(stage,'/World/FreeStrip',part)
    metric=MaterialMetric(points,faces,chart,max_edge_strain=.01,max_area_change=.01)
    rows=[];states=[];native_mat=view.owner.create_deformable_material_view('/World/FreeStrip_surface_fem_material')
    readback=as_numpy(native_mat.get_youngs_modulus()).reshape(-1).tolist()
    def record(t):
        q=as_numpy(view.get_world_positions())[0].copy();centred=q-q.mean(0)
        u,_,vt=np.linalg.svd(centred.T@rest);R=u@vt
        if np.linalg.det(R)<0:u[:,-1]*=-1;R=u@vt
        aligned=centred@R
        amplitude=float((aligned[:,1]-rest[:,1])@wave/(wave@wave))
        rows.append({'time_s':t,'modal_amplitude_m':amplitude,'material':metric.check(q)})
        states.append(q)
    record(0.)
    for step in range(round(a.seconds*a.physics_hz)):
        world.step(render=False);record((step+1)/a.physics_hz)
    errors.check();response=oscillation([r['time_s'] for r in rows],[r['modal_amplitude_m'] for r in rows])
    relative=abs(response['frequency_hz']/expected-1) if response['frequency_hz'] else None
    result.update(backend={'name':'isaacsim','version':importlib.metadata.version('isaacsim'),'solver':a.solver},
        settings=settings,expected_frequency_hz=float(expected),observed_oscillation=response,
        relative_frequency_error=relative,material_readback_youngs_modulus=readback,rows=rows,
        native_physics_errors=errors.report(),
        pass_=bool(response['pass'] and relative is not None and relative<.2 and
                   all(r['material']['pass'] for r in rows) and np.allclose(readback,settings['youngs_modulus'])))
    result['pass']=result.pop('pass_')
    np.savez_compressed(out/'states.npz',positions=states,rest=rest,faces=faces)
except Exception:
    result['error']=traceback.format_exc();result['native_physics_errors']=errors.report()
finally:
    files=[Path(__file__),Path('src/calibrated_scene_reconstruction/sim/surface_cloth.py'),Path('src/calibrated_scene_reconstruction/assets/flexure_reference.py'),out/'physics_priors_input.json']
    result['input_sha256']={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('EXTENSION',result['pass'],result.get('relative_frequency_error'),result.get('error',''),flush=True)
    errors.close();app.close()
