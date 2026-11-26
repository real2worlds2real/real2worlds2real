"""Native unforced free-strip flexure control; never a scene or task pass."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import traceback

p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
p.add_argument('--physical-priors',required=True);p.add_argument('--physical-link',required=True)
p.add_argument('--case',choices=['low','nominal','high'],default='nominal')
p.add_argument('--spacing-m',type=float,default=.002)
p.add_argument('--width-m',type=float,default=.008)
p.add_argument('--physics-hz',type=int,choices=[240,480,960,1920,3840,7680],default=3840)
p.add_argument('--solver',choices=['PGS','TGS'],default='PGS')
p.add_argument('--position-iterations',type=int,default=255)
p.add_argument('--amplitude-m',type=float,default=.0002)
p.add_argument('--reference-height-m',type=float,default=.3)
p.add_argument('--internal-viscosity-s',type=float,nargs=2,metavar=('BEND','MEMBRANE'),
               help='Diagnostic internal damping before native elasticity; explicit viscosity/modulus times in seconds')
p.add_argument('--damping-mode',choices=['disabled','material'],default='disabled',
               help='Material mode records release decay for a matched control; it cannot approve bending frequency or task feasibility')
args=p.parse_args()
if not .00005<=args.spacing_m<=.002 or not 2*args.spacing_m<=args.width_m<=.04 or not 4<=args.position_iterations<=255:p.error('Bounded material mesh and solver budget required')
if not .0001<=args.amplitude_m<=.0015 or not 0<=args.reference_height_m<=2:
    p.error('Bounded small-amplitude excitation and explicit numerical world offset required')
if args.internal_viscosity_s is not None:
    import math
    if args.damping_mode!='disabled' or any(not math.isfinite(v) or not 0<=v<=1 for v in args.internal_viscosity_s):
        p.error('Internal-viscosity control requires explicit bounded times and native damping disabled')
out=Path(args.output);out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/'source.py')
shutil.copy2(args.physical_priors,out/'physics_priors_input.json');(out/'protocol.json').write_text(json.dumps(vars(args),indent=2)+'\n')

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from pxr import PhysxSchema,UsdGeom,UsdPhysics,Vt  # noqa: E402
from calibrated_scene_reconstruction.assets.physical_priors import sample  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_rest import triangulate_grid  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_metric import MaterialMetric  # noqa: E402
from calibrated_scene_reconstruction.assets import flexure_reference  # noqa: E402
from calibrated_scene_reconstruction.sim.surface_cloth import author,SurfaceView  # noqa: E402
from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors  # noqa: E402
from calibrated_scene_reconstruction.tasks.rigid.task_checks import as_numpy  # noqa: E402

errors=PhysicsErrors(out/'native_physics_errors.json')
try:
    values=sample(json.loads((out/'physics_priors_input.json').read_text()),args.case)['bodies'][args.physical_link]['parameters']
    settings={k:values[k] for k in ['youngs_modulus','poissons_ratio','thickness','flexural_rigidity','areal_density']}
    h=settings['thickness'];settings.update(elasticity_damping=values['elasticity_damping'] if args.damping_mode=='material' else 0.,
        bend_damping=values.get('bend_damping',values['elasticity_damping']) if args.damping_mode=='material' else 0.,
        linear_damping=values['linear_damping'] if args.damping_mode=='material' else 0.,friction=0.,
        contact_offset=.75*h,rest_offset=h/2,self_collision_filter_distance=h,solver_position_iterations=args.position_iterations)
    world=World(stage_units_in_meters=1.,physics_dt=1/args.physics_hz,device='cuda:0',backend='torch');errors.guard_steps(world)
    pc=world.get_physics_context();pc.enable_gpu_dynamics(True);pc.set_broadphase_type('GPU');pc.set_solver_type(args.solver)
    stage=omni.usd.get_context().get_stage();scene=next(v for v in stage.Traverse() if v.IsA(UsdPhysics.Scene))
    UsdPhysics.Scene(scene).GetGravityMagnitudeAttr().Set(0.)
    PhysxSchema.PhysxSceneAPI.Apply(scene).CreateEnableExternalForcesEveryIterationAttr(args.solver=='TGS')
    length,width=.08,args.width_m
    x=np.linspace(-width/2,width/2,round(width/args.spacing_m)+1);s=np.linspace(0,length,round(length/args.spacing_m)+1)
    shape=flexure_reference.mode(s,length);shape-=np.polyval(np.polyfit(s,shape,1),s);shape/=abs(shape).max()
    z=args.amplitude_m*shape;slope=np.gradient(z,s)
    if abs(slope).max()>=.1:raise ValueError('Small-amplitude strip control required')
    dy=np.diff(s)*np.sqrt(1-((slope[1:]+slope[:-1])/2)**2);axial=np.r_[0,np.cumsum(dy)]
    xx,yy=np.meshgrid(x,s-length/2);rest=np.c_[xx.ravel(),yy.ravel(),np.zeros(xx.size)]
    points=rest.copy();points[:,1]=np.repeat(axial-axial.mean(),len(x));points[:,2]=np.repeat(z+args.reference_height_m,len(x))
    faces,chart,_=triangulate_grid(points,rest,len(s),len(x));part={'V':points,'F':faces,'cloth_rest':chart}
    mesh=UsdGeom.Mesh.Define(stage,'/World/FreeStrip');mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype('float32')))
    mesh.CreateFaceVertexCountsAttr([3]*len(faces));mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist());mesh.CreateSubdivisionSchemeAttr('none')
    author(stage,mesh,part,settings);stage.GetRootLayer().Export(str(out/'fixture.usda'));world.reset()
    view=SurfaceView(stage,'/World/FreeStrip',part);metric=MaterialMetric(points,faces,chart,max_edge_strain=.01,max_area_change=.01)
    internal=None
    if args.internal_viscosity_s is not None:
        from calibrated_scene_reconstruction.sim import constitutive_damping,discrete_bending,discrete_membrane,discrete_dissipation
        from calibrated_scene_reconstruction.assets import cloth_projection,cloth_curvature
        for module in [constitutive_damping,discrete_bending,discrete_membrane,discrete_dissipation,cloth_projection,cloth_curvature]:
            shutil.copy2(module.__file__,out/Path(module.__file__).name)
        bending=discrete_bending.DiscreteBending(rest,faces,np.zeros(len(cloth_curvature.adjacency(faces)[0])),settings['flexural_rigidity'],device='cuda:0')
        membrane=discrete_membrane.DiscreteMembrane(rest,faces,
            **{k:settings[k] for k in ['youngs_modulus','poissons_ratio','thickness']},device='cuda:0')
        terms=[(bending,args.internal_viscosity_s[0]),(membrane,args.internal_viscosity_s[1])]
        masses=as_numpy(view.get_particle_masses())[0]
        internal={'diagnostic_only':True,'native_elasticity_preserved':True,'positions_written':False,
                  'world_drag':False,'maximum_internal_linear_impulse_Ns':0.,'steps':0}
    basis=np.repeat(shape,len(x));basis-=basis.mean();basis/=np.sqrt(np.mean(basis**2))
    expected=flexure_reference.frequency(length,settings['flexural_rigidity'],settings['areal_density'])
    seconds=max(.8,3.2/expected)
    if seconds>3.:raise ValueError('Flexure period outside bounded independent-control interval')
    stride=args.physics_hz//240;steps=int(np.ceil(seconds*args.physics_hz/stride))*stride
    states=[];times=[];amplitudes=[];rows=[]
    def record(step):
        actual=as_numpy(view.get_world_positions())[0].copy()
        centred=actual-actual.mean(0);origin=rest-rest.mean(0)
        u,_,v=np.linalg.svd(centred.T@origin);rotation=u@v
        if np.linalg.det(rotation)<0:u[:,-1]*=-1;rotation=u@v
        aligned=centred@rotation;amplitude=float(aligned[:,2]@basis/len(basis))
        states.append(actual);times.append(step/args.physics_hz);amplitudes.append(amplitude)
        row={'time_s':times[-1],'modal_amplitude_m':amplitude,'material':metric.check(actual)};rows.append(row)
        (out/'progress.json').write_text(json.dumps(row,indent=2)+'\n')
    record(0)
    for step in range(steps):
        if internal is not None:
            import torch
            update,control=constitutive_damping.velocity_step(view.get_world_positions()[0],view.get_velocities()[0],masses,1/args.physics_hz,terms)
            view.set_velocities(update.to(dtype=torch.float32)[None].contiguous())
            internal['steps']+=1
            internal['maximum_internal_linear_impulse_Ns']=max(internal['maximum_internal_linear_impulse_Ns'],float(np.linalg.norm(control['linear_impulse_Ns'])))
        world.step(render=False)
        if (step+1)%stride==0:record(step+1)
    errors.check();response=flexure_reference.oscillation(times,amplitudes)
    relative=abs(response['frequency_hz']/expected-1) if response['frequency_hz'] else None
    from calibrated_scene_reconstruction.assets.deformable_release import modal_decay
    decay=modal_decay(times,amplitudes,expected)
    result={'pass':bool(args.damping_mode=='disabled' and internal is None and response['pass'] and relative is not None and relative<.2 and all(r['material']['pass'] for r in rows)),
        'control_complete':True,'material_valid':all(r['material']['pass'] for r in rows),
        'internal_viscosity':internal,
        'damping_mode':args.damping_mode,'release_decay':decay,
        'synthetic_control':True,'not_source_or_task_evidence':True,'gravity_enabled':False,'external_forces':False,
        'fixed_or_kinematic_nodes':0,'positions_written_after_reset':False,
        'backend':{'name':'isaacsim','version':importlib.metadata.version('isaacsim'),'solver':args.solver},
        'settings':settings,'parameters':vars(args),'expected_free_strip_frequency_hz':expected,
        'observed_oscillation':response,'relative_frequency_error':relative,'rows':rows,
        'native_physics_errors':errors.report(),
        'input_sha256':{str(q.resolve()):hashlib.sha256(q.read_bytes()).hexdigest() for q in
            [Path(__file__),Path(flexure_reference.__file__),Path('src/calibrated_scene_reconstruction/sim/surface_cloth.py'),Path('src/calibrated_scene_reconstruction/assets/deformable_release.py'),out/'physics_priors_input.json']},
        'scope':'Small-amplitude free-strip control. Disabled mode tests frequency; material mode records decay and needs a matched zero-damping comparison. Neither identifies actual material or approves source appearance, contacts or task feasibility.'}
    np.savez_compressed(out/'states.npz',points=states,times=times,amplitudes=amplitudes,rest=rest,faces=faces)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('FLEXURE_RESULT',json.dumps({k:v for k,v in result.items() if k not in ['rows','input_sha256','native_physics_errors']}),flush=True)
except Exception as error:
    (out/'error.json').write_text(json.dumps({'pass':False,'error':str(error),'traceback':traceback.format_exc()},indent=2)+'\n');raise
finally:
    errors.close();app.close()
