"""Reusable bounded-force rigid assays in the already assembled task scene.

The agent chooses setup and task conditions. This harness records native
motion/contact without prescribing poses during a trial or approving a task.
"""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

from .task_checks import as_numpy, rot
from .contact_pairs import depths
from calibrated_scene_reconstruction.tasks.mechanisms.rigid_probe import TranslationProbe, validate
from calibrated_scene_reconstruction.assets.rigid_frame_error import discrepancy, attachment_radii


def run(ctx,names,seconds,output,*,probe_plan=None,maximum_penetration_m=None):
    if getattr(ctx,'_assay_active',False):raise ValueError('Nested native assays are not supported')
    ctx._assay_active=True
    try:return _run(ctx,names,seconds,output,probe_plan=probe_plan,maximum_penetration_m=maximum_penetration_m)
    finally:ctx._assay_active=False


def _run(ctx, names, seconds, output, *, probe_plan=None,maximum_penetration_m=None):
    from isaacsim.core.prims import RigidPrim
    from pxr import Usd, UsdPhysics
    from .flexible_trace import Trace
    output=Path(output).resolve()
    if not output.is_relative_to((ctx.s.dir/'qa').resolve()):
        raise ValueError('Native assay output must stay in session QA')
    output.mkdir(parents=True,exist_ok=False)
    if not names or len(set(names))!=len(names) or not np.isfinite(seconds) or not 0<seconds<=30:
        raise ValueError('Distinct existing rigid bodies and bounded duration required')
    if maximum_penetration_m is not None and (not np.isfinite(maximum_penetration_m) or not 0<maximum_penetration_m<=.01):
        raise ValueError('Explicit positive bounded native rejection threshold required')
    bodies=[ctx.body(name) for name in names];paths=[body.prim_path for body in bodies]
    dt=float(ctx.world.get_physics_dt());steps=round(seconds/dt)
    if steps<1:raise ValueError('At least one native physics step required')
    support_name=ctx.sg['task']['support']
    roots=[f'{root}/{support_name}' for root in ['/World/Room','/World/Support','/World/Static']]
    # The default carved support route authors its static box at this path (sim.scene_stage);
    # tasks.mechanisms.tabletop.from_stage already lists it as a task-support root.
    roots.append(f'/World/Room/collider_{support_name}')
    support=[str(p.GetPath()) for p in ctx.stage.Traverse()
             if p.HasAPI(UsdPhysics.CollisionAPI) and not str(p.GetPath()).startswith('/World/Objects/')]
    if not any(any(path.startswith(root+'/') or path==root for root in roots) for path in support):
        raise ValueError('Modeled native task support is required')
    filters=paths+support
    view=RigidPrim(paths,contact_filter_prim_paths_expr=[filters for _ in paths],
                   max_contact_count=32768,reset_xform_properties=False,
                   disable_stablization=False,name='rigid_assay_'+output.name)
    view.initialize()
    if list(view.prim_paths)!=paths:raise ValueError('Native body order changed')
    controller=None;target=None;local_com=None;mass=None;inertia=None;payload_mass=0.
    if probe_plan is not None:
        probe_plan=validate(probe_plan,names,seconds)
        target=names.index(probe_plan['link']);body=bodies[target]
        native=body._rigid_prim_view
        mass,inertia=ctx.rigid_mass_properties(probe_plan['link'])
        local_com=as_numpy(native.get_coms()[0]).reshape(-1,3)[0]
        if probe_plan['link'] in ctx._partitions:
            local_com=np.array(ctx._partitions[probe_plan['link']]['metadata']['source_centre'])
        # A force applied to a rigid root also carries its genuinely bonded
        # flexible material. The fixture's feedforward estimate must include
        # that payload; actual simulation masses/inertias remain untouched.
        obj,link=probe_plan['link'].split('/')
        if any(name.startswith(obj+'/') for name in getattr(ctx,'_cloth',{})):
            from calibrated_scene_reconstruction.assets.metric_mesh import asset
            kit=asset(ctx.sg['dynamic_objects'][obj]['fix'])
            for bond in kit.get('surface_bindings',[]):
                if bond['rigid']==link:
                    payload_mass+=float(as_numpy(ctx._cloth[obj+'/'+bond['surface']].get_particle_masses()).sum())
        controller=TranslationProbe(probe_plan,mass+payload_mass,dt,inertia)
    initial=output/'initial.usda';ctx.stage.GetRootLayer().Export(str(initial))
    if Usd.Stage.Open(str(initial)) is None:raise ValueError('Native assay stage cannot be reopened')
    history=[];forces=[];torques=[];contacts=[];pair_peaks={};contact_counts={}
    contact_force_sums=[];contact_point_counts=[]
    # Equivalent fixed frames must remain equivalent during dynamics, not
    # merely preserve authored mass and inertia at setup. Bound displacement
    # over every carried root point, including rotational weld error. These
    # collider-free frames do not carry a duplicate of the owner's geometry.
    frame_checks={}
    for name,data in getattr(ctx,'_partitions',{}).items():
        if name not in names:continue
        centres=np.asarray(data['metadata']['component_centres'][1:],float)
        frame_checks[name]={'view':data['view'],'index':names.index(name),'centres':centres,
            'radii':attachment_radii(ctx.stage,data['metadata']['carriers']),
            'maximum':np.zeros(len(centres)),'history':[]}
    flexible=Trace(ctx,names,steps,output)
    # Trace may import backend-specific bulk readers. Bind those actual
    # implementations too, before the first recorded native step.
    inputs={}
    for name,module in list(sys.modules.items()):
        file=getattr(module,'__file__',None)
        if name.startswith('calibrated_scene_reconstruction.') and file and Path(file).suffix=='.py':
            file=Path(file).resolve();inputs[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
    try:
        for step in range(steps+1):
            p,q=(as_numpy(x) for x in view.get_world_poses());history.append(np.c_[p,q])
            for data in frame_checks.values():
                fp,fq=(as_numpy(x) for x in data['view'].get_world_poses())
                index=data['index']
                error=discrepancy(p[index],q[index],fp,fq,data['centres'],data['radii'])
                data['history'].append(error)
                data['maximum']=np.maximum(data['maximum'],error)
            flexible.record(step)
            if step==steps:break
            force=np.zeros(3);torque=np.zeros(3)
            if controller is not None:
                body=bodies[target];R=rot(q[target]);com=p[target]+R@local_com
                force=controller.force(step*dt,com,as_numpy(body.get_linear_velocity()))
                torque=controller.torque(step*dt,R,as_numpy(body.get_angular_velocity()))
                values=[np.asarray([v],np.float32) for v in [force,torque,com]]
                if ctx.world.backend=='torch':
                    import torch
                    values=[torch.as_tensor(v,device=ctx.world.device) for v in values]
                body._rigid_prim_view.apply_forces_and_torques_at_pos(
                    forces=values[0],torques=values[1],positions=values[2],is_global=True)
            forces.append(force);torques.append(torque);ctx.world.step(render=False)
            data=view.get_contact_force_data(dt=dt)
            shape=(len(paths),len(filters))
            counts=as_numpy(data[4]).reshape(shape)
            starts=as_numpy(data[5]).reshape(shape)
            normal_forces=as_numpy(data[0]).reshape(-1)
            force_sums=np.zeros(shape)
            for i,j in np.argwhere(counts>0):
                start,count=int(starts[i,j]),int(counts[i,j])
                selected_forces=normal_forces[start:start+count]
                if len(selected_forces)!=count or not np.isfinite(selected_forces).all():
                    raise ValueError('Native contact force buffer is invalid')
                force_sums[i,j]=np.abs(selected_forces).sum()
            contact_force_sums.append(force_sums);contact_point_counts.append(counts.copy())
            pairs=depths(as_numpy(data[3]),counts,
                         starts,paths,filters)
            for i,j in np.argwhere(counts>0):
                pair=tuple(sorted([paths[i],filters[j]]))
                contact_counts[pair]=contact_counts.get(pair,0)+int(counts[i,j])
            contacts.append(max(pairs.values(),default=0.))
            for pair,value in pairs.items():pair_peaks[pair]=max(value,pair_peaks.get(pair,0.))
            if maximum_penetration_m is not None and contacts[-1]>maximum_penetration_m:
                raise ValueError(f'Native rigid penetration {contacts[-1]:.9g}m exceeds declared {maximum_penetration_m:.9g}m; failed partial trial cannot approve its requested horizon')
            if step%max(1,round(1/dt))==0:
                print(f'[rigid.assay] {output.name}: {step*dt:.1f}/{seconds:.1f}s, max rigid penetration {max(contacts)*1000:.4f}mm',flush=True)
    except BaseException as error:
        # A failed native step must retain the actual last states, including
        # invalid readbacks. Never silently repair or substitute a pose.
        np.savez_compressed(output/'failed_physics_poses.npz',poses=np.asarray(history),
            times_s=np.arange(len(history))*dt,names=names)
        np.savez_compressed(output/'failed_forces.npz',forces_N=forces,torques_Nm=torques)
        np.savez_compressed(output/'failed_native_contacts.npz',maximum_penetration_m=contacts,
                            times_s=np.arange(1,len(contacts)+1)*dt)
        if flexible.array is not None:flexible.array.flush()
        frame_failures={}
        for name,data in frame_checks.items():
            file='failed_weld_error_'+name.replace('/','_')+'.npz'
            np.savez_compressed(output/file,source_displacement_bound_m=np.asarray(data['history']),
                                times_s=np.arange(len(data['history']))*dt)
            try:
                fp,fq=(as_numpy(x) for x in data['view'].get_world_poses())
                frame_failures[name]={'positions':fp.tolist(),'orientations_wxyz':fq.tolist(),
                    'maximum_pre_failure_displacement_bound_m':data['maximum'].tolist(),'trace':file}
            except Exception as read_error:frame_failures[name]={'read_error':str(read_error)}
        independent={}
        for name,body in zip(names,bodies):
            try:
                independent[name]={'pose':[as_numpy(x).tolist() for x in body._rigid_prim_view.get_world_poses()],
                    'velocity':as_numpy(body._rigid_prim_view.get_velocities()).tolist()}
            except Exception as read_error:independent[name]={'read_error':str(read_error)}
        (output/'native_failure.json').write_text(json.dumps({'pass':False,'task_approved':False,
            'error':str(error),'time_s':(len(history)-1)*dt,'recorded_rigid_states':len(history),
            'recorded_flexible_states':flexible.index,'independent_body_readbacks':independent,
            'equivalent_frame_readbacks':frame_failures,
            'flexible_vertex_ranges':flexible.ranges,'flexible_names':flexible.names,
            'flexible_sample_steps':flexible.sample_steps[:flexible.index],
            'contact_pairs':[{'paths':list(pair),'maximum_penetration_m':depth}
                             for pair,depth in pair_peaks.items()],
            'scope':'Failed native assay; partial states are diagnostic evidence only.'},indent=2,allow_nan=True)+'\n')
        raise
    poses=np.asarray(history);times=np.arange(len(poses))*dt
    flexible_result=flexible.finish()
    frame_report={'pass':True,'applicable':bool(frame_checks),'tolerance_m':.0001,'bodies':{},
        'scope':'Every native step: conservative carried-attachment-point displacement bound from actual owner/frame poses; actual owner geometry checked independently; no pose correction.'}
    for name,data in frame_checks.items():
        values=np.asarray(data['history']);file='weld_error_'+name.replace('/','_')+'.npz'
        np.savez_compressed(output/file,source_displacement_bound_m=values,times_s=times,
                            centres=data['centres'],radii=data['radii'])
        maximum=float(data['maximum'].max())
        frame_report['bodies'][name]={'pass':maximum<=frame_report['tolerance_m'],
            'maximum_source_displacement_bound_m':maximum,'trace':file,
            'trace_sha256':hashlib.sha256((output/file).read_bytes()).hexdigest()}
    frame_report['pass']=all(row['pass'] for row in frame_report['bodies'].values())
    (output/'rigid_frame_precision.json').write_text(json.dumps(frame_report,indent=2)+'\n')
    np.savez_compressed(output/'physics_poses.npz',poses=poses,times_s=times,names=names)
    np.savez_compressed(output/'forces.npz',forces_N=forces,torques_Nm=torques,times_s=times[:-1])
    np.savez_compressed(output/'native_contacts.npz',maximum_penetration_m=contacts,times_s=times[1:],
                        absolute_normal_force_sum_N=contact_force_sums,contact_point_counts=contact_point_counts,
                        body_paths=paths,filter_paths=filters)
    unchanged=all(Path(p).is_file() and hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in inputs.items())
    native=ctx.native_errors.report();precision=dict(ctx.native_errors.precision_audit())
    precision['rigid_frame_precision']=frame_report
    precision['pass']=bool(precision['pass'] and frame_report['pass'])
    report={'task_approved':False,'scope':'Native assembled-scene measurements; task conditions evaluated by agent check.',
            'names':names,'body_paths':paths,'dt_s':dt,'seconds':seconds,'probe':probe_plan,
            'native_mass_kg':mass,'native_inertia_local_kg_m2':None if inertia is None else inertia.tolist(),
            'bonded_flexible_payload_mass_kg':payload_mass,
            'controller_mass_model':'Aggregate native rigid mass including equivalent fixed frames, plus declared bonded surface payload; native dynamics unmodified',
            'maximum_native_penetration_m':max(contacts),'finite_poses':bool(np.isfinite(poses).all()),
            'native_physics':native,'precision':precision,'implementation_unchanged':unchanged,
            'flexible_trace':flexible_result,
            'implementation_sha256':inputs,
            'pairs':[{'paths':list(p),'maximum_penetration_m':v,
                      'recorded_contact_points':contact_counts.get(p,0)} for p,v in pair_peaks.items()]}
    if controller:
        release=times[:-1]>=probe_plan['start_s']+probe_plan['duration_s']
        report.update(peak_force_N=controller.peak_force,peak_torque_Nm=controller.peak_torque,
                      release_forces_zero=bool(release.any() and np.all(np.asarray(forces)[release]==0)),
                      release_torques_zero=bool(release.any() and np.all(np.asarray(torques)[release]==0)))
    (output/'native_result.json').write_text(json.dumps(report,indent=2)+'\n')
    return report,poses,times
