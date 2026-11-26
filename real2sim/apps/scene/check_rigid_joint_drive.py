"""Isolate a finite native joint drive on the source rigid assembly.

Flexible parts are explicitly omitted. This diagnoses the test fixture only;
no source, material, task or delivery approval is produced.
"""
import argparse
from pathlib import Path
import json
import hashlib
import traceback

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('session');p.add_argument('candidate');p.add_argument('--plan',required=True)
p.add_argument('--assembly',required=True);p.add_argument('--output',required=True)
p.add_argument('--physics-hz',type=int,default=7680)
a=p.parse_args();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False)
(out/'source.py').write_bytes(Path(__file__).read_bytes())
from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True})
import numpy as np  # noqa: E402
from copy import deepcopy  # noqa: E402
from types import SimpleNamespace  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from calibrated_scene_reconstruction.scene.session import Session  # noqa: E402
from calibrated_scene_reconstruction.assets.metric_mesh import asset  # noqa: E402
from calibrated_scene_reconstruction.assets.mechanisms import joint_frames,revolute_position  # noqa: E402
from calibrated_scene_reconstruction.sim.mechanism_stage import add_links  # noqa: E402
from calibrated_scene_reconstruction.sim.physical_priors import apply as apply_priors  # noqa: E402
from calibrated_scene_reconstruction.sim.physics import physics_material  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import add_box_collider,set_pose  # noqa: E402
from calibrated_scene_reconstruction.sim.simulation import apply as apply_profile  # noqa: E402
from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors  # noqa: E402
from calibrated_scene_reconstruction.tasks.rigid.task_checks import Ctx,as_numpy  # noqa: E402
from calibrated_scene_reconstruction.tasks.mechanisms.joint_drive_fixture import JointDriveFixture  # noqa: E402
from calibrated_scene_reconstruction.tasks.mechanisms.preparation import interpolate  # noqa: E402
from calibrated_scene_reconstruction.tasks.mechanisms.interface_contact import check as interface_check  # noqa: E402
from calibrated_scene_reconstruction.sim.surface_distance import SignedSurface,triangle_samples  # noqa: E402
errors=PhysicsErrors(out/'native_physics_errors.json')
try:
    s=Session(a.session);info=json.loads(Path(a.candidate).read_text());plan=json.loads(Path(a.plan).read_text());h=plan['hinge']
    if not out.is_relative_to(s.dir.resolve()/'qa') or not 240<=a.physics_hz<=7680:raise ValueError('Current QA output and supported clock required')
    source=Path(info['path'])/'mesh.npz';kit=asset({'mesh':str(source)})
    parts=[x for x in kit['parts'] if x['kind']=='rigid'];omitted=[x['name'] for x in kit['parts'] if x['kind']!='rigid']
    vertices=[];faces=[];links=[];nv=nf=0
    for part in parts:
        vertices.append(part['V']);faces.append(part['F']+nv)
        links.append({'name':part['name'],'kind':'rigid','vertex_start':nv,'vertex_count':len(part['V']),'face_start':nf,'triangle_count':len(part['F'])})
        nv+=len(part['V']);nf+=len(part['F'])
    full=np.concatenate(vertices);shift=(full.min(0)+full.max(0))/2;shift[2]=full[:,2].min()
    joints=deepcopy(kit['joints'])
    for joint in joints:
        for key in ['anchor0','anchor1']:joint[key]=(np.asarray(joint[key])-shift).tolist()
    mesh=out/'rigid_source.npz';np.savez(mesh,V=full-shift,F=np.concatenate(faces),links_json=np.array(json.dumps(links)),joints_json=np.array(json.dumps(joints)))
    kit=asset({'mesh':str(mesh)});g=deepcopy(s.scene_graph);spec=g['dynamic_objects'][a.assembly]
    fix={'model':True,'mesh':str(mesh),'opening':info['report']['opening'],'sdf':info['sdf'],
         'link_physics':deepcopy(info.get('link_physics',{}))};spec['fix']=fix
    dec=deepcopy(s.config('agent/mechanisms.json'));priors=deepcopy(s.config('agent/physics_priors.json'))
    refs={a.assembly+'/'+p['name'] for p in parts}
    dec['objects']={a.assembly:{'links':{k:v for k,v in dec['objects'][a.assembly]['links'].items() if a.assembly+'/'+k in refs}}}
    dec['mechanisms']=[m for m in dec['mechanisms'] if set(m['links'])<=refs]
    priors['bodies']={k:v for k,v in priors['bodies'].items() if k in refs}
    class LocalSession(Session):
        @property
        def scene_graph(self):return deepcopy(g)
    local=LocalSession(s.dir)
    profile={**s.config('simulation.json'),'physics_hz':a.physics_hz}
    world=World(stage_units_in_meters=1.,physics_dt=1/a.physics_hz,device='cuda:0',backend='torch');errors.guard_steps(world)
    pc=world.get_physics_context();pc.enable_gpu_dynamics(True);pc.set_broadphase_type('GPU')
    stage=omni.usd.get_context().get_stage();apply_profile(stage,profile,flexible=False)
    add_box_collider(stage,'/World/Support/task_desk/surface',[-.5,-.5,-.02],[.5,.5,0.])
    material=physics_material(stage,'/World/Looks/TestPhysics',.6,.45,0.)
    root=add_links(stage,'/World/Objects/'+a.assembly,kit,fix,spec,material)
    set_pose(root,shift)
    physical=apply_priors(stage,priors,dec,'nominal');stage.GetRootLayer().Export(str(out/'initial.usda'))
    print('RIGID_PHASE reset',flush=True);world.reset()
    ctx=Ctx(local,SimpleNamespace(objects={a.assembly:root}),world,stage,None)
    initial_joints=ctx.initial_articulation_state()
    (out/'initial_articulation_state.json').write_text(json.dumps(initial_joints,indent=2)+'\n')
    if not initial_joints['pass']:raise ValueError('Native reset changed the authored initial articulation state')
    print('RIGID_PHASE drive',flush=True)
    joint=next(j for j in kit['joints'] if j['parent']==h['parent'] and j['child']==h['child']);frames=joint_frames(joint)
    drive=None;rows=[];poses=[];active=h['active_s'];duration=active[1]+.5
    surfaces={p['name']:SignedSurface(p['V'],p['F']) for p in parts}
    samples={p['name']:triangle_samples(p['V'],p['F']) for p in parts}
    suppressed={frozenset([j['parent'],j['child']]) for j in kit['joints'] if not j.get('collision_enabled',False)}
    contact_pairs=[(p['name'],other['name']) for i,p in enumerate(parts) for other in parts[:i]
                   if frozenset([p['name'],other['name']]) not in suppressed]
    for step in range(round(duration*a.physics_hz)+1):
        t=step/a.physics_hz
        if active[0]<=t<active[1]:
            if drive is None:drive=JointDriveFixture(ctx,a.assembly,h['parent'],h['child'],stiffness=h['stiffness_Nm_rad'],damping=h['damping_Nms_rad'],effort_limit=h['torque_limit_Nm'])
            drive.target(float(interpolate(np.asarray(h['keyframes']),t)[0]))
        elif drive is not None and not drive.released and t>=active[1]:drive.release()
        if step%max(1,a.physics_hz//100)==0 or step==round(duration*a.physics_hz):
            current={p['name']:ctx.pose(a.assembly+'/'+p['name']) for p in parts}
            angle=revolute_position(current[h['parent']][1],current[h['child']][1],*frames)
            contacts=[]
            for x,y in contact_pairs:
                px,rx=current[x];py,ry=current[y]
                first=surfaces[y].max_penetration((samples[x]@rx.T+px-py)@ry)
                second=surfaces[x].max_penetration((samples[y]@ry.T+py-px)@rx)
                contacts.append({'parts':[x,y],'maximum_penetration_m':max(first,second)})
            rows.append({'time_s':t,'angle_rad':angle,'target_rad':None if drive is None else drive.last_target_rad,'released':bool(drive and drive.released),
                         'actual_source_contacts':contacts})
            poses.append(np.array([np.r_[current[p['name']][0],current[p['name']][1].ravel()] for p in parts]))
            (out/'progress.json').write_text(json.dumps(rows[-1],indent=2)+'\n')
        if step<round(duration*a.physics_hz):world.step(render=False)
    if drive is None or not drive.released:raise ValueError('Finite drive did not fully release')
    release=drive.verify_released();interfaces=s.config('agent/interface_checks.json',{}).get('objects',{}).get(a.assembly,[])
    seated=interface_check(ctx,a.assembly,kit['parts'],interfaces) if interfaces else None
    np.savez_compressed(out/'states.npz',poses=poses,times=[r['time_s'] for r in rows],names=[p['name'] for p in parts])
    result={'complete':True,'task_approved':False,'scope':'Rigid joint-drive control only; flexible parts omitted','omitted_flexible_links':omitted,'diagnostic_coordinate_shift_m':shift.tolist(),'physical_application':physical,'rows':rows,'native_drive':drive.initial_readback,'release':release,'interfaces':seated,'native_physics_errors':errors.report(),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'plan':plan,'native_collision_carriers':ctx._contact_carrier_checks}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print('RIGID_DRIVE_RESULT',json.dumps({'complete':True,'last_angle_rad':rows[-1]['angle_rad'],'interfaces':seated}),flush=True)
except Exception as e:
    (out/'error.json').write_text(json.dumps({'type':type(e).__name__,'error':str(e),'traceback':traceback.format_exc()},indent=2)+'\n');traceback.print_exc()
finally:app.close()
