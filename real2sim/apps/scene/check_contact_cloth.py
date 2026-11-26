"""Diagnostic cloth extraction with finite source-fingertip contact tools.

Asset contact feasibility only. No robot controller claim, cloth attachments,
nodal grasp forces, position animation or automatic delivery approval.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

p=argparse.ArgumentParser(description=__doc__)
for name in ['session','candidate','hand_plan','physics_plan','mechanism_plan','simulation_plan','assembly','output']:
    if name=='session':p.add_argument(name)
    else:p.add_argument('--'+name.replace('_','-'),required=True)
p.add_argument('--grasp-proposal',help='Optional candidate-bound initial hover proposal; otherwise derive it from the current candidate geometry')
p.add_argument('--cloth-link',default='wipe');p.add_argument('--body-link',default='body');p.add_argument('--lid-link',default='lid')
p.add_argument('--zero-friction-lift',action='store_true')
p.add_argument('--replay-positive',help='For zero-friction control: replay every source-tip command from a complete current positive trial without replanning')
p.add_argument('--record-failed-lift-attempt',action='store_true',
    help='Record the real lift even if geometric grasp witnesses fail; cannot approve the task')
p.add_argument('--opening-hold-seconds',type=float,default=0.)
p.add_argument('--opening-seconds',type=float,default=.6,help='Finite lid motion duration from observed motion or a bounded mechanism prior')
p.add_argument('--approach-seconds',type=float,default=.08)
p.add_argument('--pinch-seconds',type=float,default=.12)
p.add_argument('--pinch-hold-seconds',type=float,default=.05)
p.add_argument('--closure-mode',choices=['open_descent','coupled'],default='open_descent')
p.add_argument('--open-descent-seconds',type=float,default=.08)
p.add_argument('--closure-contact-link',action='append',default=[])
p.add_argument('--closure-contact-basis')
p.add_argument('--pinch-material-axis',type=float,nargs=3,required=True)
p.add_argument('--pinch-material-frame',choices=['spatial','opening_plane'],default='spatial')
p.add_argument('--inside-finger',choices=['either','thumb','index'],default='either')
p.add_argument('--tip-collision',choices=['convexHull','sdf'],default='sdf')
p.add_argument('--tip-sdf-resolution',type=int,default=512)
p.add_argument('--capture-motion-lead',action='store_true')
p.add_argument('--grip-force-safety-factor',type=float,default=2.)
p.add_argument('--normal-preload',action='store_true',help='Maintain finite closing load after position closure; requires independent tool qualification before acceptance')
p.add_argument('--material-contact-mode',choices=['same_component','single_patch'],default='same_component')
p.add_argument('--contact-qualification',help='Current passing source-tool coupon pair; bind material, numerics, geometry and control mode to this task')
p.add_argument('--diagnose-thickness-residual',action='store_true',help='Explicitly continue through raw thickness failures for diagnosis only; never acceptance')
args=p.parse_args();out=Path(args.output).resolve()
if args.replay_positive and not args.zero_friction_lift:p.error('Matched replay requires the explicit zero-finger-friction intervention')
if bool(args.closure_contact_link)!=bool(args.closure_contact_basis and args.closure_contact_basis.strip()):
    p.error('Expected closure contacts require link names and a source/task basis')
if (not .04<=args.open_descent_seconds<=2 or not .2<=args.opening_seconds<=3 or not 0<=args.opening_hold_seconds<=2 or not .04<=args.approach_seconds<=2
        or not .04<=args.pinch_seconds<=2 or not 0<=args.pinch_hold_seconds<=1):
    p.error('Explicit bounded fixture timings required')
if not 1<=args.grip_force_safety_factor<=16:p.error('Explicit force safety factor within1..16 required')
if args.closure_contact_link:
    candidate_data=json.loads(Path(args.candidate).read_text())
    known={part['name']for part in candidate_data['report']['parts']if part['kind']=='rigid'}
    if not set(args.closure_contact_link)<=known:p.error('Expected closure contacts must name actual rigid parts: '+', '.join(sorted(known)))
out.mkdir(parents=True,exist_ok=False)
for source,name in [(args.candidate,'candidate_input.json'),(args.hand_plan,'hand_input.json'),(__file__,'source.py')]:
    shutil.copy2(source,out/name)

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
app=SimulationApp({'headless':True})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402
from pxr import Gf,PhysxSchema,UsdPhysics  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402
from calibrated_scene_reconstruction.scene.session import Session  # noqa: E402
from calibrated_scene_reconstruction.scene.capture.articulated_model import from_usd  # noqa: E402
from calibrated_scene_reconstruction.assets.metric_mesh import asset  # noqa: E402
from calibrated_scene_reconstruction.assets.physical_priors import sample  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_native_profile import resolve  # noqa: E402
from calibrated_scene_reconstruction.assets.grasp_load import normal_force  # noqa: E402
from calibrated_scene_reconstruction.assets.response_policy import mode as response_mode,material_verdict  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_rest import area as sheet_area  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_metric import MaterialMetric  # noqa: E402
from calibrated_scene_reconstruction.assets.cloth_container_state import check as container_state  # noqa: E402
from calibrated_scene_reconstruction.assets.material_support import check as support_state  # noqa: E402
from calibrated_scene_reconstruction.assets.mechanisms import revolute_position,joint_frames  # noqa: E402
from calibrated_scene_reconstruction.sim.mechanism_stage import add_links  # noqa: E402
from calibrated_scene_reconstruction.sim.physical_priors import apply as apply_priors  # noqa: E402
from calibrated_scene_reconstruction.sim.physics import physics_material  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import add_box_collider  # noqa: E402
from calibrated_scene_reconstruction.sim.simulation import apply as apply_simulation,audit as audit_simulation  # noqa: E402
from calibrated_scene_reconstruction.sim.fingertip_fixture import FingertipFixture  # noqa: E402
from calibrated_scene_reconstruction.sim.hand_collision import Penetration,PlannedClearance,PlannedPinchContact,OpenPinchReadiness  # noqa: E402
from calibrated_scene_reconstruction.sim.hand_materials import author_friction  # noqa: E402
from calibrated_scene_reconstruction.sim.physics_errors import PhysicsErrors  # noqa: E402
from calibrated_scene_reconstruction.sim.reset_timestep import reset as native_reset  # noqa: E402
from calibrated_scene_reconstruction.sim.surface_distance import SignedSurface,triangle_samples  # noqa: E402
from calibrated_scene_reconstruction.sim.cloth_contact import self_intersections  # noqa: E402
from calibrated_scene_reconstruction.sim.cloth_clearance import measure as finite_clearance  # noqa: E402
from calibrated_scene_reconstruction.tasks.rigid.task_checks import Ctx,as_numpy  # noqa: E402
from calibrated_scene_reconstruction.tasks.mechanisms.hand_cloth import opening_assignment,pinch_axis_alignment,lift_measurement,pinch_contact,material_tangent,rim_target,clearance_offsets,capture_target  # noqa: E402
from calibrated_scene_reconstruction.tasks.mechanisms.hand_contact_isolation import material_samples  # noqa: E402
from calibrated_scene_reconstruction.tasks.mechanisms.closure_tracking import capture_lead  # noqa: E402
from calibrated_scene_reconstruction.tasks.mechanisms.hand_motion import translation_seconds,force_limited_seconds,orientation_path  # noqa: E402
from calibrated_scene_reconstruction.tasks.mechanisms.material_grasp import contact_regions,retained_regions  # noqa: E402

sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
errors=PhysicsErrors(out/'native_physics_errors.json')
try:
    s=Session(args.session);info=json.loads(Path(args.candidate).read_text())
    accuracy_mode=response_mode(s.config('task.json',{}).get('flexible_accuracy'))
    hand=json.loads(Path(args.hand_plan).read_text())
    proposal=json.loads(Path(args.grasp_proposal).read_text()) if args.grasp_proposal else None
    if proposal is not None and proposal['candidate_sha256']!=sha(args.candidate):raise ValueError('Grasp proposal belongs to another source candidate')
    priors=json.loads(Path(args.physics_plan).read_text());mechanisms=json.loads(Path(args.mechanism_plan).read_text())
    profile=json.loads(Path(args.simulation_plan).read_text());hz=int(profile['physics_hz'])
    graph=deepcopy(s.scene_graph);spec=graph['dynamic_objects'][args.assembly];requested=spec['fit']
    values=sample(priors,'nominal')['bodies'][args.assembly+'/'+args.cloth_link]['parameters']
    settings,_=resolve('surface_fem',values,requested.get('link_physics',{}).get(args.cloth_link,{}),info['cloth_physics'])
    fix={'model':True,'mesh':str(Path(info['path'])/'mesh.npz'),'opening':info['report']['opening'],
         'sdf':info.get('sdf',requested['sdf']),
         'link_physics':{**deepcopy(requested.get('link_physics',{})),**deepcopy(info.get('link_physics',{})),args.cloth_link:settings}}
    spec['fix']=fix;kit=asset(fix);parts={x['name']:x for x in kit['parts']};sheet=parts[args.cloth_link]
    if sheet.get('backend')!='surface_fem':raise ValueError('Qualify particle-specific friction control before using this interface fixture with PBD')
    qualification=None
    model=from_usd(hand['asset'],100)
    closed={j.name:float(hand['closed_state'].get(j.name,0.))for j in model.movable}
    frames=model.transforms(closed);source_pads=[]
    for finger in ['thumb','index']:
        item=hand[finger];tf=frames[item['link']]
        source_pads.append(np.asarray(hand['R'])@(tf[:3,:3]@item['pad_local']+tf[:3,3])+hand['p'])
    source_pads=np.asarray(source_pads)
    if proposal is None:
        radius=float(np.ptp(parts[args.body_link]['V'][:,:2],axis=0).mean()/2)
        def initial_assignment(point,r):
            pads=source_pads+np.array([[-.009,0,0],[.009,0,0]])
            return opening_assignment(pads@r.T+point,fix['opening'],radius,args.inside_finger)['pass']
        selected=rim_target(sheet['V'],sheet['cloth_rest']['points'],sheet['F'],fix['opening'],
            outer_radius=float(np.ptp(parts[args.body_link]['V'][:,:2],axis=0).mean()/2),
            feasible=initial_assignment,align_surface=True,material_axis=args.pinch_material_axis,material_axis_frame=args.pinch_material_frame)
        proposal={'candidate_sha256':sha(args.candidate),
            'selection':{k:v.tolist()if hasattr(v,'tolist')else v for k,v in selected.items()},
            'scope':'Current candidate geometry for initial safe hover only; re-associate actual material after opening. No task approval.'}
    (out/'initial_grasp_proposal.json').write_text(json.dumps(proposal,indent=2)+'\n')
    selection=proposal['selection'];patch=np.asarray(selection['material_patch'],int);node=int(selection['material_node'])
    if not 0<=node<len(sheet['V']) or patch.min()<0 or patch.max()>=len(sheet['V']):raise ValueError('Source material correspondence changed')
    rotation=np.asarray(selection['rotation']);centre=np.asarray(selection['point']);initial_centre=centre+[0,0,.08]
    initial_role=opening_assignment((source_pads+np.array([[-.009,0,0],[.009,0,0]]))@rotation.T+centre,
        fix['opening'],float(np.ptp(parts[args.body_link]['V'][:,:2],axis=0).mean()/2),args.inside_finger)
    if not initial_role['pass']:raise ValueError('Initial source hover does not preserve a resolved inside/outside finger assignment')
    selected_inside_finger=initial_role['inside_finger']
    class Local(Session):
        @property
        def scene_graph(self):return deepcopy(graph)
        def config(self,name,*default):
            if name=='agent/mechanisms.json':return deepcopy(mechanisms)
            if name=='agent/physics_priors.json':return deepcopy(priors)
            if name=='simulation.json':return deepcopy(profile)
            return super().config(name,*default)
    local=Local(s.dir)
    world=World(stage_units_in_meters=1.,physics_dt=1/hz,rendering_dt=1/30,device='cuda:0',backend='torch')
    errors.guard_steps(world);pc=world.get_physics_context();pc.enable_gpu_dynamics(True);pc.set_broadphase_type('GPU')
    stage=omni.usd.get_context().get_stage();apply_simulation(stage,profile,flexible=True)
    desk=add_box_collider(stage,'/World/Support/task_desk/surface',[-.5,-.5,-.02],[.5,.5,0.])
    col=PhysxSchema.PhysxCollisionAPI.Apply(desk.GetPrim());col.CreateContactOffsetAttr(.001);col.CreateRestOffsetAttr(0.)
    mat=physics_material(stage,'/World/Looks/TestPhysics',.6,.45,0.)
    root=add_links(stage,'/World/Objects/'+args.assembly,kit,fix,spec,mat)
    application=apply_priors(stage,priors,mechanisms,'nominal')
    holder=UsdPhysics.FixedJoint.Define(stage,'/World/TestFixtures/container_holder')
    holder.CreateBody1Rel().SetTargets([str(root.GetPath())+'/'+args.body_link])
    holder.CreateLocalPos0Attr(Gf.Vec3f(0));holder.CreateLocalPos1Attr(Gf.Vec3f(0))
    hinges=[p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    if len(hinges)!=1:raise ValueError('One attached source lid hinge required')
    hinge=UsdPhysics.RevoluteJoint(hinges[0]);joint=next(j for j in kit['joints'] if j['type']=='revolute')
    drive=UsdPhysics.DriveAPI.Apply(hinge.GetPrim(),'angular');drive.CreateTypeAttr('force')
    drive.CreateStiffnessAttr(.5*np.pi/180);drive.CreateDampingAttr(.0069*np.pi/180);drive.CreateMaxForceAttr(.2)
    initial_angle=joint['initial_position'];drive.CreateTargetPositionAttr(float(np.degrees(initial_angle)))
    fixture=FingertipFixture(stage,hand,model,initial_centre,rotation,out,initial_gap_m=.018,
        collision_approximation=args.tip_collision,sdf_resolution=args.tip_sdf_resolution)
    if not np.allclose(source_pads,fixture.contacts,atol=1e-10):raise ValueError('Initial planning pad frame differs from actual source fixture')
    grip_force=normal_force(values['areal_density']*sheet_area(sheet['cloth_rest']['points'],sheet['F']),
        values['dynamic_friction']*fixture.friction,safety_factor=args.grip_force_safety_factor)
    if args.contact_qualification:
        from calibrated_scene_reconstruction.tasks.mechanisms.fingertip_qualification import bind
        qualification=bind(args.contact_qualification,args.hand_plan,settings,hz,sheet['cloth_rest']['points'],sheet['F'],
            solver=profile.get('solver','PGS'),collision_approximation=args.tip_collision,normal_preload=args.normal_preload,
            normal_force_N=grip_force['normal_force_per_finger_N'],
            sdf_resolution=args.tip_sdf_resolution if args.tip_collision=='sdf' else None)
    stage.GetRootLayer().Export(str(out/'initial.usda'))
    bootstrap=native_reset(world)
    fixture.initialize(hz);ctx=Ctx(local,SimpleNamespace(objects={args.assembly:root}),world,stage,None)
    ctx.native_errors=errors;initial_joints=ctx.initial_articulation_state()
    if not initial_joints['pass']:raise ValueError('Authored source lid initial state changed')
    lid=SingleArticulation(str(root.GetPath())+'/'+args.body_link,name='contact_lid_motor');lid.initialize()
    if len(lid.dof_names)!=1:raise ValueError('One source hinge DOF required')
    support=mechanisms['objects'][args.assembly]['links'][args.cloth_link].get('initial_support')
    support_result=None
    if support:
        part=parts[support['link']]
        support_result=support_state(sheet['V'],sheet['cloth_rest']['points'],sheet['F'],part['V'],part['F'],
            support,thickness_m=values['thickness'],tolerance_m=1e-6)
        if not support_result['pass']:raise ValueError('Source cloth no longer follows the confirmed support route')
    distribution=container_state(sheet['V'],sheet['cloth_rest']['points'],sheet['F'],fix['opening'],hand['initial_material_distribution'])
    if not distribution['pass']:raise ValueError('Source initial interior/exterior distribution changed')
    files=[Path(__file__),Path('src/calibrated_scene_reconstruction/sim/reset_timestep.py'),Path('src/calibrated_scene_reconstruction/sim/fingertip_fixture.py'),Path('src/calibrated_scene_reconstruction/assets/grasp_load.py'),
           Path('src/calibrated_scene_reconstruction/sim/hand_collision.py'),Path('src/calibrated_scene_reconstruction/tasks/mechanisms/hand_cloth.py'),
           Path('src/calibrated_scene_reconstruction/assets/cloth_container_state.py'),Path(args.candidate),Path(args.hand_plan),
           Path('src/calibrated_scene_reconstruction/sim/convex_cooking.py'),Path('src/calibrated_scene_reconstruction/assets/convex_source_solids.py'),
           Path('src/calibrated_scene_reconstruction/tasks/mechanisms/closure_tracking.py'),
           Path('src/calibrated_scene_reconstruction/tasks/mechanisms/material_grasp.py'),Path('src/calibrated_scene_reconstruction/sim/pinch_load.py'),
           Path('src/calibrated_scene_reconstruction/tasks/mechanisms/hand_motion.py'),
           Path('src/calibrated_scene_reconstruction/tasks/mechanisms/contact_replay.py'),
           Path('src/calibrated_scene_reconstruction/assets/material_support.py'),Path('src/calibrated_scene_reconstruction/assets/support_route.py'),
           Path('src/calibrated_scene_reconstruction/sim/soft_surface.py'),Path('src/calibrated_scene_reconstruction/assets/tetra_boundary.py'),
           Path('src/calibrated_scene_reconstruction/sim/surface_distance.py'),Path('src/calibrated_scene_reconstruction/assets/volume_contact.py'),
           Path('src/calibrated_scene_reconstruction/sim/volume_body.py'),Path('src/calibrated_scene_reconstruction/sim/volume_initial_state.py'),
           Path('src/calibrated_scene_reconstruction/assets/volume_initial_state.py'),
           Path('src/calibrated_scene_reconstruction/sim/deformable_filter_pose.py'),Path('src/calibrated_scene_reconstruction/assets/deformable_filter_pose.py'),
           Path('src/calibrated_scene_reconstruction/tasks/mechanisms/fingertip_qualification.py'),
           Path(args.physics_plan),Path(args.mechanism_plan),Path(args.simulation_plan),
           Path(fix['mesh']),Path('src/calibrated_scene_reconstruction/sim/mechanism_stage.py'),Path('src/calibrated_scene_reconstruction/sim/surface_cloth.py')]
    if args.contact_qualification:files.append(Path(args.contact_qualification))
    if args.grasp_proposal:files.append(Path(args.grasp_proposal))
    protocol={'candidate_sha256':sha(args.candidate),'candidate_mesh_sha256':sha(fix['mesh']),
        'zero_friction_lift':args.zero_friction_lift,
        'assembly':args.assembly,'cloth_link':args.cloth_link,'body_link':args.body_link,
        'input_sha256':{str(p.resolve()):sha(p) for p in files},'diagnostic_hypothesis':True,
        'fixture':fixture.source,'cloth_forces':False,'cloth_attachments':False,'container_holder':True,
        'initial_articulation_state':initial_joints,'initial_material_distribution':distribution,
        'native_initialization':bootstrap,'simulation':audit_simulation(stage,profile),'physical_prior_application':application,
        'fixture_timing_seconds':{'opening':args.opening_seconds,'opening_hold':args.opening_hold_seconds,'approach':args.approach_seconds,
            'pinch':args.pinch_seconds,'pinch_hold':args.pinch_hold_seconds},
        'record_failed_lift_attempt':args.record_failed_lift_attempt,
        'grasp_force_proposal':grip_force,
        'closing_axis_preload':args.normal_preload,'material_contact_mode':args.material_contact_mode,
        'contact_tool_qualification':qualification,
        'initial_hover_proposal_source':'external_bound_proposal' if args.grasp_proposal else 'current_candidate_geometry',
        'diagnose_thickness_residual':args.diagnose_thickness_residual,'material_accuracy_mode':accuracy_mode,
        'closure_mode':args.closure_mode,'open_descent_seconds':args.open_descent_seconds,
        'expected_closure_contacts':{'links':args.closure_contact_link,'basis':args.closure_contact_basis,
            'native_penetration_checks_relaxed':False},
        'pinch_material_axis':args.pinch_material_axis,'pinch_material_frame':args.pinch_material_frame,
        'inside_finger':selected_inside_finger,'requested_inside_finger':args.inside_finger,
        'scope':'Asset-only source-fingertip contact fixture. Not full Sharpa joint control or task/delivery approval.'}
    if support_result is not None:protocol['initial_support_route']=support_result
    replay=None
    if args.replay_positive:
        from calibrated_scene_reconstruction.tasks.mechanisms.contact_replay import load
        source=Path(args.replay_positive).resolve()
        if source==out or not source.is_relative_to(s.dir.resolve()/'qa'):raise ValueError('Distinct current-session positive trial required')
        replay=load(source,protocol,physics_hz=hz)
        protocol['command_replay']=replay['evidence']
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (out/'initial_articulation_state.json').write_text(json.dumps(initial_joints,indent=2)+'\n')
    surfaces={n:SignedSurface(p['V'],p['F']) for n,p in parts.items() if p['kind']=='rigid'}
    from calibrated_scene_reconstruction.sim.soft_surface import from_view
    soft_surfaces={n:from_view(ctx._soft[args.assembly+'/'+n],
        stage.GetPrimAtPath(str(root.GetPath())+'/'+n+'/surface'))
        for n,p in parts.items() if p['kind'] in {'deformable','rope'}}
    soft_states={n:[] for n in soft_surfaces}
    if not set(args.closure_contact_link)<=set(surfaces):raise ValueError('Expected closure contacts must name actual rigid parts')
    closing_surfaces={n:v for n,v in surfaces.items() if n not in args.closure_contact_link}
    class ParallelTool:
        # This is the test fixture's opening, not a robot joint model.
        movable=[SimpleNamespace(name='opening')]
        def transforms(self,state):
            result={}
            for name,sign in zip(fixture.names,[-1,1]):
                matrix=np.eye(4);matrix[0,3]=sign*state['opening']/2;result[name]=matrix
            return result
    tool=ParallelTool();opened={'opening':.018};closed={'opening':0.}
    planned=PlannedClearance(fixture.rows,tool,opened,closed)
    capture=PlannedPinchContact(fixture.rows,tool,closed,fixture.names)
    open_capture=OpenPinchReadiness(fixture.rows,fixture.names)
    penetration=Penetration(fixture.rows);chart=sheet['cloth_rest']
    validity=priors['bodies'][args.assembly+'/'+args.cloth_link]['validity_hypothesis']
    metric=MaterialMetric(sheet['V'],sheet['F'],chart,max_edge_strain=validity['max_edge_strain'],max_area_change=validity['max_area_change'])
    rows=[];states=[];tools=[];times=[];commands=[];rigid={n:[] for n in surfaces};phases={};t=0.;current=initial_centre.copy();gap=.018;angle=initial_angle
    current_rotation=rotation.copy()
    grasp_regions=None;approach_patch=None
    contact_groups={finger:[name]for finger,name in zip(['thumb','index'],fixture.names)}
    body_name=args.assembly+'/'+args.body_link;cloth_name=args.assembly+'/'+args.cloth_link
    frames=joint_frames(joint);tolerance=2*values['thickness'];failure=None;lift=None;prelift=None;precontact=None;control_termination=None
    def record(phase):
        points=ctx.nodes(cloth_name).copy();poses=fixture.poses();sampled=triangle_samples(points,sheet['F'])
        crossings=self_intersections(points,sheet['F']);material=metric.check(points)
        clear=finite_clearance(points,sheet['F'],chart['points'],values['thickness'],device='cuda:0',
            local_bending_review=accuracy_mode=='task_and_semantics' and not args.diagnose_thickness_residual)
        response=material_verdict(material,accuracy_mode)
        solids={n:surface.max_penetration((sampled-ctx.pose(args.assembly+'/'+n)[0])@ctx.pose(args.assembly+'/'+n)[1]) for n,surface in surfaces.items()}
        soft_checks={}
        for n,surface in soft_surfaces.items():
            soft_checks[n]=surface.refresh()
            from calibrated_scene_reconstruction.assets.volume_contact import rigid_contacts
            soft_checks[n]['rigid_contacts']=rigid_contacts(surface.points,surface.faces,surfaces,
                {name:ctx.pose(args.assembly+'/'+name) for name in surfaces},tolerance)
            solids[n]=surface.distance.max_penetration(sampled)
            soft_states[n].append(surface.points.copy())
        contact_range=settings['contact_offset']+fixture.contact_offset_m
        contact=penetration.check(sampled,poses,fixture.names,tolerance,contact_range_m=contact_range)
        selected=penetration.check(material_samples(points,sheet['F'],patch),poses,fixture.names,tolerance,contact_range_m=contact_range)
        tool_depth=0.
        for row in fixture.rows:
            pose=poses[fixture.names.index(row['link'])];r=Rotation.from_quat(pose[3:]).as_matrix()
            world_points=triangle_samples(row['V'],row['F'])@r.T+pose[:3]
            for name,surface in surfaces.items():
                p,r=ctx.pose(args.assembly+'/'+name)
                tool_depth=max(tool_depth,surface.max_penetration((world_points-p)@r))
            for surface in soft_surfaces.values():
                tool_depth=max(tool_depth,surface.distance.max_penetration(world_points))
        valid=bool(crossings['pass'] and response['pass'] and max(solids.values())<=tolerance
                   and all(v['rigid_contacts']['pass'] for v in soft_checks.values())
                   and contact['pass'] and tool_depth<=tolerance)
        for n in rigid:
            p,r=ctx.pose(args.assembly+'/'+n);rigid[n].append(np.r_[p,r.ravel()])
        _,a=ctx.pose(body_name);_,b=ctx.pose(args.assembly+'/'+args.lid_link)
        pads=np.array([Rotation.from_quat(pose[3:]).apply(cp)+pose[:3] for pose,cp in zip(poses,fixture.contacts)])
        rows.append({'time_s':t,'phase':phase,'material':material,'self_contact':crossings,'finite_sheet_clearance':clear,
            'material_response':response,
            'non_thickness_physical_pass':valid,'physical_pass':bool(valid and clear['pass']),
            'hand_contact':contact,'selected_patch_contact':selected,'finger_pads':pads.tolist(),
            'solid_penetration_m':solids,'lid_angle_rad':float(revolute_position(a,b,*frames)),
            'native_soft_solid_checks':soft_checks,
            'tool_container_penetration_m':tool_depth,
            'body_pose':rigid[args.body_link][-1].tolist(),
            'selected_material_point':points[node].tolist(),
            'selected_material_node':node,
            'native_force_limit_N':fixture.force_limits,'task_approved':False})
        rows[-1]['commanded_tool_forces_N']=None if fixture.held is None else as_numpy(fixture.held[0]).tolist()
        if phase=='initial' and support:
            sp,sr=ctx.pose(args.assembly+'/'+support['link']);base=parts[support['link']]
            measured=support_state((points-sp)@sr,chart['points'],sheet['F'],base['V'],base['F'],support,
                thickness_m=values['thickness'],tolerance_m=tolerance)
            rows[-1]['initial_support_state']=measured
            valid=bool(valid and measured['pass'])
            rows[-1]['non_thickness_physical_pass']=valid
            rows[-1]['physical_pass']=bool(valid and clear['pass'])
        states.append(points);tools.append(poses);times.append(t)
        if grasp_regions is not None:
            rows[-1]['grasp_region_contacts']=retained_regions(points,sheet['F'],grasp_regions,penetration,
                poses,fixture.names,tolerance,contact_range,contact_groups)
        if args.zero_friction_lift and phase in {'lift','hold'}:
            native_material=as_numpy(fixture.view._physics_view.get_material_properties())
            rows[-1]['native_finger_friction_zero']=bool(np.all(native_material[...,:2]==0.))
            protocol['zero_friction_intervention']['native_readback_after_step']=rows[-1]['native_finger_friction_zero']
            if not rows[-1]['native_finger_friction_zero']:raise ValueError('Native zero finger friction did not persist')
        (out/'progress.json').write_text(json.dumps(rows[-1],indent=2)+'\n')
        return bool(valid and (clear['pass'] or args.diagnose_thickness_residual))
    def phase(name,seconds,goal,goal_gap,goal_angle):
        global t,current,gap,angle,current_rotation
        start=current.copy();g0=gap;a0=angle
        if name in {'approach','open_descent'}:
            proposed=force_limited_seconds(float(np.linalg.norm(np.asarray(goal)-start)),seconds,fixture.mass,fixture.force_limits)
            protocol.setdefault('finite_approach_timing',[]).append({'phase':name,'requested_s':seconds,'commanded_s':proposed,'force_fraction':.25,'native_arrival_still_required':True})
            seconds=proposed
        count=int(np.ceil(seconds*hz));turn=orientation_path(current_rotation,rotation)
        phases[name]=[phases.get(name,[t])[0],t+count/hz]
        print('CONTACT_CLOTH_PHASE',name,flush=True)
        for i in range(count):
            u=(i+1)/count;u=u*u*(3-2*u)
            current=start+u*(np.asarray(goal)-start);gap=g0+u*(goal_gap-g0);angle=a0+u*(goal_angle-a0)
            current_rotation=turn(u)
            loaded=args.normal_preload and name in {'pinch_hold','lift','hold'}
            commands.append(np.r_[current,current_rotation.ravel(),gap,angle,float(loaded),grip_force['normal_force_per_finger_N']])
            fixture.control(current,current_rotation,gap,normal_force_limit_N=
                grip_force['normal_force_per_finger_N'] if goal_gap==0 else None,
                normal_preload_N=grip_force['normal_force_per_finger_N'] if loaded else None)
            lid.apply_action(ArticulationAction(joint_positions=np.array([angle],np.float32)))
            world.step(render=False);t+=1/hz
            if (i+1)%(hz//20)==0 or i+1==count:
                if not record(name):raise ValueError('Actual material/self/solid validity failed in '+name)
                if name in {'lift','hold'} and prelift is not None:
                    measured=lift_measurement(prelift,states[-1],patch,before,fixture.poses()[:,:3].mean(0))
                    rows[-1]['lift_progress']=measured
                    contacts=rows[-1].get('grasp_region_contacts')
                    if (not args.record_failed_lift_attempt and measured['vertical_slip_m']>.015
                            and contacts is not None and not contacts['pass']):
                        raise ValueError('Measured material slip after loss of the captured per-finger regions')
    try:
        if replay is None:
            if not record('initial'):raise ValueError('Native captured state invalid')
            phase('opening',args.opening_seconds,initial_centre,.018,2.)
            if args.opening_hold_seconds:phase('opening_hold',args.opening_hold_seconds,initial_centre,.018,2.)
            if rows[-1]['lid_angle_rad']<1.5:raise ValueError('Actual finite lid drive did not clear the aperture')
            # Recompute the target from the actual material, never replay cloth.
            actual=ctx.nodes(cloth_name)
            poses={n:ctx.pose(args.assembly+'/'+n) for n in surfaces};chosen={};proposals=[]
            offsets,_=clearance_offsets(values['thickness'])
            radius=float(np.ptp(parts[args.body_link]['V'],axis=0)[:2].mean()/2)
            def feasible(point,r):
                pads=fixture.contacts+np.array([[-.009,0,0],[.009,0,0]])
                assignment=opening_assignment(pads@r.T+point,fix['opening'],radius,selected_inside_finger)
                if not assignment['pass']:return False
                for offset in offsets:
                    goal=point+[0,0,offset]
                    witness=capture.check(point,goal,r)
                    if not witness['pass']:continue
                    endpoint=planned.check(goal,r,closing_surfaces,poses,0.)
                    if not endpoint['pass']:continue
                    # The tools descend with a finite motor. Deformable contact
                    # during that descent is intentional; solids remain excluded.
                    turn=orientation_path(current_rotation,r)
                    path=planned.check_open_path(np.linspace(current,goal+[0,0,.002],17),
                        np.asarray([turn(alpha)for alpha in np.linspace(0,1,17)]),surfaces,poses,.0005)
                    descent=planned.check_open_path(np.linspace(goal+[0,0,.002],goal,9),
                        np.tile(r,(9,1,1)),surfaces,poses,.0005)
                    proposals.append({'point':point.tolist(),'standoff_m':float(offset),
                        'opening_assignment':assignment,'capture':witness,'endpoint':endpoint,'approach':path,'open_descent':descent})
                    if path['pass'] and (args.closure_mode=='coupled' or descent['pass']):
                        chosen.update(goal=goal,rotation=r,standoff_m=float(offset));return True
                return False
            try:
                selection=rim_target(actual,chart['points'],sheet['F'],fix['opening'],outer_radius=radius,
                    feasible=feasible,align_surface=True,material_axis=args.pinch_material_axis,material_axis_frame=args.pinch_material_frame,roll_angles=np.radians([0,15,-15,30,-30]))
            finally:
                (out/'contact_proposals.json').write_text(json.dumps(proposals,indent=2)+'\n')
            patch=np.asarray(selection['material_patch'],int);node=int(selection['material_node'])
            centre=chosen['goal'];rotation=chosen['rotation']
            protocol['postopening_selection']={k:v.tolist() if hasattr(v,'tolist') else v for k,v in selection.items()}
            phase('approach',args.approach_seconds,centre+[0,0,.002],.018,2.)
            assignment=opening_assignment(rows[-1]['finger_pads'],fix['opening'],float(np.ptp(parts[args.body_link]['V'],axis=0)[:2].mean()/2),selected_inside_finger)
            (out/'opening_assignment.json').write_text(json.dumps(assignment,indent=2)+'\n')
            if not assignment['pass']:raise ValueError('Actual source tips do not straddle the aperture before pinch')
            axis_up=None if args.pinch_material_frame=='spatial' else [0,0,1]
            if args.closure_mode=='open_descent':
                descent_log=[]
                maximum=2*max(np.ptp(row['V'],axis=0).max() for row in fixture.rows)
                intrinsic=np.asarray(chart['points'])
                patch_radius=float(np.linalg.norm(intrinsic[patch]-intrinsic[node],axis=1).max())
                descended=0.
                for attempt in range(4):
                    native=fixture.poses();pads=np.asarray(rows[-1]['finger_pads']);point=ctx.nodes(cloth_name)[node]
                    predictive=attempt==0 and args.capture_motion_lead and len(states)>1
                    if predictive:
                        lead=capture_lead(states[-2],states[-1],patch,dt=times[-1]-times[-2],closing_seconds=args.pinch_seconds,
                            up=[0,0,1],maximum_shift_m=maximum)
                        protocol['capture_motion_lead']=lead
                        predictive=np.linalg.norm(lead['shift_world_m'])>values['thickness']/2
                    actual=ctx.nodes(cloth_name)
                    def captured(index,region):
                        return (open_capture.check(actual[index],pads,native,fixture.names)['pass'] and
                            pinch_axis_alignment(pads,material_tangent(actual,chart['points'],region,args.pinch_material_axis),up=axis_up)['pass'])
                    try:
                        associated=capture_target(actual,chart['points'],sheet['F'],pads.mean(0),patch_radius=patch_radius,
                            search_radius=maximum,feasible=captured)
                        node=associated['material_node'];patch=associated['material_patch'];point=actual[node]
                        protocol['preclosure_selection']={k:v.tolist() if hasattr(v,'tolist') else v for k,v in associated.items()}
                        protocol['preclosure_selection']['time_s']=t
                        if predictive:
                            lead=capture_lead(states[-2],states[-1],patch,dt=times[-1]-times[-2],closing_seconds=args.pinch_seconds,
                                up=[0,0,1],maximum_shift_m=maximum)
                            protocol['capture_motion_lead']=lead
                            predictive=np.linalg.norm(lead['shift_world_m'])>values['thickness']/2
                    except ValueError:
                        associated=None
                    readiness=open_capture.check(point,pads,native,fixture.names)
                    if associated is not None and readiness['pass'] and not predictive:break
                    if maximum-descended<=0:break
                    predicted=point+np.asarray(lead['shift_world_m']) if predictive else point
                    candidates=open_capture.lowering_candidates(predicted,pads,native,fixture.names,
                        up=[0,0,1],step_m=values['thickness']/2,maximum_m=maximum-descended)
                    if predictive:candidates.sort(key=lambda c:abs(c['distance_m']-(pads.mean(0)[2]-predicted[2])))
                    world_tools=[]
                    for row in fixture.rows:
                        pose=native[fixture.names.index(row['link'])];r=Rotation.from_quat(pose[3:]).as_matrix()
                        world_tools.append(triangle_samples(row['V'],row['F'])@r.T+pose[:3])
                    selected=None
                    for candidate in candidates:
                        distance=candidate['distance_m'];depth=0.
                        for alpha in np.linspace(0,1,9):
                            for world_points in world_tools:
                                for name,surface in surfaces.items():
                                    p,r=ctx.pose(args.assembly+'/'+name)
                                    depth=max(depth,surface.max_penetration((world_points-[0,0,alpha*distance]-p)@r))
                        candidate['open_path_penetration_m']=depth;descent_log.append(candidate)
                        if depth<=values['thickness']/2:selected=candidate;break
                    if selected is None:break
                    distance=selected['distance_m'];descended+=distance
                    # Only finite rigid-tool commands change. Material is read-only.
                    goal=native[:,:3].mean(0)-[0,0,distance]
                    phase('open_descent',args.open_descent_seconds,goal,.018,2.)
                readiness=open_capture.check(ctx.nodes(cloth_name)[node],rows[-1]['finger_pads'],fixture.poses(),fixture.names)
                readiness['opening_assignment']=opening_assignment(rows[-1]['finger_pads'],fix['opening'],radius,selected_inside_finger)
                readiness['pass'] &= readiness['opening_assignment']['pass']
                readiness['descent_proposals']=descent_log;readiness['commanded_lowering_m']=descended
                (out/'preclosure_readiness.json').write_text(json.dumps(readiness,indent=2)+'\n')
                if not readiness['pass']:raise ValueError('Actual open tips did not reach the current material capture depth; closure was not commanded')
                centre=fixture.poses()[:,:3].mean(0)
            tangent=material_tangent(ctx.nodes(cloth_name),chart['points'],patch,args.pinch_material_axis)
            axis_up=None if args.pinch_material_frame=='spatial' else [0,0,1]
            axis=pinch_axis_alignment(rows[-1]['finger_pads'],tangent,up=axis_up)
            axis['spatial_diagnostic']=pinch_axis_alignment(rows[-1]['finger_pads'],tangent,up=None)
            (out/'preclosure_axis_check.json').write_text(json.dumps(axis,indent=2)+'\n')
            if not axis['pass']:raise ValueError('Actual source tips miss the material longitudinal axis before closure')
            phase('pinch',args.pinch_seconds,centre,0.,2.)
            if args.pinch_hold_seconds:phase('pinch_hold',args.pinch_hold_seconds,centre,0.,2.)
            prelift=ctx.nodes(cloth_name).copy();before=fixture.poses()[:,:3].mean(0)
            approach_patch=patch.copy()
            if args.material_contact_mode=='same_component':
                intrinsic=np.asarray(chart['points']);radius=float(np.linalg.norm(intrinsic[patch]-intrinsic[node],axis=1).max())
                grasp_regions=contact_regions(prelift,intrinsic,sheet['F'],rows[-1]['finger_pads'],fixture.poses(),
                    fixture.names,penetration.rows,contact_groups,
                    contact_range_m=settings['contact_offset']+fixture.contact_offset_m,
                    patch_radius=radius/2,boundary_margin=radius)
                if grasp_regions['pass']:
                    patch=np.asarray(grasp_regions['material_patch'],int)
                    if not record('pinch_hold' if args.pinch_hold_seconds else 'pinch'):
                        raise ValueError('Actual contact-region transition invalid')
                elif not args.record_failed_lift_attempt:
                    raise ValueError('Actual opposing interior material regions were not captured')
            precontact=pinch_contact(prelift,patch,rows[-1]['finger_pads'],rows[-1]['selected_patch_contact']['near_surface_samples'])
            if not precontact['pass']:
                failure='Both source fingertips did not contact the selected middle material'
                if not args.record_failed_lift_attempt:raise ValueError(failure)
                print('CONTACT_CLOTH_FAILED_GRASP_ATTEMPT',failure,flush=True)
            if args.zero_friction_lift:protocol['zero_friction_intervention']=author_friction(fixture.material,0.)
            phase('lift',translation_seconds(hand['lift_height_m'],hand['lift_speed_m_s']),
                centre+[0,0,hand['lift_height_m']],0.,2.)
            phase('hold',1.,current,0.,2.)
            lift=lift_measurement(prelift,ctx.nodes(cloth_name),patch,before,fixture.poses()[:,:3].mean(0))
        else:
            if not record('initial'):raise ValueError('Native replay capture state invalid')
            last_phase=None
            for index,(command,name)in enumerate(zip(replay['commands'],replay['labels'])):
                if name!=last_phase:
                    print('CONTACT_CLOTH_REPLAY_PHASE',name,flush=True)
                    if name=='approach':
                        selection=replay['protocol']['postopening_selection']
                        node=int(selection['material_node']);patch=np.asarray(selection['material_patch'],int)
                    if name in {'open_descent','pinch'}:
                        selection=replay['protocol'].get('preclosure_selection',replay['protocol']['postopening_selection'])
                        node=int(selection['material_node']);patch=np.asarray(selection['material_patch'],int)
                    if name=='lift':
                        grasp_regions=deepcopy(replay['result']['grasp_regions'])
                        approach_patch=replay['trace']['approach_material_patch'].copy()
                        patch=replay['trace']['material_patch'].copy()
                        node=int(replay['result']['rows'][-1]['selected_material_node'])
                        prelift=ctx.nodes(cloth_name).copy();before=fixture.poses()[:,:3].mean(0)
                        if not record(last_phase):raise ValueError('Native replay prelift state invalid')
                        precontact=pinch_contact(prelift,patch,rows[-1]['finger_pads'],rows[-1]['selected_patch_contact']['near_surface_samples'])
                        if not precontact['pass']:raise ValueError('Replay did not reproduce the positive grasp before intervention')
                        protocol['zero_friction_intervention']=author_friction(fixture.material,0.)
                    last_phase=name
                current=command[:3].copy();current_rotation=command[3:12].reshape(3,3).copy()
                gap=float(command[12]);angle=float(command[13]);loaded=bool(command[14]);normal=float(command[15])
                phases[name]=[phases.get(name,[t])[0],(index+1)/hz]
                commands.append(command.copy())
                fixture.control(current,current_rotation,gap,normal_force_limit_N=normal if gap==0. else None,
                    normal_preload_N=normal if loaded else None)
                lid.apply_action(ArticulationAction(joint_positions=np.array([angle],np.float32)))
                world.step(render=False);t=(index+1)/hz
                if index+1 in replay['record_steps']:
                    if not record(name):raise ValueError('Actual material/self/solid validity failed in replay '+name)
                    if name in {'lift','hold'}and prelift is not None:
                        measured=lift_measurement(prelift,states[-1],patch,before,fixture.poses()[:,:3].mean(0))
                        rows[-1]['lift_progress']=measured
                        contact=rows[-1].get('grasp_region_contacts')
                        if measured['vertical_slip_m']>.015 and contact is not None and not contact['pass']:
                            control_termination={'reason':'measured_slip','time_s':t,'same_command_prefix':True}
                            break
            if prelift is not None:
                lift=lift_measurement(prelift,ctx.nodes(cloth_name),patch,before,fixture.poses()[:,:3].mean(0))
    except ValueError as error:
        failure=str(error)
        if prelift is not None:
            lift=lift_measurement(prelift,ctx.nodes(cloth_name),patch,before,fixture.poses()[:,:3].mean(0))
    errors.check()
    np.savez_compressed(out/'states.npz',points=states,hand_links=tools,times=times,faces=sheet['F'],
        rest=np.asarray(chart['points']),material_patch=patch,commands=commands,
        approach_material_patch=patch if approach_patch is None else approach_patch,
        prelift_points=np.empty((0,3)) if prelift is None else prelift,
        **{n+'_poses':np.asarray(v) for n,v in rigid.items()},
        **{n+'_surface_points':np.asarray(v) for n,v in soft_states.items()},
        **{n+'_surface_faces':surface.faces for n,surface in soft_surfaces.items()})
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    result={'pass':False,'task_approved':False,'candidate_assay':True,'fixture_kind':'source_fingertip_contact',
        'complete':failure is None and control_termination is None,'failure':failure,'physical_pass':all(r['physical_pass'] for r in rows),
        'control_termination':control_termination,'executed_command_count':len(commands),
        'lift':lift,'prelift_contact':precontact,'rows':rows,'phases':phases,'hand_body_names':fixture.names,
        'grasp_regions':grasp_regions,
        'source_geometry':fixture.source,'fixture_geometry':str(out/'fixture_geometry.json'),
        'commanded_wrench_peaks':None if fixture.peak is None else as_numpy(fixture.peak).tolist(),
        'native_physics_errors':errors.report(),'scope':'Finite source-tip native contact diagnostic, not robot control or delivery approval.'}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('CONTACT_CLOTH_RESULT',json.dumps({k:result[k] for k in ['complete','failure','lift']}),flush=True)
except Exception as error:
    import traceback
    (out/'error.json').write_text(json.dumps({'pass':False,'error':str(error),'traceback':traceback.format_exc()},indent=2)+'\n')
    raise
finally:
    errors.close();app.close()
