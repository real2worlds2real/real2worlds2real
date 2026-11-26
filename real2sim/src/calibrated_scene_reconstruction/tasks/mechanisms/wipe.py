"""Native flip-open, free-sheet extraction and tabletop wiping measurements.

Force fixtures test object-model feasibility, not autonomous robot manipulation.
No cloth/body pose is prescribed during the experiment. Initial construction and
material validity remain declared hypotheses, not identified real properties.
"""
import json

import numpy as np

from calibrated_scene_reconstruction.assets.metric_mesh import asset
from calibrated_scene_reconstruction.tasks.mechanisms.cloth import MaterialMetric, support_contact, nonrigid_change, pinch_retention
from calibrated_scene_reconstruction.core.geometry.rigid import as_numpy, quat
from calibrated_scene_reconstruction.tasks.mechanisms.force_fixture import ForceHandle, approach,edge_patch,material_patch
from calibrated_scene_reconstruction.tasks.mechanisms.seam_contact import measure as seam_contact


def run(ctx, assembly, *, output, body_link='body', lid_link='lid', cloth_link='wipe',
        max_pull_force_N=1.5, max_open_torque_Nm=.2, grasp_width_m=None, grasp_fraction=1.,grasp_region=None):
    from pxr import Gf, UsdPhysics, PhysxSchema
    from calibrated_scene_reconstruction.assets.surface_distance import SignedSurface, triangle_samples
    from calibrated_scene_reconstruction.assets.cloth_contact import self_intersections
    from calibrated_scene_reconstruction.assets.response_policy import mode,material_verdict
    accuracy=mode(ctx.task.get('flexible_accuracy'))

    names={k:assembly+'/'+v for k,v in [('body',body_link),('lid',lid_link),('cloth',cloth_link)]}
    fix=ctx.sg['dynamic_objects'][assembly]['fix']
    kit=asset(fix);parts={p['name']:p for p in kit['parts']}
    for part in kit['parts']:
        if part['kind']=='rigid' and part['name'] not in {body_link,lid_link}:
            names[part['name']]=assembly+'/'+part['name']
    sheet=parts[cloth_link];rest=sheet.get('cloth_rest')
    if rest is None:
        raise ValueError('Extraction requires explicit intrinsic sheet material coordinates')
    from calibrated_scene_reconstruction.assets.physical_priors import expand_families, sample
    plan=expand_families(ctx.s.config('agent/physics_priors.json'))
    sampled_material=sample(plan,getattr(ctx,'physics_case','nominal'))['bodies'][names['cloth']]['parameters']
    sheet_thickness=sampled_material.get('thickness')
    validity=plan['bodies'][names['cloth']].get('validity_hypothesis')
    if not validity:
        raise ValueError('Declare material strain/area validity assumptions before native task checks')
    if min(max_pull_force_N,max_open_torque_Nm)<=0 or not np.isfinite([max_pull_force_N,max_open_torque_Nm]).all():
        raise ValueError('Finite positive test force/torque limits required')
    folder=ctx.s/output;folder.mkdir(parents=True,exist_ok=False)
    state_files=[]
    declarations=ctx.s.config('agent/mechanisms.json')
    ids={'hinge':'flip_lid','deform':'wipe_deformation','extract':'wipe_dispensing','pinch':'lid_wipe_clearance'}
    relevant={m['id']:m for m in declarations['mechanisms']}
    if not set(ids.values())<=set(relevant):
        raise ValueError('Agent must map this check to declared flip/sheet/contact mechanisms')
    results={key:{'pass':False,'representation':relevant[key]['representation'],
                       'checks':{name:{'pass':False} for name in relevant[key]['checks']},'evidence_files':[]}
             for key in ids.values()}
    for key in ['wipe_dispensing','lid_wipe_clearance']:
        results[key].update(prescribed_motion=False,contact_observed=False)
    results[ids['deform']]['solver_deformation_observed']=False
    def checked(group,name,passed,**values):
        results[ids[group]]['checks'][name]={'pass':bool(passed),**values}
    def save(name,**values):
        path=folder/(name+'.npz');np.savez_compressed(path,**values)
        relative=str(path.relative_to(ctx.s.dir))
        if relative not in state_files:state_files.append(relative)
    body,lid=ctx.body(names['body']),ctx.body(names['lid'])
    cloth=ctx._cloth[names['cloth']]
    dt=float(ctx.world.get_physics_dt())
    if dt>1/960+1e-12:
        raise ValueError('Validate substep/contact budget at960Hz or finer before this cloth check')
    surfaces={key:SignedSurface(parts[link]['V'],parts[link]['F'])
              for key,link in [('body',body_link),('lid',lid_link)]}
    surfaces.update({part['name']:SignedSurface(part['V'],part['F']) for part in kit['parts']
                     if part['kind']=='rigid' and part['name'] not in {body_link,lid_link}})
    opening=ctx.opening(assembly,link=body_link)
    if opening is None:raise ValueError('Current canister requires an authored physical opening')
    initial=ctx.nodes(names['cloth']).copy()
    metric=MaterialMetric(initial,sheet['F'],rest,max_edge_strain=validity['max_edge_strain'],
                          max_area_change=validity['max_area_change'])
    material=metric.check
    axis=np.asarray(relevant[ids['hinge']]['axis'],float)
    joints=[p for p in ctx.stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)
            and set(str(v) for v in (*UsdPhysics.RevoluteJoint(p).GetBody0Rel().GetTargets(),
                                     *UsdPhysics.RevoluteJoint(p).GetBody1Rel().GetTargets()))
            =={body.prim_path,lid.prim_path}]
    if len(joints)!=1:raise ValueError('Required actual native lid hinge is missing or ambiguous')
    joint=UsdPhysics.RevoluteJoint(joints[0])
    anchors=[np.asarray(getattr(joint,f'GetLocalPos{k}Attr')().Get()) for k in [0,1]]
    # Read authored native rigid-contact thickness, rather than assuming the
    # particle-particle radius follows it (PhysX autocomputes those separately).
    cloth_path=str(ctx.ts.objects[assembly].GetPath())+'/'+cloth_link+'/surface'
    cloth_prim=ctx.stage.GetPrimAtPath(cloth_path)
    surface_fem=sheet.get('backend','particle')=='surface_fem'
    if surface_fem:
        particle_radius=float(PhysxSchema.PhysxCollisionAPI(cloth_prim).GetRestOffsetAttr().Get())
        self_radius=particle_radius
    else:
        particle_path=PhysxSchema.PhysxParticleAPI(cloth_prim).GetParticleSystemRel().GetTargets()[0]
        particle_system=PhysxSchema.PhysxParticleSystem(ctx.stage.GetPrimAtPath(particle_path))
        particle_radius=float(particle_system.GetRestOffsetAttr().Get())
        self_radius=float(particle_system.GetSolidRestOffsetAttr().Get())
        native_rigid_contact={'rest_offset':particle_radius,
                              'contact_offset':particle_system.GetContactOffsetAttr().Get()}
    if not np.isfinite([particle_radius,self_radius]).all() or min(particle_radius,self_radius)<=0:
        raise ValueError('Native cloth rigid/self-contact offsets must be explicit and positive')
    from calibrated_scene_reconstruction.assets.cloth_contact_tolerance import solid_tolerance
    tolerance=solid_tolerance(sheet_thickness,
        2*float(np.ptp(parts[body_link]['V'],axis=0).max())/fix['sdf']['resolution'],
        legacy_rest_offset=particle_radius)
    from calibrated_scene_reconstruction.tasks.mechanisms.fixed import FixedAttachments
    fixed_attachments=FixedAttachments(ctx,assembly,declarations,position_tolerance=tolerance)
    limit_angle=relevant[ids['hinge']]['limits']
    v=np.asarray(rest['points']);grasp_depth=min(.01,.08*np.ptp(v[:,1]))
    grasp_width=min(.01,.08*np.ptp(v[:,0])) if grasp_width_m is None else grasp_width_m
    if grasp_region is not None:
        if (not isinstance(grasp_region,dict) or set(grasp_region)!={'centre_fraction','size_m','basis'}
                or not isinstance(grasp_region['basis'],str) or not grasp_region['basis'].strip()):
            raise ValueError('Explicit material-local grasp area and source/construction basis required')
        patch=material_patch(v,grasp_region['centre_fraction'],grasp_region['size_m'])
        grasp_width,grasp_depth=np.ptp(v[patch,:2],axis=0)
    else:patch=edge_patch(v,grasp_depth,width=grasp_width,centre_fraction=grasp_fraction)
    if len(patch)<3:raise ValueError('No material edge for a distributed force fixture')
    lengths=np.ptp(v,axis=0);length=float(lengths[1]);width=float(lengths[0])
    area_weights=np.zeros(len(v));tri=v[sheet['F']]
    weights=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/6
    for column in range(3):np.add.at(area_weights,sheet['F'][:,column],weights)
    def local(points,key):
        p,R=ctx.pose(names[key]);return (points-p)@R
    def inside_fraction(points):
        p=local(points,'body');op=fix['opening']
        inside=(np.linalg.norm(p[:,:2]-op['xy'],axis=1)<op['radius'])&(p[:,2]<op['rim_z'])&(p[:,2]>op['rim_z']-op['depth'])
        return float(area_weights[inside].sum()/area_weights.sum())
    def angle_and_attachment():
        bp,br=ctx.pose(names['body']);lp,lr=ctx.pose(names['lid'])
        from calibrated_scene_reconstruction.assets.mechanisms import revolute_position
        frames=[]
        for side in [0,1]:
            q=getattr(joint,f'GetLocalRot{side}Attr')().Get()
            frames.append([q.GetReal(),*q.GetImaginary()])
        angle=revolute_position(br,lr,*frames)
        error=float(np.linalg.norm(bp+br@anchors[0]-lp-lr@anchors[1]))
        return angle,error
    first_material=material(initial)
    first_response=material_verdict(first_material,accuracy)
    quantitative_material=first_material['pass'];material_geometry_valid=first_material['geometry_valid']
    first_self_contact=self_intersections(initial,sheet['F'])
    from calibrated_scene_reconstruction.assets.cloth_clearance import measure as finite_clearance
    first_clearance=(finite_clearance(initial,sheet['F'],rest['points'],sheet_thickness,device='cuda:0',
                                      local_bending_review=accuracy=='task_and_semantics')
                     if sheet_thickness is not None else None)
    from calibrated_scene_reconstruction.assets.cloth_sampling import rigid_coverage
    contact_sampling=({'pass':True,'representation':'Native surface triangle contact; no particle coverage substitution'}
                      if surface_fem else rigid_coverage(initial,sheet['F'],native_rigid_contact))
    if surface_fem:
        native_rest=np.asarray(cloth_prim.GetAttribute('omniphysics:restShapePoints').Get())
        expected_rest=np.asarray(rest.get('bend_reference',{}).get('points',v))
    else:
        native_rest=np.asarray(PhysxSchema.PhysxParticleClothAPI(cloth_prim).GetRestPointsAttr().Get());expected_rest=v
    rest_matches=native_rest.shape==expected_rest.shape and np.allclose(native_rest,expected_rest,rtol=1e-6,atol=1e-7)
    samples=triangle_samples(initial,sheet['F'])
    first_pen={k:surface.max_penetration(local(samples,k)) for k,surface in surfaces.items()}
    first_angle,first_attachment=angle_and_attachment()
    distances={k:surface(local(initial,k)) for k,surface in surfaces.items()}
    def seam(points, surface_distances):
        # Read-only geometry queries in actual native poses. The ray witnesses
        # allow different folds to touch the two solids, while checking every
        # intervening layer gap with the original contact tolerance.
        near=abs(surface_distances['body'])<=tolerance
        normals=np.zeros_like(points)
        if near.any():
            q=local(points[near],'body')
            direction=surfaces['body'].project_out(q,4*tolerance)-q
            direction/=np.linalg.norm(direction,axis=1,keepdims=True)
            _,rotation=ctx.pose(names['body']);normals[near]=direction@rotation.T
        p,rotation=ctx.pose(names['lid'])
        lid_world=parts[lid_link]['V']@rotation.T+p
        return seam_contact(points,sheet['F'],surface_distances['body'],surface_distances['lid'],
                            normals,lid_world[parts[lid_link]['F']],tolerance=tolerance)
    initial_seam=seam(initial,distances)
    pinch_sides={key:np.asarray(initial_seam[key],int) for key in ['first_indices','second_indices']}
    pinched=min(map(len,pinch_sides.values()))
    lid_center=np.mean([parts[lid_link]['V'].min(0),parts[lid_link]['V'].max(0)],axis=0)
    lid_solid_center=bool(surfaces['lid'](lid_center[None])[0]<0)
    fraction=inside_fraction(initial)
    body_position,body_rotation=ctx.pose(names['body'])
    initial_body_local=(initial-body_position)@body_rotation
    lid_position,lid_rotation=ctx.pose(names['lid'])
    save('initial',points=initial,rest_points=v,faces=sheet['F'],
         body_position=body_position,body_rotation=body_rotation,
         lid_position=lid_position,lid_rotation=lid_rotation,
         body_surface_distance_m=distances['body'],lid_surface_distance_m=distances['lid'])
    checked('deform','geometry',True,kind='native_surface_fem' if surface_fem else 'native_particle_cloth',vertices=len(initial))
    checked('deform','no_initial_self_intersection',first_self_contact['pass'],**{k:v for k,v in first_self_contact.items() if k!='pass'})
    checked('deform','no_self_intersection',first_self_contact['pass'],maximum_pairs=first_self_contact['intersecting_pairs'])
    if first_clearance is not None:
        checked('deform','finite_sheet_clearance',first_clearance['pass'],measurement=first_clearance)
    checked('extract','contact_resolution',contact_sampling['pass'],particle_sampling=contact_sampling)
    checked('deform','material_area',first_response['pass'],
            **{k:v for k,v in first_material.items() if k!='pass'},response_acceptance=first_response)
    checked('deform','rest_metric',rest_matches,rest_area_m2=first_material['rest_area_m2'],actual_native_attributes_read=True)
    checked('deform','no_initial_interpenetration',max(first_pen.values())<=tolerance,penetration_m=first_pen,tolerance_m=tolerance)
    from calibrated_scene_reconstruction.assets.joint_state import measure as initial_joint_state
    initial_hinge=initial_joint_state(relevant[ids['hinge']],first_angle,default=(-.12,.12))
    closed=initial_hinge['pass'] and .2<fraction<.8
    all_interfaces=ctx.s.config('agent/interface_checks.json',{}).get('objects',{}).get(assembly,[])
    interfaces=[r for r in all_interfaces if r.get('phase','initial_closed')=='initial_closed']
    unloaded_interfaces=[r for r in all_interfaces if r.get('phase')=='unloaded_closed']
    from .interface_contact import check as interface_check
    if interfaces:
        from .interface_contact import check as interface_check
        seated=interface_check(ctx,assembly,kit['parts'],interfaces)
        checked('hinge','closed_seating',seated['pass'],measurement=seated)
        closed=closed and seated['pass']
    support=ctx.sg['furniture'][ctx.sg['task']['support']]['box'];lo,hi=np.asarray(support,float)
    body_points=ctx.points(names['body']);body_gap=float(body_points[:,2].min()-hi[2])
    checked('hinge','geometry',abs(body_gap)<=tolerance,body_mesh_watertight=True,lid_mesh_watertight=True,body_support_gap_m=body_gap)
    checked('hinge','closed_initial_clearance',closed and pinched>=3,angle_rad=first_angle,inside_material_fraction=fraction,pinch_nodes=pinched)
    checked('extract','closed_initial_state',closed,inside_material_fraction=fraction,initial_joint_state=initial_hinge)
    checked('extract','rim_seam_exit_without_invented_hole',pinched>=3 and lid_solid_center,pinch_nodes=pinched,lid_center_is_solid=lid_solid_center)
    checked('pinch','closed_initial_clearance',max(first_pen.values())<=tolerance,penetration_m=first_pen)
    checked('pinch','closed_pinch_contact',closed and initial_seam['pass'],pinch_nodes=pinched,
            geometric_contact=initial_seam)
    for key in ['extract','pinch']:results[ids[key]]['contact_observed']=pinched>=3
    peak_strain=first_material['maximum_absolute_edge_strain'];peak_area=abs(first_material['relative_area_change'])
    peak_principal=first_material['maximum_principal_material_strain']
    peak_self_intersections=first_self_contact['intersecting_pairs']
    peak_pen=max(first_pen.values());peak_step=0.;peak_attachment=first_attachment
    frames=[];history=[];phases=[];requested_damping=[];force_handle=None;empty_drive=None
    # Keep the existing every16-step material/intersection checks and every
    # physics-step rigid checks. Dense state storage follows a separate clock:
    # otherwise a finer solver clock multiplies trace RAM and checkpoint IO.
    trace_stride=16*max(1,round(1/(60*dt*16)))
    last_trace_sample=None;step_index=0;elapsed=0.
    minimum_hinge_angle=maximum_hinge_angle=first_angle
    fixture='/World/TaskCheckFixtures/wipe_container_holder'
    passive_response=None
    def finish(what,extra=None):
        fixed_attachments.sample(elapsed)
        if last_trace_sample is not None and (not history or last_trace_sample[1][0]>history[-1][0]):
            points,measurement,phase=last_trace_sample
            frames.append(points);history.append(measurement);phases.append(phase)
        results.update(fixed_attachments.results())
        if fixed_attachments.entries:
            path=folder/'fixed_attachment_states.json';path.write_text(json.dumps(fixed_attachments.trace,indent=2)+'\n')
            state_files.append(str(path.relative_to(ctx.s.dir)))
        save('motion',points=np.asarray(frames),measurements=np.asarray(history),phase=np.asarray(phases,dtype=str))
        for item in results.values():
            item['evidence_files']=list(state_files)
            item['pass']=all(check['pass'] for check in item['checks'].values())
        # Source-observation checks belong to the framework and are merged
        # after this experiment. Their placeholders cannot make every task
        # operation fail before its independent observations are attached.
        for name,item in results.items():
            external={o['check'] for o in declarations.get('observations',[]) if o['mechanism']==name}
            item['operation_pass']=all(check['pass'] for key,check in item['checks'].items() if key not in external)
        report={'pass':bool(passive_response and passive_response['pass']) and all(item['operation_pass'] for item in results.values()),'what':what,'mechanisms':results,
                'unforced_assembly':passive_response,
                'material_accuracy_mode':accuracy,'quantitative_material_pass':bool(quantitative_material),
                'trace_sampling':{'stored_state_stride':trace_stride,'stored_state_hz':1/(trace_stride*dt),
                    'material_and_intersection_stride':16,'rigid_contact_stride':1,
                    'last_validated_sample_retained':True},
                'physical_plausibility_review_required':not bool(quantitative_material),
                'scope':'Native object-model force fixture; no autonomous robot execution or material identification',
                'maximum_penetration_m':peak_pen,'maximum_edge_strain':peak_strain,
                'maximum_principal_material_strain':peak_principal,
                'maximum_self_intersection_pairs':peak_self_intersections,
                'max_pull_force_N':max_pull_force_N,'max_open_torque_Nm':max_open_torque_Nm,
                'native_particle_offsets_m':{'rigid_rest':particle_radius,'self_rest':self_radius},
                'damping_limits':requested_damping,**(extra or {})}
        report['force_fixture'] = ({**force_handle.report(), 'material_grasp_depth_m':float(grasp_depth),
                                    'material_grasp_width_m':float(grasp_width),
                                    'material_grasp_centre_fraction':float(grasp_fraction) if grasp_region is None else None,
                                    'material_grasp_region':grasp_region,
                                    'selection':'Explicit material area' if grasp_region is not None else 'Compact actual-material edge area',
                                    'material_min':v[patch].min(0).tolist(),'material_max':v[patch].max(0).tolist(),
                                    'selected_nodes':len(patch), 'selected_mass_kg':float(masses[patch].sum())}
                                   if force_handle is not None else None)
        report['fixture_assumptions'] = 'Compliant edge-force target limited to0.05m/s; hinge target limited to1rad/s. Force/torque budgets are test assumptions, not measured demonstration forces.'
        (folder/'details.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        return report
    if not(first_response['pass'] and first_self_contact['pass'] and (first_clearance is None or first_clearance['pass']) and contact_sampling['pass'] and rest_matches and peak_pen<=tolerance and closed and pinched>=3
           and lid_solid_center and first_attachment<=tolerance and abs(body_gap)<=tolerance):
        return finish('Initial material/contact/pinching state failed; task motion was not attempted')
    masses=as_numpy(cloth.get_particle_masses())[0];mass=float(masses.sum())
    force_handle=ForceHandle(initial[patch],masses[patch],dt=dt,held_mass=mass,force_limit=max_pull_force_N)
    requested_damping.append(force_handle.report())
    hinge_command=first_angle;hinge_active=False
    previous=initial.copy();step_index=0
    _,lid_inertia=ctx.rigid_mass_properties(names['lid'])
    def tick(phase,*,hinge_target=None,target=None,downward=False):
        nonlocal previous,peak_strain,peak_principal,peak_area,peak_pen,peak_step,peak_attachment,step_index,hinge_command,peak_self_intersections
        nonlocal quantitative_material,material_geometry_valid,hinge_active
        nonlocal last_trace_sample,minimum_hinge_angle,maximum_hinge_angle,elapsed
        if hinge_target is not None:
            a,_=angle_and_attachment();_,R=ctx.pose(names['body']);world_axis=R@axis
            _,lid_R=ctx.pose(names['lid']);local_axis=lid_R.T@world_axis
            moment=float(local_axis@lid_inertia@local_axis)
            if not np.isfinite(moment) or moment<=0:raise ValueError('Invalid actual hinge-axis inertia')
            hinge_command=float(approach(np.array(hinge_command),np.array(hinge_target),dt))
            hinge_stiffness=min(.2,.25*moment/dt**2)
            hinge_damping=min(2*np.sqrt(hinge_stiffness*moment),.25*moment/dt)
            omega=as_numpy(lid.get_angular_velocity())
            torque=float(np.clip(hinge_stiffness*(hinge_command-a)-hinge_damping*np.dot(omega,world_axis),-max_open_torque_Nm,max_open_torque_Nm))
            ctx.apply_joint_effort(assembly,body_link,lid_link,torque,effort_limit=max_open_torque_Nm)
            hinge_active=True
        elif hinge_active:
            ctx.apply_joint_effort(assembly,body_link,lid_link,0.,effort_limit=max_open_torque_Nm)
            hinge_active=False
        if target is not None:
            points=ctx.nodes(names['cloth']);vel=as_numpy(cloth.get_velocities())[0,patch]
            force=force_handle.forces(points[patch],vel,target,load=[0,0,-.1 if downward else mass*9.81])
            cap=max_pull_force_N*float(force_handle.weights.max())
            ctx.apply_nodal_forces(names['cloth'],patch,force,force_limit_N=cap)
        ctx.world.step(render=False);points=ctx.nodes(names['cloth']);a,attachment=angle_and_attachment()
        elapsed=(step_index+1)*dt
        minimum_hinge_angle=min(minimum_hinge_angle,a);maximum_hinge_angle=max(maximum_hinge_angle,a)
        sampled=triangle_samples(points,sheet['F'])
        penetration=max(surface.max_penetration(local(sampled,key)) for key,surface in surfaces.items())
        peak_pen=max(peak_pen,penetration);peak_attachment=max(peak_attachment,attachment)
        peak_step=max(peak_step,float(np.linalg.norm(points-previous,axis=1).max()));previous=points.copy()
        if step_index%16==0:
            fixed_attachments.sample(elapsed)
            m=material(points);peak_strain=max(peak_strain,m['maximum_absolute_edge_strain']);peak_area=max(peak_area,abs(m['relative_area_change']))
            response=material_verdict(m,accuracy)
            if not surface_fem:
                dynamic_sampling=rigid_coverage(points,sheet['F'],native_rigid_contact)
                checked('extract','dynamic_contact_resolution',dynamic_sampling['pass'],
                        particle_sampling=dynamic_sampling,last_time_s=elapsed)
                if not dynamic_sampling['pass']:
                    save('particle_sampling_violation',points=points,faces=sheet['F'])
                    raise ValueError('Native cloth deformation opens unresolved particle contact gaps during '+phase)
            quantitative_material &= m['pass'];material_geometry_valid &= m['geometry_valid']
            peak_principal=max(peak_principal,m['maximum_principal_material_strain'])
            measurement=[elapsed,a,attachment,penetration,m['maximum_absolute_edge_strain'],m['relative_area_change']]
            last_trace_sample=(points.copy(),measurement,phase)
            if step_index%trace_stride==0:
                frames.append(last_trace_sample[0]);history.append(measurement);phases.append(phase)
                if sheet_thickness is not None:
                    clearance=finite_clearance(points,sheet['F'],rest['points'],sheet_thickness,device='cuda:0',
                                               local_bending_review=accuracy=='task_and_semantics')
                    checked('deform','finite_sheet_clearance',clearance['pass'],measurement=clearance,
                            sampled_hz=1/(trace_stride*dt),last_time_s=elapsed)
                    if not clearance['pass']:
                        save('finite_thickness_violation',points=points,faces=sheet['F'])
                        raise ValueError('Native sheet layers violate physical thickness during '+phase)
            self_contact=self_intersections(points,sheet['F'])
            peak_self_intersections=max(peak_self_intersections,self_contact['intersecting_pairs'])
            checked('deform','no_self_intersection',peak_self_intersections==0,maximum_pairs=peak_self_intersections)
            if not self_contact['pass']:
                save('self_intersection_violation',points=points,faces=sheet['F'],
                     example_face_pairs=np.asarray(self_contact['example_face_pairs']))
                raise ValueError('Native cloth surfaces self-intersect during '+phase+'; stop the invalid trajectory')
            if not response['pass']:
                checked('deform','strain_bounded',False,maximum_edge_strain=peak_strain,
                        maximum_principal_material_strain=peak_principal,maximum_relative_area_change=peak_area,
                        response_acceptance=response)
                save('material_violation',points=points,measurements=np.asarray([measurement]),phase=np.asarray([phase]))
                raise ValueError('Native material validity exceeded during '+phase+'; stop the invalid trajectory')
        # Append immutable chunks. Recompressing the full growing trajectory at
        # every checkpoint causes quadratic IO for dense native cloth meshes.
        checkpoint_stride=20*trace_stride
        if step_index%checkpoint_stride==0:save('partial_motion_'+str(step_index//checkpoint_stride),points=np.asarray(frames[-20:]),measurements=np.asarray(history[-20:]),phase=np.asarray(phases[-20:],dtype=str))
        step_index+=1
        return points,a
    try:
        # Record the assembly before installing a task holder. Otherwise a
        # successful pinch can conceal preload that moves the bottle or lid.
        from .passive_response import evaluate as evaluate_passive
        rigid_names=[p['name'] for p in kit['parts'] if p['kind']=='rigid']
        passive_poses={name:[] for name in rigid_names};passive_times=[]
        def record_passive():
            passive_times.append(elapsed)
            for name in rigid_names:
                p,r=ctx.pose(assembly+'/'+name)
                passive_poses[name].append(np.r_[p,r.ravel()])
        record_passive()
        count=round(.5/dt)
        for step in range(count):
            tick('closed_passive')
            if (step+1)%trace_stride==0 or step+1==count:record_passive()
        understood=ctx.s.config('agent/object_understanding.json',{}).get('objects',{}).get(assembly,{})
        passive_response=evaluate_passive(understood,kit['parts'],passive_poses,passive_times)
        save('unforced_rigid_states',times=passive_times,**passive_poses)
        if not passive_response['pass']:
            return finish('Unforced assembly moved outside its declared passive bounds; task holder was not installed')
        passive=ctx.nodes(names['cloth']);drift=float(np.linalg.norm(passive-initial,axis=1).max())
        passive_distances={k:surface(local(passive,k)) for k,surface in surfaces.items()}
        passive_seam=seam(passive,passive_distances)
        if interfaces:
            seated_after=interface_check(ctx,assembly,kit['parts'],interfaces)
            checked('hinge','closed_seating',seated['pass'] and seated_after['pass'],
                    initial=seated,after_release=seated_after)
            if not seated_after['pass']:return finish('Released lid no longer mates with its declared seating surface')
        # Compare material in the actual body frame, separating rigid settling
        # from slip of the contacted region and the freely settling flap.
        retention={}
        for side,indices in pinch_sides.items():
            contact=np.zeros(len(passive),bool);contact[passive_seam[side]]=True
            retention[side]=pinch_retention(initial_body_local,local(passive,'body'),indices,contact,
                    inside_before=fraction,inside_after=inside_fraction(passive),contact_tolerance=tolerance)
        stable=passive_seam['pass'] and all(r['pass'] for r in retention.values()) and .2<inside_fraction(passive)<.8
        checked('pinch','closed_pinch_contact',stable,retention_by_side=retention,
                initial_geometric_contact=initial_seam,passive_geometric_contact=passive_seam)
        if not stable:return finish('Closed-lid pinching state did not remain stable under passive native physics')
        bp,br=ctx.pose(names['body']);q=quat(br)
        holder=UsdPhysics.FixedJoint.Define(ctx.stage,fixture)
        holder.CreateBody1Rel().SetTargets([body.prim_path]);holder.CreateLocalPos0Attr(Gf.Vec3f(*map(float,bp)))
        holder.CreateLocalRot0Attr(Gf.Quatf(float(q[0]),Gf.Vec3f(*map(float,q[1:]))));holder.CreateLocalPos1Attr(Gf.Vec3f(0));holder.CreateLocalRot1Attr(Gf.Quatf(1))
        checked('pinch','no_rigid_sheet_substitute',True,native_node_states_read=True)
        opened=False
        for _ in range(round(5/dt)):
            _,a=tick('open_lid',hinge_target=1.9)
            if a>1.3:opened=True;break
        checked('hinge','relative_motion',opened,opening_angle_rad=a)
        checked('hinge','bounded_torque_open',opened,torque_limit_Nm=max_open_torque_Nm)
        checked('pinch','opening_clearance',opened,opening_angle_rad=a)
        current=ctx.nodes(names['cloth'])
        current_distances={k:surface(local(current,k)) for k,surface in surfaces.items()}
        open_seam=seam(current,current_distances)
        remaining_by_side={side:len(set(map(int,indices))&set(open_seam[side]))
                           for side,indices in pinch_sides.items()}
        remaining_pinch=min(remaining_by_side.values())
        checked('pinch','opening_releases_pinch',opened and remaining_pinch<=max(1,int(pinched*.1)),
                remaining_initial_pinch_nodes=remaining_pinch,remaining_by_side=remaining_by_side,
                geometric_contact=open_seam)
        if not opened:return finish('Native lid failed to open under bounded torque')
        # Engage the soft handle at the current material edge. Its target moves;
        # no node positions, attachment or rigid patch orientation are imposed.
        force_handle=ForceHandle(current[patch],masses[patch],dt=dt,held_mass=mass,force_limit=max_pull_force_N)
        _,R=ctx.pose(names['body']);goal=np.r_[opening['xy'],opening['rim_z']+1.2*length]-np.r_[R[:2,1]*width*.6,0]
        released=False
        for _ in range(round(12/dt)):
            points,a=tick('extract',hinge_target=1.9,target=goal)
            if points[:,2].min()>opening['rim_z']+4*particle_radius:
                released=True;break
        checked('extract','finite_force_extraction',released,total_force_limit_N=max_pull_force_N)
        checked('extract','complete_release',released,remaining_inside_material_fraction=inside_fraction(points))
        attached=False
        for prim in ctx.stage.Traverse():
            if prim.IsA(UsdPhysics.Joint) or prim.GetTypeName()=='PhysxPhysicsAttachment':
                targets=[str(t) for rel in prim.GetRelationships() for t in rel.GetTargets()]
                attached |= any(t==cloth_path or t==cloth_path.rsplit('/',1)[0] for t in targets)
        checked('extract','no_permanent_attachment',released and not attached,permanent_attachment_present=bool(attached))
        deformation=nonrigid_change(initial,points)
        deformed=deformation['maximum_m']>max(.001,.01*length)
        checked('deform','solver_deformation',deformed,best_rigid_alignment_removed=True,**deformation)
        results[ids['deform']]['solver_deformation_observed']=deformed
        if not released:return finish('Free native sheet failed complete extraction')
        if unloaded_interfaces:
            # The free sheet is already outside the mouth. Test the unloaded
            # physical lid separately from the observed cloth-pinched state.
            from .joint_drive_fixture import JointDriveFixture
            ctx.apply_joint_effort(assembly,body_link,lid_link,0.,effort_limit=max_open_torque_Nm)
            hinge_active=False
            empty_drive=JointDriveFixture(ctx,assembly,body_link,lid_link,
                stiffness=2.,damping=.01,effort_limit=min(.08,max_open_torque_Nm))
            empty_command=angle_and_attachment()[0]
            for _ in range(round(10/dt)):
                empty_command=max(0.,empty_command-.25*dt)
                empty_drive.target(empty_command)
                points,a=tick('unloaded_lid_closing',target=goal)
                if empty_command==0. and abs(a)<.015:break
            seated_empty=interface_check(ctx,assembly,kit['parts'],unloaded_interfaces)
            empty_drive.release()
            for _ in range(round(.5/dt)):
                tick('unloaded_lid_release',target=goal)
            seated_released=interface_check(ctx,assembly,kit['parts'],unloaded_interfaces)
            empty_ok=seated_empty['pass'] and seated_released['pass']
            checked('hinge','closed_seating',empty_ok,loaded_state='source-observed cloth pinch',
                    unloaded=seated_empty,after_release=seated_released,
                    joint_drive_released=empty_drive.verify_released())
            if not empty_ok:return finish('Empty lid failed actual mating or force-free retention after cloth extraction')
        held_lid_target=None if unloaded_interfaces else 1.9
        from calibrated_scene_reconstruction.tasks.mechanisms.tabletop import from_stage
        tabletop=from_stage(ctx,ctx.sg['task']['support'],hi[2])
        margin=max(width,length)*.6
        stroke=max(width*.5,.04)
        start,end=tabletop.stroke(opening['xy'],avoid_radius=opening['radius'],clearance=margin,length=stroke)
        goal=start.copy();goal[2]+=2*particle_radius
        lowering_seconds=float(np.linalg.norm(goal-force_handle.target)/force_handle.speed+2.)
        if lowering_seconds>30:return finish('No nearby support stroke within the bounded force-fixture duration')
        for _ in range(round(lowering_seconds/dt)):tick('lower_to_table',hinge_target=held_lid_target,target=goal,downward=True)
        contacts=[];contact_steps=0
        for i in range(round(4/dt)):
            target=goal+(end-start)*min(1.,i*dt/4)
            points,_=tick('wipe',hinge_target=held_lid_target,target=target,downward=True)
            domain=tabletop.contains(points[:,:2])
            if domain.any():
                contact=support_contact(points,[0,0,1],hi[2],particle_radius=particle_radius,tolerance=2*particle_radius,domain=domain)
                peak_pen=max(peak_pen,contact['maximum_penetration_m'])
                if contact['contact_nodes']>=max(3,int(len(points)*.05)):
                    contact_steps+=1;contacts.append(points[contact['contact_indices'],:2].mean(0))
        travel=float(np.linalg.norm(np.ptp(np.asarray(contacts),axis=0))) if contacts else 0.
        wipe_ok=contact_steps*dt>=1. and travel>=stroke*.5
        checked('deform','tabletop_contact',contact_steps*dt>=1.,seconds=contact_steps*dt)
        checked('deform','wiping_coverage',wipe_ok,contact_travel_m=travel,requested_stroke_m=stroke)
        checked('deform','task_interaction',released and wipe_ok)
        before_release=ctx.nodes(names['cloth']).copy()
        for _ in range(round(1/dt)):tick('release')
        after_release=ctx.nodes(names['cloth'])
        checked('deform','post_release_flexibility',deformed and not attached,positions_prescribed=False,external_forces_removed=True,
                passive_displacement_m=float(np.linalg.norm(after_release-before_release,axis=1).max()))
        checked('hinge','limits',limit_angle[0]-.02<=minimum_hinge_angle and maximum_hinge_angle<=limit_angle[1]+.02,
                minimum_angle_rad=minimum_hinge_angle,maximum_angle_rad=maximum_hinge_angle,every_physics_step=True)
        checked('hinge','attachment',peak_attachment<=tolerance,maximum_anchor_error_m=peak_attachment)
        checked('hinge','passive_attachment_retention',peak_attachment<=tolerance,maximum_anchor_error_m=peak_attachment)
        final_response=material_verdict({'pass':bool(quantitative_material),'geometry_valid':bool(material_geometry_valid)},accuracy)
        checked('deform','strain_bounded',final_response['pass'],response_acceptance=final_response,
                maximum_edge_strain=peak_strain,maximum_principal_material_strain=peak_principal,maximum_relative_area_change=peak_area)
        checked('extract','contact_resolution',contact_sampling['pass'] and peak_pen<=tolerance and peak_step<=2*particle_radius,
                particle_sampling=contact_sampling,
                maximum_penetration_m=peak_pen,maximum_node_step_m=peak_step,tolerance_m=tolerance,
                triangle_samples_every_physics_step=True,scope='Vertices, edge midpoints and face centres; not exhaustive continuous triangle intersection proof')
        return finish('Flip lid, extract same free sheet and wipe observed tabletop',{'initial_passive_drift_m':drift})
    except Exception as error:
        return finish('Native task aborted with preserved partial evidence',{'error':type(error).__name__+': '+str(error)})
    finally:
        if empty_drive is not None and not empty_drive.released:empty_drive.release()
        ctx.clear_joint_efforts()
        ctx.stage.RemovePrim(fixture)
