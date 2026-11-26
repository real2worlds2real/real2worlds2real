"""Reuse a reviewed native scene, change initial state and estimate bounded inputs.

The source scene's rigid, articulated and particle models remain intact. All
paths, material observations and input pulse windows belong to the agent plan.
"""
import cv2
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

from calibrated_scene_reconstruction.physics.process.contract import read,require,write




def create(world,stage,parameters,inputs):
    import omni.usd
    import omni.physx
    import carb
    if getattr(world,'raw_physx',False):
        from calibrated_scene_reconstruction.sim.process_world import SingleRigidPrim
    else:
        from isaacsim.core.prims import SingleRigidPrim,RigidPrim
    from pxr import Gf,Sdf,Usd,UsdGeom,UsdPhysics,UsdShade,PhysxSchema,PhysicsSchemaTools,Vt

    cfg=read(inputs['scene_setup']);prefix='/World/Replay'
    supported={'object.mass','pair.kinetic_friction','pair.static_ratio','pair.restitution',
               'push.peak_force','push.skew','push.heading','push.onset','push.release','push.height'}
    contact_parameters=cfg.get('contact_parameters',{})
    mass_parameters=cfg.get('mass_parameters',{})
    initial_pose_parameters=cfg.get('initial_pose_parameters',{})
    grip_couples=cfg.get('grip_couples',{})
    lateral_parameter=cfg.get('lateral_offset_parameter')
    optional_parameters={lateral_parameter} if lateral_parameter else set()
    require(set(parameters)==supported|set(contact_parameters)|set(mass_parameters)|optional_parameters|set(initial_pose_parameters)|set(grip_couples),'Select or extend the model explicitly for additional physical quantities')
    require(all(item['body']==cfg['target'] and parameters[k]>=0 for k,item in grip_couples.items()),
            'Declare nonnegative grip forces on the explicitly selected native target')
    lateral_offset=parameters[lateral_parameter] if lateral_parameter else 0.
    require(np.isfinite(lateral_offset) and abs(lateral_offset)<cfg['contact_radius_m'],
            'Bounded lateral contact offset must lie on the declared source contact surface')
    root=UsdGeom.Xform.Define(stage,prefix)
    root.GetPrim().GetReferences().AddReference(inputs['native_scene'],'/World')
    source_stage=Usd.Stage.Open(inputs['native_scene'])
    source_physics=next(p for p in source_stage.Traverse() if p.IsA(UsdPhysics.Scene))
    owned_physics=next(p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene))
    for attr in source_physics.GetAttributes():
        if attr.HasAuthoredValueOpinion() and attr.GetName().startswith('physxScene:') and attr.GetName()!='physxScene:timeStepsPerSecond':
            owned_physics.CreateAttribute(attr.GetName(),attr.GetTypeName(),custom=False).Set(attr.Get())
    owned_physics.CreateAttribute('physxScene:timeStepsPerSecond',Sdf.ValueTypeNames.UInt,custom=False).Set(round(1/world.get_physics_dt()))
    # A /World reference omits a source scene authored beside /World. USD
    # discards those external targets; reconnect the same native scene explicitly.
    for original in Usd.PrimRange(source_stage.GetPrimAtPath('/World')):
        prim=stage.GetPrimAtPath(prefix+str(original.GetPath())[len('/World'):])
        for relationship in original.GetRelationships():
            targets=relationship.GetTargets()
            if source_physics.GetPath() in targets:
                prim.GetRelationship(relationship.GetName()).SetTargets([
                    owned_physics.GetPath() if p==source_physics.GetPath() else
                    p.ReplacePrefix(Sdf.Path('/World'),Sdf.Path(prefix)) for p in targets])

    def matrix(prim,value):
        xform=UsdGeom.Xformable(prim);xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(Gf.Matrix4d(np.asarray(value,float).T.tolist()))

    matrix(root.GetPrim(),cfg['world_from_source'])
    if 'initial_state' in inputs:
        state=np.load(inputs['initial_state'],allow_pickle=False)
        def pose_matrix(p):
            value=np.eye(4);value[:3,:3]=Rotation.from_quat(p[[4,5,6,3]]).as_matrix();value[:3,3]=p[:3];return value
        saved=dict(zip(state['rigid_names'],state['rigid_poses']))
        current_target=stage.GetPrimAtPath(prefix+cfg['bodies'][cfg['target']]['path'])
        if 'initial_anchor_pose' in cfg:
            parent=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(current_target.GetParent())).T
            matrix(current_target,np.linalg.inv(parent)@np.asarray(cfg['initial_anchor_pose']))
        target_matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(current_target)).T
        angle=Rotation.from_euler('z',cfg.get('contents_azimuth',0.)).as_matrix()
        center=np.array(cfg.get('contents_rotation_center',[0,0,0]),float)
        spin=np.eye(4);spin[:3,:3]=angle;spin[:3,3]=center-angle@center
        transfer=target_matrix@spin@np.linalg.inv(pose_matrix(saved[cfg['state_anchor']]))
        for change in cfg.get('state_assemblies',[]):
            anchor=stage.GetPrimAtPath(prefix+change['anchor']);assembly=stage.GetPrimAtPath(prefix+change['assembly'])
            cache=UsdGeom.XformCache();old=np.asarray(cache.GetLocalToWorldTransform(anchor)).T
            desired=transfer@pose_matrix(saved[change['state_name']])
            parent_world=np.asarray(cache.GetLocalToWorldTransform(assembly.GetParent())).T
            assembly_world=np.asarray(cache.GetLocalToWorldTransform(assembly)).T
            matrix(assembly,np.linalg.inv(parent_world)@desired@np.linalg.inv(old)@assembly_world)
        for item in cfg['bodies'].values():
            name=item.get('state_name')
            if not name or name==cfg['state_anchor']:continue
            prim=stage.GetPrimAtPath(prefix+item['path'])
            desired=transfer@pose_matrix(saved[name])
            parent=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent())).T
            matrix(prim,np.linalg.inv(parent)@desired)
        for name,(start,count) in zip(state['particle_names'],state['particle_ranges']):
            prim=stage.GetPrimAtPath(prefix+cfg['state_particle_paths'][str(name)])
            desired=state['particle_positions'][start:start+count]@transfer[:3,:3].T+transfer[:3,3]
            M=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T
            local=(desired-M[:3,3])@np.linalg.inv(M[:3,:3]).T
            prim.GetAttribute('points').Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(local,np.float32)))
    for change in cfg.get('initial_assemblies',[]):
        anchor=stage.GetPrimAtPath(prefix+change['anchor']);parent=stage.GetPrimAtPath(prefix+change['assembly'])
        cache=UsdGeom.XformCache()
        current=np.asarray(cache.GetLocalToWorldTransform(anchor)).T
        delta=np.asarray(change['pose'])@np.linalg.inv(current)
        parent_world=np.asarray(cache.GetLocalToWorldTransform(parent)).T
        ancestor=np.asarray(cache.GetLocalToWorldTransform(parent.GetParent())).T
        matrix(parent,np.linalg.inv(ancestor)@delta@parent_world)

    context=world.get_physics_context()
    if cfg.get('gpu_dynamics'):
        context.enable_gpu_dynamics(True);context.set_broadphase_type('GPU')
    context.set_solver_type('TGS')
    bindings={name:{'path':prefix+item['path'],'kind':item['kind']} for name,item in cfg['bodies'].items()}
    if initial_pose_parameters:
        from calibrated_scene_reconstruction.physics.process.initial_pose import apply
        pose_audit=apply(stage,bindings,initial_pose_parameters,parameters,prefix=prefix)
        write(Path(inputs['_process_context']['trial_directory'])/'initial_pose_parameters.json',pose_audit)
    target=bindings[cfg['target']]['path']
    from calibrated_scene_reconstruction.physics.process.mass_bindings import apply as bind_masses,audit as audit_masses
    mass_rows=bind_masses(stage,bindings,mass_parameters,parameters,reserved=[cfg['target']])
    if mass_rows:write(Path(inputs['_process_context']['trial_directory'])/'additional_mass_bindings.json',mass_rows)
    contact_readbacks={};contact_links=[];contact_targets=set()
    for name,item in contact_parameters.items():
        require(item['attribute'] in {'physics:staticFriction','physics:dynamicFriction','physics:restitution'}
                and item.get('materials') and parameters[name]>=0,'Declare bounded contact material parameters explicitly')
        values=[]
        for relative in item['materials']:
            prim=stage.GetPrimAtPath(prefix+relative)
            require(bool(prim) and prim.HasAPI(UsdPhysics.MaterialAPI),'Contact parameter target must be an existing physics material')
            require((relative,item['attribute']) not in contact_targets,'Contact parameter bindings overlap')
            contact_targets.add((relative,item['attribute']))
            attribute=prim.GetAttribute(item['attribute']);require(bool(attribute),'Missing source contact attribute')
            attribute.Set(float(parameters[name]));values.append(float(attribute.Get()))
            for link in item.get('linked',[]):
                require(link['attribute'] in {'physics:staticFriction','physics:dynamicFriction','physics:restitution'}
                        and np.isfinite(link['factor']) and link['factor']>0,'Declare a positive linked material prior ratio')
                require((relative,link['attribute']) not in contact_targets,'Linked contact parameter bindings overlap')
                contact_targets.add((relative,link['attribute']))
                attr=prim.GetAttribute(link['attribute']);require(bool(attr),'Missing linked contact attribute')
                attr.Set(float(parameters[name]*link['factor']))
                contact_links.append({'parameter':name,'path':str(prim.GetPath()),'attribute':link['attribute'],
                                      'factor':link['factor'],'value':float(attr.Get())})
        contact_readbacks[name]=values
    mass=UsdPhysics.MassAPI(stage.GetPrimAtPath(target));previous=float(mass.GetMassAttr().Get())
    requested=parameters['object.mass'];mass.GetMassAttr().Set(requested)
    inertia=mass.GetDiagonalInertiaAttr().Get()
    if inertia is not None:mass.GetDiagonalInertiaAttr().Set(Gf.Vec3f(*((np.asarray(inertia)*requested/previous).tolist())))
    mu=parameters['pair.kinetic_friction'];static=mu*parameters['pair.static_ratio']
    material=UsdShade.Material.Define(stage,'/World/ReplayPairMaterial')
    physical=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physical.CreateStaticFrictionAttr(static);physical.CreateDynamicFrictionAttr(mu);physical.CreateRestitutionAttr(parameters['pair.restitution'])
    pair_schema=PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    binding_mode=cfg.get('pair_material_binding','both_average')
    require(binding_mode in {'both_average','support_min'},'Explicit supported contact-material binding required')
    source_materials=[]
    if binding_mode=='support_min':
        for prim in Usd.PrimRange(stage.GetPrimAtPath(target)):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
            original=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')[0]
            require(bool(original),'Target contact material prior is missing')
            api=UsdPhysics.MaterialAPI(original.GetPrim());combine=PhysxSchema.PhysxMaterialAPI(original.GetPrim()).GetFrictionCombineModeAttr().Get()
            values={'path':str(original.GetPath()),'dynamic':float(api.GetDynamicFrictionAttr().Get()),
                    'static':float(api.GetStaticFrictionAttr().Get()),'combine':combine or 'average'}
            require(values['dynamic']>=mu and values['static']>=static and values['combine'] in {'average','min'},
                    'Support-min binding cannot realize requested pair friction while preserving target contact priors')
            source_materials.append(values)
        pair_schema.CreateFrictionCombineModeAttr('min');pair_schema.CreateRestitutionCombineModeAttr('min')
        binding_paths=[prefix+cfg['support']]
    else:
        pair_schema.CreateFrictionCombineModeAttr('average');binding_paths=[target,prefix+cfg['support']]
    for path in binding_paths:
        for prim in Usd.PrimRange(stage.GetPrimAtPath(path)):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,UsdShade.Tokens.strongerThanDescendants,'physics')

    marker=np.load(inputs['marker_mesh'],allow_pickle=False)
    vertices,faces=marker['V'],marker['F']
    camera=dict(np.load(inputs['process_camera'],allow_pickle=False))
    group_faces={};group_surfaces={}
    landmarks={**cfg.get('initial_landmarks',{}),**cfg.get('rigid_landmarks',{})}
    material_binding_checks=[]
    for source in cfg.get('material_feature_bindings',[]):
        from calibrated_scene_reconstruction.physics.process.surface_features import validate_rendered_binding
        require(source in inputs,'Native material correspondence receipt must be a declared input')
        receipt=read(inputs[source]);validate_rendered_binding(receipt,landmarks)
        material_binding_checks.append(source)
    contact_patch_rows=[]
    for row in cfg.get('contact_patch_priors',[]):
        require(row.get('reason') and row['maximum_radius_m']>0 and row['path'].startswith('/World/'),
                'Finite native contact-patch priors need a reviewed local geometry bound')
        prim=stage.GetPrimAtPath(prefix+row['path'][len('/World'):])
        require(prim and prim.HasAPI(UsdPhysics.CollisionAPI),'Contact patch must bind an actual native collision shape')
        values={name:float(prim.GetAttribute(name).Get()) for name in row['values']}
        require(set(values)=={'physxCollision:torsionalPatchRadius','physxCollision:minTorsionalPatchRadius'}
                and all(0<=v<=row['maximum_radius_m'] and np.isclose(v,row['values'][k]) for k,v in values.items()),
                'Native contact-patch readback differs from declared bounded prior')
        contact_patch_rows.append({'path':str(prim.GetPath()),'values':values,'reason':row['reason']})
    feature_probes={}
    for name,item in cfg.get('feature_sets',{}).items():
        require(np.allclose(camera['dist'],0,rtol=0,atol=1e-9),
                'Straight projected-line observations require undistorted source images/camera; do not fit radial curvature as part motion')
        require(item['encoding']=='line_probes' and item.get('members') and len(set(item['members']))==len(item['members'])
                and all(k in landmarks and len(landmarks[k]['points'])==2 for k in item['members']),
                'Line probe sets need explicit two-point rigid landmarks')
        if item.get('require_distinct_bodies',False):
            require(len({landmarks[k]['body'] for k in item['members']})==len(item['members']),
                    'Independent line observations must bind independent physical bodies')
        data=np.load(inputs[item['probes']],allow_pickle=False)
        require(data['y'].shape==(len(camera['t']),2) and np.isfinite(data['y']).all()
                and np.allclose(data['t'],camera['t'],rtol=0,atol=1e-8),'Source line measurement grid/timebase mismatch')
        feature_probes[name]=data['y']
    for name,group in cfg.get('landmark_groups',{}).items():
        group_triangles=[];offset=0
        for key in group['bodies']:
            prim=stage.GetPrimAtPath(bindings[key]['path'])
            face=np.asarray(prim.GetAttribute('faceVertexIndices').Get(),int).reshape(-1,3)
            group_triangles.append(face+offset);offset+=len(prim.GetAttribute('points').Get())
        group_surfaces[name]=[]
        for surface in group.get('surfaces',[]):
            require(bindings[surface['body']]['kind']=='rigid','Declare rigid material surfaces separately from flexible point groups')
            mesh=np.load(inputs[surface['mesh']],allow_pickle=False);V=np.asarray(mesh['V']);F=np.asarray(mesh['F'],int)
            require(V.ndim==2 and V.shape[1]==3 and np.isfinite(V).all() and F.ndim==2 and F.shape[1]==3
                    and F.min()>=0 and F.max()<len(V),'Invalid observed native material subset mesh')
            group_surfaces[name].append((surface['body'],V));group_triangles.append(F+offset);offset+=len(V)
        group_faces[name]=np.concatenate(group_triangles)
    initial_measurements=[]
    body=None;contact_view=None;commands=[];contact_depths=[];contact_unknown=[];contact_counts=[];contact_peaks=[];warm_contact_depths=[]
    couple_commands={k:[] for k in grip_couples}
    rigid_paths=[item['path'] for item in bindings.values() if item['kind']=='rigid']
    if getattr(world,'raw_physx',False):
        world.contact_diagnostic_path=Path(inputs['_process_context']['trial_directory'])/'unknown_contact_examples.json'
        for path in rigid_paths:
            PhysxSchema.PhysxContactReportAPI.Apply(stage.GetPrimAtPath(path)).CreateThresholdAttr(0.)
    static_colliders=[str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)
                      and not any(str(p.GetPath()).startswith(path+'/') for path in rigid_paths)]
    filters=rigid_paths+static_colliders
    def numpy(value):
        return value.detach().cpu().numpy() if hasattr(value,'detach') else np.asarray(value)

    def initialize():
        nonlocal body,contact_view
        body=SingleRigidPrim(target,reset_xform_properties=False);body.initialize()
        if getattr(world,'raw_physx',False):
            world.contact_monitor()
            return
        contact_view=RigidPrim(rigid_paths,contact_filter_prim_paths_expr=[filters for _ in rigid_paths],
            max_contact_count=32768,reset_xform_properties=False,disable_stablization=False,name='process_contacts')
        contact_view.initialize()

    def after_step():
        if getattr(world,'raw_physx',False):
            contact_depths.append(max(0.,-min(world.contacts,default=0.)))
            contact_unknown.append(len(world.contact_unknown));contact_counts.append(len(world.contacts))
            if world.contact_cache and contact_depths[-1]>max((r['depth_m'] for r in contact_peaks),default=0.):
                key,item=min(world.contact_cache.items(),key=lambda pair:min(pair[1]['distances']))
                contact_peaks.append({'t':len(contact_depths)*world.get_physics_dt(),'depth_m':contact_depths[-1],
                    'actors':[str(PhysicsSchemaTools.intToSdfPath(p)) for p in key[:2]],
                    'colliders':[str(PhysicsSchemaTools.intToSdfPath(p)) for p in key[2:]],
                    'metric':'Native reported contact separation; not an independent post-solver exact-mesh distance.'})
            return
        data=contact_view.get_contact_force_data(dt=world.get_physics_dt())
        distances=numpy(data[3]).reshape(-1);counts=numpy(data[4]).reshape(len(rigid_paths),len(filters))
        starts=numpy(data[5]).reshape(counts.shape);peak=0.
        for i,j in np.argwhere(counts>0):
            start,count=int(starts[i,j]),int(counts[i,j]);require(start>=0 and start+count<=len(distances),'Invalid native contact ranges')
            if rigid_paths[i]==filters[j]:continue
            peak=max(peak,max(0.,-float(np.min(distances[start:start+count]))))
        contact_depths.append(peak)

    def drive(t):
        onset,release=parameters['push.onset'],parameters['push.release']
        u=np.clip((t-onset)/(release-onset),0,1)
        magnitude=parameters['push.peak_force']*np.sin(np.pi*u)*(1+parameters['push.skew']*(1-2*u))
        if not onset<=t<=release:magnitude=0.
        direction=np.array([np.cos(parameters['push.heading']),np.sin(parameters['push.heading']),0.])
        p,q=body.get_world_pose();p,q=numpy(p),numpy(q)
        R=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
        local=np.asarray(cfg.get('contact_center_local',[0,0,0]))+np.array([0,0,parameters['push.height']])
        from calibrated_scene_reconstruction.physics.process.wrench import body_side_contact,circular_grip
        point=body_side_contact(p,R,local,direction,cfg['contact_radius_m'],lateral_offset)
        force=magnitude*direction
        if magnitude:
            omni.physx.get_physx_simulation_interface().apply_force_at_pos(omni.usd.get_context().get_stage_id(),
                PhysicsSchemaTools.sdfPathToInt(Sdf.Path(target)),carb.Float3(*force.tolist()),carb.Float3(*point.tolist()),'Force')
        commands.append([t,*force,*point])
        for key,item in grip_couples.items():
            envelope=magnitude/parameters['push.peak_force']
            points,forces=circular_grip(p,R,item['center_local'],item['normal_local'],item['radii'],
                                       item['heights'],parameters[key]*envelope)
            for position,grip_force in zip(points,forces):
                if np.any(grip_force):
                    omni.physx.get_physx_simulation_interface().apply_force_at_pos(omni.usd.get_context().get_stage_id(),
                        PhysicsSchemaTools.sdfPathToInt(Sdf.Path(target)),carb.Float3(*grip_force.tolist()),carb.Float3(*position.tolist()),'Force')
            couple_commands[key].append([t,*forces[0],*points[0],*forces[1],*points[1]])

    def after_settle_step():
        after_step()
        warm_contact_depths.append(contact_depths[-1])

    def observe(t,state):
        p=state[cfg['target']]
        R=Rotation.from_quat(p[[4,5,6,3]]).as_matrix()
        frame=int(np.argmin(abs(camera['t']-t)))
        uv,extent=project_material(vertices,faces,p[:3],R,(camera['K'],camera['dist'],camera['T_world_cam'][frame]),
                                  opening=cfg['observation_opening'])
        output={'marker_uv':uv,'marker_extent':extent}
        for name,group in cfg.get('landmark_groups',{}).items():
            pieces=[state[k].reshape(-1,3) for k in group['bodies']]
            for key,V in group_surfaces[name]:
                pose=state[key];R=Rotation.from_quat(pose[[4,5,6,3]]).as_matrix();pieces.append(V@R.T+pose[:3])
            points=np.concatenate(pieces)
            center,extent=project_material(points,group_faces[name],np.zeros(3),np.eye(3),
                (camera['K'],camera['dist'],camera['T_world_cam'][frame]),opening=group.get('opening',(3,3)))
            output[name]=center
            if group.get('extent_output'):output[group['extent_output']]=extent
        projected_landmarks={}
        for name,item in landmarks.items():
            pose=state[item['body']];rotation=Rotation.from_quat(pose[[4,5,6,3]]).as_matrix()
            xyz=np.asarray(item['points'])@rotation.T+pose[:3]
            T=camera['T_world_cam'][frame];Rcw=T[:3,:3].T
            uv=cv2.projectPoints(xyz,cv2.Rodrigues(Rcw)[0],-Rcw@T[:3,3],camera['K'],camera['dist'])[0].reshape(-1,2)
            projected_landmarks[name]=uv
            if item['feature']=='point':output[name]=uv[0]
            elif item['feature']=='axis_angle':
                delta=uv[1]-uv[0];output[name]=np.array([np.arctan2(delta[0],delta[1])])
            else:raise ValueError('Unsupported source initial landmark feature: '+item['feature'])
        for name,item in cfg.get('feature_sets',{}).items():
            from calibrated_scene_reconstruction.physics.process.line_features import line_probes
            values=[]
            for member in item['members']:
                values.append(line_probes(projected_landmarks[member],feature_probes[name][frame]))
            output[name]=np.asarray(values)
        if cfg.get('initial_checks'):initial_measurements.append({'t':t,**output})
        return output

    def checks(trace):
        command=np.asarray(commands)
        np.savez_compressed(Path(inputs['_process_context']['trial_directory'])/'external_wrench.npz',commands=command)
        np.savez_compressed(Path(inputs['_process_context']['trial_directory'])/'native_contacts.npz',
                            maximum_penetration_m=contact_depths,unknown_pairs=contact_unknown,contact_counts=contact_counts)
        from calibrated_scene_reconstruction.physics.process.contract import write
        write(Path(inputs['_process_context']['trial_directory'])/'contact_peaks.json',contact_peaks)
        if getattr(world,'raw_physx',False):
            write(Path(inputs['_process_context']['trial_directory'])/'unknown_contact_examples.json',
                  list(world.contact_unknown_examples.values()))
        outside=(command[:,0]<parameters['push.onset'])|(command[:,0]>parameters['push.release'])
        clearance=[]
        for record in trace:
            for name,binding in bindings.items():
                if binding['kind']!='rigid':continue
                # Source-grounded body origins alone do not certify contact surfaces.
                clearance.append(record['bodies'][name][2])
        limit=inputs['_process_context']['physical_limits']['max_rigid_penetration_m']
        checks={'finite_native_response':{'pass':bool(np.isfinite(clearance).all()),'reason':'Recorded source rigid and particle states remain finite.'},
            'rigid_contacts':{'pass':bool(contact_depths and max(contact_depths)<=limit
                                          and not any(contact_unknown)
                                          and (not getattr(world,'raw_physx',False) or world.contact_events>0)),
                'reason':f'Native every-step rigid contact maximum={max(contact_depths,default=0.):.8g} m; declared limit={limit:.8g} m; unknown contact steps={sum(n>0 for n in contact_unknown)}.'},
            'contact_window':{'pass':bool(np.all(command[outside,1:4]==0)),
                              'reason':'External push is zero outside the source-bounded contact window.'}}
        if mass_rows:
            checks['additional_mass_readback']={'pass':audit_masses(stage,mass_rows),
                'reason':'Declared additional rigid mass priors/estimates and principal inertias bound together and read back from native USD schemas. COM and principal axes unchanged; this alone does not establish unique mass identification.'}
        if grip_couples:
            arrays={k:np.asarray(v) for k,v in couple_commands.items()}
            np.savez_compressed(Path(inputs['_process_context']['trial_directory'])/'external_couples.npz',**arrays)
            finite=all(np.isfinite(v).all() for v in arrays.values())
            balanced=all(np.allclose(v[:,1:4]+v[:,7:10],0,rtol=0,atol=1e-10) for v in arrays.values())
            bounded=all(np.all(v[outside,1:4]==0) and np.all(v[outside,7:10]==0) for v in arrays.values())
            checks['contact_window']['pass'] &= bool(bounded)
            checks['external_contact_couples']={'pass':bool(finite and balanced and bounded),
                'reason':'Recorded paired inward forces on reviewed circular side sections are finite, equal/opposite, and zero outside the source contact window. Contact geometry and force magnitude remain conditional human-grip priors, not simulated hand measurements.'}
        if cfg.get('initial_checks'):
            from calibrated_scene_reconstruction.physics.process.initial_state import source_metrics
            source=np.load(inputs['initial_source_features'],allow_pickle=False);rows={}
            for item in cfg['initial_checks']:
                predicted=np.asarray([r[item['predicted']] for r in initial_measurements])
                observed=np.asarray(source[item['observed']])
                metrics=source_metrics(predicted,observed,cfg['initial_split'],angular=item.get('angular',False))
                rows[item['id']]={'pass':max(metrics.values())<=item['tolerance'],'tolerance':item['tolerance'],**metrics}
            write(Path(inputs['_process_context']['trial_directory'])/'initial_source_checks.json',rows)
            checks['initial_state_source']={'pass':bool(rows) and all(r['pass'] for r in rows.values()),
                'reason':'Independent initial part tip/axis and flexible-region source observations checked after native settling: '+str(rows)}
        if cfg.get('aperture_check'):
            from calibrated_scene_reconstruction.physics.process.initial_state import aperture_clearance,mesh_aperture_clearance
            aperture=cfg['aperture_check'];clearances={}
            for item in aperture['shafts']:
                values=[]
                mesh=np.load(inputs[item['mesh']],allow_pickle=False) if item.get('mesh') else None
                for record in trace:
                    states=record['bodies'];container=states[aperture['container']];pose=states[item['body']]
                    Rc=Rotation.from_quat(container[[4,5,6,3]]).as_matrix()
                    R=Rotation.from_quat(pose[[4,5,6,3]]).as_matrix()
                    points=(np.asarray(mesh['V'] if mesh is not None else item['points'])@R.T+pose[:3]-container[:3])@Rc
                    if mesh is not None:
                        value=mesh_aperture_clearance(points,mesh['F'],center=aperture['center'],height=aperture['height'],opening_radius=aperture['radius'])
                    else:
                        value=aperture_clearance(*points,item['radius'],center=aperture['center'],height=aperture['height'],opening_radius=aperture['radius'])
                    values.append(value)
                clearances[item['body']]=min(values)
            write(Path(inputs['_process_context']['trial_directory'])/'aperture_entries.json',clearances)
            checks['aperture_entries']={'pass':bool(clearances) and min(clearances.values())>=-limit,
                'reason':'Finite shaft sections remain inside the modeled open rim during native response. Minimum conservative clearances (m): '+str(clearances)}
        if binding_mode=='support_min':
            preserved=True
            for saved in source_materials:
                api=UsdPhysics.MaterialAPI(stage.GetPrimAtPath(saved['path']))
                preserved &= np.isclose(api.GetDynamicFrictionAttr().Get(),saved['dynamic']) and np.isclose(api.GetStaticFrictionAttr().Get(),saved['static'])
            write(Path(inputs['_process_context']['trial_directory'])/'contact_parameter_scope.json',
                {'binding':binding_mode,'target_material_priors':source_materials,'target_priors_preserved':bool(preserved),
                 'requested_support_pair_dynamic':mu,'requested_support_pair_static':static,
                 'scope':'Support material uses min combine within target coefficients. Support fitting does not rewrite target/contents materials; any separately declared contact parameters are reported independently.'})
            checks['contact_parameter_scope']={'pass':bool(source_materials) and bool(preserved),
                'reason':'Target material prior retained; support friction bound only to support with explicit min combine. Separately planned material parameters are read back independently.'}
        if contact_parameters:
            write(Path(inputs['_process_context']['trial_directory'])/'contact_parameter_readbacks.json',contact_readbacks)
            write(Path(inputs['_process_context']['trial_directory'])/'contact_parameter_links.json',contact_links)
            checks['contact_parameter_readback']={'pass':all(np.allclose(v,parameters[k],rtol=1e-6) for k,v in contact_readbacks.items())
                and all(np.isclose(r['value'],parameters[r['parameter']]*r['factor'],rtol=1e-6) for r in contact_links),
                'reason':'Explicit agent-planned contact material parameters bound and read back from their declared native materials; unspecified materials unchanged.'}
        if cfg.get('check_initial_contacts'):
            write(Path(inputs['_process_context']['trial_directory'])/'initial_native_contacts.json',
                  {'maximum_penetration_m':max(warm_contact_depths,default=None),'steps':len(warm_contact_depths),
                   'scope':'Native rigid contact reports during every preparation settling step, before source time zero.'})
            checks['initial_native_contacts']={'pass':bool(warm_contact_depths) and max(warm_contact_depths)<=limit,
                'reason':'Every-step native rigid contacts include the settling phase; maximum='+str(max(warm_contact_depths,default=None))+' m.'}
        if material_binding_checks:
            checks['material_feature_correspondence']={'pass':True,
                'reason':'Asset-local material points match independent native rendered identities; bound evidence hashes are current: '+str(material_binding_checks)}
        write(Path(inputs['_process_context']['trial_directory'])/'contact_patch_priors.json',contact_patch_rows)
        checks['contact_patch_prior_readback']={'pass':True,
            'reason':('Explicit finite-contact torsional friction priors read back on native colliders, bounded by reviewed part radius. No added constraints or restoring forces.'
                      if contact_patch_rows else 'No additional finite-contact patch priors declared; reviewed source contact settings retained.')}
        return checks

    handles={'bodies':bindings,'driver_bodies':[],'consumed_parameters':list(parameters),
        'after_reset':initialize,'after_step':after_step,'drive':drive,'settle':lambda:None,'observe':observe,'checks':checks,
        'effective_parameters':{'values':parameters,'source':inputs['native_scene'],'limitations':cfg['limitations']}}
    if cfg.get('marker_region'):
        from calibrated_scene_reconstruction.physics.process.region_observations import definition
        marker_region=cfg['marker_region']
        require(marker_region['scope']=='complete_part',
                'Material projection cannot advertise a photometric subset or occluded surface measurement')
        handles['observation_definitions']={'marker_uv':definition(marker_region['part'],marker_region['scope'],
            'centroid',cfg['observation_opening'])}
    if cfg.get('check_initial_contacts'):handles['after_settle_step']=after_settle_step
    return handles

from calibrated_scene_reconstruction.physics.process.material_projection import project_material
