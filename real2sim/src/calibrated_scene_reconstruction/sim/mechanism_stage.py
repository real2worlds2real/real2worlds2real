"""Isaac Sim 5.1 authoring for physical links and real joints.

Contact mechanisms (snaps, gears) have no kinematic joint between their
mating bodies. Tests may apply forces; delivered assets contain no task drives.
"""
import numpy as np
import trimesh
import hashlib
import json
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade, Vt

from calibrated_scene_reconstruction.assets.material_mass import link_mass
from calibrated_scene_reconstruction.assets.metric_mesh import sdf_settings
from calibrated_scene_reconstruction.sim.usd import add_mesh, color_material, textured_material
from calibrated_scene_reconstruction.core.paths import resolve


def add_links(stage, path, kit, spec, object_spec, physics_material, *, look_base=None):
    root = UsdGeom.Xform.Define(stage, path).GetPrim()
    names = {part['name'] for part in kit['parts']}
    from calibrated_scene_reconstruction.assets.mechanisms import joint_components
    components = joint_components({p['name']: {'kind': p.get('kind', 'rigid')} for p in kit['parts']},
                                  kit.get('joints', []))
    whole_tree = len(components) == 1 and set(components[0]['links']) == names
    if whole_tree:
        UsdPhysics.ArticulationRootAPI.Apply(root)
        PhysxSchema.PhysxArticulationAPI.Apply(root).CreateEnabledSelfCollisionsAttr(True)
    volumes = {p['name']: float(trimesh.Trimesh(p['V'], p['F'], process=True).volume)
               for p in kit['parts'] if p.get('kind', 'rigid') == 'rigid'}
    face_uv, face_offset = None, 0
    look = spec.get('look')
    if look:
        if not look.get('preserves_links') or look_base is None:
            raise ValueError('A flattened appearance mesh cannot replace declared moving links')
        if hashlib.sha256(resolve(spec['mesh']).read_bytes()).hexdigest() != look.get('source_mesh_sha256'):
            raise ValueError('Link appearance belongs to a different physical surface')
        with np.load(look_base/look['mesh']) as data:
            face_uv = data['face_uv']
        if len(face_uv) != sum(len(p['F']) for p in kit['parts']):
            raise ValueError('Link appearance face count differs from physical geometry')
    for part in kit['parts']:
        body_path = f"{path}/{part['name']}"
        body = UsdGeom.Xform.Define(stage, body_path).GetPrim()
        material = part['material']
        mat = color_material(stage, f"/World/Looks/{path.split('/')[-1]}_{part['name']}",
                             material['color'], roughness=material.get('roughness', .5))
        uv = part.get('UV')
        if face_uv is not None:
            uv = face_uv[face_offset:face_offset+len(part['F'])]
            mat = textured_material(stage, f"/World/Looks/{path.split('/')[-1]}_{part['name']}_texture",
                                     look_base/look['albedo'], roughness=look.get('roughness', .5),
                                     tint=(float(look.get('gain', 1.)),)*3)
        face_offset += len(part['F'])
        mesh = add_mesh(stage, body_path + '/surface', part['V'], part['F'], uv, mat,normals=part.get('normals'))
        from calibrated_scene_reconstruction.sim.object_materials import bind_surface_materials
        bind_surface_materials(stage, mesh, part, look=look if face_uv is not None else None, look_base=look_base)
        if part.get('kind', 'rigid') != 'rigid':
            add_deformable(stage, mesh, part, spec.get('link_physics', {}).get(part['name'], {}), physics_material)
            continue
        # Collision cooking may normalize or partition a source solid. Mass
        # remains based on the original physical surface, not cooking coordinates.
        mesh.GetPrim().SetCustomDataByKey('calibrated_scene_reconstruction:mass_source_surface', True)
        UsdPhysics.RigidBodyAPI.Apply(body)
        PhysxSchema.PhysxContactReportAPI.Apply(body).CreateThresholdAttr(0.)
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(body)
        rb.CreateEnableCCDAttr(True)
        rb.CreateSolverPositionIterationCountAttr(32)
        rb.CreateSolverVelocityIterationCountAttr(1)
        volume = volumes[part['name']]
        link_settings=spec.get('link_physics', {}).get(part['name'], {})
        kg, why = link_mass(volume, object_spec, link_settings, sum(volumes.values()))
        UsdPhysics.MassAPI.Apply(body).CreateMassAttr(kg)
        from .source_moments import author as source_moments
        source_moments(body,part['V'],part['F'],kg)
        body.SetCustomDataByKey('calibrated_scene_reconstruction:mass_from', why)
        settings = sdf_settings(link_settings.get('sdf', spec.get('sdf')))
        approximation=link_settings.get('collision_approximation','sdf')
        if approximation not in {'sdf','convexDecomposition'}:
            raise ValueError('Explicit supported rigid mesh collision approximation required')
        colliders=[(mesh,approximation,link_settings.get('convex_cooking'),1.)]
        representation=link_settings.get('collision_representation','surface')
        slicing=link_settings.get('collision_slices')
        if slicing is not None and (not isinstance(slicing,dict) or set(slicing)-{'max_span_m','axis','maximum_pieces','dimensions'} or 'max_span_m' not in slicing):
            raise ValueError('Explicit spatial collision slicing parameters required')
        if slicing is not None and (link_settings.get('collision_groups') is not None or representation!='surface' or approximation!='sdf'):
            raise ValueError('Spatial SDF cuts cannot be combined with another collision partition')
        if representation not in {'surface','revolved_source_convex'}:
            raise ValueError('Unknown source collision representation')
        if representation=='revolved_source_convex' and link_settings.get('collision_groups') is not None:
            raise ValueError('Choose source revolution cells or material partitions, not both')
        if link_settings.get('collision_groups') is not None or representation=='revolved_source_convex' or slicing is not None:
            from calibrated_scene_reconstruction.assets.collision_groups import partition
            colliders=[]
            groups=[]
            if representation=='revolved_source_convex':
                from calibrated_scene_reconstruction.assets.revolved_source_cells import cells as revolution_cells
                conditioning=link_settings.get('revolution_normalization_scale','minimum_extent')
                if conditioning not in {'minimum_extent','maximum_extent','axis_extents'}:
                    raise ValueError('Explicit supported source-cell coordinate conditioning required')
                source_groups=[{**cell,'name':'source_cell_'+str(i),'materials':part.get('material_names',[]),
                                'approximation':'convexHull','normalize_coordinates':True,
                                'align_principal_axes':True,'normalization_scale':conditioning,
                                'cooking':{'hull_vertex_limit':64,'minimum_thickness_m':1e-7}}
                               for i,cell in enumerate(revolution_cells(part['V'],part['F'],axis=link_settings.get('revolution_axis')))]
            elif slicing is not None:
                from calibrated_scene_reconstruction.assets.collision_slices import partition as spatial_partition
                source_groups,split_report=spatial_partition(part,**slicing)
                body.SetCustomDataByKey('calibrated_scene_reconstruction:spatial_collision_slices',json.dumps(split_report))
            else:source_groups=partition(part,link_settings['collision_groups'])
            for group in source_groups:
                if group.get('max_cell_aspect_ratio') is None:
                    groups.append(group)
                else:
                    from calibrated_scene_reconstruction.assets.convex_source_solids import cells
                    for index,cell in enumerate(cells(group['V'],group['F'],max_aspect_ratio=group['max_cell_aspect_ratio'])):
                        groups.append({**group,**cell,'name':group['name']+'_cell_'+str(index)})
            for group in groups:
                vertices=group['V'];extent=1.;centre=np.zeros(3)
                if group.get('normalize_coordinates',False):
                    from calibrated_scene_reconstruction.assets.collision_groups import normalize
                    vertices,centre,basis,extent=normalize(vertices,principal_axes=group.get('align_principal_axes',False),
                                                          scale_mode=group.get('normalization_scale','maximum_extent'))
                collider=add_mesh(stage,body_path+'/colliders/'+group['name'],vertices,group['F'],None,mat)
                if group.get('normalize_coordinates',False):
                    transform=UsdGeom.Xformable(collider)
                    transform.AddTranslateOp().Set(Gf.Vec3d(*map(float,centre)))
                    if group.get('align_principal_axes',False):
                        from scipy.spatial.transform import Rotation
                        q=Rotation.from_matrix(basis).as_quat()
                        transform.AddOrientOp().Set(Gf.Quatf(float(q[3]),Gf.Vec3f(*map(float,q[:3]))))
                    scale=np.broadcast_to(extent,(3,))
                    transform.AddScaleOp().Set(Gf.Vec3f(*map(float,scale)))
                collider.CreateVisibilityAttr('invisible')
                collider.GetPrim().SetCustomDataByKey('calibrated_scene_reconstruction:source_materials',','.join(group['materials']))
                colliders.append((collider,group['approximation'],group.get('cooking'),extent))
        for collider,method,cooking,coordinate_scale in colliders:
            prim=collider.GetPrim()
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(method)
            if method=='sdf':
                PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim).CreateSdfResolutionAttr(settings['resolution'])
            if cooking is not None:
                if method=='convexHull':
                    from calibrated_scene_reconstruction.sim.convex_cooking import apply_hull
                    apply_hull(prim,cooking,coordinate_scale)
                elif method=='convexDecomposition':
                    from calibrated_scene_reconstruction.sim.convex_cooking import apply as apply_convex_cooking
                    apply_convex_cooking(prim,cooking)
                else:raise ValueError('Convex cooking controls require a convex approximation')
            collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collision.CreateContactOffsetAttr(settings['contact_offset'])
            collision.CreateRestOffsetAttr(settings['rest_offset'])
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(physics_material, UsdShade.Tokens.strongerThanDescendants, 'physics')
    for index, joint in enumerate(kit.get('joints', [])):
        kind = joint.get('type')
        if kind not in {'revolute', 'prismatic', 'fixed'}:
            raise ValueError('Only actual fixed/revolute/prismatic joints are authored; contact is not a joint')
        a, b = joint.get('parent'), joint.get('child')
        if a not in names or b not in names or a == b:
            raise ValueError('Joint parent/child must reference two declared links')
        cls = {'revolute': UsdPhysics.RevoluteJoint, 'prismatic': UsdPhysics.PrismaticJoint,
               'fixed': UsdPhysics.FixedJoint}[kind]
        usd_joint = cls.Define(stage, f'{path}/joints/joint_{index}')
        usd_joint.CreateBody0Rel().SetTargets([f'{path}/{a}'])
        usd_joint.CreateBody1Rel().SetTargets([f'{path}/{b}'])
        for side in (0, 1):
            point = joint.get(f'anchor{side}')
            if point is None or np.asarray(point).shape != (3,) or not np.isfinite(point).all():
                raise ValueError('Joint anchors required in each link local frame, metres')
            getattr(usd_joint, f'CreateLocalPos{side}Attr')(Gf.Vec3f(*point))
        from calibrated_scene_reconstruction.assets.mechanisms import joint_frames
        for side,q in enumerate(joint_frames(joint)):
            getattr(usd_joint,f'CreateLocalRot{side}Attr')(Gf.Quatf(float(q[0]),Gf.Vec3f(*q[1:])))
        if kind != 'fixed':
            axis = np.asarray(joint.get('axis'), float)
            limits = np.asarray(joint.get('limits'), float)
            if axis.shape != (3,) or not np.isfinite(axis).all() or abs(np.linalg.norm(axis)-1) > 1e-6:
                raise ValueError('Joint axis must be normalized')
            if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] >= limits[1]:
                raise ValueError('Finite ordered joint limits required')
            usd_joint.CreateAxisAttr('X')
            factor = 180 / np.pi if kind == 'revolute' else 1.
            usd_joint.CreateLowerLimitAttr(float(limits[0]*factor))
            usd_joint.CreateUpperLimitAttr(float(limits[1]*factor))
            if 'initial_position' in joint:
                state=PhysxSchema.JointStateAPI.Apply(usd_joint.GetPrim(),'angular' if kind=='revolute' else 'linear')
                state.CreatePositionAttr(float(joint['initial_position']*factor))
                state.CreateVelocityAttr(0.)
        # Only adjacent genuinely joined links suppress collision. All contact
        # mechanism pairs and other links keep their real surface collisions.
        usd_joint.CreateCollisionEnabledAttr(bool(joint.get('collision_enabled', False)))
    if not whole_tree:
        # A floating articulation can explicitly root at its rigid base. Keep
        # loose cloth/rigid siblings outside it without changing link paths.
        for component in components:
            prim = stage.GetPrimAtPath(f"{path}/{component['root']}")
            UsdPhysics.ArticulationRootAPI.Apply(prim)
            PhysxSchema.PhysxArticulationAPI.Apply(prim).CreateEnabledSelfCollisionsAttr(True)
    # A USD per-joint collision flag cannot enable adjacent articulation
    # contacts in PhysX. Keep requested contacts effective with an equivalent
    # welded collision carrier, preserving source mass moments and all DOFs.
    from .contact_carrier import author_requested
    author_requested(stage,path,kit.get('joints',[]),
                     protected_owners=[path+'/'+binding['rigid'] for binding in kit.get('surface_bindings',[])])
    from calibrated_scene_reconstruction.sim.surface_attachment import author_many as root_bonds
    records=[]
    for index, binding in enumerate(kit.get('surface_bindings',[])):
        part=next(p for p in kit['parts'] if p['name']==binding['surface'])
        surface=f"{path}/{binding['surface']}/surface"
        rigid=f"{path}/{binding['rigid']}"
        matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(surface)))
        points=np.asarray(part['V'])@matrix[:3,:3]+matrix[3,:3]
        owner=next(p for p in kit['parts'] if p['name']==binding['rigid'])
        records.append({'path':f'{path}/root_bond_{index}','surface':surface,'rigid':rigid,'points':points,
            'faces':part['F'],'indices':binding['vertices'],'basis':binding['basis'],'rigid_geometry':owner,
            'clamp':binding.get('clamp'),
            'backend':spec.get('link_physics',{}).get(binding['rigid'],{}).get('surface_root_backend','direct')})
    particle=[];surface=[]
    for record in records:
        part=next(p for p in kit['parts'] if record['surface']==f"{path}/{p['name']}/surface")
        (particle if part.get('backend','particle')=='particle' else surface).append(record)
    if particle:
        from .particle_attachment import author_many as particle_bonds
        particle_bonds(stage,particle)
    if surface:root_bonds(stage,surface)
    from .contact_carrier import audit_requested
    root.SetCustomDataByKey('calibrated_scene_reconstruction:joint_collision_requests',json.dumps(audit_requested(stage,path)))
    return root


def add_deformable(stage, mesh, part, settings, physics_material):
    """Native 5.1 FEM solid/rope or particle cloth; no animated skin substitute.

    Production uses the legacy solid FEM and particle APIs shipped with5.1.
    Its beta surface-FEM APIs are evaluated separately before any explicit
    backend change; spring and continuum material constants are not aliases.
    """
    from omni.physx.scripts import deformableUtils, particleUtils
    scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if len(scenes) != 1:
        raise ValueError('Flexible bodies require one explicit physics scene')
    # Avoid the legacy TGS velocity-averaging path, which can strongly attenuate
    # localized flexible-body impulses even with all material damping disabled.
    # This setting is serialized with the scene; runtime force adapters audit it.
    scene_api = PhysxSchema.PhysxSceneAPI.Apply(scenes[0])
    scene_api.CreateEnableExternalForcesEveryIterationAttr(scene_api.GetSolverTypeAttr().Get() == 'TGS')
    path = mesh.GetPath()
    if part['kind']=='cloth' and part.get('cloth_rest'):
        mesh.GetPrim().SetCustomDataByKey('calibrated_scene_reconstruction:cloth_rest',json.dumps(part['cloth_rest']))
        from calibrated_scene_reconstruction.assets.constitutive_features import usage
        mesh.GetPrim().SetCustomDataByKey('calibrated_scene_reconstruction:constitutive_authoring',
            json.dumps(usage(part['cloth_rest'],part.get('backend','particle'))))
    if part['kind']=='cloth' and part.get('backend','particle')=='surface_fem':
        from calibrated_scene_reconstruction.sim.surface_cloth import author
        author(stage,mesh,part,settings)
        mesh.GetPrim().SetCustomDataByKey('calibrated_scene_reconstruction:physical_kind','cloth')
        return
    if part['kind'] in {'deformable', 'rope'}:
        if part.get('backend','legacy')=='volume_fem':
            from calibrated_scene_reconstruction.sim.volume_initial_state import author
            if not part.get('volume_initial'):raise ValueError('Explicit conforming volume reference and initial state required')
            author(stage,mesh,part['volume_initial'],settings,backend='volume_fem')
            return
        required = {'youngs_modulus', 'poissons_ratio', 'density', 'dynamic_friction', 'resolution'}
        if not required.issubset(settings):
            raise ValueError(f"{part['name']}: explicit FEM material and resolution required; state assumptions")
        E, nu, rho, friction = (settings[k] for k in ('youngs_modulus', 'poissons_ratio', 'density', 'dynamic_friction'))
        if not np.isfinite([E, nu, rho, friction]).all() or E <= 0 or not -1 < nu < .5 or rho <= 0 or friction < 0:
            raise ValueError('Invalid FEM material')
        resolution = settings['resolution']
        if type(resolution) is not int or not 4 <= resolution <= 128:
            raise ValueError('FEM resolution must be an integer 4..128')
        from calibrated_scene_reconstruction.assets.volume_contact import settings as volume_contact_settings
        contact_profile=volume_contact_settings(settings)
        ok = deformableUtils.add_physx_deformable_body(
            stage, path, kinematic_enabled=False, simulation_hexahedral_resolution=resolution,
            collision_simplification=False, solver_position_iteration_count=contact_profile['solver_position_iterations'], self_collision=True)
        material_path = str(path) + '_fem_material'
        ok = ok and deformableUtils.add_deformable_body_material(
            stage, material_path, youngs_modulus=E, poissons_ratio=nu, density=rho,
            dynamic_friction=friction, damping_scale=settings.get('damping_scale', .05))
        if not ok:
            raise RuntimeError('Isaac 5.1 FEM authoring failed')
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(
            UsdShade.Material.Get(stage, material_path), UsdShade.Tokens.weakerThanDescendants, 'physics')
        if 'contact_offset' in contact_profile:
            collider=PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
            collider.CreateContactOffsetAttr(contact_profile['contact_offset'])
            collider.CreateRestOffsetAttr(contact_profile['rest_offset'])
            mesh.GetPrim().SetCustomDataByKey('calibrated_scene_reconstruction:volume_contact_profile',json.dumps(contact_profile))
    elif part['kind'] == 'cloth':
        required = {'contact_offset', 'stretch_stiffness', 'bend_stiffness', 'shear_stiffness', 'damping', 'friction'}
        if not required.issubset(settings) or not np.isfinite([settings[k] for k in required]).all():
            raise ValueError('Explicit finite cloth material, thickness/contact offset and mass required')
        if min(settings[k] for k in required) <= 0:
            raise ValueError('Cloth parameters must be positive')
        if len({'mass', 'areal_density'}.intersection(settings)) != 1:
            raise ValueError('Cloth needs exactly one explicit mass or areal density')
        from calibrated_scene_reconstruction.assets.cloth_rest import area
        material_points = (part.get('cloth_rest') or {}).get('points', part['V'])
        material_area = area(material_points, part['F'])
        mass_kg = settings.get('mass', settings.get('areal_density', 0.)*material_area)
        if not np.isfinite(mass_kg) or mass_kg <= 0:
            raise ValueError('Cloth material prior gives invalid mass')
        iterations = settings.get('solver_position_iterations', 32)
        if type(iterations) is not int or not 32 <= iterations <= 256:
            raise ValueError('Cloth solver_position_iterations must be an integer32..256')
        rest_offset = settings.get('rest_offset', settings['contact_offset']*.5)
        solid_rest_offset = settings.get('solid_rest_offset', rest_offset)
        if (not np.isfinite([rest_offset, solid_rest_offset]).all()
                or not 0 < rest_offset < settings['contact_offset']
                or not 0 < solid_rest_offset < settings['contact_offset']):
            raise ValueError('Explicit cloth rigid/self-contact rest offsets must lie between0 and contact_offset')
        owner = next((p.GetPath() for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)), None)
        if owner is None:
            raise ValueError('Cloth requires a GPU physics scene')
        maximum_depenetration=settings.get('max_depenetration_velocity',.05)
        if (not np.isfinite(maximum_depenetration) or not 0<maximum_depenetration<=10):
            raise ValueError('Cloth needs a finite positive native depenetration speed at most10m/s')
        from .particle_contacts import system as contact_system,group as contact_group
        system=contact_system(stage,path,settings,owner)
        particle_group=contact_group(stage,system) if settings.get('shared_particle_system') else None
        rest = part.get('cloth_rest')
        if rest is None:
            particleUtils.add_physx_particle_cloth(
                stage, path, None, system.GetPath(), spring_stretch_stiffness=settings['stretch_stiffness'],
                spring_bend_stiffness=settings['bend_stiffness'], spring_shear_stiffness=settings['shear_stiffness'],
                spring_damping=settings['damping'], self_collision=True, self_collision_filter=True,particle_group=particle_group)
        else:
            from calibrated_scene_reconstruction.assets.cloth_rest import validate, constraints
            validate(rest, part['V'], part['F'])
            pairs, lengths, stiffness, kinds = constraints(rest, settings)
            normals = trimesh.Trimesh(part['V'], part['F'], process=False).vertex_normals
            particleUtils.add_physx_particle_cloth_with_constraints(
                stage, path, system.GetPath(), Vt.Vec3fArray.FromNumpy(np.asarray(part['V'], np.float32)),
                Vt.Vec3fArray.FromNumpy(np.asarray(normals, np.float32)),
                Vt.Vec3fArray.FromNumpy(np.asarray(rest['points'], np.float32)),
                Vt.Vec3fArray.FromNumpy(np.zeros_like(part['V'], dtype=np.float32)),
                Vt.IntArray(np.asarray(part['F']).ravel().tolist()),
                Vt.Vec2iArray.FromNumpy(pairs), Vt.FloatArray(stiffness),
                Vt.FloatArray([float(settings['damping'])]*len(pairs)), Vt.FloatArray(lengths.tolist()),
                self_collision=True, self_collision_filter=True,particle_group=particle_group)
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
            mesh.GetPrim().SetCustomDataByKey('calibrated_scene_reconstruction:cloth_spring_kinds', json.dumps(kinds))
            mesh.GetPrim().SetCustomDataByKey('calibrated_scene_reconstruction:cloth_spring_model', rest.get('spring_model','triangle_edges'))
            if rest.get('bend_memory') is not None:
                mesh.GetPrim().SetCustomDataByKey('calibrated_scene_reconstruction:cloth_bend_memory', json.dumps(rest['bend_memory']))
        from calibrated_scene_reconstruction.assets.cloth_mass import authoring_mass
        mass_cooking=authoring_mass(mass_kg,material_points,part['F'])
        UsdPhysics.MassAPI.Apply(mesh.GetPrim()).CreateMassAttr(mass_cooking['authoring_mass_kg'])
        mesh.GetPrim().SetCustomDataByKey('calibrated_scene_reconstruction:cloth_mass_cooking',json.dumps(mass_cooking))
    else:
        raise ValueError(f"Unsupported physical kind {part['kind']}; rigid fallback refused")
    mesh.GetPrim().SetCustomDataByKey('calibrated_scene_reconstruction:physical_kind', part['kind'])


def audit(stage, declaration,*,sampling_tolerances=None):
    """Inspect the exported stage, rather than trusting only source metadata."""
    bodies, paths = {}, {}
    if any(link['kind'] != 'rigid' for obj in declaration['objects'].values() for link in obj['links'].values()):
        scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
        if len(scenes) != 1:
            raise ValueError('Flexible task scene must have one physics scene')
        scene_api = PhysxSchema.PhysxSceneAPI(scenes[0])
        expected = scene_api.GetSolverTypeAttr().Get() == 'TGS'
        if bool(scene_api.GetEnableExternalForcesEveryIterationAttr().Get()) != expected:
            raise ValueError('Flexible scene external-force iteration setting differs from its native solver')
    for obj, spec in declaration['objects'].items():
        root = stage.GetPrimAtPath(f'/World/Objects/{obj}')
        if not root:
            raise ValueError(f'Missing declared object {obj}')
        for name, link in spec['links'].items():
            candidate = root if name == 'model' and root.HasAPI(UsdPhysics.RigidBodyAPI) else stage.GetPrimAtPath(
                f'/World/Objects/{obj}/{name}')
            if not candidate:
                raise ValueError(f'Missing declared physical link {obj}/{name}')
            key = f'{obj}/{name}'
            if link['kind'] == 'rigid':
                if not candidate.HasAPI(UsdPhysics.RigidBodyAPI):
                    raise ValueError(f'{key}: no actual rigid body')
                rigid = PhysxSchema.PhysxRigidBodyAPI(candidate)
                position_iterations = rigid.GetSolverPositionIterationCountAttr().Get()
                velocity_iterations = rigid.GetSolverVelocityIterationCountAttr().Get()
                if position_iterations < 32 or velocity_iterations < 1:
                    raise ValueError(f'{key}: rigid mechanism needs at least32 position/1 velocity solver iterations')
            else:
                surface = stage.GetPrimAtPath(str(candidate.GetPath())+'/surface')
                api = PhysxSchema.PhysxParticleClothAPI if link['kind'] == 'cloth' else PhysxSchema.PhysxDeformableBodyAPI
                actual_surface=(link.get('backend','particle')=='surface_fem' and link['kind']=='cloth')
                actual_volume=(link.get('backend','legacy')=='volume_fem' and link['kind'] in {'deformable','rope'})
                supported=surface and (surface.HasAPI('OmniPhysicsSurfaceDeformableSimAPI') if actual_surface else
                                       surface.HasAPI('OmniPhysicsVolumeDeformableSimAPI') if actual_volume else surface.HasAPI(api))
                if not supported or candidate.HasAPI(UsdPhysics.RigidBodyAPI):
                    raise ValueError(f'{key}: missing native flexible representation or rigid substitution')
            paths[key] = str(candidate.GetPath())
            bodies[key] = {'path': paths[key], 'kind': link['kind']}
            if link['kind']!='rigid' and (actual_surface or actual_volume):
                from .deformable_filter_pose import audit as audit_filter_pose
                bodies[key]['self_collision_filter_pose']=audit_filter_pose(surface)
            if link['kind']=='cloth':
                bodies[key]['backend']=link.get('backend','particle')
                authored=surface.GetCustomDataByKey('calibrated_scene_reconstruction:constitutive_authoring')
                if authored:bodies[key]['constitutive_authoring']=json.loads(authored)
            if link['kind'] in {'deformable','rope'}:
                if link.get('backend')=='volume_fem':bodies[key]['backend']='volume_fem'
                explicit=surface.GetCustomDataByKey('calibrated_scene_reconstruction:volume_contact_profile')
                if explicit:
                    requested=json.loads(explicit)
                    collision=PhysxSchema.PhysxCollisionAPI(surface)
                    actual={'contact_offset':collision.GetContactOffsetAttr().Get(),
                            'rest_offset':collision.GetRestOffsetAttr().Get(),
                            'solver_position_iterations':(surface.GetAttribute('physxDeformableBody:solverPositionIterationCount').Get()
                                if link.get('backend')=='volume_fem' else PhysxSchema.PhysxDeformableBodyAPI(surface).GetSolverPositionIterationCountAttr().Get())}
                    if any(not np.isclose(actual[name],value,rtol=1e-6,atol=1e-10)
                           for name,value in requested.items()):
                        raise ValueError('Explicit native volume contact profile changed: '+key)
                    bodies[key]['volume_contact_profile']=actual
            if link['kind'] == 'rigid':
                bodies[key]['solver_iterations'] = [position_iterations, velocity_iterations]
                from .contact_carrier import verify as verify_contact_carrier
                carrier=verify_contact_carrier(stage,paths[key])
                if carrier:bodies[key]['contact_carrier']=carrier
            elif link['kind']=='cloth' and link.get('backend','particle')=='surface_fem':
                surface=stage.GetPrimAtPath(paths[key]+'/surface')
                required={'selfCollision':True,'enableSpeculativeCCD':True}
                if any(surface.GetAttribute('physxDeformableBody:'+name).Get()!=value for name,value in required.items()):
                    raise ValueError('Native surface FEM contact must remain enabled')
                count=surface.GetAttribute('physxDeformableBody:solverPositionIterationCount').Get()
                if not isinstance(count,int) or count<32:raise ValueError('Native surface FEM solver budget missing')
                bodies[key]['surface_solver_position_iterations']=count
                corrective=surface.GetAttribute('physxDeformableBody:maxDepenetrationVelocity').Get()
                if corrective is None or not np.isfinite(corrective) or not 0<corrective<=10:
                    raise ValueError('Surface FEM contact-correction velocity must be explicit and finite')
                bodies[key]['maximum_depenetration_velocity_m_s']=float(corrective)
                mesh=UsdGeom.Mesh(surface)
                from calibrated_scene_reconstruction.assets.cloth_contact import self_intersections
                crossing=self_intersections(np.asarray(mesh.GetPointsAttr().Get()),np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3))
                bodies[key]['initial_self_intersection']=crossing
                if not crossing['pass']:raise ValueError('Native surface FEM source mesh self-intersects')
            elif link['kind'] == 'cloth':
                surface = stage.GetPrimAtPath(paths[key]+'/surface')
                targets = PhysxSchema.PhysxParticleAPI(surface).GetParticleSystemRel().GetTargets()
                if len(targets) != 1:
                    raise ValueError('Native cloth must use one particle system')
                particle = PhysxSchema.PhysxParticleSystem(stage.GetPrimAtPath(targets[0]))
                actual = particle.GetSolverPositionIterationCountAttr().Get()
                if type(actual) is not int or not 32 <= actual <= 256:
                    raise ValueError('Native cloth particle solver iteration budget is absent or invalid')
                bodies[key]['particle_solver_position_iterations'] = actual
                offsets = {'contact_offset': particle.GetContactOffsetAttr().Get(),
                           'particle_contact_offset': particle.GetParticleContactOffsetAttr().Get(),
                           'rest_offset': particle.GetRestOffsetAttr().Get(),
                           'solid_rest_offset': particle.GetSolidRestOffsetAttr().Get()}
                if (not np.isfinite(list(offsets.values())).all()
                        or not 0 < offsets['rest_offset'] < offsets['contact_offset']
                        or not 0 < offsets['solid_rest_offset'] < offsets['particle_contact_offset']):
                    raise ValueError('Native cloth offsets must be explicit; automatic thickness is not verified')
                bodies[key]['particle_offsets_m'] = offsets
                depenetration=particle.GetMaxDepenetrationVelocityAttr().Get()
                if not np.isfinite(depenetration) or not 0<depenetration<=10:
                    raise ValueError('Native cloth corrective velocity budget is not explicit and finite')
                bodies[key]['maximum_depenetration_velocity_m_s']=float(depenetration)
                from calibrated_scene_reconstruction.assets.cloth_sampling import rigid_coverage
                surface_mesh=UsdGeom.Mesh(surface)
                sampling=rigid_coverage(np.asarray(surface_mesh.GetPointsAttr().Get()),
                                  np.asarray(surface_mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3),
                                  offsets,maximum_gap_m=(sampling_tolerances or {}).get(key,0.))
                bodies[key]['particle_contact_sampling']=sampling
                if not sampling['pass']:
                    raise ValueError(f'{key}: particle contact undersamples the sheet beyond its declared accuracy; refine the mesh without inflating physical thickness')
                material, _ = UsdShade.MaterialBindingAPI(particle.GetPrim()).ComputeBoundMaterial('physics')
                pbd = PhysxSchema.PhysxPBDMaterialAPI(material.GetPrim())
                ambient = {name: float(getattr(pbd, 'Get'+name.title()+'Attr')().Get())
                           for name in ('damping', 'drag', 'lift')}
                if any(value != 0. for value in ambient.values()):
                    raise ValueError('Undeclared global cloth damping/drag/lift in native material')
                bodies[key]['ambient_particle_coefficients'] = ambient
                from calibrated_scene_reconstruction.assets.cloth_contact import self_intersections
                mesh = UsdGeom.Mesh(surface)
                crossing = self_intersections(np.asarray(mesh.GetPointsAttr().Get()),
                    np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3))
                bodies[key]['initial_self_intersection'] = crossing
                if not crossing['pass']:
                    raise ValueError(f"{key}: exported cloth has {crossing['intersecting_pairs']} self-intersecting triangle pairs")
    # Inspect the authored geometry before World.reset can move penetrations
    # away. A settled runtime state cannot conceal an invalid delivered start.
    cloth_refs=[key for key,value in bodies.items() if value['kind']=='cloth']
    if cloth_refs:
        from pxr import Usd
        from calibrated_scene_reconstruction.assets.surface_distance import SignedSurface,triangle_samples
        cache=UsdGeom.XformCache()
        rigid_surfaces={}
        for key,value in bodies.items():
            if value['kind']!='rigid':continue
            mesh=UsdGeom.Mesh(stage.GetPrimAtPath(value['path']+'/surface'))
            transform=np.asarray(cache.GetLocalToWorldTransform(mesh.GetPrim()))
            rigid_surfaces[key]=(SignedSurface(np.asarray(mesh.GetPointsAttr().Get()),
                np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)),np.linalg.inv(transform))
        for key in cloth_refs:
            mesh=UsdGeom.Mesh(stage.GetPrimAtPath(bodies[key]['path']+'/surface'))
            intrinsic=mesh.GetPrim().GetCustomDataByKey('calibrated_scene_reconstruction:cloth_rest')
            validity=mesh.GetPrim().GetCustomDataByKey('calibrated_scene_reconstruction:cloth_validity')
            if declaration.get('material_state_contract',0)>=2 and (not intrinsic or not validity):
                raise ValueError(f'{key}: authored cloth requires intrinsic material and declared validity limits')
            if intrinsic and validity:
                from calibrated_scene_reconstruction.assets.cloth_metric import MaterialMetric
                limits=json.loads(validity)
                material=MaterialMetric(np.asarray(mesh.GetPointsAttr().Get()),
                    np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3),json.loads(intrinsic),
                    max_edge_strain=limits['max_edge_strain'],max_area_change=limits['max_area_change'])
                bodies[key]['authored_material_state']=material.check(np.asarray(mesh.GetPointsAttr().Get()))
                if not bodies[key]['authored_material_state']['pass']:
                    raise ValueError(f'{key}: authored sheet already exceeds material validity before native initialization')
            transform=np.asarray(cache.GetLocalToWorldTransform(mesh.GetPrim()))
            points=np.asarray(mesh.GetPointsAttr().Get())
            faces=np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
            sample=triangle_samples(points,faces)
            sample=sample@transform[:3,:3]+transform[3,:3]
            obj,link=key.split('/')
            connection=next((b for b in declaration['objects'][obj].get('surface_connections',[])
                             if b['surface']==link),None)
            roots=json.loads(mesh.GetPrim().GetCustomDataByKey('calibrated_scene_reconstruction:physical_root_indices') or '[]')
            penetration={}
            for name,(surface,inverse) in rigid_surfaces.items():
                query=sample
                if connection and name==obj+'/'+connection['rigid']:
                    # A finite glued patch is embedded in its real owner.
                    # The attachment/filter audit below verifies these nodes;
                    # every exposed face retains the original contact check.
                    free=faces[~np.isin(faces,roots).any(axis=1)]
                    if not len(free):raise ValueError('Root bond leaves no exposed material surface')
                    used,inverse_faces=np.unique(free,return_inverse=True)
                    query=triangle_samples(points[used],inverse_faces.reshape(-1,3))@transform[:3,:3]+transform[3,:3]
                penetration[name]=surface.max_penetration(query@inverse[:3,:3]+inverse[3,:3])
            bodies[key]['authored_rigid_penetration_m']=penetration
            if any(value>1e-6 for value in penetration.values()):
                raise ValueError(f'{key}: authored cloth penetrates rigid geometry before native initialization: {penetration}')
    for mechanism in declaration['mechanisms']:
        if mechanism['type'] == 'deformation':
            continue
        pair = {paths[r] for r in mechanism['links']}
        matching = []
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Joint):
                joint = UsdPhysics.Joint(prim)
                targets = {str(p) for p in (*joint.GetBody0Rel().GetTargets(), *joint.GetBody1Rel().GetTargets())}
                if targets == pair:
                    matching.append(prim)
        if mechanism['type'] == 'contact' and matching:
            raise ValueError('Exported contact mechanism was replaced by a joint')
        if mechanism['type'] in {'fixed', 'revolute', 'prismatic'} and mechanism['role'] != 'secondary':
            schema = {'fixed': UsdPhysics.FixedJoint, 'revolute': UsdPhysics.RevoluteJoint,
                      'prismatic': UsdPhysics.PrismaticJoint}[mechanism['type']]
            if len(matching) != 1 or not matching[0].IsA(schema):
                raise ValueError('Required native joint absent or wrong type in exported stage')
            if mechanism['type'] != 'fixed':
                joint = schema(matching[0])
                factor = np.pi/180 if mechanism['type'] == 'revolute' else 1.
                actual = np.array([joint.GetLowerLimitAttr().Get(), joint.GetUpperLimitAttr().Get()])*factor
                if not np.allclose(actual, mechanism['limits']):
                    raise ValueError('Native joint limits differ from declared mechanism')
    rooted={};root_filters={}
    for obj,spec in declaration['objects'].items():
        root=stage.GetPrimAtPath(f'/World/Objects/{obj}')
        from pxr import Usd
        actual=[];vertices={};partitions={}
        from .rigid_partition import OWNER,verify
        for prim in Usd.PrimRange(root):
            if 'Attachment' not in prim.GetTypeName():continue
            if prim.GetTypeName()=='PhysxPhysicsAttachment':
                from .particle_attachment import audit_bond
                data=audit_bond(stage,prim)
                if data is not None:
                    pair,ids=data;actual.append(pair);vertices.setdefault(pair,[]).extend(ids)
                continue
            if prim.GetTypeName()!='OmniPhysicsVtxXformAttachment':raise ValueError('Unreviewed native physical attachment type')
            a=prim.GetRelationship('omniphysics:src0').GetTargets();b=prim.GetRelationship('omniphysics:src1').GetTargets()
            indices=prim.GetAttribute('omniphysics:vtxIndicesSrc0').Get()
            if len(a)!=1 or len(b)!=1 or not indices:raise ValueError('Root attachment lacks actual physical endpoints')
            endpoint=stage.GetPrimAtPath(b[0]);physical_owner=endpoint.GetCustomDataByKey(OWNER)
            if physical_owner:
                if physical_owner not in partitions:partitions[physical_owner]=verify(stage,physical_owner)
                info=partitions[physical_owner]
                if not info or str(b[0]) not in info['carriers'] or prim.GetCustomDataByKey(OWNER)!=physical_owner:
                    raise ValueError('Unverified numerical frame at physical root')
            pair=(str(a[0]),physical_owner or str(b[0]));actual.append(pair)
            vertices.setdefault(pair,[]).extend(map(int,indices))
        expected=[(f"/World/Objects/{obj}/{b['surface']}/surface",f"/World/Objects/{obj}/{b['rigid']}")
                  for b in spec.get('surface_connections',[])]
        if sorted(set(actual))!=sorted(expected):raise ValueError('Exported root bonds differ from physical construction plan')
        for pair,ids in vertices.items():
            declared=stage.GetPrimAtPath(pair[0]).GetCustomDataByKey('calibrated_scene_reconstruction:physical_root_indices')
            if len(ids)!=len(set(ids)) or declared and sorted(ids)!=sorted(json.loads(declared)):
                raise ValueError('Physical root vertex coverage changed or duplicated')
        from .surface_attachment import audit_filters
        from .particle_attachment import audit_filters as particle_filters
        root_filters[obj]=[(particle_filters if stage.GetPrimAtPath(pair[0]).HasAPI(PhysxSchema.PhysxParticleClothAPI)
                           else audit_filters)(stage,pair[0],pair[1],ids) for pair,ids in vertices.items()]
        rooted[obj]=len(actual)
    from .contact_carrier import audit_requested
    joint_contacts={obj:audit_requested(stage,f'/World/Objects/{obj}') for obj in declaration['objects']}
    return {'pass': True, 'links': bodies, 'rooted_connections':rooted,'joint_collision_requests':joint_contacts,
            'root_contact_filters':root_filters,'contact_pairs_have_no_substitute_joint': True}


def passive_audit(stage, declaration,*,sampling_tolerances=None):
    result = audit(stage, declaration,sampling_tolerances=sampling_tolerances)
    for obj in declaration['objects']:
        from pxr import Usd
        root = stage.GetPrimAtPath('/World/Objects/'+obj)
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get():
                raise ValueError('Delivered task bodies cannot use kinematic pose control')
            for attr in prim.GetAttributes():
                if attr.GetNumTimeSamples() and (attr.GetName().startswith('xformOp:') or attr.GetName() == 'points'):
                    raise ValueError('Delivered mechanisms cannot be animated pose/shape substitutes')
            for axis in ('angular', 'linear'):
                if prim.HasAPI(UsdPhysics.DriveAPI, axis):
                    drive = UsdPhysics.DriveAPI(prim, axis)
                    if drive.GetTargetVelocityAttr().Get() not in (None, 0.):
                        raise ValueError('Remove active task-test drives before delivery')
    fixture = stage.GetPrimAtPath('/World/TaskCheckFixtures')
    if fixture and fixture.GetChildren():
        raise ValueError('Remove temporary task-test fixtures before delivery')
    return {**result, 'passive': True, 'test_controls_removed': True}
