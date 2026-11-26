"""Apply explicit agent physical estimates to actual Isaac 5.1 bodies/materials."""
import numpy as np
import json
from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade, Vt, Sdf

from calibrated_scene_reconstruction.assets.physical_priors import sample, validate
from calibrated_scene_reconstruction.sim.physics import physics_material, bind_colliders


def body_path(stage, ref):
    obj, link = ref.split('/')
    root = stage.GetPrimAtPath('/World/Objects/'+obj)
    return str(root.GetPath()) if root and link == 'model' and root.HasAPI(UsdPhysics.RigidBodyAPI) else f'/World/Objects/{obj}/{link}'


def apply(stage, plan, mechanisms, case='nominal'):
    values = sample(validate(plan, mechanisms), case)
    report = {'case': case, 'bodies': {}, 'mechanisms': {}, 'supports': {},
              'identification': 'prior estimates; feasibility does not identify real properties'}
    # One native particle system has one friction material. Reject a sampled
    # plan that asks shared cloth sets to have different friction, before any
    # material is overwritten by whichever link happens to run last.
    shared_friction={};shared_thickness={}
    from .particle_contacts import KEY as shared_key
    for ref,estimate in values['bodies'].items():
        prim=stage.GetPrimAtPath(body_path(stage,ref)+'/surface')
        if not prim or not prim.HasAPI(PhysxSchema.PhysxParticleAPI):continue
        targets=PhysxSchema.PhysxParticleAPI(prim).GetParticleSystemRel().GetTargets()
        if len(targets)!=1:continue
        system=stage.GetPrimAtPath(targets[0])
        if not system.GetCustomDataByKey(shared_key):continue
        friction=float(estimate['parameters']['dynamic_friction'])
        name=str(targets[0]);before=shared_friction.setdefault(name,friction)
        if not np.isclose(before,friction,rtol=0.,atol=1e-12):
            raise ValueError('One shared particle contact system cannot implement different per-cloth friction priors')
        if 'thickness' in estimate['parameters']:
            h=float(estimate['parameters']['thickness']);before=shared_thickness.setdefault(name,h)
            if not np.isclose(before,h,rtol=0.,atol=1e-12):
                raise ValueError('One shared particle contact system cannot implement different per-cloth thickness priors')
    for ref, estimate in values['bodies'].items():
        path = body_path(stage, ref)
        prim = stage.GetPrimAtPath(path)
        if not prim:
            raise ValueError('Physical prior body absent: '+ref)
        p = estimate['parameters']
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            from calibrated_scene_reconstruction.sim.mass_properties import rigid_volume
            volume = rigid_volume(prim)
            kg = p['mass'] if 'mass' in p else p['density']*volume
            if not np.isfinite(kg) or kg <= 0:
                raise ValueError('Physical prior gives invalid actual mass')
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(kg))
            from .source_moments import update as update_source_moments
            source_moments=update_source_moments(prim,kg)
            rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            rb.CreateLinearDampingAttr(float(p['linear_damping']))
            rb.CreateAngularDampingAttr(float(p['angular_damping']))
            from .rigid_partition import update as update_partition
            update_partition(stage,path,kg,linear_damping=p['linear_damping'],angular_damping=p['angular_damping'])
            from .contact_carrier import update as update_contact_carrier
            carrier=update_contact_carrier(stage,path,kg,linear_damping=p['linear_damping'],angular_damping=p['angular_damping'])
            mat = physics_material(stage, path+'_physical_material', p['static_friction'], p['dynamic_friction'],
                                   p['restitution'], friction_combine='average')
            bind_colliders(stage, [path,*([carrier['carrier']] if carrier else [])], mat)
            from calibrated_scene_reconstruction.assets.compliant_contact import settings as compliant_contact
            compliant = compliant_contact(p)
            if compliant:
                # A declared finite contact stiffness (assets.compliant_contact): the native implicit
                # contact spring of this body's material. Restitution is then unused by the backend.
                native_material = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
                native_material.CreateCompliantContactStiffnessAttr(compliant['stiffness_N_per_m'])
                native_material.CreateCompliantContactDampingAttr(compliant['damping_N_s_per_m'])
            report['bodies'][ref] = {**p, 'actual_mass_kg': float(kg), 'solid_volume_m3': float(volume),
                                     'effective_density_kg_m3': float(kg/volume) if volume > 0 else None,
                                     'inertia_model': 'Uniform density on generated solid; unseen material layout is not identified'}
            if compliant:report['bodies'][ref]['compliant_contact']=compliant
            if source_moments is not None:report['bodies'][ref]['source_mass_moments']=source_moments
            if carrier:report['bodies'][ref]['contact_carrier']=carrier
        else:
            mesh = stage.GetPrimAtPath(path+'/surface')
            if mechanisms['objects'][ref.split('/')[0]]['links'][ref.split('/')[1]]['kind']=='cloth':
                validity=plan['bodies'][ref].get('validity_hypothesis')
                if mechanisms.get('material_state_contract',0)>=2 and (not validity or not {'max_edge_strain','max_area_change'}<=validity.keys()):
                    raise ValueError('Cloth priors require explicit strain and area validity limits')
                if validity:mesh.SetCustomDataByKey('calibrated_scene_reconstruction:cloth_validity',json.dumps(validity))
            material, _ = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial('physics')
            if mesh.HasAPI('OmniPhysicsSurfaceDeformableSimAPI'):
                from calibrated_scene_reconstruction.sim.surface_cloth import apply_material
                from calibrated_scene_reconstruction.assets.surface_contact_profile import sample as contact_sample
                reference=mesh.GetCustomDataByKey('calibrated_scene_reconstruction:surface_contact_reference')
                if reference is None:
                    reference={'thickness':material.GetPrim().GetAttribute('omniphysics:surfaceThickness').Get(),
                               'contact_offset':PhysxSchema.PhysxCollisionAPI(mesh).GetContactOffsetAttr().Get(),
                               'rest_offset':PhysxSchema.PhysxCollisionAPI(mesh).GetRestOffsetAttr().Get(),
                               'self_filter_distance':mesh.GetAttribute('physxDeformableBody:selfCollisionFilterDistance').Get()}
                    mesh.SetCustomDataByKey('calibrated_scene_reconstruction:surface_contact_reference',json.dumps(reference))
                else:reference=json.loads(reference)
                contacts=contact_sample(reference,p['thickness'])
                collision=PhysxSchema.PhysxCollisionAPI.Apply(mesh)
                collision.CreateContactOffsetAttr(contacts['contact_offset']);collision.CreateRestOffsetAttr(contacts['rest_offset'])
                mesh.GetAttribute('physxDeformableBody:selfCollisionFilterDistance').Set(contacts['self_filter_distance'])
                surface=UsdGeom.Mesh(mesh);rest=np.asarray(mesh.GetAttribute('omniphysics:restShapePoints').Get())
                faces=np.asarray(surface.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
                material=apply_material(stage,mesh,{**p,'friction':p['dynamic_friction']},rest,faces)
                report['bodies'][ref]={**p,'native_total_mass_kg':material['mass_kg'],
                    'rest_area_m2':material['rest_area_m2'],'mass_model':'Uniform native node masses, validated by independent rigid-contact controls; not direct mass readback',
                    'sampled_surface_contacts':contacts,
                    'backend':'surface_fem'}
                continue
            if mesh.HasAPI('OmniPhysicsVolumeDeformableSimAPI'):
                from calibrated_scene_reconstruction.assets.tetra_boundary import determinants
                for name,attribute in [('density','density'),('youngs_modulus','youngsModulus'),
                                       ('poissons_ratio','poissonsRatio'),('dynamic_friction','dynamicFriction')]:
                    material.GetPrim().GetAttribute('omniphysics:'+attribute).Set(float(p[name]))
                mesh.GetAttribute('physxDeformableBody:linearDamping').Set(float(p['linear_damping']))
                rest=np.asarray(mesh.GetAttribute('omniphysics:restShapePoints').Get(),float)
                cells=np.asarray(mesh.GetAttribute('omniphysics:restTetVtxIndices').Get(),int)
                volume=float(abs(determinants(rest,cells)).sum()/6);kg=volume*p['density']
                mesh.GetAttribute('omniphysics:mass').Set(float(kg))
                report['bodies'][ref]={**p,'backend':'volume_fem','reference_volume_m3':volume,
                    'authored_total_mass_kg':float(kg),'direct_mass_readback':False}
                continue
            if mesh.HasAPI(PhysxSchema.PhysxDeformableBodyAPI):
                api = PhysxSchema.PhysxDeformableBodyMaterialAPI.Apply(material.GetPrim())
                for name, setter in (('youngs_modulus', 'CreateYoungsModulusAttr'),
                                     ('poissons_ratio', 'CreatePoissonsRatioAttr'), ('density', 'CreateDensityAttr'),
                                     ('damping_scale', 'CreateDampingScaleAttr'), ('dynamic_friction', 'CreateDynamicFrictionAttr')):
                    getattr(api, setter)(float(p[name]))
            elif mesh.HasAPI(PhysxSchema.PhysxParticleClothAPI):
                from calibrated_scene_reconstruction.assets.cloth_rest import area
                cloth = PhysxSchema.PhysxParticleClothAPI(mesh)
                rest = cloth.GetRestPointsAttr().Get()
                surface = UsdGeom.Mesh(mesh)
                if rest is None or not len(rest):
                    rest = surface.GetPointsAttr().Get()
                faces = np.asarray(surface.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
                material_area = area(np.asarray(rest), faces)
                kg = p['mass'] if 'mass' in p else p['areal_density']*material_area
                if not np.isfinite(kg) or kg <= 0:
                    raise ValueError('Cloth material prior gives invalid native mass')
                kinds_json = mesh.GetCustomDataByKey('calibrated_scene_reconstruction:cloth_spring_kinds')
                if kinds_json is not None:
                    from calibrated_scene_reconstruction.assets.cloth_rest import KINDS
                    kinds = json.loads(kinds_json)
                    api = PhysxSchema.PhysxParticleClothAPI(mesh)
                    if (len(kinds) != len(api.GetSpringIndicesAttr().Get())
                            or any(kind not in KINDS for kind in kinds)):
                        raise ValueError('Explicit cloth material springs differ from their physical prior mapping')
                    api.CreateSpringStiffnessesAttr(Vt.FloatArray([float(p[k+'_stiffness']) for k in kinds]))
                    api.CreateSpringDampingsAttr(Vt.FloatArray([float(p['spring_damping'])]*len(kinds)))
                else:
                    api = PhysxSchema.PhysxAutoParticleClothAPI(mesh)
                    for name, setter in (('stretch_stiffness', 'CreateSpringStretchStiffnessAttr'),
                                         ('bend_stiffness', 'CreateSpringBendStiffnessAttr'),
                                         ('shear_stiffness', 'CreateSpringShearStiffnessAttr'),
                                         ('spring_damping', 'CreateSpringDampingAttr')):
                        getattr(api, setter)(float(p[name]))
                from calibrated_scene_reconstruction.assets.cloth_mass import authoring_mass
                mass_cooking=authoring_mass(kg,np.asarray(rest),faces)
                UsdPhysics.MassAPI.Apply(mesh).CreateMassAttr(mass_cooking['authoring_mass_kg'])
                mesh.SetCustomDataByKey('calibrated_scene_reconstruction:cloth_mass_cooking',json.dumps(mass_cooking))
                system_path = PhysxSchema.PhysxParticleAPI(mesh).GetParticleSystemRel().GetTargets()[0]
                system = stage.GetPrimAtPath(system_path)
                if 'thickness' in p:
                    from calibrated_scene_reconstruction.assets.particle_contact_profile import sample as particle_contacts
                    system_api=PhysxSchema.PhysxParticleSystem(system)
                    reference=system.GetCustomDataByKey('calibrated_scene_reconstruction:particle_contact_reference')
                    if reference is None:
                        reference={'thickness':plan['bodies'][ref]['parameters']['thickness']['nominal'],
                            'rest_offset':system_api.GetRestOffsetAttr().Get(),
                            'solid_rest_offset':system_api.GetSolidRestOffsetAttr().Get(),
                            'contact_offset':system_api.GetContactOffsetAttr().Get(),
                            'particle_contact_offset':system_api.GetParticleContactOffsetAttr().Get()}
                        system.SetCustomDataByKey('calibrated_scene_reconstruction:particle_contact_reference',json.dumps(reference))
                    else:reference=json.loads(reference)
                    contacts=particle_contacts(reference,p['thickness'])
                    for name,value in contacts.items():
                        setter='Create'+''.join(word.title() for word in name.split('_'))+'Attr'
                        getattr(system_api,setter)(value)
                pbd_material, _ = UsdShade.MaterialBindingAPI(system).ComputeBoundMaterial('physics')
                PhysxSchema.PhysxPBDMaterialAPI.Apply(pbd_material.GetPrim()).CreateFrictionAttr(float(p['dynamic_friction']))
            else:
                raise ValueError('Flexible prior has no native solver representation')
            report['bodies'][ref] = dict(p)
            if mesh.HasAPI(PhysxSchema.PhysxParticleClothAPI):
                report['bodies'][ref].update(native_total_mass_kg=float(kg), rest_area_m2=material_area,
                                            effective_areal_density_kg_m2=float(kg/material_area),
                                            native_mass_cooking=mass_cooking)
                if 'thickness' in p:report['bodies'][ref]['sampled_particle_contacts']=contacts
    for mechanism in mechanisms['mechanisms']:
        name = mechanism['id']
        if mechanism['type'] == 'contact':
            a, b = [values['bodies'][r]['parameters'] for r in mechanism['links']]
            if 'static_friction' in a and 'static_friction' in b:
                report['mechanisms'][name] = {'effective_static_friction': (a['static_friction']+b['static_friction'])/2,
                                              'effective_dynamic_friction': (a['dynamic_friction']+b['dynamic_friction'])/2,
                                              'combine': 'average'}
            else:
                report['mechanisms'][name] = {'combine': 'native flexible contact; measured by task checks'}
        elif name in values.get('mechanisms', {}) and mechanism['type'] in {'revolute', 'prismatic'}:
            targets = {body_path(stage, ref) for ref in mechanism['links']}
            matches = []
            for prim in stage.Traverse():
                if prim.IsA(UsdPhysics.Joint):
                    joint = UsdPhysics.Joint(prim)
                    pair = {str(p) for p in (*joint.GetBody0Rel().GetTargets(), *joint.GetBody1Rel().GetTargets())}
                    if pair == targets:
                        matches.append(prim)
            if len(matches) != 1:
                raise ValueError('Joint physical prior requires one matching native joint')
            p = values['mechanisms'][name]['parameters']
            axis = 'angular' if mechanism['type'] == 'revolute' else 'linear'
            prim = matches[0]
            if UsdPhysics.Joint(prim).GetExcludeFromArticulationAttr().Get():
                raise ValueError('Native per-axis friction requires an articulation; use a qualified collision carrier for adjacent contacts')
            # jointFriction is a legacy dimensionless coefficient, whereas the
            # declared prior is a force/torque. Isaac5.1's effort API is codeless:
            # its Python PhysxSchema class may be absent despite native support.
            if not prim.ApplyAPI('PhysxJointAxisAPI', axis):
                raise ValueError('Native per-axis friction effort unavailable; cannot reinterpret SI effort as a coefficient')
            PhysxSchema.PhysxJointAPI.Apply(prim).CreateJointFrictionAttr(0.)
            damping = p['joint_damping']*(np.pi/180 if axis == 'angular' else 1.)
            for attr, value in (('staticFrictionEffort', p['joint_friction']),
                                ('dynamicFrictionEffort', p['joint_friction']),
                                ('viscousFrictionCoefficient', damping)):
                prim.CreateAttribute(f'physxJointAxis:{axis}:{attr}', Sdf.ValueTypeNames.Float).Set(float(value))
            drive = UsdPhysics.DriveAPI.Apply(matches[0], axis)
            drive.CreateTypeAttr('force')
            drive.CreateStiffnessAttr(0.)
            drive.CreateTargetPositionAttr(0.)
            drive.CreateTargetVelocityAttr(0.)
            drive.CreateDampingAttr(0.)  # Viscous friction above already supplies damping.
            report['mechanisms'][name] = {**p, 'passive_joint_friction_only': True,
                'friction_model': 'Native per-axis SI dry-friction effort and viscous coefficient',
                'static_equals_dynamic': 'One inferred dry-friction prior; difference is not identified',
                'legacy_friction_coefficient': 0., 'native_viscous_coefficient': float(damping)}
    for name, support in values.get('supports', {}).items():
        p = support['parameters']
        material = physics_material(stage, '/World/Looks/prior_support_'+name, p['static_friction'],
                                    p['dynamic_friction'], p.get('restitution', 0.), friction_combine='average')
        roots = []
        for root in ('/World/Room', '/World/Support', '/World/Static'):
            scope = stage.GetPrimAtPath(root)
            if scope:
                roots.extend(str(prim.GetPath()) for prim in scope.GetChildren()
                             if prim.GetName() in {name, 'collider_'+name} or prim.GetName().startswith(name+'_'))
        count = bind_colliders(stage, roots, material)
        if not count:
            raise ValueError('Task support physical prior has no matching colliders: '+name)
        report['supports'][name] = {**p, 'colliders': count}
    return report
