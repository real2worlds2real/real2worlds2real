"""Reuse explicitly compatible native particle-contact systems, not cloth bodies."""
import json
from pxr import UsdGeom,UsdShade,PhysxSchema
from calibrated_scene_reconstruction.assets.particle_contacts import declaration,compatible,optional_numerics

KEY='calibrated_scene_reconstruction:shared_particle_contact_domain'


def system(stage,surface,settings,owner):
    from omni.physx.scripts import particleUtils
    contract=declaration(settings,owner)
    path=(('/World/ClothContacts/'+contract['name']) if contract else str(surface)+'_particles')
    if contract:
        UsdGeom.Scope.Define(stage,'/World/ClothContacts')
        prim=stage.GetPrimAtPath(path)
        if prim:
            stored=prim.GetCustomDataByKey(KEY)
            if stored is None:raise ValueError('Shared particle domain path is already owned by another prim')
            compatible(json.loads(stored),contract)
            return PhysxSchema.PhysxParticleSystem(prim)
    rest=settings.get('rest_offset',settings['contact_offset']*.5)
    explicit=optional_numerics(settings)
    particle_offset=explicit.pop('particle_contact_offset',settings['contact_offset'])
    result=particleUtils.add_physx_particle_system(stage,path,simulation_owner=owner,
        contact_offset=settings['contact_offset'],rest_offset=rest,
        solid_rest_offset=settings.get('solid_rest_offset',rest),
        particle_contact_offset=particle_offset,
        solver_position_iterations=settings.get('solver_position_iterations',32),
        enable_ccd=True,max_depenetration_velocity=float(settings.get('max_depenetration_velocity',.05)),**explicit)
    if not result:raise RuntimeError('Particle contact-system authoring failed')
    mat_path=path+'_material'
    particleUtils.add_pbd_particle_material(stage,mat_path,friction=settings['friction'],damping=0.,drag=0.,lift=0.)
    UsdShade.MaterialBindingAPI.Apply(result.GetPrim()).Bind(UsdShade.Material.Get(stage,mat_path),UsdShade.Tokens.weakerThanDescendants,'physics')
    if 'thickness' in settings:
        reference={'thickness':float(settings['thickness']),'contact_offset':float(settings['contact_offset']),
                   'particle_contact_offset':float(particle_offset),'rest_offset':float(rest),
                   'solid_rest_offset':float(settings.get('solid_rest_offset',rest))}
        result.GetPrim().SetCustomDataByKey('calibrated_scene_reconstruction:particle_contact_reference',json.dumps(reference))
    if contract:result.GetPrim().SetCustomDataByKey(KEY,json.dumps(contract,sort_keys=True))
    return result


def group(stage,system):
    """Distinct cloth sets keep local self-filtering separate from other cloths."""
    path=system.GetPath();existing=[]
    for prim in stage.Traverse():
        if not prim.HasAPI(PhysxSchema.PhysxParticleAPI):continue
        api=PhysxSchema.PhysxParticleAPI(prim)
        if path in api.GetParticleSystemRel().GetTargets():existing.append(api.GetParticleGroupAttr().Get())
    return max([int(x) for x in existing if x is not None]+[0])+1
