"""Read actual native particle contact and material parameters."""
import numpy as np
from calibrated_scene_reconstruction.assets.particle_binding import compare_profile

def read(view, parameters):
    """Read USD system settings and native tensor mass/spring arrays after reset."""
    from pxr import PhysxSchema,UsdShade
    from calibrated_scene_reconstruction.core.geometry.rigid import as_numpy
    from calibrated_scene_reconstruction.assets.cloth_rest import area
    from calibrated_scene_reconstruction.assets.contact_control_profile import edge_range
    import json
    prim=view.prim;stage=prim.GetStage()
    cloth=PhysxSchema.PhysxParticleClothAPI(prim)
    paths=PhysxSchema.PhysxParticleAPI(prim).GetParticleSystemRel().GetTargets()
    if len(paths)!=1:raise ValueError('Exactly one native particle contact system required')
    system=PhysxSchema.PhysxParticleSystem(stage.GetPrimAtPath(paths[0]))
    contact={name:getattr(system,'Get'+''.join(w.title() for w in name.split('_'))+'Attr')().Get()
             for name in ['contact_offset','rest_offset','solid_rest_offset','particle_contact_offset',
                          'max_neighborhood','max_depenetration_velocity']}
    contact['solver_position_iterations']=system.GetSolverPositionIterationCountAttr().Get()
    material=UsdShade.MaterialBindingAPI(system.GetPrim()).ComputeBoundMaterial('physics')[0]
    api=PhysxSchema.PhysxPBDMaterialAPI(material.GetPrim())
    authored=np.asarray(cloth.GetSpringStiffnessesAttr().Get(),float)
    damping=np.asarray(cloth.GetSpringDampingsAttr().Get(),float)
    kinds=json.loads(prim.GetCustomDataByKey('calibrated_scene_reconstruction:cloth_spring_kinds'))
    expected=np.array([parameters[k+'_stiffness'] for k in kinds])
    native=as_numpy(view.view.get_stretch_stiffnesses()).ravel()
    native_damping=as_numpy(view.view.get_spring_dampings()).ravel()
    spring_ok=(len(authored)==len(expected)==len(native)==len(damping)==len(native_damping)
               and np.allclose(authored,expected,rtol=1e-5,atol=1e-12)
               and np.allclose(np.sort(native),np.sort(expected),rtol=1e-5,atol=1e-12)
               and np.allclose(damping,parameters['spring_damping'],rtol=1e-5,atol=1e-15)
               and np.allclose(native_damping,parameters['spring_damping'],rtol=1e-5,atol=1e-15))
    rest_area=area(view.rest,view.faces)
    mass=float(as_numpy(view.get_particle_masses()).sum())
    target=parameters.get('native_total_mass_kg',parameters.get('mass',parameters.get('areal_density',0.)*rest_area))
    return {'native_contact_settings':contact,
            'spring_settings':{k:parameters[k] for k in ['stretch_stiffness','shear_stiffness','bend_stiffness']},
            'spring_damping_kg_s':parameters['spring_damping'],
            'physical_thickness_m':parameters['thickness'], 'areal_density_kg_m2':mass/rest_area,
            'native_spring_readback':bool(spring_ok),
            'native_mass_readback':bool(mass>0 and np.isclose(mass,target,rtol=1e-4,atol=1e-10)),
            'friction':api.GetFrictionAttr().Get(),'particle_friction_scale':api.GetParticleFrictionScaleAttr().Get(),
            'self_collision':PhysxSchema.PhysxParticleAPI(prim).GetSelfCollisionAttr().Get(),
            'self_collision_filter':cloth.GetSelfCollisionFilterAttr().Get(),
            'spring_model':prim.GetCustomDataByKey('calibrated_scene_reconstruction:cloth_spring_model') or 'triangle_edges',
            'material_edge_range_m':edge_range(view.rest,view.faces)}
