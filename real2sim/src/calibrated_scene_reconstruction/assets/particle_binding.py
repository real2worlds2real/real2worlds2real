"""Bind a particle contact control to the actual running cloth representation."""
import numpy as np


def compare_profile(control, runtime):
    """No control can certify a different spring network, padding or sampling."""
    from calibrated_scene_reconstruction.assets.contact_control_profile import resolution_matches
    if control.get('experimental_internal_shell') is not None:
        raise ValueError('Experimental internal forces cannot qualify native particle springs')
    if control.get('native_spring_readback') is not True:
        raise ValueError('Contact control needs actual native spring readback')
    if runtime.get('native_spring_readback') is not True or runtime.get('native_mass_readback') is not True:
        raise ValueError('Runtime particle springs and masses require native readback')
    if not runtime.get('self_collision') or not runtime.get('self_collision_filter'):
        raise ValueError('Particle contact filtering differs from the qualified control')
    if runtime['spring_model'] != control['spring_model']:
        raise ValueError('Particle spring topology differs from contact control')
    for key in ['native_contact_settings','spring_settings']:
        if set(runtime[key]) != set(control[key]):
            raise ValueError('Incomplete runtime particle profile: '+key)
        for name,value in control[key].items():
            if not np.isclose(runtime[key][name],value,rtol=1e-5,atol=1e-10):
                raise ValueError('Particle contact profile differs: '+name)
    for key in ['areal_density_kg_m2','physical_thickness_m','spring_damping_kg_s']:
        if not np.isclose(runtime[key],control[key],rtol=1e-5,atol=1e-14):
            raise ValueError('Particle material differs from contact control: '+key)
    if (not np.isclose(runtime['friction'],control['parameters']['friction'],rtol=1e-5,atol=1e-8)
            or not np.isclose(runtime['particle_friction_scale'],1.,rtol=0,atol=1e-8)):
        raise ValueError('Particle friction or self-friction scale differs from control')
    if not resolution_matches(control['material_edge_range_m'],runtime['material_edge_range_m']):
        raise ValueError('Particle material sampling differs from contact control')
    return {'pass':True,'runtime_profile':runtime,
            'scope':'Matched native contact profile only; source, material response and task remain separate'}
