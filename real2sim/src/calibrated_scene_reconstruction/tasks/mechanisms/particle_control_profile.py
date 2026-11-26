"""Explicit native particle-cloth control settings, without scene assumptions."""
import numpy as np


def validate(profile):
    keys={'version','physics_hz','spacing_m','physical_thickness_m','areal_density',
          'spring_settings','spring_damping','spring_model','contact'}
    if not isinstance(profile,dict) or set(profile)!=keys or profile['version']!=1:
        raise ValueError('Complete version1 native particle control profile required')
    hz=profile['physics_hz']
    if type(hz) is not int or not 240<=hz<=7680 or hz%40:
        raise ValueError('Supported native clock and recording interval required')
    for key in ['spacing_m','physical_thickness_m','areal_density','spring_damping']:
        if not np.isfinite(profile[key]) or profile[key]<=0:
            raise ValueError('Positive finite particle material and sampling values required')
    if not .0001<=profile['spacing_m']<=.002:
        raise ValueError('Resolved particle coupon spacing within its construction budget required')
    if profile['spring_model'] not in {'triangle_edges','balanced_square_lattice','triangle_surface'}:
        raise ValueError('Explicit native spring-network topology required')
    springs=profile['spring_settings']
    if (not isinstance(springs,dict) or set(springs)!={'stretch_stiffness','shear_stiffness','bend_stiffness'}
            or not np.isfinite(list(springs.values())).all() or min(springs.values())<=0):
        raise ValueError('Explicit positive native spring constants required; no implicit continuum conversion')
    contact=profile['contact']
    if (not isinstance(contact,dict) or set(contact)!={'contact_offset','rest_offset','solid_rest_offset',
            'particle_contact_offset','solver_position_iterations','max_neighborhood','max_depenetration_velocity'}
            or not np.isfinite(list(contact.values())).all()):
        raise ValueError('Complete finite particle contact profile required')
    h=profile['physical_thickness_m']
    if (min(contact['rest_offset'],contact['solid_rest_offset'])<h/2
            or contact['contact_offset']<=contact['rest_offset']
            or contact['particle_contact_offset']<=contact['solid_rest_offset']
            or max(contact['contact_offset'],contact['particle_contact_offset'])>.01
            or not 0<contact['max_depenetration_velocity']<=10):
        raise ValueError('Explicit numerical margins must preserve physical sheet thickness')
    for key,lo,hi in [('solver_position_iterations',32,255),('max_neighborhood',1,1024)]:
        if type(contact[key]) is not int or not lo<=contact[key]<=hi:
            raise ValueError('Bounded integer particle solver budgets required')
    return profile
