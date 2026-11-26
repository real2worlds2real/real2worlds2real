"""Explicit shared native particle-contact domains.

Native particle systems do not establish cross-system cloth contacts. Sharing
one requires compatible system-wide contact/material settings; individual
cloth spring networks and masses remain separate.
"""
import re
import numpy as np


def optional_numerics(settings):
    """Explicit settings used by qualified controls; preserve legacy defaults."""
    result={}
    if 'particle_contact_offset' in settings:
        value=settings['particle_contact_offset']
        rest=settings.get('solid_rest_offset',settings.get('rest_offset',settings['contact_offset']*.5))
        if not np.isfinite(value) or value<=rest:
            raise ValueError('Particle contact offset must exceed its solid rest radius')
        result['particle_contact_offset']=float(value)
    if 'max_neighborhood' in settings:
        value=settings['max_neighborhood']
        if type(value) is not int or not 1<=value<=1024:
            raise ValueError('Explicit bounded integer particle neighbourhood required')
        result['max_neighborhood']=value
    return result


def declaration(settings,owner):
    group=settings.get('shared_particle_system')
    if group is None:return None
    if not isinstance(group,str) or re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}',group) is None:
        raise ValueError('Shared particle contact domain needs a stable simple name')
    rest=settings.get('rest_offset',settings['contact_offset']*.5)
    values={'contact_offset':settings['contact_offset'],'rest_offset':rest,
            'solid_rest_offset':settings.get('solid_rest_offset',rest),
            'particle_contact_offset':settings['contact_offset'],
            'solver_position_iterations':settings.get('solver_position_iterations',32),
            'max_depenetration_velocity':settings.get('max_depenetration_velocity',.05),
            'friction':settings['friction']}
    values.update(optional_numerics(settings))
    if not np.isfinite(list(values.values())).all():raise ValueError('Finite shared particle-contact settings required')
    return {'name':group,'simulation_owner':str(owner),'settings':values}


def compatible(existing,requested):
    if existing!=requested:
        raise ValueError('Shared particle system requires identical contact offsets, friction, solver settings and owner; distinct materials need a validated contact alternative')
    return existing
