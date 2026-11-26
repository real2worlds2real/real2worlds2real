"""Separate native cloth material parameters from numerical candidate changes."""
from copy import deepcopy

import numpy as np


def resolve(backend,material,declared,overrides=None):
    """Resolve nominal authoring; prior sampling still happens exactly once.

    A particle spring network is not a surface-FEM material alias. Particle
    contact settings must be explicit because they govern both sampling and
    cross-layer interaction; no automatic radius inferred from a scene name.
    """
    if backend=='surface_fem':
        settings={**material,'friction':material['dynamic_friction'],
            'contact_offset':.75*material['thickness'],'rest_offset':material['thickness']/2,
            'self_collision_filter_distance':material['thickness'],
            'solver_position_iterations':255,'collision_pair_updates':16,'collision_iteration_multiplier':2,
            'max_depenetration_velocity':.1,'max_linear_velocity':10.}
        numerical=['contact_offset','rest_offset','self_collision_filter_distance','solver_position_iterations',
                   'collision_pair_updates','collision_iteration_multiplier','max_depenetration_velocity','max_linear_velocity']
    elif backend=='particle':
        required=['thickness','areal_density','stretch_stiffness','shear_stiffness','bend_stiffness',
                  'spring_damping','dynamic_friction']
        if not all(k in material for k in required):
            raise ValueError('Particle cloth requires explicit spring, thickness and mass priors; FEM constants cannot substitute')
        settings={k:material[k] for k in required if k not in ['spring_damping','dynamic_friction']}
        settings.update(damping=material['spring_damping'],friction=material['dynamic_friction'])
        numerical=['contact_offset','rest_offset','solid_rest_offset','particle_contact_offset',
                   'solver_position_iterations','max_neighborhood','max_depenetration_velocity']
        if not all(k in declared for k in numerical):
            raise ValueError('Explicit qualified particle contact and numerical profile required')
    else:
        raise ValueError('Choose an explicitly supported native cloth backend')
    for key in numerical:
        if key in declared:settings[key]=declared[key]
    baseline=deepcopy(settings)
    for key in numerical:
        if key in (overrides or {}):settings[key]=overrides[key]
    if not np.isfinite([settings[k] for k in numerical]).all():
        raise ValueError('Finite native cloth numerical settings required')
    changes={k:{'declared':baseline[k],'candidate':settings[k]}
             for k in numerical if settings[k]!=baseline[k]}
    return settings,changes
