"""Finite contact stiffness as a declared prior of a rigid body.

Ideal rigid Coulomb contact has no unique static normal force in a self-locking
wedge (a taper whose half-angle is below the friction angle, such as two nested
cups of almost equal taper): the native solver can then accumulate normal force
and penetration without bound. Real thin-walled parts resolve this through the
elastic compliance of their walls. A rigid link may therefore declare
`contact_stiffness` and `contact_damping` in `agent/physics_priors.json`; the
simulation layer authors them as the native implicit contact spring and damper
of that link's physics material. Values are per native contact point, so their
mapping to a physical wall stiffness depends on the collider's contact sampling
and must be measured for the actual representation. The native model ignores
restitution for such a material. These are priors: a task pass identifies
neither the stiffness nor the damping of the real object.
"""

PARAMETERS = ('contact_stiffness', 'contact_damping')


def priors(plan, mechanisms):
    """Both parameters or neither; rigid links only; strictly positive lower bounds."""
    kinds = {f'{obj}/{link}': spec['kind'] for obj, body in mechanisms['objects'].items()
             for link, spec in body['links'].items()}
    for ref, body in plan.get('bodies', {}).items():
        declared = [name for name in PARAMETERS if name in body.get('parameters', {})]
        if not declared:
            continue
        if len(declared) != len(PARAMETERS):
            raise ValueError(f'{ref}: compliant contact needs contact_stiffness together with contact_damping')
        if kinds.get(ref) != 'rigid':
            raise ValueError(f'{ref}: compliant contact priors apply to rigid links only')
        for name in PARAMETERS:
            value = body['parameters'][name]
            bounds = value.get('bounds') if isinstance(value, dict) else None
            if not isinstance(bounds, list) or len(bounds) != 2 or not bounds[0] > 0:
                raise ValueError(f'{ref}: {name} needs a strictly positive prior interval')
    for name, support in plan.get('supports', {}).items():
        if any(key in support.get('parameters', {}) for key in PARAMETERS):
            raise ValueError(f'{name}: a static support keeps rigid contact; declare compliance on the task body')


def settings(parameters):
    """Sampled native settings of one rigid body, or None when it keeps rigid contact."""
    if not any(name in parameters for name in PARAMETERS):
        return None
    stiffness, damping = (float(parameters[name]) for name in PARAMETERS)
    if not stiffness > 0 or not damping > 0:
        raise ValueError('Sampled compliant contact stiffness and damping must be positive')
    return {'stiffness_N_per_m': stiffness, 'damping_N_s_per_m': damping,
            'model': 'native implicit contact spring and damper per contact point',
            'restitution_active': False}
