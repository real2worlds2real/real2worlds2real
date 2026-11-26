"""Numerical trial suggestions, separate from materials and task acceptance.

Changing a solver setting is a hypothesis, not evidence of convergence. Keep
the original results and run the same geometry, material and physical checks.
"""
from copy import deepcopy


TGS_REFERENCE = 'https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.0/dev_guide/simulation_control/simulation_control.html'


def cautions(profile):
    values = profile.get('rigid_iterations')
    result = []
    if profile.get('solver', 'TGS') == 'TGS' and values is not None and values[1] > 4:
        result.append(dict(code='tgs_many_velocity_iterations',
                           reason='TGS velocity and position iterations are different operations; increasing both does not establish convergence.',
                           action='Compare zero or few velocity iterations independently, retaining geometry, material, timestep and acceptance criteria.',
                           reference=TGS_REFERENCE))
    return result


def trials(profile):
    """Suggest single-coordinate controls; never install a new runtime profile."""
    result = []
    values = profile.get('rigid_iterations')
    if profile.get('solver', 'TGS') == 'TGS' and values is not None and values[1] > 1:
        candidate = deepcopy(profile)
        candidate['rigid_iterations'][1] = 1
        result.append(dict(changed='rigid_velocity_iterations', profile=candidate))
    if isinstance(profile.get('physics_hz'), int) and profile['physics_hz'] <= 1920:
        candidate = deepcopy(profile)
        candidate['physics_hz'] *= 2
        result.append(dict(changed='physics_hz', profile=candidate))
    if values is not None and values[0] <= 128:
        candidate = deepcopy(profile)
        candidate['rigid_iterations'][0] *= 2
        result.append(dict(changed='rigid_position_iterations', profile=candidate))
    return [dict(**row, diagnostic_only=True, accepted=False,
                 instruction='Run unchanged source geometry, material hypotheses, task fixtures and acceptance checks; compare native traces.')
            for row in result]
