"""Diagnostic energy in an authored distance-spring cloth network.

This is the analytic network evaluated at recorded coordinates, not a native
solver energy/stress readback or a material identification. It introduces no
acceptance threshold and does not change cloth state or rest lengths.
"""
import numpy as np

from .cloth_rest import constraints


def measure(points, chart, settings, *, mass_kg=None, gravity_m_s2=9.81):
    p = np.asarray(points, float)
    rest = np.asarray(chart['points'], float)
    if p.ndim != 2 or p.shape != rest.shape or p.shape[1:] != (3,) or not np.isfinite(p).all():
        raise ValueError('Finite positions matching the immutable material chart required')
    pairs, lengths, stiffness, kinds = constraints(chart, settings)
    pairs = np.asarray(pairs, int); lengths = np.asarray(lengths, float)
    stiffness = np.asarray(stiffness, float); kinds = np.asarray(kinds)
    if (not len(pairs) or pairs.min() < 0 or pairs.max() >= len(p)
            or not np.isfinite(lengths).all() or (lengths <= 0).any()
            or not np.isfinite(stiffness).all() or (stiffness < 0).any()):
        raise ValueError('Positive rest lengths and finite nonnegative spring constants required')
    distance = np.linalg.norm(p[pairs[:, 1]]-p[pairs[:, 0]], axis=1)
    strain = distance/lengths-1
    energy = .5*stiffness*(distance-lengths)**2
    result = {'analytic_spring_energy_J': float(energy.sum()), 'groups': {},
        'native_energy_readback': False, 'material_identified': False, 'task_approved': False,
        'scope': 'Diagnostic authored network energy; no threshold, force or state change'}
    for name in sorted(set(kinds)):
        chosen = kinds == name
        result['groups'][str(name)] = {'springs': int(chosen.sum()),
            'energy_J': float(energy[chosen].sum()),
            'rms_strain': float(np.sqrt(np.mean(strain[chosen]**2))),
            'maximum_absolute_strain': float(np.max(abs(strain[chosen])))}
    if mass_kg is not None:
        if not np.isfinite([mass_kg, gravity_m_s2]).all() or min(mass_kg, gravity_m_s2) <= 0:
            raise ValueError('Finite positive total mass and gravitational reference required')
        result['mass_kg'] = float(mass_kg)
        result['energy_over_weight_m'] = float(energy.sum()/(mass_kg*gravity_m_s2))
        result['energy_over_weight_scope'] = 'Scale comparison only; not a prediction of release height'
    return result
