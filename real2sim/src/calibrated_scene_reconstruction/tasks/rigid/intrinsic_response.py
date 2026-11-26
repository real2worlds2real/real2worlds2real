"""Resolve declared intrinsic rigid metrics from independent native recordings.

This measures an authored rigid approximation, not unknown real compliance or
material composition. Source appearance and task acceptance remain separate.
"""
import numpy as np

from calibrated_scene_reconstruction.tasks.mechanisms.passive_response import rigid_drift
from calibrated_scene_reconstruction.tasks.mechanisms.passive_extent import initial_extent


def measure(parts, poses, times, link, probe, *, thickness_m, expected_mass_kg):
    """Use complete unforced object geometry and a resolved external contact.

The caller must bind source and raw native files and prove that the passive
interval had no target actuation. ``thickness_m`` is the actual modeled solid
section at the probe, never real-world size truth or a numerical contact offset.
    """
    times = np.asarray(times, float)
    if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all()
            or (np.diff(times) <= 0).any() or link not in parts
            or set(parts) != set(poses) or any(p['kind'] != 'rigid' for p in parts.values())):
        raise ValueError('Complete rigid object and ordered passive native samples required')
    if (not np.isfinite([thickness_m, expected_mass_kg]).all()
            or thickness_m <= 0 or expected_mass_kg <= 0):
        raise ValueError('Actual positive modeled section and prior-based mass required')
    if (probe.get('pass') is not True or probe.get('task_actuation') is not False
            or probe.get('test_body_pose_commands') is not False
            or probe.get('rigid_surface_unchanged') is not True
            or probe.get('target_contact_steps', 0) < 2
            or probe.get('plan', {}).get('link') != link):
        raise ValueError('Resolved independent native external contact required')
    mass = probe.get('native_mass_kg', {}).get(link)
    if mass is None or not np.isclose(mass, expected_mass_kg, rtol=1e-5, atol=1e-8):
        raise ValueError('Native target mass disagrees with the unchanged declared prior')
    extent = initial_extent(parts, poses, {}, len(times))
    drift = rigid_drift(parts[link]['V'], poses[link])
    values = {'normalized_unforced_drift': {
                  'value': drift['maximum_surface_displacement_m']/extent['value_m'], 'unit': 'object_extent'},
              'penetration_over_local_thickness': {
                  'value': float(probe['maximum_native_penetration_m'])/thickness_m, 'unit': 'ratio'},
              'normalized_extent_change': {'value': 0., 'unit': 'ratio'}}
    return {'measurements': values, 'passive_surface_displacement_m': drift['maximum_surface_displacement_m'],
            'initial_object_extent': extent, 'actual_solid_section_m': float(thickness_m),
            'native_mass_kg': float(mass), 'expected_prior_mass_kg': float(expected_mass_kg),
            'scope': 'Unforced carried-surface motion, actual contact depth and unchanged native rigid '
                     'surface coordinates under the recorded external load. Zero rigid deformation is '
                     'a representation check, not identification of real stiffness.'}
