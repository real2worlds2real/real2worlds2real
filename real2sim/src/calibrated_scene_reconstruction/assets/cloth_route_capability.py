"""Describe natural-curvature representation without approving native behavior."""
import numpy as np

from .cloth_curvature import measured
from .cloth_rest import surface_chart
from .spring_energy import measure


def inspect(backend, chart, faces):
    if backend not in {'particle', 'surface_fem'}:
        raise ValueError('Explicit supported cloth backend required')
    reference = chart.get('bend_reference', {}).get('points', chart['points'])
    _, angles = measured(reference, faces)
    curved = np.abs(angles) > 1e-7  # Numerical flatness diagnostic, no acceptance tolerance.
    return {'backend': backend, 'natural_nonflat_edges': int(curved.sum()),
        'natural_abs_dihedral_median_deg': float(np.median(abs(angles))) if len(angles) else 0.,
        'bend_reference_authored_as_signed_rest_shape': backend == 'surface_fem' and 'bend_reference' in chart,
        'distance_bend_reflection_ambiguity': backend == 'particle',
        'formed_shape_review_required': backend == 'particle' and bool(curved.any()),
        'native_verification_required': True, 'object_approved': False,
        'scope': 'Capability diagnostic only. Particle distance springs cannot distinguish opposite '
                 'fold signs and do not consume bend_reference as signed curvature. Native FEM '
                 'rest-shape authoring does not establish correct response without release, '
                 'contact, material and semantic checks. Ordinary particle cloth remains available.'}


def distance_bend_control(rest_degrees=35.):
    """Synthetic isometric hinge: opposite fold signs share every spring length.

    This tests the authored law analytically, not an Isaac trajectory or a claim
    that a particular object will cross the intervening finite energy barrier.
    """
    if not np.isfinite(rest_degrees) or not 0 < abs(rest_degrees) < 170:
        raise ValueError('Resolved nonflat synthetic rest angle required')
    faces = np.array([[0, 1, 2], [1, 0, 3]])
    def hinge(degrees):
        theta = np.deg2rad(degrees)
        return np.array([[-.01, 0, 0], [.01, 0, 0], [0, .02, 0],
                         [0, -.02*np.cos(theta), -.02*np.sin(theta)]])
    natural = hinge(rest_degrees)
    chart = surface_chart(natural, faces, source='Independent synthetic hinge',
                          basis='Isometric fold sign control, not an observed material')
    settings = {'stretch_stiffness': 120., 'shear_stiffness': 60., 'bend_stiffness': 8.}
    states = []
    for degrees in [rest_degrees, 0., -rest_degrees]:
        points = hinge(degrees)
        states.append({'angle_deg': degrees, 'signed_dihedral_deg': measured(points, faces)[1].tolist(),
                       **measure(points, chart, settings)})
    return {'states': states, 'native_experiment': False, 'material_identified': False,
            'object_approved': False, 'scope': 'Analytic signed-fold ambiguity control; unchanged native physics.'}
