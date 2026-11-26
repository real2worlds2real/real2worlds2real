"""Release diagnostics, separate from stiffness and task acceptance.

Energy below is a discrete hinge estimate, not a PhysX energy readback.
No observation, rest shape, material parameter or acceptance limit is inferred.
"""
import numpy as np

from calibrated_scene_reconstruction.assets.cloth_curvature import adjacency, measured
from calibrated_scene_reconstruction.assets.cloth_rest import material_inverse


def membrane_budget(points,faces,rest,youngs_modulus,poissons_ratio,thickness,mass,gravity=9.81):
    """Plane-stress StVK estimate of prestrain, not a native energy readback."""
    points,rest=np.asarray(points,float),np.asarray(rest,float);faces=np.asarray(faces)
    if (points.shape!=rest.shape or points.ndim!=2 or points.shape[1:]!=(3,)
            or not np.isfinite([points,rest]).all()
            or not np.isfinite([youngs_modulus,poissons_ratio,thickness,mass,gravity]).all()
            or min(youngs_modulus,thickness,mass,gravity)<=0 or not -1<poissons_ratio<.5):
        raise ValueError('Finite matched sheet and valid continuum parameters required')
    tri=points[faces];ds=np.stack([tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]],axis=2)
    deformation=ds@material_inverse(rest,faces)
    strain=(np.swapaxes(deformation,1,2)@deformation-np.eye(2))/2
    u,v,w=strain[:,0,0],strain[:,1,1],strain[:,0,1];nu=poissons_ratio
    density=youngs_modulus*thickness/(2*(1-nu*nu))*(u*u+v*v+2*nu*u*v+2*(1-nu)*w*w)
    rt=rest[faces];area=np.linalg.norm(np.cross(rt[:,1]-rt[:,0],rt[:,2]-rt[:,0]),axis=1)/2
    energy=float(density@area)
    return {'estimated_membrane_energy_J':energy,'equivalent_gravity_height_m':energy/(mass*gravity),
            'equivalent_speed_m_s':float(np.sqrt(2*energy/mass)),
            'scope':'Plane-stress StVK diagnostic; no native energy or material identification. An admissible strain ceiling is not a relaxed initial-state certificate.'}


def bending_budget(points, faces, rest, rigidity, mass, gravity=9.81, natural=None):
    points, rest = np.asarray(points, float), np.asarray(rest, float)
    faces = np.asarray(faces)
    if (points.shape != rest.shape or points.ndim != 2 or points.shape[1] != 3
            or not np.isfinite([points, rest]).all()
            or not np.isfinite([rigidity, mass, gravity]).all()
            or min(rigidity, mass, gravity) <= 0):
        raise ValueError('Finite matched sheet coordinates and positive physical parameters required')
    pairs, edges = adjacency(faces)
    _, angles = measured(points, faces)
    _, reference = measured(rest if natural is None else natural, faces)
    delta = np.radians((angles-reference+180)%360-180)
    triangles = rest[faces]
    areas = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                    triangles[:, 2]-triangles[:, 0]), axis=1)/2
    square = np.sum((rest[edges[:, 1]]-rest[edges[:, 0]])**2, axis=1)
    energy = float(.5*rigidity*np.sum(square/areas[pairs].sum(1)*delta**2))
    from calibrated_scene_reconstruction.assets.cloth_curvature import orientation_diagnostic
    return {'estimated_bending_energy_J': energy,
            'equivalent_gravity_height_m': energy/(mass*gravity),
            'equivalent_speed_m_s': float(np.sqrt(2*energy/mass)),
            'max_rest_angle_difference_deg': float(np.max(np.abs(np.degrees(delta)), initial=0)),
            'natural_fold_orientation':orientation_diagnostic(points,faces,rest if natural is None else natural),
            'scope': 'Discrete hinge estimate only; no native energy, constitutive identification or pass. Compare different packing/rest hypotheses before increasing stiffness.'}


def modal_decay(times, amplitudes, reference_frequency_hz):
    """Measure a release envelope without requiring it to keep oscillating.

    The harmonic modal proxy includes displacement AND velocity: a zero
    crossing alone cannot appear to be damping. Numerical damping remains
    inseparable without a matched zero-material-damping control.
    """
    t, q = np.asarray(times, float), np.asarray(amplitudes, float)
    if (t.ndim != 1 or q.shape != t.shape or len(t) < 20
            or not np.isfinite([t, q]).all() or (np.diff(t) <= 0).any()
            or not np.isfinite(reference_frequency_hz) or reference_frequency_hz <= 0):
        raise ValueError('Finite ordered release trace and positive independent frequency required')
    period = 1/reference_frequency_hz
    if t[-1]-t[0] < 2*period or np.max(np.diff(t)) > period/10:
        raise ValueError('Release needs at least two periods and ten samples per period')
    velocity = np.gradient(q, t, edge_order=2)
    proxy = q*q+(velocity/(2*np.pi*reference_frequency_hz))**2
    start = (t <= t[0]+period/2)
    end = (t >= t[-1]-period/2)
    initial, final = float(np.mean(proxy[start])), float(np.mean(proxy[end]))
    if initial <= 1e-16:
        raise ValueError('No resolved initial release excitation')
    return {'initial_modal_proxy_m2': initial, 'final_modal_proxy_m2': final,
            'final_to_initial_proxy_ratio': final/initial,
            'max_proxy_to_initial_ratio': float(proxy.max()/initial),
            'window_duration_s': period/2,
            'scope': 'Harmonic modal proxy, not total physical energy. Compare identical undamped controls; no task or material identification.'}
