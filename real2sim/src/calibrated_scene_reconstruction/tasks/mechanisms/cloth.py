"""Measured material/contact checks for native sheet states, without pose control."""
import numpy as np

from calibrated_scene_reconstruction.assets.cloth_metric import MaterialMetric


def nonrigid_change(before, after):
    """Residual after the best rigid alignment; rotation alone is not deformation."""
    before,after=np.asarray(before,float),np.asarray(after,float)
    if (before.ndim!=2 or before.shape[1:]!=(3,) or len(before)<3 or after.shape!=before.shape
            or not np.isfinite(before).all() or not np.isfinite(after).all()):
        raise ValueError('Matching finite native point states required')
    a,b=before-before.mean(0),after-after.mean(0)
    u,_,vt=np.linalg.svd(a.T@b)
    if np.linalg.det(u@vt)<0:u[:,-1]*=-1
    residual=np.linalg.norm(a@(u@vt)-b,axis=1)
    return {'maximum_m':float(residual.max()),'rms_m':float(np.sqrt(np.mean(residual**2)))}



def material_check(points, faces, rest, *, max_edge_strain, max_area_change):
    """Check stretch/shear and material area, without identifying real material."""
    return MaterialMetric(points, faces, rest, max_edge_strain=max_edge_strain,
                          max_area_change=max_area_change).check(points)


def pinch_retention(before, after, initial_indices, current_contact, *, inside_before, inside_after,
                    contact_tolerance, max_material_transfer=.02, minimum_contact_fraction=.8):
    """Hold the contacted material, without requiring a rigid free cloth flap.

    Points are expressed in the holding body's frame. Strain, penetration and
    contact resolution remain separate required checks. Whole-sheet displacement
    is reported even when only free material settles under gravity.
    """
    before, after = np.asarray(before, float), np.asarray(after, float)
    indices, contact = np.asarray(initial_indices), np.asarray(current_contact)
    scalars = [inside_before, inside_after, contact_tolerance, max_material_transfer, minimum_contact_fraction]
    if (before.ndim != 2 or before.shape[1:] != (3,) or after.shape != before.shape
            or not len(before) or not np.isfinite(before).all() or not np.isfinite(after).all()
            or indices.ndim != 1 or indices.dtype.kind not in 'iu' or len(indices) < 3
            or indices.min() < 0 or indices.max() >= len(before) or len(set(indices)) != len(indices)
            or contact.shape != (len(before),) or contact.dtype.kind != 'b'
            or not np.isfinite(scalars).all() or not 0 <= inside_before <= 1 or not 0 <= inside_after <= 1
            or contact_tolerance <= 0 or not 0 <= max_material_transfer < 1
            or not 0 < minimum_contact_fraction <= 1):
        raise ValueError('Finite matching cloth/body-frame states, contact material and retention limits required')
    displacement = np.linalg.norm(after-before, axis=1)
    retained = float(contact[indices].mean())
    drift = float(displacement[indices].max())
    transfer = abs(inside_after-inside_before)
    return {'pass': bool(drift <= contact_tolerance and retained >= minimum_contact_fraction
                         and transfer <= max_material_transfer),
            'initial_contact_nodes': len(indices), 'retained_contact_fraction': retained,
            'maximum_contact_material_drift_m': drift,
            'whole_sheet_maximum_displacement_m': float(displacement.max()),
            'inside_material_fraction_before': float(inside_before), 'inside_material_fraction_after': float(inside_after),
            'inside_material_fraction_delta': float(transfer),
            'limits': {'contact_drift_m': float(contact_tolerance), 'material_transfer': max_material_transfer,
                       'minimum_contact_fraction': minimum_contact_fraction},
            'scope': 'Retention of originally contacted material; free cloth may settle, not a rigid-sheet constraint'}


def support_contact(points, normal, offset, *, particle_radius, tolerance, domain=None):
    """Planar surface contact in an independently specified support footprint.

    Does not prove friction, wiping coverage, or contact with a different mesh.
    """
    points, normal = np.asarray(points, float), np.asarray(normal, float)
    if (points.ndim != 2 or points.shape[1:] != (3,) or not len(points) or not np.isfinite(points).all()
            or normal.shape != (3,) or not np.isfinite(normal).all() or not np.isclose(np.linalg.norm(normal), 1)
            or not np.isfinite([offset, particle_radius, tolerance]).all() or particle_radius < 0 or tolerance <= 0):
        raise ValueError('Finite native nodes, unit surface normal, contact radius and tolerance required')
    domain = np.ones(len(points), bool) if domain is None else np.asarray(domain)
    if domain.dtype.kind != 'b' or domain.shape != (len(points),) or not domain.any():
        raise ValueError('Support domain must select actual nodes within the support footprint')
    gap = points@normal-offset-particle_radius
    contact = domain & (abs(gap) <= tolerance)
    return {'contact_indices': np.flatnonzero(contact).tolist(), 'contact_nodes': int(contact.sum()),
            'domain_nodes': int(domain.sum()), 'maximum_penetration_m': float(max(0., -gap[domain].min())),
            'minimum_surface_gap_m': float(gap[domain].min())}
