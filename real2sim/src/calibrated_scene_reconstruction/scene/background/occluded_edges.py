"""Use reviewed source-edge visibility without shrinking behind an occluder."""
import numpy as np

EDGES = ('left', 'top', 'right', 'bottom')


def hidden(row):
    if 'occluded_edges' not in row:
        partial = row.get('partially_occluded', False) or row.get('geometry_fit_scope') == 'occluded_object'
        return np.full(4, bool(partial))
    edges = row['occluded_edges']
    if (not isinstance(edges, list) or any(not isinstance(e, str) for e in edges)
            or len(set(edges)) != len(edges) or not set(edges) <= set(EDGES)
            or not isinstance(row.get('occlusion_review'), str) or not row['occlusion_review'].strip()):
        raise ValueError('Explicit distinct occluded edges and source visibility review required')
    return np.array([edge in edges for edge in EDGES])


def residual_factor(delta, occluded):
    delta, occluded = np.asarray(delta), np.asarray(occluded)
    if delta.shape != (4,) or occluded.shape != (4,) or occluded.dtype != bool or not np.isfinite(delta).all():
        raise ValueError('Finite projected bounds and reviewed edge flags required')
    undershoot = delta*np.array([1., 1., -1., -1.]) > 0
    return np.where(undershoot, 1., np.where(occluded, 0., .35))
