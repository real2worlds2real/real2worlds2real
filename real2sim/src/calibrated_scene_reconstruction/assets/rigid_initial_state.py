"""Screen authored independent rigid parts before native relaxation.

Measures both directions against the actual closed triangle solids. Empty
cavities remain empty. Edge samples supplement face centres and vertices;
passing this finite sampling screen is not a proof of disjoint interiors or
a substitute for collision-shape and native-state validation.
"""
import numpy as np


def surface_samples(vertices, faces, spacing, *, max_samples=2_000_000):
    from calibrated_scene_reconstruction.assets.surface_distance import triangle_samples
    vertices, faces = np.asarray(vertices, float), np.asarray(faces)
    base = triangle_samples(vertices, faces)
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('Positive finite surface sample spacing required')
    edges = np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]],
                                              faces[:, [2, 0]]]), axis=1), axis=0)
    ends = vertices[edges]
    counts = np.maximum(1, np.ceil(np.linalg.norm(ends[:, 1]-ends[:, 0], axis=1)/spacing)).astype(int)
    if len(base)+int(np.maximum(0, counts-1).sum()) > max_samples:
        raise ValueError('Rigid preflight sample budget exceeded; do not silently coarsen it')
    extra = [a+(b-a)*np.arange(1, n)[:, None]/n for (a, b), n in zip(ends, counts) if n > 1]
    return np.concatenate([base, *extra])


def measure(parts, *, spacing_m, tolerance_m=1e-6, device='cpu'):
    """Parts must share a frame; use only pairs whose collisions are enabled."""
    from calibrated_scene_reconstruction.assets.surface_distance import SignedSurface
    if not np.isfinite(tolerance_m) or tolerance_m < 0:
        raise ValueError('Nonnegative finite penetration tolerance required')
    names = [p['name'] for p in parts]
    if len(set(names)) != len(names):
        raise ValueError('Unique physical part names required')
    samples = [surface_samples(p['V'], p['F'], spacing_m) for p in parts]
    surfaces = [SignedSurface(p['V'], p['F'], device=device) for p in parts]
    pairs = []
    for i in range(len(parts)):
        for j in range(i):
            depths = [surfaces[j].max_penetration(samples[i]),
                      surfaces[i].max_penetration(samples[j])]
            pairs.append({'parts': [names[i], names[j]],
                          'sampled_penetration_m': dict(zip([names[i], names[j]], depths)),
                          'pass': max(depths) <= tolerance_m})
    return {'pass': all(p['pass'] for p in pairs), 'pairs': pairs,
            'spacing_m': float(spacing_m), 'tolerance_m': float(tolerance_m),
            'sample_counts': dict(zip(names, map(len, samples))),
            'scope': 'Authored closed solids; finite surface sampling, not exhaustive intersections, native cooking, stability or task approval'}
