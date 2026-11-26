"""Source silhouette hull proposals for shapes poorly described by a few radii.

This estimates an outer envelope, not hidden cavities, material or collision
validity. The agent must retain openings and functional parts explicitly and
run the unchanged source, geometry and native task checks after Blender authoring.
"""
import numpy as np


def votes(points, observations):
    """Count positive and independently known pixels; occlusion is unknown."""
    cameras, frames, masks, windows = observations
    positive = np.zeros(len(points), np.uint16)
    known = np.zeros(len(points), np.uint16)
    for k, (i, mask, window) in enumerate(zip(frames, masks, windows)):
        local = (points-cameras['centre'][i]) @ cameras['R_world_cam'][i]
        homogeneous = local @ cameras['K'][i].T
        depth = homogeneous[:, 2]
        xy = homogeneous[:, :2]/np.maximum(depth[:, None], 1e-12)
        xy = np.rint((xy-window[:2])*window[4]).astype(np.int64)
        valid = (depth > 0) & (xy[:, 0] >= 0) & (xy[:, 0] < mask.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < mask.shape[0])
        ids = np.flatnonzero(valid); x, y = xy[ids].T
        domain = cameras.get('_object_domains')
        visible = domain[k][y, x] if domain is not None else np.ones(len(ids), bool)
        known[ids] += visible
        positive[ids] += visible & mask[y, x]
    return positive, known


def propose(observations, bounds, spacing, *, agreement=.9, min_views=8):
    from scipy.ndimage import label
    from skimage.measure import marching_cubes
    box = np.asarray(bounds, float)
    if (box.shape != (2, 3) or not np.isfinite(box).all() or np.any(box[1] <= box[0])
            or not np.isfinite(spacing) or spacing <= 0 or not .5 < agreement <= 1
            or type(min_views) is not int or min_views < 4):
        raise ValueError('Finite source search bounds, positive spacing and at least four known views required')
    axes = [np.arange(a, b+spacing*.5, spacing) for a, b in zip(*box)]
    shape = tuple(map(len, axes)); count = int(np.prod(shape))
    if min(shape) < 3 or count > 20_000_000:
        raise ValueError('Use at least3 samples per axis and at most20 million total samples')
    occupied = np.zeros(count, bool)
    for start in range(0, count, 131072):
        ids = np.arange(start, min(count, start+131072))
        index = np.unravel_index(ids, shape)
        points = np.column_stack([axis[indices] for axis, indices in zip(axes, index)])
        positive, known = votes(points, observations)
        occupied[ids] = (known >= min_views) & (positive >= agreement*known)
    occupied = occupied.reshape(shape)
    components, n = label(occupied)
    if not n:
        raise ValueError('No source-supported hull in search bounds')
    sizes = np.bincount(components.ravel()); sizes[0] = 0
    selected = int(sizes.argmax()); occupied = components == selected
    boundary = {f'{axis}_{side}': bool(np.take(occupied, index, axis=axis).any())
                for axis in range(3) for side, index in [('low', 0), ('high', -1)]}
    vertices, faces, _, _ = marching_cubes(np.pad(occupied, 1), .5, spacing=(spacing,)*3,
                                          gradient_direction='ascent')
    vertices += box[0]-spacing
    return vertices, faces, {
        'proposal_only': True, 'bounds': box.tolist(), 'spacing_m': float(spacing),
        'agreement': float(agreement), 'minimum_known_views': min_views,
        'grid_shape': list(shape), 'components': int(n),
        'retained_voxels': int(occupied.sum()), 'discarded_component_voxels': int(sizes.sum()-occupied.sum()),
        'touches_search_boundary': boundary,
        'scope': 'Outer silhouette envelope only; source review, openings, actual material, contacts and task checks remain required',
    }
