"""Resample an explicitly parameterized thin strip for uniform native masses.

This is geometry preparation, not a material estimate or physics approval.
Vertices stay on the supplied triangular surface. Cross-section counts follow
physical width instead of putting the same number of nodes in a pointed tip
and a broad middle. Independent strips can request area-proportional counts.
"""
import numpy as np


def _distance(vertices, faces, samples):
    """Use Blender's BVH in its bundled Python, or trimesh in the QA env."""
    try:
        from mathutils.bvhtree import BVHTree
    except ImportError:
        import trimesh
        return trimesh.Trimesh(vertices, faces, process=False).nearest.on_surface(samples)[1]
    tree = BVHTree.FromPolygons(vertices.tolist(), faces.tolist(), all_triangles=True)
    return np.array([tree.find_nearest(tuple(p))[3] for p in samples])


def _surface(grid, uv):
    shape = np.array(grid.shape[:2]) - 1
    q = np.clip(np.asarray(uv, float), 0., 1.) * shape
    ij = np.minimum(np.floor(q).astype(int), shape - 1)
    t, u = (q - ij).T
    i, j = ij.T
    a, b = grid[i, j], grid[i, j + 1]
    c, d = grid[i + 1, j + 1], grid[i + 1, j]
    return np.where((u >= t)[:, None],
                    (1-u)[:, None]*a + (u-t)[:, None]*b + t[:, None]*c,
                    (1-t)[:, None]*a + u[:, None]*c + (t-u)[:, None]*d)


def _sample(grid, spacing, root_samples, maximum_deviation):
    rows, columns = np.array(grid.shape[:2]) - 1
    centre = grid[:, grid.shape[1] // 2]
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(centre, axis=0), axis=1))]
    if (np.diff(distance) <= 0).any():
        raise ValueError('The strip needs an ordered, noncollapsed centreline')
    steps = max(2, int(np.ceil(distance[-1] / spacing)))
    longitudinal = np.interp(np.linspace(0., distance[-1], steps + 1),
                             distance, np.linspace(0., 1., rows + 1))
    # Preserve tapered boundaries where uniform centreline spacing alone
    # would cut off a curved tip or the flare immediately above a narrow root.
    native_t = np.linspace(0., 1., rows + 1)
    probes_u = np.array([0., .5, 1.])
    for _ in range(20):
        additions = []
        for left, right in zip(longitudinal, longitudinal[1:]):
            inner = native_t[(native_t > left+1e-12) & (native_t < right-1e-12)]
            if not len(inner):
                continue
            ends = _surface(grid, np.c_[np.repeat([left, right], 3), np.tile(probes_u, 2)]).reshape(2, 3, 3)
            actual = _surface(grid, np.c_[np.repeat(inner, 3), np.tile(probes_u, len(inner))]).reshape(-1, 3, 3)
            fraction = ((inner-left)/(right-left))[:, None, None]
            error = np.linalg.norm(actual-((1-fraction)*ends[0]+fraction*ends[1]), axis=2).max(1)
            index = int(error.argmax())
            if error[index] > maximum_deviation:
                additions.append(inner[index])
        if not additions:
            break
        longitudinal = np.unique(np.r_[longitudinal, additions])
    uv, rings, count = [], [], 0
    for index, t in enumerate(longitudinal):
        native_u = np.linspace(0., 1., columns + 1)
        cross = _surface(grid, np.c_[np.full(len(native_u), t), native_u])
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(cross, axis=0), axis=1))]
        if (np.diff(arc) <= 0).any():
            raise ValueError('A resolved noncollapsed strip boundary is required')
        n = root_samples if index == 0 else max(2, int(np.ceil(arc[-1] / spacing)) + 1)
        u = np.interp(np.linspace(0., arc[-1], n), arc, native_u)
        uv.extend(np.c_[np.full(n, t), u].tolist())
        rings.append((np.arange(count, count + n), u))
        count += n
    faces = []
    for (a, au), (b, bu) in zip(rings, rings[1:]):
        i = j = 0
        while i < len(a)-1 or j < len(b)-1:
            av = au[i+1] if i+1 < len(a) else np.inf
            bv = bu[j+1] if j+1 < len(b) else np.inf
            if abs(av-bv) < 1e-12:
                faces.extend([[a[i], a[i+1], b[j+1]], [a[i], b[j+1], b[j]]])
                i += 1
                j += 1
            elif av < bv:
                faces.append([a[i], a[i+1], b[j]])
                i += 1
            else:
                faces.append([a[i], b[j+1], b[j]])
                j += 1
    uv, faces = np.asarray(uv), np.asarray(faces, int)
    return _surface(grid, uv), faces, uv


def resample(grid, spacing, *, root_samples=3, target_vertices=None):
    """Return a boundary-preserving strip mesh and auditable sampling metadata.

``grid`` has ordered length/width axes and the standard a-b-c/a-c-d triangle
split. Root nodes are all vertices of its first cross-section, not a selected
subset of a denser edge. ``target_vertices`` can be round(area/spacing**2) to
give disconnected pieces of a homogeneous sheet consistent native masses.
Native contact coverage and the original source gates remain separate checks.
    """
    grid = np.asarray(grid, float)
    if (grid.ndim != 3 or grid.shape[2] != 3 or min(grid.shape[:2]) < 3
            or not np.isfinite(grid).all() or not np.isfinite(spacing) or spacing <= 0
            or type(root_samples) is not int or not 2 <= root_samples <= grid.shape[1]):
        raise ValueError('Finite resolved strip geometry, spacing and root sampling required')
    if target_vertices is not None and (type(target_vertices) is not int or target_vertices < 12):
        raise ValueError('A resolved positive target node count is required')
    selected_spacing = spacing
    maximum_deviation = spacing * .04
    best = _sample(grid, selected_spacing, root_samples, maximum_deviation)
    if target_vertices is not None:
        lo, hi = spacing * .5, spacing * 2.
        error = abs(len(best[0]) - target_vertices)
        for _ in range(15):
            trial_spacing = (lo + hi) / 2
            trial = _sample(grid, trial_spacing, root_samples, maximum_deviation)
            e = abs(len(trial[0]) - target_vertices)
            if e < error:
                best, selected_spacing, error = trial, trial_spacing, e
            if len(trial[0]) > target_vertices:
                lo = trial_spacing
            else:
                hi = trial_spacing
    vertices, faces, uv = best
    tri = vertices[faces]
    area = np.linalg.norm(np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]), axis=1) / 2
    if not len(area) or area.min() <= 1e-14:
        raise ValueError('Resampling made a degenerate material triangle')
    # Sample both edges and interiors against the same source parameterization.
    # This is explicitly a sampled deviation, not a Hausdorff certificate.
    fractions = [np.array([1/3]*3), np.array([.5, .5, 0.]),
                 np.array([0., .5, .5]), np.array([.5, 0., .5])]
    parameter_maximum = 0.
    samples = []
    for weights in fractions:
        source_uv = np.einsum('j,njk->nk', weights, uv[faces])
        proposed = np.einsum('j,njk->nk', weights, tri)
        samples.append(proposed)
        parameter_maximum = max(parameter_maximum, float(np.linalg.norm(_surface(grid, source_uv)-proposed, axis=1).max()))
    # UV interpolation is not geometric distance: changing strip width causes
    # tangential shifts even on a perfectly planar material surface. Measure
    # actual nearest surfaces in both directions. Normalize coordinates before
    # trimesh's triangle query to avoid its absolute tolerance on small leaves.
    cols = grid.shape[1]
    a = np.arange((grid.shape[0]-1)*cols).reshape(-1, cols)[:, :-1].ravel()
    source_faces = np.concatenate([np.stack([a, a+1, a+cols+1], axis=1),
                                   np.stack([a, a+cols+1, a+cols], axis=1)])
    source_vertices = grid.reshape(-1, 3)
    origin = source_vertices.mean(0)
    scale = np.linalg.norm(np.ptp(source_vertices, axis=0))
    forward = _distance((source_vertices-origin)/scale, source_faces,
                        (np.concatenate(samples)-origin)/scale)
    reverse = _distance((vertices-origin)/scale, faces, (source_vertices-origin)/scale)
    maximum = float(max(forward.max(), reverse.max())*scale)
    report = {'vertices': len(vertices), 'triangles': len(faces),
              'requested_spacing_m': spacing, 'selected_spacing_m': selected_spacing,
              'target_vertices': target_vertices, 'actual_area_m2': float(area.sum()),
              'sampled_source_surface_deviation_m': maximum,
              'sampled_parameter_interpolation_deviation_m': parameter_maximum,
              'root_vertices': list(range(root_samples)), 'root_nodes_all_bound_required': True,
              'scope': 'Source-surface resampling only; native mass, contact, shape and task gates remain required.'}
    return vertices, faces, uv, report
