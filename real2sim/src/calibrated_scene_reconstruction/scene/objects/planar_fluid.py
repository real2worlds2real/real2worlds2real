"""Source-only planar liquid footprints in independently calibrated cameras.

The supporting plane comes from scene geometry. Multi-view mask consensus
estimates the visible footprint, never film thickness or physical viscosity.
Held-out masks only evaluate the proposal and cannot alter it.
"""
import cv2
import numpy as np


def seed_film(footprint, *, thickness_m, thickness_basis):
    """Extrude a reviewed planar footprint into resolved native seed layers.

    Thickness is a separately disclosed hypothesis, never inferred from an RGB
    silhouette. This is source preparation, not a proof of fluid convergence.
    A single PBD layer can collapse into clumps despite small velocities; retain
    at least three layers and require the caller's native resolution control.
    """
    points = np.asarray(footprint['points'], float)
    report = footprint['report']
    spacing = float(report['spacing_m'])
    plane = report['plane']
    normal = np.asarray(plane['normal'], float)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 8
            or not np.isfinite(points).all() or normal.shape != (3,)
            or not np.isfinite(normal).all() or not np.isclose(normal@normal, 1.)
            or not np.isfinite(spacing) or spacing <= 0
            or not np.isfinite(thickness_m) or thickness_m <= 0
            or not isinstance(thickness_basis, str) or not thickness_basis.strip()):
        raise ValueError('Finite planar sampling and an explicit thickness hypothesis required')
    if np.max(np.abs(points@normal-plane['offset'])) > max(1e-9, spacing*1e-3):
        raise ValueError('Footprint points must lie on their independently observed support plane')
    layers = round(thickness_m/spacing)
    if layers < 3 or not np.isclose(layers*spacing, thickness_m, rtol=1e-6, atol=1e-10):
        raise ValueError('Resample the footprint to at least three complete layers across film thickness')
    if len(points)*layers > 2000000:
        raise ValueError('Film exceeds the explicit two million particle sampling budget')
    seed = (points[:, None, :] + normal[None, None, :]
            * ((np.arange(layers)+.5)*spacing)[None, :, None]).reshape(-1, 3)
    return dict(points=seed, report=dict(accepted=False, layers=layers, spacing_m=spacing,
        thickness_m=float(thickness_m), thickness_basis=thickness_basis,
        nominal_seed_volume_m3=float(len(seed)*spacing**3),
        source_footprint_area_m2=float(len(points)*spacing**2),
        scope='Native seed proposal only; thickness, density, resolution, source footprint and task checks remain separate'))


def propose(cameras, observations, plane, *, spacing_m=.0008, maximum_cells=2000000):
    names = list(map(str, cameras['names']))
    n = np.asarray(plane['normal'], float)
    d = float(plane['offset'])
    if n.shape != (3,) or not np.isfinite(np.r_[n, d]).all() or not np.isclose(n@n, 1.) or n[2] <= .2:
        raise ValueError('An independently observed upward support plane is required')
    if not np.isfinite(spacing_m) or not 1e-5 <= spacing_m <= .01:
        raise ValueError('Finite metric planar sampling resolution required')
    if type(maximum_cells) is not int or maximum_cells < 1:
        raise ValueError('Positive bounded sampling budget required')
    origin = n*d
    ex = np.array([1., 0., -n[0]/n[2]])
    ex /= np.linalg.norm(ex)
    axes = np.stack([ex, np.cross(n, ex)], axis=1)
    rows, bounds, seen = [], [], set()
    for row in observations:
        name = row['frame']
        if name not in names or name in seen or type(row.get('training')) is not bool:
            raise ValueError('Distinct registered views with explicit train/held-out roles required')
        seen.add(name)
        i = names.index(name)
        mask = np.asarray(row['mask'])
        w, h = map(int, cameras['size'][i])
        if mask.dtype != bool or mask.shape != (h, w) or not mask.any():
            raise ValueError('Reviewed nonempty masks must match the full calibrated source raster')
        if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
            raise ValueError('A cropped liquid boundary cannot establish its complete footprint')
        R, C, K = (np.asarray(cameras[key][i]) for key in ('R_world_cam', 'centre', 'K'))
        # Homography from metric plane coordinates to this pinhole image.
        H = K @ R.T @ np.column_stack([axes, origin-C])
        if abs(np.linalg.det(H)) < 1e-12:
            raise ValueError('Support plane is degenerate in a source view')
        if row['training']:
            contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pixels = np.vstack(contours).reshape(-1, 2)
            q = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(H).T
            if not np.isfinite(q).all() or np.any(abs(q[:, 2]) < 1e-10):
                raise ValueError('Footprint rays do not intersect a finite plane patch')
            xy = q[:, :2]/q[:, 2:]
            world = xy @ axes.T + origin
            if np.any(((world-C) @ R)[:, 2] <= 0):
                raise ValueError('Liquid boundary intersects behind a source camera')
            bounds.append(xy)
        rows.append((name, mask, H, row['training']))
    training_count = sum(row[3] for row in rows)
    if training_count < 3 or len(rows)-training_count < 2:
        raise ValueError('At least three training views and two held-out views required')
    contour = np.concatenate(bounds)
    lo = np.floor(contour.min(0)/spacing_m)*spacing_m-2*spacing_m
    hi = np.ceil(contour.max(0)/spacing_m)*spacing_m+2*spacing_m
    shape = np.ceil((hi-lo)/spacing_m).astype(int)+1
    if np.prod(shape.astype(float)) > maximum_cells or np.any(shape < 2):
        raise ValueError('Projected liquid footprint exceeds the explicit sampling budget')
    xx, yy = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    xy = np.c_[xx.ravel(), yy.ravel()]*spacing_m+lo
    coordinates = np.c_[xy, np.ones(len(xy))]
    votes = np.zeros(len(xy), int)
    visibility = np.zeros(len(xy), int)
    for _, mask, H, training in rows:
        if not training:
            continue
        q = coordinates @ H.T
        uv = q[:, :2]/q[:, 2:]
        valid = (q[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < mask.shape[1]-1) & (uv[:, 1] >= 0) & (uv[:, 1] < mask.shape[0]-1)
        ids = np.flatnonzero(valid)
        ix, iy = np.rint(uv[ids]).astype(int).T
        visibility[ids] += 1
        votes[ids] += mask[iy, ix]
    selected = (visibility == training_count) & (votes > training_count/2)
    if selected.sum() < 8:
        raise ValueError('Insufficient multi-view liquid footprint consensus')
    grid = selected.reshape(shape[1], shape[0]).astype('uint8')
    grid_to_plane = np.array([[spacing_m, 0, lo[0]], [0, spacing_m, lo[1]], [0, 0, 1.]])
    checks, rendered = [], {}
    for name, mask, H, training in rows:
        predicted = cv2.warpPerspective(grid, H @ grid_to_plane, mask.shape[::-1], flags=cv2.INTER_NEAREST) > 0
        checks.append(dict(frame=name, training=training,
                           iou=float((mask & predicted).sum()/max(1, (mask | predicted).sum())),
                           source_pixels=int(mask.sum()), predicted_pixels=int(predicted.sum())))
        rendered[name] = predicted
    return dict(points=xy[selected] @ axes.T + origin, grid=grid, axes=axes, origin=origin,
                grid_to_plane=grid_to_plane, rendered=rendered,
                report=dict(accepted=False, area_m2=float(selected.sum()*spacing_m**2),
                            cells=int(selected.sum()), spacing_m=spacing_m, views=checks,
                            scope=__doc__.strip(), plane=plane))
