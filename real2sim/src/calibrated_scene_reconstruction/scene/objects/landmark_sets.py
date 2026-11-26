"""Unordered source landmarks for initializing repeated rigid parts.

Only calibrated camera rays and independently reviewed image points enter this
proposal. A fit to these points never replaces full-object or native checks.
"""
from itertools import combinations, permutations

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment


def mask_line(mask):
    """Propose an image centreline from a reviewed narrow straight-part mask.

    End caps and partial occlusion do not identify physical endpoints. Median
    cross sections reduce boundary quantization without changing source labels;
    callers must still score the actual full mesh against every source view.
    """
    mask = np.asarray(mask, bool)
    if mask.ndim != 2 or mask.sum() < 20:
        raise ValueError('A resolved reviewed narrow-part mask is required')
    y, x = np.nonzero(mask); points = np.column_stack([x, y]).astype(float)
    centre = np.median(points, axis=0)
    _, singular, axes = np.linalg.svd(points-centre, full_matrices=False)
    if singular[0] < 3*singular[1]:
        raise ValueError('Source part is not distinctly elongated')
    along = (points-centre) @ axes[0]; across = (points-centre) @ axes[1]
    edges = np.linspace(*np.quantile(along, [.1, .9]), 9); sections = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        keep = (along >= lo) & (along <= hi)
        if keep.sum() >= 3:
            sections.append([np.median(along[keep]), np.median(across[keep])])
    if len(sections) < 4:
        raise ValueError('Too few visible shaft cross sections')
    sections = np.asarray(sections)
    initial = np.polyfit(sections[:, 0], sections[:, 1], 1)
    fitted = least_squares(lambda p: sections[:, 0]*p[0]+p[1]-sections[:, 1],
                           initial, loss='soft_l1', f_scale=.5)
    samples = np.array([sections[:, 0].min(), sections[:, 0].max()])
    pixels = centre + samples[:, None]*axes[0] + (samples*fitted.x[0]+fitted.x[1])[:, None]*axes[1]
    return {'pixels': pixels.tolist(), 'cross_section_residual_px':
            (sections[:, 0]*fitted.x[0]+fitted.x[1]-sections[:, 1]).tolist(),
            'proposal_only': True, 'physical_endpoints': False}


def line_plane_uncertainty(views, point, direction, plane_point, plane_normal, *, pixel_sigma=1.):
    """Local uncertainty of a source axis where it crosses a chosen plane.

    Assumes independent source endpoint errors with the stated pixel sigma.
    Camera errors, repeated-frame correlations and misidentified edges are
    extra. This is not clearance approval or a reason to relax a source gate.
    """
    point, direction, plane_point, plane_normal = [np.asarray(x, float) for x in
                                                (point, direction, plane_point, plane_normal)]
    if (any(x.shape != (3,) for x in (point, direction, plane_point, plane_normal))
            or not np.isfinite(np.r_[point, direction, plane_point, plane_normal, pixel_sigma]).all()
            or pixel_sigma <= 0 or np.linalg.norm(direction) < 1e-10
            or np.linalg.norm(plane_normal) < 1e-10 or len(views) < 4):
        raise ValueError('Finite axis, crossing plane, four views and positive pixel uncertainty required')
    direction = direction/np.linalg.norm(direction)
    plane_normal = plane_normal/np.linalg.norm(plane_normal)
    if abs(direction@plane_normal) < 1e-5:
        raise ValueError('Axis is parallel to the crossing plane')
    basis = np.linalg.svd(direction[None], full_matrices=True)[2][1:]
    # Place the pivot at the requested crossing, avoiding large arbitrary axis
    # anchors and mixing long-axis gauge with actual localization uncertainty.
    pivot = point+direction*((plane_point-point)@plane_normal)/(direction@plane_normal)
    rows = []
    for view in views:
        k, r, c, p = (np.asarray(view[key], float) for key in ('K', 'R', 'C', 'pixels'))
        if (k.shape != (3, 3) or r.shape != (3, 3) or c.shape != (3,) or p.shape != (2, 2)
                or not all(np.isfinite(x).all() for x in (k, r, c, p))
                or abs(np.linalg.det(k)) < 1e-12
                or not np.allclose(r.T@r, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(r), 1.)):
            raise ValueError('Calibrated cameras and two finite line samples required')
        if np.linalg.norm(p[0]-p[1]) < 1e-8:
            raise ValueError('Distinct source line samples required')
        rows.append((k, r, c, np.c_[p, np.ones(2)]))

    def values(q):
        anchor = pivot+q[:2]@basis
        axis = direction+q[2:]@basis
        errors = []
        for k, r, c, pixels in rows:
            line = np.linalg.solve(k.T, np.cross((anchor-c)@r, axis@r))
            if np.linalg.norm(line[:2]) < 1e-12:
                raise ValueError('Axis projects through a camera centre')
            errors.extend(pixels@line/np.linalg.norm(line[:2]))
        crossing = anchor+axis*((plane_point-anchor)@plane_normal)/(axis@plane_normal)
        return np.array(errors), crossing

    columns = []; crossing_columns = []
    for j in range(4):
        q = np.zeros(4); q[j] = 1e-6
        plus, cp = values(q); minus, cm = values(-q)
        columns.append((plus-minus)/2e-6); crossing_columns.append((cp-cm)/2e-6)
    jac = np.array(columns).T
    _, singular, vt = np.linalg.svd(jac, full_matrices=False)
    resolved = bool(singular[-1] > max(1e-10, singular[0]*1e-8))
    covariance = None
    if resolved:
        parameter_covariance = (vt.T/singular**2)@vt*pixel_sigma**2
        cross_jac = np.array(crossing_columns).T
        covariance = cross_jac@parameter_covariance@cross_jac.T
    return {'crossing': pivot.tolist(), 'resolved': resolved,
            'pixel_sigma': float(pixel_sigma), 'jacobian_singular_values': singular.tolist(),
            'crossing_covariance_m2': None if covariance is None else covariance.tolist(),
            'largest_crossing_sigma_m': None if covariance is None else
                float(np.sqrt(max(0., np.linalg.eigvalsh(covariance)[-1]))),
            'scope': 'Local independent endpoint noise only; calibration, temporal correlation, edge identity and physical contacts remain unchecked',
            'proposal_only': True}


def distal_tip(mask, origin, *, native_scale=1., cap_pixels=1.5):
    """A distal visible cap, with tolerance in native source pixels.

    An area percentile retreats along a thin shaft as its attached body grows.
    This uses distance from the distal boundary instead. Identity, visibility
    and whether the cap itself is occluded still require source review.
    """
    mask = np.asarray(mask, bool); origin = np.asarray(origin, float)
    if (mask.ndim != 2 or not mask.any() or origin.shape != (2,)
            or not np.isfinite(origin).all() or not np.isfinite([native_scale, cap_pixels]).all()
            or native_scale <= 0 or cap_pixels <= 0):
        raise ValueError('Visible source mask, body origin and positive native pixel sampling required')
    y, x = np.nonzero(mask); points = np.column_stack([x, y])
    distance = np.linalg.norm(points-origin, axis=1)
    return np.median(points[distance >= distance.max()-cap_pixels*native_scale], axis=0)


def triangulate_lines(views):
    """Intersect calibrated source interpretation planes; return an infinite line.

    Image endpoints define each observed line, not the physical shaft endpoints.
    This proposal does not identify radius, length, flexibility or support.
    """
    if len(views) < 4:
        raise ValueError('At least four independently observed lines required')
    normals = []; offsets = []
    for view in views:
        k, r, c, pixels = (np.asarray(view[key], float) for key in ['K', 'R', 'C', 'pixels'])
        if (k.shape != (3, 3) or r.shape != (3, 3) or c.shape != (3,) or pixels.shape != (2, 2)
                or not all(np.isfinite(v).all() for v in [k, r, c, pixels])
                or not np.allclose(r.T@r, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(r), 1.) or abs(np.linalg.det(k)) < 1e-12):
            raise ValueError('Finite calibrated cameras and two distinct source line pixels required')
        line = np.cross(np.r_[pixels[0], 1.], np.r_[pixels[1], 1.])
        n = r @ (k.T@line)
        if np.linalg.norm(n) < 1e-10:
            raise ValueError('Source line endpoints must be distinct')
        n /= np.linalg.norm(n); normals.append(n); offsets.append(n@c)
    normals = np.asarray(normals); offsets = np.asarray(offsets)
    _, singular, vt = np.linalg.svd(normals, full_matrices=False)
    if singular[1] < 1e-4:
        raise ValueError('Source line planes do not resolve3D direction')
    direction = vt[-1]
    point = np.linalg.lstsq(np.vstack([normals, direction]), np.r_[offsets, 0.], rcond=None)[0]
    # Refine the infinite line in image space, rather than connecting separately
    # triangulated physical tips to a fitted point. Segment endpoints below are
    # observations of a line, not correspondences of physical endpoints.
    origin = np.mean([np.asarray(v['C'], float) for v in views], axis=0)
    basis = vt[:2]
    initial_direction = direction.copy()
    initial_point = point-origin-direction*np.dot(point-origin, direction)
    initial = np.r_[basis@initial_point, 0., 0.]

    def unpack(values):
        axis = initial_direction+values[2:]@basis
        axis /= np.linalg.norm(axis)
        return origin+values[:2]@basis, axis

    def residual(values):
        anchor, axis = unpack(values)
        errors = []
        for view in views:
            k, r, c, pixels = (np.asarray(view[key], float) for key in ['K', 'R', 'C', 'pixels'])
            normal = np.linalg.solve(k.T, np.cross((anchor-c)@r, axis@r))
            denominator = np.linalg.norm(normal[:2])
            if denominator < 1e-12:
                # A line through a camera has no resolved image-line normal.
                errors.extend([1e6, 1e6])
            else:
                errors.extend(np.c_[pixels, np.ones(2)]@normal/denominator)
        return np.asarray(errors)

    fitted = least_squares(residual, initial, loss='soft_l1', f_scale=1., max_nfev=500)
    point, direction = unpack(fitted.x)
    errors = abs(residual(fitted.x)).reshape(-1, 2)
    return {'point': point.tolist(), 'direction': direction.tolist(),
            'plane_distance_m': (normals@point-offsets).tolist(),
            'plane_direction_residual': (normals@direction).tolist(),
            'segment_endpoint_line_errors_px': errors.tolist(),
            'median_line_error_px': float(np.median(errors)),
            'p95_line_error_px': float(np.percentile(errors, 95)),
            'optimizer_converged': bool(fitted.success),
            'interpretation_plane_singular_values': singular.tolist(),
            'source_views_removed': False,
            'scope': 'Infinite axis proposal only; source identity, finite tips, radius, hidden extent and physical support require independent checks',
            'proposal_only': True, 'model_or_task_dimensions_read': False}


def triangulate_line_sets(views, count, *, refinements=6):
    """Propose repeated infinite axes from unordered visible source segments.

    Each view supplies 1..count distinct segments, with shape (N,2,2).
    Deduplication and visible identity must be reviewed from source images.
    Missing segments are not negatives. No tip, length or contact hypothesis
    participates; all supplied views remain in the residual report.
    """
    if type(count) is not int or not 1 <= count <= 5 or len(views) < 4:
        raise ValueError('Require 1..5 axes and at least four source views')
    if type(refinements) is not int or refinements < 1:
        raise ValueError('Positive number of proposal refinements required')
    rows = []
    for view in views:
        k, r, c, pixels = (np.asarray(view[key], float) for key in ['K', 'R', 'C', 'pixels'])
        if (k.shape != (3, 3) or r.shape != (3, 3) or c.shape != (3,)
                or pixels.ndim != 3 or pixels.shape[1:] != (2, 2) or not 1 <= len(pixels) <= count
                or not all(np.isfinite(v).all() for v in [k, r, c, pixels])
                or not np.allclose(r.T@r, np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(r), 1.) or abs(np.linalg.det(k)) < 1e-12):
            raise ValueError('Finite calibrated cameras and distinct source segments required')
        homogeneous = np.concatenate([pixels, np.ones((*pixels.shape[:2], 1))], axis=2)
        lines = np.cross(homogeneous[:, 0], homogeneous[:, 1])
        normals = lines@k@r.T
        lengths = np.linalg.norm(normals, axis=1)
        if np.any(lengths < 1e-10):
            raise ValueError('Source segment endpoints must be distinct')
        rows.append(dict(K=k, R=r, C=c, pixels=homogeneous, normals=normals/lengths[:, None]))
    full = [i for i, row in enumerate(rows) if len(row['pixels']) == count]
    if len(full) < 2:
        raise ValueError('Two complete source views required for unordered axis initialization')
    origin = np.mean([r['C'] for r in rows], axis=0)

    def errors(axes, row):
        normals = np.array([np.cross((point-row['C'])@row['R'], direction@row['R'])
                            for point, direction in axes])@np.linalg.inv(row['K'])
        lengths = np.linalg.norm(normals[:, :2], axis=1)
        normals /= np.maximum(lengths[:, None], 1e-12)
        values = np.einsum('sec,ac->sae', row['pixels'], normals)
        values[:, lengths < 1e-12] = 1e6
        observed, assigned = linear_sum_assignment(np.mean(abs(values), axis=2))
        return values[observed, assigned], assigned

    def residual(axes):
        return np.concatenate([errors(axes, row)[0].ravel() for row in rows])

    seeds = []
    for ia, ib in combinations(full, 2):
        a, b = rows[ia], rows[ib]
        if np.linalg.norm(a['C']-b['C']) < 1e-8:
            continue
        for order in permutations(range(count)):
            axes = []
            for j, match in enumerate(order):
                normals = np.array([a['normals'][j], b['normals'][match]])
                direction = np.cross(*normals)
                length = np.linalg.norm(direction)
                if length < 1e-4:
                    break
                direction /= length
                offset = [normals[0]@(a['C']-origin), normals[1]@(b['C']-origin), 0.]
                point = origin+np.linalg.solve(np.vstack([normals, direction]), offset)
                axes.append((point, direction))
            if len(axes) == count:
                value = residual(axes)
                seeds.append((float(np.mean(np.sqrt(1+value**2)-1)), axes))
    if not seeds:
        raise ValueError('Source line planes do not resolve an unordered 3D proposal')
    best = None
    for _, seed in sorted(seeds, key=lambda item: item[0])[:refinements]:
        bases = [np.linalg.svd(direction[None], full_matrices=True)[2][1:] for _, direction in seed]

        def unpack(values):
            axes = []
            for q, (point, direction), basis in zip(values.reshape(count, 4), seed, bases):
                axis = direction+q[2:]@basis
                axes.append((point+q[:2]@basis, axis/np.linalg.norm(axis)))
            return axes

        result = least_squares(lambda q: residual(unpack(q)), np.zeros(count*4),
                               loss='soft_l1', f_scale=1., max_nfev=250)
        axes = unpack(result.x); value = residual(axes)
        cost = float(np.mean(np.sqrt(1+value**2)-1))
        if best is None or cost < best[0]:
            best = cost, axes, result.success
    cost, axes, converged = best
    per_view = [errors(axes, row) for row in rows]
    absolute = abs(residual(axes))
    return {'axes': [{'point': p.tolist(), 'direction': d.tolist()} for p, d in axes],
            'assignments': [a.tolist() for _, a in per_view],
            'segment_endpoint_line_errors_px': [abs(e).tolist() for e, _ in per_view],
            'median_line_error_px': float(np.median(absolute)),
            'p95_line_error_px': float(np.percentile(absolute, 95)),
            'optimizer_converged': bool(converged), 'source_views_removed': False,
            'proposal_only': True, 'model_or_task_dimensions_read': False,
            'scope': 'Unordered infinite axes only; source identity/coverage, finite extent, radius and support remain unverified'}


def triangulate_sets(views, count, *, max_median_px=4., min_ray_angle_deg=1.):
    """Match equal-sized, unordered point sets without reading a model.

    Ambiguous or missing point sets must be kept in the caller's source review;
    this routine requires at least four complete views. Cardinality is an
    observed identity hypothesis, not a target imposed on segmentation.
    """
    if (type(count) is not int or not 1 <= count <= 5 or len(views) < 4
            or not np.isfinite([max_median_px, min_ray_angle_deg]).all()
            or max_median_px <= 0 or not 0 < min_ray_angle_deg < 90):
        raise ValueError('Require1..5 observed landmarks, four views and finite positive tolerances')
    rows=[]
    for view in views:
        row={key: np.asarray(view[key], float) for key in ['K', 'R', 'C', 'pixels']}
        if (any(row[k].shape != shape for k, shape in
                [('K',(3,3)),('R',(3,3)),('C',(3,)),('pixels',(count,2))])
                or not all(np.isfinite(value).all() for value in row.values())
                or not np.allclose(row['R'].T@row['R'],np.eye(3),atol=1e-5)
                or not np.isclose(np.linalg.det(row['R']),1.,atol=1e-5)
                or abs(np.linalg.det(row['K']))<1e-12):
            raise ValueError('Finite source pixels, intrinsics and world-from-camera rotation required')
        rays=np.c_[row['pixels'],np.ones(count)]@np.linalg.inv(row['K']).T@row['R'].T
        row['rays']=rays/np.linalg.norm(rays,axis=1)[:,None]
        rows.append(row)

    def project(points,row):
        camera=(points-row['C'])@row['R'];pixel=camera@row['K'].T
        return pixel[:,:2]/np.maximum(pixel[:,2,None],1e-12)

    def residual(values, selected):
        points=values.reshape(count,3);result=[]
        for row in selected:
            predicted=project(points,row)
            a,b=linear_sum_assignment(np.linalg.norm(predicted[:,None]-row['pixels'][None],axis=2))
            result.extend((predicted[a]-row['pixels'][b]).ravel())
        return np.asarray(result)

    best=None
    identity=np.eye(3)
    for ia,ib in combinations(range(len(rows)),2):
        first,second=rows[ia],rows[ib]
        for ordering in permutations(range(count)):
            left,right=first['rays'],second['rays'][list(ordering)]
            angles=np.degrees(np.arccos(np.clip(np.abs((left*right).sum(1)),-1.,1.)))
            if angles.min()<min_ray_angle_deg:continue
            points=[]
            for a,b in zip(left,right):
                pa,pb=identity-np.outer(a,a),identity-np.outer(b,b)
                points.append(np.linalg.lstsq(np.vstack([pa,pb]),
                              np.r_[pa@first['C'],pb@second['C']],rcond=None)[0])
            points=np.asarray(points)
            if any((((points-row['C'])@row['R'])[:,2]<=0).any() for row in rows):continue
            error=float(np.median(np.linalg.norm(residual(points,rows).reshape(-1,2),axis=1)))
            if best is None or error<best[0]:best=(error,points,[ia,ib])
    if best is None:raise ValueError('No source landmark baseline resolves depth')
    fit=least_squares(lambda x:residual(x,rows),best[1].ravel(),loss='soft_l1',f_scale=2.,max_nfev=300)
    points=fit.x.reshape(count,3)
    errors=np.linalg.norm(residual(points,rows).reshape(len(rows),count,2),axis=2)
    positive=all((((points-row['C'])@row['R'])[:,2]>0).all() for row in rows)
    median=float(np.median(errors));p95=float(np.percentile(errors,95))
    return {'points':points.tolist(),'median_reprojection_px':median,
            'p95_reprojection_px':p95,'per_view_errors_px':errors.tolist(),
            'seed_views':best[2],'proposal_ok':bool(positive and median<=max_median_px and p95<=3*max_median_px),
            'proposal_only':True,'model_or_task_dimensions_read':False,
            'scope':'Initialization only; preserve all original source, geometry, contact and task gates'}
