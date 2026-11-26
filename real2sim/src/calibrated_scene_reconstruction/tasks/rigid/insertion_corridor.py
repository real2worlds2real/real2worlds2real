"""Geometry proposals for inserting a straight shaft past loose contents.

This does not move bodies or approve a task. Capsules must enclose the source
obstacles, and circular envelope constraints must fit inside the actual cavity.
The complete tool/object mesh and release dynamics still require native tests.
"""
import numpy as np


def stop_before_plane(points, translation, *, normal, offset, clearance):
    """Shorten a straight tool route using the entire source surface.

    Allowed half-space is ``normal @ point >= offset + clearance``. In
    particular, a tilted flat-ended shaft must stop on its lowest rim vertex,
    not its tip centre. This is geometry planning, not a contact certificate.
    """
    points=np.asarray(points,float);translation=np.asarray(translation,float)
    normal=np.asarray(normal,float)
    if (points.ndim!=2 or points.shape[1]!=3 or not len(points)
            or translation.shape!=(3,) or normal.shape!=(3,)
            or not np.isfinite(points).all() or not np.isfinite(translation).all()
            or not np.isfinite(normal).all() or not np.isfinite([offset,clearance]).all()
            or clearance<0 or not np.isclose(np.linalg.norm(normal),1.,atol=1e-10,rtol=0)):
        raise ValueError('Finite complete surface, translation, unit normal and nonnegative clearance required')
    initial=float(np.min(points@normal))-offset-clearance
    if initial < -1e-10:raise ValueError('Initial surface violates the stopping half-space')
    approach=float(translation@normal)
    fraction=min(1.,max(0.,initial)/-approach) if approach<0 else 1.
    return {'translation_m':(translation*fraction).tolist(),'fraction':float(fraction),
            'initial_clearance_m':initial+clearance,
            'final_clearance_m':initial+clearance+approach*fraction,
            'proposal_only':True,'task_approved':False}


def swept_point_clearance(points, translation, obstacles):
    """Minimum capsule clearance of sampled surface points under translation.

    Finite segment distances include interior/interior crossings, not merely
    endpoints. This is a route proposal; unsampled triangles and moving contents
    still require full native checks.
    """
    points=np.asarray(points,float);u=np.asarray(translation,float)
    if points.ndim!=2 or points.shape[1]!=3 or not len(points) or u.shape!=(3,) or not np.isfinite(points).all() or not np.isfinite(u).all():
        raise ValueError('Finite surface points and translation required')
    aa=float(u@u);result=[]
    for obstacle in obstacles:
        ends=np.asarray(obstacle['axis'],float);radius=float(obstacle['radius'])
        if ends.shape!=(2,3) or not np.isfinite(ends).all() or not np.isfinite(radius) or radius<=0:
            raise ValueError('Finite obstacle axis and positive radius required')
        c,d=ends;v=d-c;cc=float(v@v);bb=float(u@v)
        def point_segment(p,a,direction,norm):
            t=np.clip(np.einsum('ij,j->i',p-a,direction)/norm,0,1) if norm>0 else np.zeros(len(p))
            return np.linalg.norm(p-a-t[:,None]*direction,axis=1)
        values=[point_segment(points,c,v,cc),point_segment(points+u,c,v,cc),
                point_segment(np.broadcast_to(c,points.shape),points,u,aa),
                point_segment(np.broadcast_to(d,points.shape),points,u,aa)]
        det=aa*cc-bb*bb
        if det>1e-12*aa*cc:
            w=points-c;dd=w@u;ee=w@v;s=(bb*ee-cc*dd)/det;t=(aa*ee-bb*dd)/det
            distance=np.linalg.norm(w+s[:,None]*u-t[:,None]*v,axis=1)
            values.append(np.where((s>=0)&(s<=1)&(t>=0)&(t<=1),distance,np.inf))
        result.append(float(np.min(values))-radius)
    if not result:raise ValueError('At least one obstacle required')
    return np.asarray(result)


def segment_distance(first, second):
    """Minimum distance between two finite 3-D segments, including parallel ones."""
    first, second = np.asarray(first, float), np.asarray(second, float)
    if (first.shape != (2, 3) or second.shape != (2, 3)
            or not np.isfinite(first).all() or not np.isfinite(second).all()):
        raise ValueError('Two finite 3-D endpoint pairs required')
    a, b = first; c, d = second
    u, v, w = b-a, d-c, a-c
    aa, bb, cc = u@u, u@v, v@v

    def point_distance(p, x, direction, length2):
        t = np.clip((p-x)@direction/length2, 0., 1.) if length2 > 0 else 0.
        return np.linalg.norm(p-x-t*direction)

    distances = [point_distance(p, c, v, cc) for p in first]
    distances += [point_distance(p, a, u, aa) for p in second]
    determinant = aa*cc-bb*bb
    if determinant > 1e-12*aa*cc:
        dd, ee = u@w, v@w
        s, t = (bb*ee-cc*dd)/determinant, (aa*ee-bb*dd)/determinant
        if 0 <= s <= 1 and 0 <= t <= 1:
            distances.append(np.linalg.norm(w+s*u-t*v))
    return float(min(distances))


def margins(parameters, *, centre_xy, aperture_z, swept_z, shaft_radius,
            envelope, obstacles):
    """Clearances of one coaxial translation; envelope rows are (z, radius).

    Parameters are mouth offset x/y and slope dx/dz, dy/dz. Circular envelope
    sections constrain the elliptical intersection of a tilted cylinder. All
    geometry uses one shared metric frame. No task-specific dimensions occur.
    """
    p = np.asarray(parameters, float); centre = np.asarray(centre_xy, float)
    span = np.asarray(swept_z, float); sections = np.asarray(envelope, float)
    if (p.shape != (4,) or centre.shape != (2,) or span.shape != (2,)
            or sections.ndim != 2 or sections.shape[1] != 2 or not len(sections)
            or not all(np.isfinite(a).all() for a in [p, centre, span, sections])
            or not np.isfinite(aperture_z) or not np.isfinite(shaft_radius)
            or shaft_radius <= 0 or span[1] <= span[0] or (sections[:, 1] <= 0).any()):
        raise ValueError('Finite metric shaft, sweep and cavity envelope required')
    mouth = centre+p[:2]
    line = np.c_[mouth+(span-aperture_z)[:, None]*p[2:], span]
    section_offsets = p[:2]+(sections[:, 0]-aperture_z)[:, None]*p[2:]
    ellipse_radius = shaft_radius*np.sqrt(1+p[2:]@p[2:])
    values = (sections[:, 1]-np.linalg.norm(section_offsets, axis=1)-ellipse_radius).tolist()
    for obstacle in obstacles:
        radius = float(obstacle['radius'])
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError('Positive enclosing obstacle radius required')
        values.append(segment_distance(line, obstacle['axis'])-shaft_radius-radius)
    return np.asarray(values)


def propose(*, maximum_slope, seed=0, **geometry):
    """Find a corridor proposal; an obstructed result remains explicitly failed."""
    from scipy.optimize import differential_evolution
    if not np.isfinite(maximum_slope) or maximum_slope < 0:
        raise ValueError('Finite nonnegative slope bound required')
    # Validate before optimization; never convert malformed geometry to a loss.
    margins(np.zeros(4), **geometry)
    sections = np.asarray(geometry['envelope'], float)
    width = float(sections[:, 1].min())
    bounds = [(-width, width)]*2+[(-maximum_slope, maximum_slope)]*2
    result = differential_evolution(lambda p: -float(margins(p, **geometry).min()),
                                    bounds, seed=seed, maxiter=400, popsize=20, tol=1e-8)
    clearances = margins(result.x, **geometry)
    return {'proposal_only': True, 'task_approved': False,
            'parameters': result.x.tolist(), 'clearances_m': clearances.tolist(),
            'proxy_clearance_m': float(clearances.min()),
            'proxy_clear': bool(clearances.min() > 0),
            'scope': 'Static capsule/envelope proposal; full mesh, native contact and release checks required'}
