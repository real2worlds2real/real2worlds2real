"""Optional explicit contact numerics for existing native volume/rope bodies."""
import math


def rigid_contacts(points,faces,surfaces,poses,maximum_penetration_m):
    """Sample an actual soft boundary against actual rigid partner frames."""
    import numpy as np
    from calibrated_scene_reconstruction.assets.surface_distance import triangle_samples
    if (set(surfaces)!=set(poses) or not np.isfinite(maximum_penetration_m)
            or maximum_penetration_m<0):
        raise ValueError('Matching actual rigid surfaces/poses and finite penetration budget required')
    vertices,triangles=np.asarray(points,float),np.asarray(faces)
    if (vertices.ndim!=2 or vertices.shape[1:]!=(3,) or triangles.ndim!=2
            or triangles.shape[1:]!=(3,) or not len(triangles) or triangles.dtype.kind not in 'iu'
            or triangles.min()<0 or triangles.max()>=len(vertices)):
        raise ValueError('Actual indexed soft boundary required')
    # Legacy native buffers include unused allocation padding. Query only the
    # physical boundary, never those unrelated (possibly undefined) nodes.
    used,inverse=np.unique(triangles,return_inverse=True)
    probes=triangle_samples(vertices[used],inverse.reshape(-1,3));depth={}
    for name,surface in surfaces.items():
        position,rotation=[np.asarray(x,float) for x in poses[name]]
        if (position.shape!=(3,) or rotation.shape!=(3,3) or not np.isfinite(position).all()
                or not np.isfinite(rotation).all() or not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-6)
                or not np.isclose(np.linalg.det(rotation),1.,atol=1e-6)):
            raise ValueError('Proper actual rigid partner frames required')
        value=float(surface.max_penetration((probes-position)@rotation))
        if not np.isfinite(value) or value<0:raise ValueError('Actual nonnegative penetration depth required')
        depth[name]=value
    return {'pass':all(v<=maximum_penetration_m for v in depth.values()),
            'penetration_m':depth,'maximum_penetration_m':float(maximum_penetration_m),
            'scope':'Sampled actual soft-boundary vertices, triangle centres and edge midpoints against rigid surfaces; no normal-pressure, retention or task approval.'}


def settings(value):
    iterations=value.get('solver_position_iterations',32)
    if type(iterations) is not int or not 4<=iterations<=255:
        raise ValueError('Volume solver iterations must be an integer in4..255')
    keys={'contact_offset','rest_offset'}
    present=keys & value.keys()
    result={'solver_position_iterations':iterations}
    if present:
        if present!=keys:raise ValueError('Explicit volume contact and rest offsets must be paired')
        contact,rest=value['contact_offset'],value['rest_offset']
        if (any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) for x in [contact,rest])
                or not 0<=rest<contact<=.1):
            raise ValueError('Volume offsets require finite0<=rest<contact<=0.1m')
        result.update(contact_offset=float(contact),rest_offset=float(rest))
    return result
