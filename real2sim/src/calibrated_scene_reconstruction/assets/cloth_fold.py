"""Developable corrugation proposals; no change to the intrinsic/rest material."""
import numpy as np


def corrugation(points, *, panels, radius, turn_fraction=1., maximum_turn_degrees=None):
    points=np.asarray(points,float)
    if (points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all()
            or not np.allclose(points[:,2],points[0,2]) or type(panels) is not int
            or not 2<=panels<=12 or not np.isfinite([radius,turn_fraction]).all() or radius<=0
            or not 0<=turn_fraction<=1):
        raise ValueError('Planar material chart and bounded explicit corrugation proposal required')
    x=np.unique(points[:,0]);length=float(np.ptp(x));arc=np.pi*radius
    if arc>length/panels/2:raise ValueError('Fold radius consumes the proposed panel')
    cuts=np.linspace(x.min(),x.max(),panels+1)[1:-1]
    middle=(x[1:]+x[:-1])/2;theta=np.zeros(len(middle))
    for i,cut in enumerate(cuts):
        theta+=(-1 if i%2 else 1)*np.pi*turn_fraction*np.clip((middle-cut)/arc+.5,0,1)
    if maximum_turn_degrees is not None:
        if (not np.isfinite(maximum_turn_degrees) or not 0<maximum_turn_degrees<=90):
            raise ValueError('Resolved crease angle bound must be positive and at most90degrees')
        turns=np.diff(np.r_[0.,theta,np.pi*turn_fraction*((panels-1)%2)])
        if abs(turns).max(initial=0)>np.radians(maximum_turn_degrees)+1e-12:
            raise ValueError('Material sampling cannot resolve the rounded crease; refine across the fold before constructing it')
    step=np.diff(x)[:,None]*np.c_[np.cos(theta),np.sin(theta)]
    cross=np.vstack([np.zeros(2),np.cumsum(step,axis=0)])
    cross-=(cross.min(0)+cross.max(0))/2
    result=np.c_[cross[np.searchsorted(x,points[:,0])],points[:,1]]
    return result


def rounded_layered(points, *, panels, layer_gap, maximum_turn_degrees=60.):
    """Resolved accordion in width/length/thickness coordinates.

    Uniform or anisotropic material sampling must resolve each turn; this
    refuses a sharp one-edge substitute. Discrete panel separation is measured
    independently, rather than assumed equal to the requested circle diameter.
    This remains a geometric/history proposal, not native material approval.
    """
    if not np.isfinite(layer_gap) or layer_gap<=0:
        raise ValueError('Positive finite proposed layer gap required')
    return corrugation(points,panels=panels,radius=layer_gap/2,
                       maximum_turn_degrees=maximum_turn_degrees)[:,[0,2,1]]


def layered(points, *, panels, layer_gap):
    """Discrete isometric accordion with explicit separation between panels.

    Each transition occupies one existing material edge. This is a folded
    mid-surface hypothesis, not a resolved circular crease or inferred memory.
    It avoids silently placing a 90-degree edge across a sub-grid bend radius.
    Native convergence and finite-thickness contact must still be checked.
    """
    points=np.asarray(points,float)
    if (points.ndim!=2 or points.shape[1:]!=(3,) or not np.isfinite(points).all()
            or not np.allclose(points[:,2],points[0,2]) or type(panels) is not int
            or not 2<=panels<=12 or not np.isfinite(layer_gap) or layer_gap<=0):
        raise ValueError('Planar material chart, bounded panel count and positive gap required')
    x=np.unique(points[:,0]);dx=np.diff(x);mid=(x[1:]+x[:-1])/2
    cuts=np.linspace(x[0],x[-1],panels+1)[1:-1]
    indices=np.array([np.argmin(abs(mid-cut)) for cut in cuts])
    if len(np.unique(indices))!=len(indices) or (dx[indices]<=layer_gap).any():
        raise ValueError('Existing crease edges cannot resolve the proposed layer gap')
    angles=np.zeros(len(dx))
    for i,index in enumerate(indices):
        angles[index+1:]=np.pi if i%2==0 else 0.
        rise=np.arcsin(layer_gap/dx[index])
        angles[index]=np.pi-rise if i%2==0 else rise
    cross=np.r_[np.zeros((1,2)),np.cumsum(dx[:,None]*np.c_[np.cos(angles),np.sin(angles)],axis=0)]
    cross-=(cross.min(0)+cross.max(0))/2
    return np.c_[cross[np.searchsorted(x,points[:,0]),0],points[:,1],cross[np.searchsorted(x,points[:,0]),1]]
