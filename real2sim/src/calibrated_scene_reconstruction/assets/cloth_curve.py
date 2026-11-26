"""Sample a proposed folded-sheet centreline without changing material length."""
import numpy as np


def longitudinal_frame(reference,material,*,axis=1):
    """Align a straight history axis with the retained material correspondence.

    A history frame can reverse the material route. Correct a reversal by a
    proper half-turn, including its thickness axis, never by reflecting just
    the length coordinate. Intrinsic dimensions and natural fold signs stay
    unchanged. Curved or scaled history axes require a different explicit map.
    """
    r,m=np.asarray(reference,float),np.asarray(material,float)
    if (r.shape!=m.shape or r.ndim!=2 or r.shape[1:]!=(3,)or len(r)<3
            or not np.isfinite([r,m]).all()or type(axis)is not int or axis not in (0,1)):
        raise ValueError('Matching finite history/material coordinates and an intrinsic route axis required')
    x=m[:,axis];y=r[:,1]
    if np.ptp(x)<=1e-8:raise ValueError('Resolved longitudinal material extent required')
    slope=float((x-x.mean())@(y-y.mean())/np.sum((x-x.mean())**2));offset=float(y.mean()-slope*x.mean())
    if not np.isclose(abs(slope),1.,atol=1e-7,rtol=0.)or np.max(abs(y-(slope*x+offset)))>1e-7:
        raise ValueError('History length is not an unscaled straight material route')
    rotation=np.eye(3)if slope>0 else np.diag([1.,-1.,-1.])
    return r@rotation.T,{'rotation':rotation.tolist(),'route_reversed':slope<0,
                         'material_changed':False,'proposal_only':True,'task_approved':False}


def sample(points,distances,*,smooth_frames=False):
    points,distances=np.asarray(points,float),np.asarray(distances,float)
    if (points.ndim!=2 or points.shape[1:]!=(2,) or len(points)<2
            or distances.ndim!=1 or not len(distances) or not np.isfinite(points).all()
            or not np.isfinite(distances).all() or distances.min()<0 or np.any(np.diff(distances)<0)):
        raise ValueError('Finite planar curve and ordered nonnegative material distances required')
    edges=np.diff(points,axis=0);lengths=np.linalg.norm(edges,axis=1)
    if lengths.min()<=1e-12:raise ValueError('Curve samples must have positive segment lengths')
    arc=np.r_[0.,np.cumsum(lengths)]
    epsilon=64*np.finfo(float).eps*max(1.,abs(points).max(),arc[-1])
    if distances.max()>arc[-1]+epsilon:raise ValueError('Proposed curve does not cover the required material length')
    distances=np.minimum(distances,arc[-1])
    centres=np.column_stack([np.interp(distances,arc,points[:,j]) for j in range(2)])
    indices=np.clip(np.searchsorted(arc,distances+epsilon,side='right')-1,0,len(edges)-1)
    tangent=edges[indices]/lengths[indices,None]
    if type(smooth_frames) is not bool:raise ValueError('Explicit curve-frame policy required')
    if smooth_frames:
        # A polyline normal jumps at every vertex. Sweeping a thick folded
        # section through those jumps adds artificial longitudinal strain.
        # Smooth only an explicitly densely sampled turning path; real sharp
        # corners retain the legacy segment-frame route.
        angles=np.unwrap(np.arctan2(edges[:,1],edges[:,0]))
        if np.max(abs(np.diff(angles)),initial=0)>np.pi/4:
            raise ValueError('Smooth frames require resolved turns; retain or refine intentional sharp corners')
        vertex=np.r_[angles[0],(angles[:-1]+angles[1:])/2,angles[-1]]
        angle=np.interp(distances,arc,vertex)
        tangent=np.c_[np.cos(angle),np.sin(angle)]
    normal=np.column_stack([-tangent[:,1],tangent[:,0]])
    return centres,normal


def parallel_stations(reference,curve,*,anchor='start'):
    """Match longitudinal distances along each offset curve, allowing ply slip.

    Constant-height sections retain their own arc length through a turn.
    Rounded transverse creases can still shear where folds intersect: this
    is a geometry proposal, not proof of full-sheet isometry or rest history.
    """
    reference,curve=np.asarray(reference,float),np.asarray(curve,float)
    if (reference.ndim!=2 or reference.shape[1:]!=(3,)or not len(reference)
            or curve.ndim!=2 or curve.shape[1:]!=(2,)or len(curve)<3
            or not np.isfinite(reference).all()or not np.isfinite(curve).all()
            or anchor not in {'start','end'}):
        raise ValueError('Finite material section, resolved curve and explicit arc anchor required')
    edge=np.diff(curve,axis=0);length=np.linalg.norm(edge,axis=1)
    if (length<=1e-12).any():raise ValueError('Positive resolved curve segments required')
    angles=np.unwrap(np.arctan2(edge[:,1],edge[:,0]))
    if np.max(abs(np.diff(angles)),initial=0)>np.pi/4:
        raise ValueError('Parallel arcs require a resolved turning curve')
    theta=np.r_[angles[0],(angles[:-1]+angles[1:])/2,angles[-1]]
    arc=np.r_[0.,np.cumsum(length)];distance=reference[:,1]-reference[:,1].min();span=float(distance.max())
    stations=np.empty(len(reference));epsilon=64*np.finfo(float).eps*max(1.,arc[-1])
    for offset in np.unique(reference[:,2]):
        ids=np.flatnonzero(reference[:,2]==offset)
        # With sweep's proper right normal, d(C+offset*N)/ds=(1+offset*kappa)*T.
        parallel=arc+offset*(theta-theta[0])
        if (np.diff(parallel)<=epsilon).any():
            raise ValueError('Folded section turns through its offset radius; use a larger bend')
        wanted=distance[ids] if anchor=='start' else parallel[-1]-(span-distance[ids])
        if wanted.min()<-epsilon or wanted.max()>parallel[-1]+epsilon:
            raise ValueError('Curve does not cover every parallel material length')
        stations[ids]=np.interp(np.clip(wanted,0.,parallel[-1]),parallel,arc)
    return stations


def sweep(reference, curve,*,smooth_frames=False,parallel_arcs=False,anchor='start'):
    """Place a width/length/thickness chart along an XZ centreline.

    Width maps to world Y. The thickness axis is chosen so that the local
    frame is a *proper* rotation of the material frame. Using the left-hand
    planar curve normal here reflects the sheet and reverses its natural
    signed folds. This helper only proposes geometry: curved, offset layers
    still need intrinsic-strain and contact checks before simulation.
    """
    reference=np.asarray(reference,float)
    if (reference.ndim!=2 or reference.shape[1:]!=(3,) or not len(reference)
            or not np.isfinite(reference).all()):
        raise ValueError('Finite width/length/thickness material coordinates required')
    if type(parallel_arcs)is not bool or anchor not in {'start','end'}:
        raise ValueError('Explicit parallel-arc and anchoring policy required')
    if parallel_arcs and not smooth_frames:
        raise ValueError('Parallel arcs require smooth frames')
    distances=(parallel_stations(reference,curve,anchor=anchor)if parallel_arcs else
               reference[:,1]-reference[:,1].min())
    if not parallel_arcs and anchor=='end':
        points=np.asarray(curve,float)
        if (points.ndim!=2 or points.shape[1:]!=(2,) or len(points)<2
                or not np.isfinite(points).all()):
            raise ValueError('Finite planar curve required')
        # A common station preserves transverse correspondence. Independently
        # matching every offset's arc length can introduce large shear where
        # a transverse crease meets the longitudinal turn. Both policies are
        # proposals and still require full material and contact validation.
        total=np.linalg.norm(np.diff(points,axis=0),axis=1).sum()
        if total<distances.max()-1e-12:
            raise ValueError('Curve does not cover the required material length')
        distances=distances+max(0.,total-distances.max())
    order=np.argsort(distances)
    centres,normals=sample(curve,distances[order],smooth_frames=smooth_frames)
    placed=np.empty_like(reference)
    # sample() returns (-t_z, t_x). With width=Y, right handedness requires
    # thickness=(t_z, 0, -t_x), i.e. the opposite planar normal.
    section=centres-reference[order,2,None]*normals
    placed[order]=np.c_[section[:,0],reference[order,0],section[:,1]]
    return placed
