"""Bound local volume replacement to an actual closed convex native body."""
import numpy as np
from scipy.spatial import ConvexHull


def mask(vertices,faces,points,margin):
    from calibrated_scene_reconstruction.assets.topology import closed_oriented_surface
    V=np.asarray(vertices,float);F=np.asarray(faces,int);P=np.asarray(points,float)
    radii=np.broadcast_to(np.asarray(margin,float),(len(P),))
    if (V.ndim!=2 or V.shape[1]!=3 or F.ndim!=2 or F.shape[1]!=3 or P.shape!=(len(P),3)
            or not np.isfinite(V).all() or not np.isfinite(P).all()
            or not np.isfinite(radii).all() or np.any(radii<0) or np.any(radii>.20)):
        raise ValueError('Finite closed geometry and bounded replacement radii required')
    if not closed_oriented_surface(V,np.full(len(F),3),F.ravel()):raise ValueError('Closed consistently oriented replacement body required')
    hull=ConvexHull(V);tri=V[F]-V.mean(0)
    volume=abs(np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6)
    if hull.volume<=1e-10 or not np.isclose(volume,hull.volume,rtol=1e-4,atol=1e-10):
        raise ValueError('Native replacement body must be convex; do not fill a concavity or gap')
    result=np.zeros(len(P),bool)
    for start in range(0,len(P),8192):
        distances=P[start:start+8192]@hull.equations[:,:3].T+hull.equations[:,3]
        result[start:start+8192]=np.all(distances<=radii[start:start+8192,None]+1e-8,axis=1)
    return result
