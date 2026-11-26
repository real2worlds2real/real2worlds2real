"""Oriented boundary triangles of an explicit conforming tetrahedral solid."""
import numpy as np


def determinants(points, tetrahedra):
    v=np.asarray(points,float);t=np.asarray(tetrahedra)
    if (v.ndim!=2 or v.shape[1:]!=(3,) or not np.isfinite(v).all()
            or t.ndim!=2 or t.shape[1:]!=(4,) or not len(t) or t.dtype.kind not in 'iu'
            or t.min()<0 or t.max()>=len(v)):
        raise ValueError('Finite volume points and valid tetrahedron indices required')
    p=v[t]
    return np.linalg.det(np.stack([p[:,1]-p[:,0],p[:,2]-p[:,0],p[:,3]-p[:,0]],axis=2))


def surface(points, tetrahedra):
    v=np.asarray(points,float);t=np.asarray(tetrahedra)
    if (v.ndim!=2 or v.shape[1:]!=(3,) or not np.isfinite(v).all()
            or t.ndim!=2 or t.shape[1:]!=(4,) or not len(t) or t.dtype.kind not in 'iu'
            or t.min()<0 or t.max()>=len(v)):
        raise ValueError('Finite volume points and valid tetrahedron indices required')
    p=v[t];det=np.linalg.det(np.stack([p[:,1]-p[:,0],p[:,2]-p[:,0],p[:,3]-p[:,0]],axis=2))
    if np.any(abs(det)<1e-20):raise ValueError('Degenerate tetrahedron cannot define a solid boundary')
    t=t.copy();reverse=det<0;t[reverse]=t[reverse][:,[1,0,2,3]]
    f=np.concatenate([t[:,[0,2,1]],t[:,[0,1,3]],t[:,[1,2,3]],t[:,[2,0,3]]])
    _,inverse,counts=np.unique(np.sort(f,axis=1),axis=0,return_inverse=True,return_counts=True)
    if counts.max()>2:raise ValueError('Nonmanifold volume has overlapping tetrahedral ownership')
    parity=1-2*(((f[:,0]>f[:,1]).astype(int)+(f[:,0]>f[:,2])+(f[:,1]>f[:,2]))%2)
    orientation=np.bincount(inverse,weights=parity,minlength=len(counts))
    if np.any(orientation[counts==2]!=0):
        raise ValueError('Shared volume face has duplicate ownership on the same side')
    return f[counts[inverse]==1]
