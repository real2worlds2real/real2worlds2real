"""Retain valid authored shading, derive missing normals from real triangles."""
import numpy as np


def complete(vertices,faces,normals):
    v=np.asarray(vertices,float);f=np.asarray(faces);n=np.array(normals,float,copy=True)
    if (v.ndim!=2 or v.shape[1:]!=(3,) or f.ndim!=2 or f.shape[1:]!=(3,)
            or n.shape!=(len(f),3,3) or f.dtype.kind not in 'iu' or not len(f)
            or f.min()<0 or f.max()>=len(v) or not np.isfinite(v).all()):
        raise ValueError('Actual triangles and corresponding corner normals required')
    length=np.linalg.norm(n,axis=2);bad=~np.isfinite(n).all(2)|(length<=1e-12)
    valid=~bad;n[valid]/=length[valid,None]
    if bad.any():
        geometric=np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]])
        magnitude=np.linalg.norm(geometric,axis=1)
        if (magnitude[bad.any(1)]<=0).any():raise ValueError('A zero-area surface has no geometric shading normal')
        geometric/=np.maximum(magnitude[:,None],np.finfo(float).tiny)
        n[bad]=np.broadcast_to(geometric[:,None],n.shape)[bad]
    return n,{'geometric_fallback_corners':int(bad.sum()),'geometry_changed':False,
              'scope':'Only missing/nonfinite shading normals; valid authored normals retained'}
