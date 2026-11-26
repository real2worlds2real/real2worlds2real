"""Conforming tetrahedra for an explicitly triangulated extruded section."""
import numpy as np

from .tetra_boundary import determinants,surface


def extrude(section,triangles,levels):
    p,f,z=np.asarray(section,float),np.asarray(triangles),np.asarray(levels,float)
    if (p.ndim!=2 or p.shape[1:]!=(2,) or len(p)<3 or not np.isfinite(p).all()
            or f.ndim!=2 or f.shape[1:]!=(3,) or f.dtype.kind not in 'iu' or not len(f)
            or f.min()<0 or f.max()>=len(p) or z.ndim!=1 or len(z)<2
            or not np.isfinite(z).all() or (np.diff(z)<=0).any()):
        raise ValueError('Finite indexed planar triangles and increasing extrusion levels required')
    vertices=np.concatenate([np.c_[p,np.full(len(p),height)] for height in z])
    ordered=np.sort(f,axis=1);cells=[];n=len(p)
    for k in range(len(z)-1):
        a,b,c=(ordered+k*n).T;A,B,C=a+n,b+n,c+n
        # Every vertical face uses bottom-low-id to top-high-id diagonal,
        # independently of the neighbouring triangle's orientation.
        cells.extend([np.c_[a,b,c,C],np.c_[a,b,B,C],np.c_[a,A,B,C]])
    tets=np.concatenate(cells);det=determinants(vertices,tets)
    if (abs(det)<1e-20).any():raise ValueError('Degenerate extruded material cells')
    negative=det<0;tets[negative]=tets[negative][:,[1,0,2,3]]
    surface(vertices,tets)  # Reject inconsistent shared material ownership.
    return vertices,tets
