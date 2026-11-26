"""Conforming edge subdivision preserving the original planar collision surface."""
import numpy as np


_PATTERNS={
    0:[[0,1,2]],1:[[0,3,2],[3,1,2]],2:[[0,1,4],[0,4,2]],
    4:[[0,1,5],[1,2,5]],3:[[1,4,3],[0,3,2],[3,4,2]],
    5:[[0,3,5],[3,1,2],[3,2,5]],6:[[2,5,4],[0,1,5],[1,4,5]],
    7:[[0,3,5],[3,1,4],[5,4,2],[3,4,5]]}


def subdivide(vertices,faces,max_edge_m,*,max_triangles=2_000_000):
    v=np.asarray(vertices,float);f=np.asarray(faces)
    if (v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all() or f.ndim!=2 or f.shape[1]!=3
            or f.dtype.kind not in 'iu' or not len(f) or f.min()<0 or f.max()>=len(v)):
        raise ValueError('Finite valid source triangles required')
    if not np.isfinite(max_edge_m) or not .001<=max_edge_m<=.1:
        raise ValueError('Explicit metric collision edge budget must be .001.. .1m')
    if type(max_triangles) is not int or len(f)>max_triangles:
        raise ValueError('Collision tessellation triangle budget exceeded')
    for _ in range(20):
        edges=np.sort(f[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1)
        unique,inverse=np.unique(edges,axis=0,return_inverse=True)
        split=np.linalg.norm(v[unique[:,0]]-v[unique[:,1]],axis=1)>max_edge_m*(1+1e-10)
        if not split.any():return v,f
        masks=split[inverse].reshape(-1,3);codes=masks@np.array([1,2,4])
        count=int(sum(np.count_nonzero(codes==code)*len(pattern) for code,pattern in _PATTERNS.items()))
        if count>max_triangles:raise ValueError('Collision tessellation triangle budget exceeded')
        ids=np.full(len(unique),-1,int);ids[split]=np.arange(split.sum())+len(v)
        mids=v[unique[split]].mean(1);vertices_for_faces=np.c_[f,ids[inverse].reshape(-1,3)]
        f=np.concatenate([vertices_for_faces[codes==code][:,pattern].reshape(-1,3)
                          for code,pattern in _PATTERNS.items() if (codes==code).any()])
        v=np.concatenate([v,mids])
    raise ValueError('Collision subdivision did not converge within its iteration budget')
