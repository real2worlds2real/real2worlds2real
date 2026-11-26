"""Position-aware surface topology checks, including duplicated UV seam vertices."""
import numpy as np


def closed_oriented_surface(points,counts,indices):
    """Require two oppositely directed faces per edge after merging positions."""
    points=np.asarray(points);counts=np.asarray(counts,int);indices=np.asarray(indices,int)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all() or counts.sum()!=len(indices):return False
    _,unique=np.unique(np.rint(points*1e6).astype('int64'),axis=0,return_inverse=True)
    edges={};offset=0
    for count in counts:
        face=unique[indices[offset:offset+count]];offset+=count
        if count<3 or len(set(face))!=count:return False
        for a,b in zip(face,np.roll(face,-1)):
            key=(min(a,b),max(a,b));n,balance=edges.get(key,(0,0));edges[key]=(n+1,balance+(1 if a<b else -1))
    return bool(edges) and all(n==2 and balance==0 for n,balance in edges.values())
