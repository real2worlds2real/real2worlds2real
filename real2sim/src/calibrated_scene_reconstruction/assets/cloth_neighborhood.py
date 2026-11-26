"""Topology-only surface-neighbour proposals; never filter distant material."""
import numpy as np
from scipy.sparse import coo_matrix,eye,triu

from .cloth_curvature import adjacency


def face_pairs(faces,*,rings=2):
    faces=np.asarray(faces)
    if type(rings) is not int or not 1<=rings<=3:
        raise ValueError('One to three explicit edge-neighbour rings required')
    pairs,_=adjacency(faces);count=len(faces)
    rows=np.r_[pairs[:,0],pairs[:,1]];cols=np.r_[pairs[:,1],pairs[:,0]]
    graph=coo_matrix((np.ones(len(rows),bool),(rows,cols)),shape=(count,count)).tocsr()
    reach=eye(count,dtype=bool,format='csr');step=reach
    for _ in range(rings):step=step@graph;reach=reach+step
    result=triu(reach,k=1,format='coo')
    return np.c_[result.row,result.col].astype(np.uint32)
