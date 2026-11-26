"""Triangle connectivity diagnostics, independent of a cloth's folded pose."""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def measure(vertex_count,faces):
    """Count actual material surfaces; nearby plies never become connected.

    This measures indexed mesh construction, not source fold count, contact,
    stitching, physical ply bonding or appearance. It grants no acceptance.
    """
    f=np.asarray(faces)
    if (type(vertex_count) is not int or vertex_count<3 or f.ndim!=2 or f.shape[1:]!=(3,)
            or not len(f) or f.dtype.kind not in 'iu' or f.min()<0 or f.max()>=vertex_count
            or (np.diff(np.sort(f,axis=1),axis=1)==0).any()):
        raise ValueError('Valid noncollapsed indexed material triangles required')
    edges=np.stack([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]],axis=1).reshape(-1,2)
    unique,inverse,counts=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True,return_counts=True)
    order=np.argsort(inverse,kind='stable');offset=np.r_[0,np.cumsum(counts)]
    # Connect each incident triangle to one representative; nonmanifold edges
    # are reported explicitly rather than building quadratic face cliques.
    first=order[np.repeat(offset[:-1],counts)]
    adjacency=coo_matrix((np.ones(len(order)),(order//3,first//3)),shape=(len(f),len(f)))
    count,labels=connected_components(adjacency,directed=False)
    duplicate_count=len(f)-len(np.unique(np.sort(f,axis=1),axis=0))
    shared=np.flatnonzero(counts==2)
    left=order[offset[shared]];right=order[offset[shared]+1]
    forward=edges[:,0]<edges[:,1]
    inconsistent=int((forward[left]==forward[right]).sum())
    return {'edge_connected_surfaces':int(count),
            'triangles_per_surface':np.sort(np.bincount(labels))[::-1].tolist(),
            'unused_vertices':int(vertex_count-len(np.unique(f))),
            'boundary_edges':int((counts==1).sum()),
            'nonmanifold_edges':int((counts>2).sum()),
            'inconsistent_shared_edge_winding':inconsistent,
            'duplicate_triangles':duplicate_count,
            'euler_characteristic':int(len(np.unique(f))-len(unique)+len(f)),
            'scope':'Indexed material surface connectivity only; no source construction, ply bonding, shape or task approval'}
