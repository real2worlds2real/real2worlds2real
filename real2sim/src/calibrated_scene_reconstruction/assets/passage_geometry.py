"""Finite segment probes of source-reviewed solid and empty regions."""
import numpy as np
import trimesh


def occupancy(vertices, faces, starts, ends):
    v,f,a,b=map(np.asarray,[vertices,faces,starts,ends])
    if (a.ndim!=2 or a.shape[1:]!=(3,) or b.shape!=a.shape or not 1<=len(a)<=4096
            or not all(np.isfinite(x).all() for x in [v,a,b])):
        raise ValueError('Finite source mesh and one to4096 finite segment pairs required')
    mesh=trimesh.Trimesh(v,f,process=False)
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise ValueError('Occupancy requires the actual closed, consistently oriented source solid')
    lengths=np.linalg.norm(b-a,axis=1)
    if (lengths<1e-8).any():raise ValueError('Resolved source probe segments required')
    directions=(b-a)/lengths[:,None]
    points,rays,_=mesh.ray.intersects_location(a,directions,multiple_hits=True)
    counts=np.zeros(len(a),int)
    if len(points):
        distances=np.einsum('ij,ij->i',points-a[rays],directions[rays])
        valid=(distances>=-1e-9)&(distances<=lengths[rays]+1e-9)
        np.add.at(counts,rays[valid],1)
    occupied=(counts>0)|mesh.contains(a)|mesh.contains(b)
    return {'occupied':occupied.tolist(),'surface_intersections':counts.tolist(),
            'scope':'Finite reviewed segments only; no exhaustive passage or native-contact proof'}
