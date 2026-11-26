"""Explicit planar capping of a reviewed protruding surface patch.

This is geometry proposal construction, not automatic contents recognition.
The caller selects faces. Only a single planar convex boundary is supported;
all retained source triangles and their material indexing are preserved.
"""
import numpy as np
import trimesh
from scipy.spatial import ConvexHull


def planar_cap(vertices, faces, selected, *, tolerance_m=1e-6):
    v=np.asarray(vertices,float);f=np.asarray(faces)
    selected=np.asarray(selected)
    if (v.ndim!=2 or v.shape[1:]!=(3,) or f.ndim!=2 or f.shape[1:]!=(3,)
            or f.dtype.kind not in 'iu' or not np.isfinite(v).all()
            or f.min()<0 or f.max()>=len(v) or selected.dtype!=bool or selected.shape!=(len(f),)
            or not selected.any() or selected.all()
            or not np.isfinite(tolerance_m) or tolerance_m<=0):
        raise ValueError('Finite indexed source mesh, explicit nonempty face patch and tolerance required')
    source=trimesh.Trimesh(v,f,process=False)
    if not source.is_watertight or not source.is_winding_consistent or source.volume<=0:
        raise ValueError('Consistent closed positive source mesh required')
    kept=f[~selected]
    edges=np.concatenate([kept[:,[0,1]],kept[:,[1,2]],kept[:,[2,0]]])
    _,inverse,counts=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True,return_counts=True)
    boundary=edges[counts[inverse]==1]
    if len(boundary)<3 or len(np.unique(boundary[:,0]))!=len(boundary):
        raise ValueError('Patch must leave a single nonbranching boundary')
    following=dict(boundary.tolist());cycle=[int(boundary[0,0])]
    while following.get(cycle[-1])!=cycle[0]:
        nxt=following.get(cycle[-1])
        if nxt is None or nxt in cycle or len(cycle)>=len(boundary):
            raise ValueError('Patch boundary is not a single closed cycle')
        cycle.append(nxt)
    if len(cycle)!=len(boundary):raise ValueError('Multiple patch boundaries are not supported')
    p=v[cycle];centre=p.mean(0)
    _,_,axes=np.linalg.svd(p-centre,full_matrices=False)
    if abs((p-centre)@axes[2]).max()>tolerance_m:
        raise ValueError('Patch boundary is not planar')
    xy=(p-centre)@axes[:2].T
    if len(ConvexHull(xy).vertices)!=len(cycle):
        raise ValueError('Patch boundary must be strictly convex')
    # Opposite each retained oriented boundary edge, preserving outward winding.
    cap=np.c_[boundary[:,1],boundary[:,0],np.full(len(boundary),len(v))]
    output=trimesh.Trimesh(np.r_[v,centre[None]],np.r_[kept,cap],process=False)
    output.remove_unreferenced_vertices()
    if not output.is_watertight or not output.is_winding_consistent or output.volume<=0:
        raise ValueError('Capped proposal is not a consistent positive closed solid')
    return {'V':np.asarray(output.vertices),'F':np.asarray(output.faces),
            'source_face_indices':np.r_[np.flatnonzero(~selected),np.full(len(cap),-1)],
            'removed_volume_m3':float(source.volume-output.volume),
            'boundary_vertex_count':len(cycle),'retained_triangles_changed':False,
            'source_approved':False,'task_approved':False}
