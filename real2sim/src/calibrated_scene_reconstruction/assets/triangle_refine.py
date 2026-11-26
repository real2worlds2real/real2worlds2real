"""Conforming linear triangle refinement; no smoothing or physical fitting."""
import numpy as np


def refine(points,faces,divisions,*,vertex_fields=None,uv=None):
    """Preserve geometry, holes and seams while resolving material/contact mesh.

    Vertex fields are intensive coordinates such as the intrinsic rest chart,
    not masses or other extensive quantities. Recompute physical lumped data.
    UVs are per-face corners, so a shared geometric edge never welds a UV seam.
    """
    p,f=np.asarray(points,float),np.asarray(faces)
    if (p.ndim!=2 or p.shape[1:]!=(3,) or not np.isfinite(p).all() or f.ndim!=2 or f.shape[1:]!=(3,)
            or not len(f) or f.dtype.kind not in 'iu' or f.min()<0 or f.max()>=len(p)
            or type(divisions) is not int or not 1<=divisions<=8 or len(np.unique(f))!=len(p)):
        raise ValueError('Finite fully used triangle vertices and bounded integer refinement required')
    fields={name:np.asarray(value,float) for name,value in (vertex_fields or {}).items()}
    if any(value.ndim<1 or len(value)!=len(p) or not np.isfinite(value).all() for value in fields.values()):
        raise ValueError('Finite source-indexed intensive vertex fields required')
    if uv is not None:
        uv=np.asarray(uv,float)
        if uv.shape!=(len(f),3,2) or not np.isfinite(uv).all():raise ValueError('Finite per-corner UVs required')
    weights=[];grid={};local_faces=[];n=divisions
    for i in range(n+1):
        for j in range(n+1-i):
            grid[i,j]=len(weights);weights.append([n-i-j,i,j])
    for i in range(n):
        for j in range(n-i):
            local_faces.append([grid[i,j],grid[i+1,j],grid[i,j+1]])
            if i+j<n-1:local_faces.append([grid[i+1,j],grid[i+1,j+1],grid[i,j+1]])
    integer=np.asarray(weights,int);barycentric=integer/n;local_faces=np.asarray(local_faces,int)
    locations=[point for point in p];new_fields={k:list(v) for k,v in fields.items()}
    lookup={((i,n),):i for i in range(len(p))};indices=np.empty((len(f),len(integer)),int)
    for face_index,face in enumerate(f):
        for local,weight in enumerate(integer):
            key=tuple(sorted((int(i),int(w)) for i,w in zip(face,weight) if w))
            node=lookup.get(key)
            if node is None:
                node=len(locations);lookup[key]=node;blend=weight/n
                locations.append(blend@p[face])
                for name,values in fields.items():new_fields[name].append(np.tensordot(blend,values[face],axes=(0,0)))
            indices[face_index,local]=node
    new_faces=indices[:,local_faces].reshape(-1,3)
    new_uv=None if uv is None else np.einsum('ijk,fkl->fijl',barycentric[local_faces],uv).reshape(-1,3,2)
    return np.asarray(locations),new_faces,{k:np.asarray(v) for k,v in new_fields.items()},new_uv,{
        'divisions':n,'source_vertices':len(p),'vertices':len(locations),'source_triangles':len(f),'triangles':len(new_faces),
        'source_vertices_unchanged':True,'smoothing_applied':False,'material_extent_changed':False,
        'task_approved':False,'scope':'Linear meshing only; native material, contact and task validation remain required.'}
