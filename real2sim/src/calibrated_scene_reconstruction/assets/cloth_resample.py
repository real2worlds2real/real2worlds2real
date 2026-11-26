"""Resample a rectangular material chart without changing sheet dimensions."""
import numpy as np


def surface(points,material,faces,new_material,*,maximum_extrapolation_m=0.):
    """Transfer a pose through its actual material triangles before simulation.

    The caller supplies and independently validates the new chart/topology.
    Default queries must lie on the existing material surface, including cuts.
    Explicit bounded extrapolation can propose a finer source boundary; it is
    never source, physical or native approval and cannot change intrinsic data.
    """
    p,r,f,q=map(np.asarray,[points,material,faces,new_material])
    if (p.ndim!=2 or p.shape[1:]!=(3,) or r.shape!=p.shape or not np.isfinite([p,r]).all()
            or f.ndim!=2 or f.shape[1:]!=(3,) or f.dtype.kind not in 'iu' or not len(f)
            or f.min()<0 or f.max()>=len(p) or q.ndim!=2 or q.shape[1:]!=(3,) or not len(q)
            or not np.isfinite(q).all() or not np.isfinite(maximum_extrapolation_m)
            or not 0<=maximum_extrapolation_m<=.002):
        raise ValueError('Finite source triangles and an explicitly bounded target material chart required')
    import warp as wp
    from .support_route import _nearest
    wp.init();device='cpu'
    mesh=wp.Mesh(points=wp.array(r.astype('float32'),dtype=wp.vec3,device=device),
                 indices=wp.array(f.astype('int32').ravel(),dtype=int,device=device))
    nearest=wp.empty(len(q),dtype=wp.vec3,device=device);indices=wp.empty(len(q),dtype=int,device=device)
    wp.launch(_nearest,dim=len(q),inputs=[mesh.id,wp.array(q.astype('float32'),dtype=wp.vec3,device=device),nearest,indices],device=device)
    ids=indices.numpy()
    if (ids<0).any():raise ValueError('Actual material triangle lookup failed')
    distance=np.linalg.norm(q-nearest.numpy(),axis=1)
    if distance.max()>maximum_extrapolation_m+1e-7:
        raise ValueError('Requested material point lies outside the explicit source extrapolation budget')
    tri=r[f[ids]];edge=np.stack([tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]],axis=2)
    gram=np.swapaxes(edge,1,2)@edge
    if (np.linalg.det(gram)<=1e-24).any():raise ValueError('Resolved nondegenerate source material triangles required')
    uv=np.linalg.solve(gram,np.einsum('nij,ni->nj',edge,q-tri[:,0])[...,None])[...,0]
    weights=np.c_[1-uv.sum(1),uv]
    result=np.einsum('ni,nij->nj',weights,p[f[ids]])
    return result,{'native_states_changed':False,'task_approved':False,'source_approved':False,
        'source_triangles_used':int(len(np.unique(ids))),'maximum_material_extrapolation_m':float(distance.max()),
        'maximum_allowed_extrapolation_m':float(maximum_extrapolation_m),
        'scope':'Offline pose proposal through actual source material triangles; target outline, topology, material and contact checks remain required.'}


def rows(points,material,shape,new_distances):
    """Interpolate material rows along existing surface-column edges.

    This changes numerical sampling, not the material's width, length or flat
    area. New cell faces/springs must be generated and independently checked.
    Surface interiors between these new nodes are a geometry proposal.
    """
    points,material=np.asarray(points,float),np.asarray(material,float)
    r,c=shape;new=np.asarray(new_distances,float)
    if (type(r) is not int or type(c) is not int or min(r,c)<3
            or points.shape!=(r*c,3) or material.shape!=points.shape
            or not np.isfinite(points).all() or not np.isfinite(material).all()
            or new.ndim!=1 or len(new)<3 or not np.isfinite(new).all() or (np.diff(new)<=0).any()):
        raise ValueError('Finite structured sheet and increasing material row distances required')
    chart=material.reshape(r,c,3);state=points.reshape(r,c,3)
    old=np.linalg.norm(chart[:,0]-chart[0,0],axis=1)
    axis=(chart[-1,0]-chart[0,0])/old[-1]
    if ((np.diff(old)<=0).any() or not np.allclose(chart,chart[:1]+old[:,None,None]*axis,atol=1e-10)
            or not np.isclose(new[0],0.,atol=1e-12) or not np.isclose(new[-1],old[-1],atol=1e-12)):
        raise ValueError('Straight material rows and unchanged physical endpoints required')
    source=np.clip(np.searchsorted(old,new,side='right')-1,0,r-2)
    weight=((new-old[source])/(old[source+1]-old[source]))[:,None,None]
    vertices=(1-weight)*state[source]+weight*state[source+1]
    rest=(1-weight)*chart[source]+weight*chart[source+1]
    return vertices.reshape(-1,3),rest.reshape(-1,3)
