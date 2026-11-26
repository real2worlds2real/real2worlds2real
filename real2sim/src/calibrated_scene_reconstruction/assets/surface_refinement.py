"""Conforming refinement of sheet triangles without moving their surface.

Use before defining a native material chart. Shared edge midpoints preserve
the complete piecewise-linear surface and real boundary ownership; no face is
deleted and no gaps are bridged. This is not a dynamic remeshing adapter.
"""
import numpy as np


def uniform(points,faces,roots,levels=1,*,maximum_vertices=250000):
    """Conforming four-way subdivision; geometry and root edges are retained."""
    v=np.asarray(points,float).copy();f=np.asarray(faces,int).copy();root=set(map(int,roots))
    if (type(levels) is not int or not 0<=levels<=5 or v.ndim!=2 or v.shape[1:]!=(3,)
            or f.ndim!=2 or f.shape[1:]!=(3,) or not len(f) or not np.isfinite(v).all()
            or f.min()<0 or f.max()>=len(v)):
        raise ValueError('Finite actual source triangles and bounded subdivision required')
    before=(len(v),len(f))
    for _ in range(levels):
        edges,inverse=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0,return_inverse=True)
        if len(v)+len(edges)>maximum_vertices:raise ValueError('Uniform subdivision exceeds the declared node budget')
        ids=inverse.reshape(3,len(f)).T+len(v)
        root.update(len(v)+i for i,(a,b) in enumerate(edges) if a in root and b in root)
        v=np.r_[v,(v[edges[:,0]]+v[edges[:,1]])/2]
        a,b,c=f.T;m,n,o=ids.T
        f=np.stack([np.c_[a,m,o],np.c_[m,b,n],np.c_[o,n,c],np.c_[m,n,o]],axis=1).reshape(-1,3)
    return v,f,sorted(root),{'vertices_before':before[0],'vertices_after':len(v),
        'triangles_before':before[1],'triangles_after':len(f),'levels':levels,
        'source_surface_changed':False,'old_vertices_moved':False,'new_cross_component_connections':False}


def refine(points,faces,roots,target_vertices,*,maximum_vertices=250000):
    v=np.asarray(points,float).copy();f=np.asarray(faces,int).copy();root=set(map(int,roots))
    if (v.ndim!=2 or v.shape[1:]!=(3,) or not np.isfinite(v).all() or f.ndim!=2 or f.shape[1:]!=(3,)
            or not len(f) or f.min()<0 or f.max()>=len(v) or type(target_vertices) is not int
            or not len(v)<=target_vertices<=maximum_vertices):
        raise ValueError('Finite triangles and bounded nondecreasing target node count required')
    initial_vertices=len(v);initial_faces=len(f)
    while len(v)<target_vertices:
        edges=np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1)
        edges=np.unique(edges,axis=0);length=np.linalg.norm(v[edges[:,0]]-v[edges[:,1]],axis=1)
        count=min(target_vertices-len(v),max(1,len(v)//2),len(edges))
        selected=edges[np.argsort(length)[-count:]];mapping={tuple(e):len(v)+i for i,e in enumerate(selected)}
        for edge,index in mapping.items():
            if all(i in root for i in edge):root.add(index)
        v=np.r_[v,(v[selected[:,0]]+v[selected[:,1]])/2];result=[]
        for tri in f:
            a,b,c=map(int,tri);ids=[mapping.get(tuple(sorted(pair))) for pair in ((a,b),(b,c),(c,a))]
            active=[i for i,x in enumerate(ids) if x is not None]
            if not active:result.append([a,b,c]);continue
            if len(active)==1:
                k=active[0];a,b,c=map(int,np.roll(tri,-k));m=ids[k]
                result.extend([[a,m,c],[m,b,c]])
            elif len(active)==2:
                missing=next(i for i,x in enumerate(ids) if x is None);k=(missing+1)%3
                a,b,c=map(int,np.roll(tri,-k));m,n=ids[k],ids[(k+1)%3]
                result.extend([[b,n,m],[a,m,c],[m,n,c]])
            else:
                m,n,o=ids;result.extend([[a,m,o],[m,b,n],[o,n,c],[m,n,o]])
        f=np.asarray(result,int)
    return v,f,sorted(root),{'vertices_before':initial_vertices,'vertices_after':len(v),
        'triangles_before':initial_faces,'triangles_after':len(f),'existing_points_moved':False,
        'surface_changed':False,'scope':'Conforming straight-edge midpoint subdivision of the original triangle surface'}


def group_by_area(parts,*,maximum_vertices=100000):
    """Group same-material islands without changing their surfaces or density.

    Native uniform node masses require node counts proportional to sheet area.
    Refine existing triangles only; preserve all old vertices and real roots.
    Material compatibility and common physical ownership are caller obligations.
    """
    from .surface_mass import distribution
    if not parts or type(maximum_vertices) is not int or maximum_vertices<3:
        raise ValueError('Actual separate material parts and a bounded node budget required')
    areas=[]
    for part in parts:
        v=np.asarray(part['V'],float);f=np.asarray(part['F'],int)
        if not len(v) or not len(f):raise ValueError('Every material island needs actual geometry')
        tri=v[f];area=float(np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1).sum()/2)
        if not np.isfinite(area) or area<=0:raise ValueError('Positive finite material area required')
        areas.append(area)
    density=max(len(p['V'])/a for p,a in zip(parts,areas))
    counts=[max(len(p['V']),int(np.ceil(a*density-1e-9))) for p,a in zip(parts,areas)]
    if sum(counts)>maximum_vertices:
        raise ValueError('Area-consistent compound surface exceeds node budget; keep separate bodies or smaller groups')
    vertices=[];faces=[];roots=[];reports=[];offset=0
    for part,count in zip(parts,counts):
        v,f,r,report=refine(part['V'],part['F'],part['roots'],count,maximum_vertices=maximum_vertices)
        reports.append({**report,'name':part['name'],'vertex_start':offset,'roots':len(r)})
        vertices.append(v);faces.append(f+offset);roots.extend((np.asarray(r,int)+offset).tolist());offset+=len(v)
    v=np.concatenate(vertices);f=np.concatenate(faces)
    # Unit total mass checks relative per-island allocation independently of
    # the unmeasured material's actual areal-density prior.
    mass=distribution(v,f,1.)
    if not mass['pass'] or len(mass['components'])!=len(parts):
        raise ValueError('Compound refinement did not retain independent homogeneous sheet masses')
    return v,f,roots,{'components':reports,'mass_distribution':mass,'surface_changed':False,
        'scope':'Same-material grouping only. Original piecewise-linear sheets and root ownership preserved by conforming subdivision; no added bridges, mass or density adjustment.'}
