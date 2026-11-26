"""Shape-bounded isotropic meshing for curved, rooted shell patches.

Isaac's tested surface backend distributes body mass uniformly over nodes.
Uniform parameter grids with submicron cut cells are unsuitable for that
model. Remesh construction geometry before defining natural coordinates;
never repair a running simulation or fill missing semantic structure.
"""
import numpy as np


def patch(points,faces,roots,*,spacing_m,error_m,chart=None):
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    from mathutils.geometry import delaunay_2d_cdt,barycentric_transform
    points=np.asarray(points,float);faces=np.asarray(faces,int);roots=set(map(int,roots))
    if not 0<error_m<spacing_m/2:raise ValueError('Explicit resolved spacing and shape-error budget required')
    parent=np.arange(len(points))
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for tri in faces:
        for i in tri[1:]:parent[find(int(i))]=find(int(tri[0]))
    components={}
    for i in range(len(points)):components.setdefault(find(i),[]).append(i)
    if len(components)>1:
        vertices=[];triangles=[];attached=[];reports=[];offset=0
        for ids in components.values():
            ids=np.asarray(ids);selected=faces[np.isin(faces,ids).all(1)]
            mapping=np.full(len(points),-1,int);mapping[ids]=np.arange(len(ids))
            rr=[int(mapping[x]) for x in roots if mapping[x]>=0]
            if not len(selected) or not rr:raise ValueError('Every remeshed material island needs actual faces and real roots')
            v,f,r,report=patch(points[ids],mapping[selected],rr,spacing_m=spacing_m,error_m=error_m,
                               chart=None if chart is None else np.asarray(chart)[ids])
            vertices.extend(v);triangles.extend(f+offset);attached.extend([x+offset for x in r]);offset+=len(v);reports.append(report)
        return np.asarray(vertices),np.asarray(triangles),attached,{'components':reports,
            'maximum_sampled_surface_error_m':max(x['maximum_sampled_surface_error_m'] for x in reports),
            'scope':'Every separate rooted material component remeshed independently; no material bridges'}
    centre=points.mean(0);_,_,basis=np.linalg.svd(points-centre,full_matrices=False)
    xy=(points-centre)@basis[:2].T if chart is None else np.asarray(chart,float)
    if xy.shape!=(len(points),2) or not np.isfinite(xy).all():raise ValueError('Finite one-to-one construction chart required')
    area=np.cross(xy[faces[:,1]]-xy[faces[:,0]],xy[faces[:,2]]-xy[faces[:,0]])
    if not (np.all(area>1e-16) or np.all(area<-1e-16)):
        raise ValueError('Curved patch needs multiple charts; single projection folds over')
    edges=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    unique,counts=np.unique(np.sort(edges,axis=1),axis=0,return_counts=True)
    boundary=unique[counts==1];neighbors={}
    for a,b in boundary:neighbors.setdefault(int(a),[]).append(int(b));neighbors.setdefault(int(b),[]).append(int(a))
    if any(len(v)!=2 for v in neighbors.values()):raise ValueError('Simple manifold boundary required')
    start=min(neighbors);loop=[start];last=-1;current=start
    while True:
        nxt=next(x for x in neighbors[current] if x!=last)
        if nxt==start:break
        if nxt in loop:raise ValueError('Boundary is not a simple loop')
        loop.append(nxt);last,current=current,nxt
    if len(loop)!=len(neighbors):raise ValueError('Multiple patch boundaries require separate chart construction')
    if np.cross(xy[loop],np.roll(xy[loop],-1,axis=0)).sum()<0:loop.reverse()
    original_loop=loop.copy()
    # Keep material root/free transitions and all source-visible curvature.
    for _ in range(len(loop)):
        changed=False
        for j,b in enumerate(loop):
            a,c=loop[j-1],loop[(j+1)%len(loop)]
            if len({x in roots for x in (a,b,c)})>1:continue
            delta=points[c]-points[a];t=np.clip((points[b]-points[a])@delta/(delta@delta),0,1)
            if (np.linalg.norm(points[b]-points[a]-t*delta)<=error_m/2
                    and np.linalg.norm(delta)<=spacing_m*1.5 and len(loop)>4):
                loop.pop(j);changed=True;break
        if not changed:break
    lo=xy.min(0);hi=xy.max(0);rows=[]
    for row,y in enumerate(np.arange(lo[1]+spacing_m*.5,hi[1],spacing_m*np.sqrt(3)/2)):
        x=np.arange(lo[0]+spacing_m*(.5+.5*(row%2)),hi[0],spacing_m)
        rows.extend(np.c_[x,np.full(len(x),y)])
    grid=np.asarray(rows).reshape(-1,2);inside=np.zeros(len(grid),bool);minedge=np.full(len(grid),np.inf)
    polygon=xy[loop]
    for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
        inside^=((a[1]>grid[:,1])!=(b[1]>grid[:,1]))&(grid[:,0]<(b[0]-a[0])*(grid[:,1]-a[1])/(b[1]-a[1]+1e-30)+a[0])
        delta=b-a;t=np.clip(((grid-a)@delta)/(delta@delta),0,1);minedge=np.minimum(minedge,np.linalg.norm(grid-a-t[:,None]*delta,axis=1))
    inputs=np.r_[polygon,grid[inside&(minedge>spacing_m*.45)]]
    flat=np.c_[xy,np.zeros(len(xy))];tree=BVHTree.FromPolygons(flat.tolist(),faces.tolist(),all_triangles=True)
    oldtree=BVHTree.FromPolygons(points.tolist(),faces.tolist(),all_triangles=True)
    for iteration in range(24):
        verts,_,triangles,origins,_,face_origins=delaunay_2d_cdt([Vector(x) for x in inputs],[],[list(range(len(loop)))],1,1e-10,True)
        # CDT may retain bounded regions outside a concave supplied face.
        # Only triangles owned by that material polygon belong to this sheet.
        triangles=np.asarray([f for f,owned in zip(triangles,face_origins) if 0 in owned],int)
        if not len(triangles):raise ValueError('CDT lost the supplied material polygon')
        used=np.unique(triangles);mapping=np.full(len(verts),-1,int);mapping[used]=np.arange(len(used))
        uv=np.array(verts)[used];origins=[origins[i] for i in used];triangles=mapping[triangles];result=[]
        for q in uv:
            _,_,index,distance=tree.find_nearest(Vector((*q,0.)))
            if distance>1e-7:raise ValueError('Remesh vertex lies outside original material patch')
            tri=faces[index];a,b,c=xy[tri];weights=np.linalg.solve(np.c_[b-a,c-a],q-a)
            result.append(points[tri[0]]+(points[tri[1]]-points[tri[0]])*weights[0]+(points[tri[2]]-points[tri[0]])*weights[1])
        result=np.asarray(result);root_indices=[]
        if chart is None:
            chart_error=float(np.linalg.norm((result-centre)@basis[:2].T-uv,axis=1).max())
            if chart_error>1e-7:raise ValueError(f'Curved chart inverse mismatch: {chart_error:.9g}m')
        for j,original in enumerate(origins):
            if any(i<len(loop) and loop[i] in roots for i in original):root_indices.append(j)
        if not root_indices:raise ValueError('Construction remesh lost actual bonded roots')
        if np.all(area<0):triangles=triangles[:,::-1]
        newtree=BVHTree.FromPolygons(result.tolist(),triangles.tolist(),all_triangles=True)
        maximum=0.;refine=[];worst=None
        for v,f,target in ((points,faces,newtree),(result,triangles,oldtree)):
            samples=np.r_[v,v[f].mean(1),(v[f[:,0]]+v[f[:,1]])/2,(v[f[:,1]]+v[f[:,2]])/2,(v[f[:,2]]+v[f[:,0]])/2]
            for q in samples:
                near,_,face_index,distance=target.find_nearest(Vector(q))
                if distance>maximum:maximum=distance;worst=(q.tolist(),list(near),'old_to_new' if target is newtree else 'new_to_old')
                if distance>error_m*.8:
                    # For a new chord use its corresponding point on the old
                    # curved surface. For old samples retain their own chart.
                    original=q if target is newtree else np.asarray(near)
                    if chart is None:candidate=(original-centre)@basis[:2].T
                    else:
                        _,_,index,_=oldtree.find_nearest(Vector(original));tri=faces[index]
                        value=barycentric_transform(Vector(original),*[Vector(p) for p in points[tri]],*[Vector((*p,0.)) for p in xy[tri]])
                        candidate=np.asarray(value)[:2]
                    if np.linalg.norm(inputs-candidate,axis=1).min()>1e-7:refine.append(candidate)
        if maximum<=error_m:break
        if not refine:raise ValueError(f'Remesh shape error cannot be resolved: {maximum:.9g}m; witness={worst}; triangles={len(triangles)}; nodes={len(uv)}')
        inputs=np.r_[inputs,np.unique(np.round(refine,10),axis=0)]
    if maximum>error_m:raise ValueError(f'Remesh shape error exceeds construction budget: {maximum:.9g}m')
    return result,triangles,root_indices,{'maximum_sampled_surface_error_m':maximum,'spacing_m':spacing_m,
        'error_budget_m':error_m,'vertices_before':len(points),'vertices_after':len(result),'boundary_vertices_before':len(original_loop),
        'scope':'Bidirectional vertex/edge/triangle-centre distance screen; independent source, root, finite thickness and native tests remain required'}
