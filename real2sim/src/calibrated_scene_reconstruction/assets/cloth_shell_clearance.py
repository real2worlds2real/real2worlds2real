"""Independent diagnostic of finite thickness at a local material bend.

The broad midpoint-distance screen can flag a continuous bend just beyond its
normal-angle cutoff. This diagnostic examines a normal extrusion of the same
sheet, without changing thickness, state, material or the raw screen verdict.
It is not by itself a native contact, constitutive or task certificate.
"""
import numpy as np
import heapq

from calibrated_scene_reconstruction.assets.cloth_contact import self_intersections


def _adjacency(faces):
    edges=np.stack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]],axis=1).reshape(-1,2)
    _,inverse,counts=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True,return_counts=True)
    if (counts>2).any():
        raise ValueError('Manifold material edges required for a local shell review')
    order=np.argsort(inverse,kind='stable');offset=np.r_[0,np.cumsum(counts)]
    shared=np.flatnonzero(counts==2)
    left,right=order[offset[shared]],order[offset[shared]+1]
    neighbours=np.full((len(faces),3),-1,int)
    neighbours[left//3,left%3]=right//3;neighbours[right//3,right%3]=left//3
    return neighbours


def _expand(seed,neighbours,rings):
    selected=set(map(int,seed));front=selected.copy()
    for _ in range(rings):
        new={int(n) for face in front for n in neighbours[face] if n>=0}-selected
        selected.update(new);front=new
    return selected


def _material_graph(points,faces):
    """Continuous paths inside material triangles using shared edge sites.

    Vertices and edge midpoints avoid anisotropic face-centre zigzags. Every
    segment stays inside a real triangle; paths cannot jump holes or components.
    """
    edges=np.stack([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]],axis=1)
    unique,inverse,counts=np.unique(np.sort(edges.reshape(-1,2),axis=1),axis=0,return_inverse=True,return_counts=True)
    if (counts>2).any():raise ValueError('Manifold material graph required')
    sites=np.r_[points,points[unique].mean(1)]
    face_sites=np.c_[faces,inverse.reshape(-1,3)+len(points)]
    columns=np.array([(i,j) for i in range(6) for j in range(6) if i!=j])
    left=face_sites[:,columns[:,0]].ravel();right=face_sites[:,columns[:,1]].ravel()
    weight=np.linalg.norm(sites[left]-sites[right],axis=1);order=np.argsort(left,kind='stable')
    degree=np.bincount(left,minlength=len(sites));offset=np.r_[0,np.cumsum(degree)]
    owner=face_sites.ravel();sequence=np.argsort(owner,kind='stable')
    owner_offset=np.r_[0,np.cumsum(np.bincount(owner,minlength=len(sites)))]
    return sites,face_sites,offset,right[order],weight[order],owner_offset,sequence//6


def _material_search(seed,graph,centres,radius,point=None,limit=1024):
    sites,face_sites,offset,neighbours,lengths,owner_offset,owners=graph
    start=centres[seed] if point is None else point
    distances={int(i):float(np.linalg.norm(sites[i]-start)) for i in face_sites[seed]}
    queue=[(distance,i) for i,distance in distances.items()];heapq.heapify(queue)
    region={int(seed)};visited=set()
    while queue:
        distance,node=heapq.heappop(queue)
        if node in visited:continue
        if distance>radius:break
        visited.add(node);region.update(map(int,owners[owner_offset[node]:owner_offset[node+1]]))
        if len(region)>limit or len(visited)>6*limit:
            raise ValueError('Local material neighbourhood exceeds diagnostic budget')
        for i in range(offset[node],offset[node+1]):
            other=int(neighbours[i]);candidate=distance+float(lengths[i])
            if candidate<=radius and candidate<distances.get(other,np.inf):
                distances[other]=candidate;heapq.heappush(queue,(candidate,other))
    return distances,region


def _material_neighbourhood(seed,graph,centres,radius,limit=1024):
    distances,region=_material_search(seed,graph,centres,radius,limit=limit)
    sites,face_sites=graph[:2]
    return {face for face in region if min(distances.get(int(i),np.inf)+np.linalg.norm(sites[i]-centres[face])
                                          for i in face_sites[face])<=radius}


def _minimum_jacobian(points,normals,faces,thickness):
    """Exact minimum of the linear director map over each triangle prism.

The determinant is linear in material barycentric coordinates and quadratic
through thickness. Evaluate the three material corners and every interior
minimum of that quadratic, in addition to both outer surfaces.
"""
    q=points[faces];n=normals[faces]
    e1,e2=q[:,1]-q[:,0],q[:,2]-q[:,0]
    d1,d2=n[:,1]-n[:,0],n[:,2]-n[:,0]
    a=np.einsum('ij,ikj->ik',np.cross(d1,d2),n)
    b=np.einsum('ij,ikj->ik',np.cross(e1,d2)+np.cross(d1,e2),n)
    c=np.einsum('ij,ikj->ik',np.cross(e1,e2),n)
    if (c<=0).any():return -1.
    h=thickness/2
    values=np.minimum(a*h*h+b*h+c,a*h*h-b*h+c)
    critical=np.divide(-b,2*a,out=np.zeros_like(a),where=a>0)
    inside=(a>0)&(abs(critical)<h)
    values=np.where(inside,np.minimum(values,a*critical**2+b*critical+c),values)
    return float((values/c).min())


def review(points,faces,clearance_report,*,device='cpu',material_points=None):
    """Diagnose fully enumerated local failures, preserving raw rejection.

Triangles connected by at most three material edges retain their legacy path.
For finer sampling, a bounded path within four physical thicknesses in the
intrinsic chart also identifies a local bend. This only selects a diagnostic
patch: it does not loosen the physical thickness or extrusion tests.
Remote folds and disconnected plies cannot acquire an exemption. A six-ring
neighbourhood is extruded on both sides by half the physical thickness; its
closed boundary must not intersect and its full director-map Jacobian must
stay positive. A partial example list is insufficient.
"""
    p=np.asarray(points,float);f=np.asarray(faces)
    thickness=clearance_report.get('required_clearance_m')
    examples=clearance_report.get('examples',[])
    if (p.ndim!=2 or p.shape[1:]!=(3,) or not np.isfinite(p).all()
            or f.ndim!=2 or f.shape[1:]!=(3,) or f.dtype.kind not in 'iu'
            or not len(f) or f.min()<0 or f.max()>=len(p)
            or thickness is None or not np.isfinite(thickness) or thickness<=0):
        raise ValueError('Finite indexed material surface and physical thickness required')
    result={'local_shell_valid':False,'raw_clearance_pass':clearance_report.get('pass') is True,
            'physical_thickness_m':float(thickness),'pairs':[],
            'scope':'Diagnostic local normal extrusion; raw midpoint-distance rejection is preserved. '
                    'No native force, material identification, contact or task approval.'}
    if (clearance_report.get('pass') is not False or not examples
            or clearance_report.get('violating_pairs')!=len(examples)):
        return {**result,'reason':'A complete, nonempty list of failing pairs is required'}
    pairs=np.asarray([row['triangles'] for row in examples],int)
    if pairs.shape!=(len(examples),2) or pairs.min()<0 or pairs.max()>=len(f):
        raise ValueError('Valid failed triangle pairs required')
    if len(np.unique(np.sort(pairs,axis=1),axis=0))!=len(pairs):
        return {**result,'reason':'Duplicate examples cannot certify all failing pairs'}
    neighbours=_adjacency(f)
    centres=None;material_cache={}
    if material_points is not None:
        rest=np.asarray(material_points,float)
        if rest.shape!=p.shape or not np.isfinite(rest).all():raise ValueError('Matching finite intrinsic material chart required')
        centres=rest[f].mean(1)
        result['intrinsic_local_path_limit_m']=4*float(thickness)
    tri=p[f];fn=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    normals=np.zeros_like(p)
    for col in range(3):np.add.at(normals,f[:,col],fn)
    lengths=np.linalg.norm(normals,axis=1)
    normals=np.divide(normals,lengths[:,None],out=np.zeros_like(normals),where=lengths[:,None]>0)
    graph=None;witnesses=None;records=[];parent={}
    def find(face):
        face=int(face);parent.setdefault(face,face)
        while parent[face]!=face:
            parent[face]=parent[parent[face]];face=parent[face]
        return face
    def union(a,b):parent[find(a)]=find(b)
    for pair_index,(first,second) in enumerate(pairs):
        row={'triangles':[int(first),int(second)],'local_shell_valid':False}
        nearby=_expand([first],neighbours,3);material_region=set()
        if second not in nearby and centres is not None:
            try:
                if graph is None:
                    from calibrated_scene_reconstruction.assets.cloth_clearance import contact_witnesses
                    graph=_material_graph(rest,f)
                    witnesses=contact_witnesses(p,f,pairs)
                    starts=(rest[witnesses['faces_a']]*witnesses['weights_a'][:,:,None]).sum(1)
                    ends=(rest[witnesses['faces_b']]*witnesses['weights_b'][:,:,None]).sum(1)
                    result['intrinsic_path_method']='Bounded paths between closest material witnesses through vertex and edge sites'
                start,end=starts[pair_index],ends[pair_index]
                key=(int(first),*map(float,start))
                if key not in material_cache:
                    material_cache[key]=_material_search(first,graph,centres,4*thickness,point=start)
                distances,material_region=material_cache[key];sites,face_sites=graph[:2]
                distance=min(distances.get(int(i),np.inf)+np.linalg.norm(sites[i]-end) for i in face_sites[second])
                if distance<=4*thickness:nearby.add(int(second))
            except ValueError as error:
                row['reason']=str(error);result['pairs'].append(row);continue
        result['pairs'].append(row)
        if second not in nearby:
            row['reason']='Not a local edge-connected material bend';continue
        union(first,second);records.append((row,int(first),int(second),material_region))
    # A single stronger extrusion can cover many overlapping local witnesses.
    # Keep per-pair fallback on failure, preserving existing accepted cases.
    for face in list(parent):
        for other in neighbours[face]:
            if int(other) in parent:union(face,int(other))
    groups={}
    for record in records:groups.setdefault(find(record[1]),[]).append(record)
    def extrude(region):
        region=np.array(sorted(region),int)
        ids,inverse=np.unique(f[region],return_inverse=True);local_faces=inverse.reshape(-1,3)
        q,n=p[ids],normals[ids]
        if (lengths[ids]<=1e-14).any():
            return {'local_shell_valid':False,'reason':'Undefined material directors'}
        jacobian=_minimum_jacobian(q,n,local_faces,thickness)
        count=len(q);v=np.r_[q+thickness/2*n,q-thickness/2*n]
        boundary=np.r_[local_faces,local_faces[:,::-1]+count]
        edges=np.concatenate([local_faces[:,[0,1]],local_faces[:,[1,2]],local_faces[:,[2,0]]])
        _,index,counts=np.unique(np.sort(edges,axis=1),axis=0,return_index=True,return_counts=True)
        sides=[]
        for a,b in edges[index[counts==1]]:sides.extend([[b,a,a+count],[b,a+count,b+count]])
        if sides:boundary=np.r_[boundary,sides]
        crossing=self_intersections(v,boundary,device=device)
        return {'minimum_relative_jacobian':jacobian,'boundary_intersections':crossing,
                'material_faces':len(region),'physical_thickness_m':float(thickness),
                'local_shell_valid':bool(jacobian>1e-6 and crossing['pass'])}
    result['extrusion_patches']=[]
    for group in groups.values():
        seeds={face for _,first,second,_ in group for face in [first,second]}
        region=_expand(seeds,neighbours,6)
        for _,_,_,material_region in group:region|=material_region
        witness=extrude(region);index=len(result['extrusion_patches'])
        result['extrusion_patches'].append({'failed_pairs_covered':len(group),**witness})
        for row,first,second,material_region in group:
            measured=witness if witness['local_shell_valid'] else extrude(_expand([first,second],neighbours,6)|material_region)
            row.update(measured,extrusion_patch=index,batch_witness_used=witness['local_shell_valid'])
    result['local_shell_valid']=all(row['local_shell_valid'] for row in result['pairs'])
    return result
