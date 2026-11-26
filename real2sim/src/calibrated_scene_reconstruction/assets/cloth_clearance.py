"""Nonlocal sheet-surface clearance, separate from triangle intersections.

Finite collision thickness can overlap even when zero-thickness triangles do
not cross. Rest-neighbour regions are explicitly excluded; native contact and
source/task response checks remain necessary.
"""
import numpy as np
import warp as wp


@wp.func
def _point_segment(p:wp.vec3,a:wp.vec3,b:wp.vec3):
    edge=b-a;t=wp.clamp(wp.dot(p-a,edge)/wp.max(wp.dot(edge,edge),1.e-24),0.,1.)
    return wp.length_sq(p-(a+t*edge))


@wp.func
def _point_triangle(p:wp.vec3,a:wp.vec3,b:wp.vec3,c:wp.vec3):
    u=b-a;v=c-a;n=wp.cross(u,v)
    q=p-n*(wp.dot(p-a,n)/wp.max(wp.dot(n,n),1.e-24))
    w=q-a;uu=wp.dot(u,u);uv=wp.dot(u,v);vv=wp.dot(v,v)
    divisor=wp.max(uu*vv-uv*uv,1.e-24)
    s=(vv*wp.dot(w,u)-uv*wp.dot(w,v))/divisor
    t=(uu*wp.dot(w,v)-uv*wp.dot(w,u))/divisor
    result=wp.min(_point_segment(p,a,b),wp.min(_point_segment(p,b,c),_point_segment(p,c,a)))
    if s>=0. and t>=0. and s+t<=1.:result=wp.min(result,wp.length_sq(p-q))
    return result


@wp.func
def _segments(a:wp.vec3,b:wp.vec3,c:wp.vec3,d:wp.vec3):
    u=b-a;v=d-c;r=a-c;aa=wp.dot(u,u);ee=wp.dot(v,v);ff=wp.dot(v,r)
    s=float(0.);t=float(0.)
    if aa<=1.e-24:
        t=wp.clamp(ff/wp.max(ee,1.e-24),0.,1.)
    else:
        cc=wp.dot(u,r)
        if ee<=1.e-24:s=wp.clamp(-cc/aa,0.,1.)
        else:
            bb=wp.dot(u,v);denominator=aa*ee-bb*bb
            if denominator>1.e-24:s=wp.clamp((bb*ff-cc*ee)/denominator,0.,1.)
            t=(bb*s+ff)/ee
            if t<0.:
                t=0.;s=wp.clamp(-cc/aa,0.,1.)
            elif t>1.:
                t=1.;s=wp.clamp((bb-cc)/aa,0.,1.)
    return wp.length_sq((a+s*u)-(c+t*v))


@wp.func
def _vertex(a:wp.vec3,b:wp.vec3,c:wp.vec3,i:int):
    point=a
    if i==1:point=b
    elif i==2:point=c
    return point


@wp.kernel
def _clearance(mesh:wp.uint64,points:wp.array(dtype=wp.vec3),rest:wp.array(dtype=wp.vec3),faces:wp.array(dtype=int),
               radius:float,required:float,exclusion:float,tolerance:float,opposition_cosine:float,
               violations:wp.array(dtype=int),minimum:wp.array(dtype=float),partner:wp.array(dtype=int),
               collect:bool,pair_count:wp.array(dtype=int),all_pairs:wp.array2d(dtype=int),capacity:int,
               only_violations:bool,pair_gaps:wp.array(dtype=float),
               groups:wp.array(dtype=int),cross_only:bool):
    i=wp.tid();a=faces[3*i];b=faces[3*i+1];c=faces[3*i+2]
    p=points[a];q=points[b];r=points[c];centre=(rest[a]+rest[b]+rest[c])/3.
    query=wp.mesh_query_aabb(mesh,wp.min(wp.min(p,q),r)-wp.vec3(radius),wp.max(wp.max(p,q),r)+wp.vec3(radius))
    j=int(0);best=float(1.e10);selected=int(-1);count=int(0)
    while wp.mesh_query_aabb_next(query,j):
        if j>i and (not cross_only or groups[i]!=groups[j]):
            x=faces[3*j];y=faces[3*j+1];z=faces[3*j+2]
            shared=(a==x or a==y or a==z or b==x or b==y or b==z or c==x or c==y or c==z)
            other=(rest[x]+rest[y]+rest[z])/3.
            aa=points[x];bb=points[y];cc=points[z]
            # Material-near coplanar regions belong to one continuous sheet.
            # A fold can turn these same neighbours into opposing surfaces;
            # a geodesic/ring exclusion must not hide their finite thickness.
            normal_a=wp.cross(q-p,r-p);normal_b=wp.cross(bb-aa,cc-aa)
            opposed=wp.dot(normal_a,normal_b)<opposition_cosine*wp.length(normal_a)*wp.length(normal_b)
            if not shared and (groups[i]!=groups[j] or wp.length(centre-other)>exclusion or opposed):
                distance=wp.min(_point_triangle(p,aa,bb,cc),wp.min(_point_triangle(q,aa,bb,cc),_point_triangle(r,aa,bb,cc)))
                distance=wp.min(distance,wp.min(_point_triangle(aa,p,q,r),wp.min(_point_triangle(bb,p,q,r),_point_triangle(cc,p,q,r))))
                for edge0 in range(3):
                    for edge1 in range(3):
                        distance=wp.min(distance,_segments(_vertex(p,q,r,edge0),_vertex(p,q,r,(edge0+1)%3),
                                                           _vertex(aa,bb,cc,edge1),_vertex(aa,bb,cc,(edge1+1)%3)))
                gap=wp.sqrt(wp.max(0.,distance))
                if collect and gap<radius and (not only_violations or gap<required-tolerance):
                    slot=wp.atomic_add(pair_count,0,1)
                    if slot<capacity:
                        all_pairs[slot,0]=i;all_pairs[slot,1]=j
                        pair_gaps[slot]=gap
                if gap<radius and gap<best:best=gap;selected=j
                if gap<required-tolerance:count+=1
    violations[i]=count;minimum[i]=best;partner[i]=selected


def contact_witnesses(points,faces,pairs):
    """Closest features for a *proposal* penalty, never native attachments.

    Weights reconstruct both closest points at this state. They are frozen
    linear coordinates for a subsequent proposal step, not a clearance proof
    for another state; recompute and run the independent triangle screen.
    Use only disjoint triangles already screened for intersections.
    """
    import trimesh
    points,faces,pairs=np.asarray(points,float),np.asarray(faces),np.asarray(pairs)
    if (points.ndim!=2 or points.shape[1:]!=(3,) or not np.isfinite(points).all()
            or faces.ndim!=2 or faces.shape[1:]!=(3,) or faces.dtype.kind not in 'iu'
            or not len(faces) or faces.min()<0 or faces.max()>=len(points)
            or pairs.ndim!=2 or pairs.shape[1:]!=(2,) or pairs.dtype.kind not in 'iu'
            or not len(pairs) or pairs.min()<0 or pairs.max()>=len(faces)
            or np.any(pairs[:,0]==pairs[:,1])):
        raise ValueError('Finite geometry and distinct triangle-pair indices required')
    vertices=points-points.mean(0);a,b=vertices[faces[pairs[:,0]]],vertices[faces[pairs[:,1]]]
    if np.any(np.linalg.norm(np.cross(a[:,1]-a[:,0],a[:,2]-a[:,0]),axis=1)<1e-14) or np.any(
            np.linalg.norm(np.cross(b[:,1]-b[:,0],b[:,2]-b[:,0]),axis=1)<1e-14):
        raise ValueError('Nondegenerate contact triangles required')
    # Trimesh's closest-feature tolerances are absolute. Normalize each pair
    # before those queries so millimetre/submillimetre triangles do not fall
    # into a different numerical branch from the same geometry at unit scale.
    origin=(a.mean(1)+b.mean(1))/2
    scale=np.maximum(np.linalg.norm(a-np.roll(a,1,axis=1),axis=2).max(1),
                     np.linalg.norm(b-np.roll(b,1,axis=1),axis=2).max(1))
    a=(a-origin[:,None])/scale[:,None,None];b=(b-origin[:,None])/scale[:,None,None]
    best=np.full(len(pairs),np.inf);wa=np.zeros((len(pairs),3));wb=wa.copy()
    def retain(left,right):
        delta=(a*left[:,:,None]).sum(1)-(b*right[:,:,None]).sum(1)
        distance=np.einsum('ij,ij->i',delta,delta);take=distance<best
        best[take]=distance[take];wa[take]=left[take];wb[take]=right[take]
    for k in range(3):
        unit=np.broadcast_to(np.eye(3)[k],(len(pairs),3))
        retain(unit,trimesh.triangles.points_to_barycentric(b,trimesh.triangles.closest_point(b,a[:,k])))
        retain(trimesh.triangles.points_to_barycentric(a,trimesh.triangles.closest_point(a,b[:,k])),unit)
    for i in range(3):
        for j in range(3):
            u=a[:,(i+1)%3]-a[:,i];v=b[:,(j+1)%3]-b[:,j];r=a[:,i]-b[:,j]
            aa=(u*u).sum(1);ee=(v*v).sum(1);bb=(u*v).sum(1)
            cc=(u*r).sum(1);ff=(v*r).sum(1);denominator=aa*ee-bb*bb
            s=np.clip(np.divide(bb*ff-cc*ee,denominator,out=np.zeros_like(aa),where=denominator>1e-24),0,1)
            t=(bb*s+ff)/ee;low=t<0;high=t>1
            s[low]=np.clip(-cc[low]/aa[low],0,1);s[high]=np.clip((bb[high]-cc[high])/aa[high],0,1)
            t=np.clip(t,0,1);left=np.zeros_like(wa);right=np.zeros_like(wb)
            left[:,i]=1-s;left[:,(i+1)%3]=s;right[:,j]=1-t;right[:,(j+1)%3]=t
            retain(left,right)
    delta=(a*wa[:,:,None]).sum(1)-(b*wb[:,:,None]).sum(1);gap=np.sqrt(best)
    if np.any(gap<=1e-10):raise ValueError('Intersecting/coincident proposal features need a new admissible seed')
    return {'faces_a':faces[pairs[:,0]],'faces_b':faces[pairs[:,1]],
            'weights_a':wa,'weights_b':wb,'normal':delta/gap[:,None],'gap_m':gap*scale}


def rest_filter_diagnostic(rest_points,faces,clearance_report,filter_distance):
    """Relate failed local gaps to an authored native distance filter.

    This inspects reported examples, not the native engine's actual pair list.
    It never converts a failed clearance test to a pass.
    """
    if not np.isfinite(filter_distance) or filter_distance<0:
        raise ValueError('Finite nonnegative authored self-collision filter required')
    examples=clearance_report.get('examples',[])
    result={'filter_distance_m':float(filter_distance),'examples':[],
            'scope':'Example-pair rest distances only; native pair filtering is not observed. Compare mesh/filter and timestep controls before attributing a failure. No clearance or task approval.'}
    if not examples:return result
    pairs=np.asarray([e['triangles'] for e in examples],int)
    try:distances=contact_witnesses(rest_points,faces,pairs)['gap_m']
    except ValueError as error:
        return {**result,'unresolved_rest_distance':str(error)}
    result['examples']=[{'triangles':pair.tolist(),'current_gap_m':row['gap_m'],
        'rest_gap_m':float(gap),'within_authored_filter':bool(gap<filter_distance)}
        for pair,row,gap in zip(pairs,examples,distances)]
    thickness=clearance_report.get('required_clearance_m')
    if thickness is not None:
        if not np.isfinite(thickness) or thickness<=0:
            raise ValueError('Positive finite physical clearance required for filter diagnosis')
        bound=(float(distances.min())-float(thickness))/2
        result['padding_filter_tradeoff']={
            'physical_thickness_m':float(thickness),
            'per_side_padding_upper_bound_m':max(0.,bound),
            'positive_padding_hypothesis_possible':bool(bound>0),
            'basis':'The native filter must be at least twice the rest offset. '
                    'To avoid filtering these example pairs, twice the offset must also '
                    'remain strictly below their smallest native-reference gap. '
                    'A smaller numerical padding is only a new hypothesis: preserve physical '
                    'thickness and independently repeat native contact, stability and task checks. '
                    'These examples do not certify all mesh pairs.'}
    return result


def measure(points,faces,material_points,required,*,device='cpu',tolerance=1e-6,neighbour_factor=3.,example_limit=8,proposal_candidates=False,search_padding=0.,candidate_capacity=1000000,face_groups=None,cross_only=True,collect_all_pairs=True,opposition_cosine=0.,local_bending_review=False):
    """Screen nonlocal, nonintersecting triangle pairs at an explicit thickness.

    Combine with the independent intersection check: vertex/edge features give
    exact closest distances for disjoint triangles. This function alone cannot
    certify intersecting geometry, local fold thickness, CCD or task behavior.
    """
    if type(local_bending_review) is not bool or (local_bending_review and proposal_candidates):
        raise ValueError('Local shell review is a measurement, not a proposal-constraint relaxation')
    points,material=np.asarray(points,float),np.asarray(material_points,float);faces=np.asarray(faces)
    if type(collect_all_pairs) is not bool:raise ValueError('Explicit boolean pair-collection policy required')
    if not np.isfinite(opposition_cosine) or not 0<=opposition_cosine<1:
        raise ValueError('Conservative local-fold normal cosine must be in[0,1)')
    if (points.ndim!=2 or points.shape[1:]!=(3,) or material.shape!=points.shape
            or not np.isfinite(points).all() or not np.isfinite(material).all()
            or faces.ndim!=2 or faces.shape[1:]!=(3,) or faces.dtype.kind not in 'iu' or not len(faces)
            or faces.min()<0 or faces.max()>=len(points) or not np.isfinite([required,tolerance,neighbour_factor]).all()
            or required<=4*tolerance or tolerance<=0 or neighbour_factor<0
            or type(example_limit) is not int or not 0<=example_limit<=100):
        raise ValueError('Finite matching material/state triangles, resolved positive clearance and numeric tolerance required')
    tri=material[faces];edge=max(np.linalg.norm(tri[:,1]-tri[:,0],axis=1).max(),np.linalg.norm(tri[:,2]-tri[:,1],axis=1).max(),np.linalg.norm(tri[:,0]-tri[:,2],axis=1).max())
    if edge<=0:raise ValueError('Degenerate material triangles')
    groups=np.zeros(len(faces),dtype='int32') if face_groups is None else np.asarray(face_groups)
    if groups.shape!=(len(faces),) or groups.dtype.kind not in 'iu':raise ValueError('One integer physical owner per triangle required')
    wp.init();vertices=wp.array((points-points.mean(0)).astype('float32'),dtype=wp.vec3,device=device)
    indices=wp.array(faces.astype('int32').ravel(),dtype=int,device=device)
    chart=wp.array((material-material.mean(0)).astype('float32'),dtype=wp.vec3,device=device)
    mesh=wp.Mesh(points=vertices,indices=indices);counts=wp.zeros(len(faces),dtype=int,device=device)
    minimum=wp.empty(len(faces),dtype=float,device=device);partners=wp.empty(len(faces),dtype=int,device=device)
    if (not np.isfinite(search_padding) or search_padding<0 or
            type(candidate_capacity) is not int or not 1<=candidate_capacity<=2000000):
        raise ValueError('Nonnegative search padding and bounded candidate capacity required')
    radius=1.2*required+tolerance+search_padding
    pair_count=wp.zeros(1,dtype=int,device=device)
    collect=bool(proposal_candidates and collect_all_pairs)
    capacity=candidate_capacity if collect else 1
    all_pairs=wp.empty((capacity,2),dtype=int,device=device)
    pair_gaps=wp.empty(capacity,dtype=float,device=device)
    wp.launch(_clearance,dim=len(faces),inputs=[mesh.id,vertices,chart,indices,float(radius),float(required),
        float(neighbour_factor*edge),float(tolerance),float(opposition_cosine),counts,minimum,partners,collect,pair_count,all_pairs,capacity,False,pair_gaps,
        wp.array(groups.astype('int32'),dtype=int,device=device),face_groups is not None and cross_only],device=device)
    count,gap,partner=counts.numpy(),minimum.numpy(),partners.numpy();seen=partner>=0
    candidates=np.flatnonzero(count);selected=candidates[np.argsort(gap[candidates])][:example_limit]
    result={'pass':not bool(count.sum()),'required_clearance_m':float(required),'violating_pairs':int(count.sum()),
        'minimum_gap_within_search_m':float(gap[seen].min()) if seen.any() else None,
        'search_radius_m':float(radius),'material_neighbour_exclusion_m':float(neighbour_factor*edge),
        'numerical_tolerance_m':float(tolerance),'opposed_material_neighbours_checked':True,
        'local_fold_normal_cosine':float(opposition_cosine),
        'cross_surface_only':face_groups is not None and cross_only,
        'material_neighbour_exclusion_within_owner_only':face_groups is not None,
        'examples':[{'triangles':[int(i),int(partner[i])],'gap_m':float(gap[i])} for i in selected],
        'scope':'Nonlocal surface-clearance screen only. Shared-vertex faces and non-opposed nearby material regions excluded; opposing folded neighbours included; independent intersection, native finite-contact and source/task checks remain required.'}
    if proposal_candidates:
        result['closest_violating_pairs']=np.c_[candidates,partner[candidates]].tolist()
        nearby=np.flatnonzero(seen)
        result['closest_nearby_pairs']=np.c_[nearby,partner[nearby]].tolist()
        if collect:
            total=int(pair_count.numpy()[0])
            if total>capacity:raise ValueError(f'Full nearby-contact candidate budget exceeded ({total}>{capacity}); refusing truncated constraints')
            result['all_nearby_pairs']=all_pairs.numpy()[:total].tolist()
        result['proposal_pair_collection']='all' if collect else 'closest_per_face'
        result['proposal_candidate_scope']='Full clearance scan; optional closest-per-face proposal constraints are not an exhaustive linearized contact set. Recompute and run the full independent screen after every proposal.'
    if local_bending_review:
        result['raw_mid_surface_clearance_pass']=result['pass']
        if not result['pass'] and face_groups is None:
            from calibrated_scene_reconstruction.assets.cloth_contact import self_intersections
            crossing = self_intersections(points, faces, device=device, example_limit=example_limit)
            if not crossing['pass']:
                # A local finite-thickness extrusion cannot rehabilitate an
                # already intersecting mid-surface. Avoid enumerating and
                # reviewing hundreds of thousands of invalid local patches.
                # Retain the full raw clearance count and an actual negative
                # intersection witness; no incomplete review can become a pass.
                result['local_bending_review'] = {
                    'local_shell_valid': False,
                    'reason': 'Mid-surface self-intersections invalidate local shell review',
                    'self_intersection': crossing,
                    'complete_pair_review': False,
                    'task_approved': False,
                }
                result['scope'] += ' Local shell review rejected by its nonintersecting-surface prerequisite.'
                return result
            from calibrated_scene_reconstruction.assets.cloth_shell_clearance import review
            # A closest-per-face diagnostic is not an exhaustive contact list.
            # Re-run only failed states, collecting every violating pair from
            # the same predicate; never approve a partial local-shell review.
            capacity=min(candidate_capacity,int(count.sum()))
            all_pairs=wp.empty((capacity,2),dtype=int,device=device)
            pair_gaps=wp.empty(capacity,dtype=float,device=device)
            pair_count.zero_()
            wp.launch(_clearance,dim=len(faces),inputs=[mesh.id,vertices,chart,indices,float(radius),float(required),
                float(neighbour_factor*edge),float(tolerance),float(opposition_cosine),counts,minimum,partners,
                True,pair_count,all_pairs,capacity,True,pair_gaps,
                wp.array(groups.astype('int32'),dtype=int,device=device),False],device=device)
            total=int(pair_count.numpy()[0])
            if total!=int(count.sum()) or total>capacity:
                result['local_bending_review']={'local_shell_valid':False,
                    'reason':'Complete violating-pair collection unavailable within the declared budget',
                    'pair_count':total,'capacity':capacity}
            else:
                pairs=all_pairs.numpy()[:total];gaps=pair_gaps.numpy()[:total]
                order=np.lexsort((pairs[:,1],pairs[:,0]))
                exhaustive=[{'triangles':pairs[i].tolist(),'gap_m':float(gaps[i])} for i in order]
                result['local_bending_review']=review(points,faces,{**result,'examples':exhaustive},device=device,
                                                       material_points=material_points)
                result['local_bending_review']['complete_pair_count']=total
            result['pass']=result['local_bending_review']['local_shell_valid']
        result['scope']+=' Local-bend mode retains raw distances and requires an independent, nonintersecting, positive-Jacobian finite-thickness extrusion for every failed pair; remote and cross-owner layer contacts remain strict.'
    return result


if __name__=='__main__':
    import argparse
    import hashlib
    import json
    from pathlib import Path
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mesh');parser.add_argument('link');parser.add_argument('--clearance',required=True,type=float)
    parser.add_argument('--device',default='cpu');parser.add_argument('--output',required=True)
    args=parser.parse_args();source=Path(args.mesh);output=Path(args.output)
    with np.load(source) as data:
        parts=json.loads(str(data['links_json']));part=next(p for p in parts if p['name']==args.link)
        a,n=part['vertex_start'],part['vertex_count'];b,count=part['face_start'],part['triangle_count']
        rest=json.loads(str(data['cloth_rest_json']))[args.link]
        result=measure(data['V'][a:a+n],data['F'][b:b+count]-a,rest['points'],args.clearance,device=args.device)
    result.update(diagnostic_only=True,task_approved=False,mesh_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),link=args.link)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in {'examples','scope'}},indent=2))
