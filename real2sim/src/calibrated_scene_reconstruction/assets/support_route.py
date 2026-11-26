"""Initial sheet routing above a real support; not a contact-force test."""
import numpy as np
import trimesh
from scipy.spatial import ConvexHull
import warp as wp


@wp.kernel
def _nearest(mesh:wp.uint64,points:wp.array(dtype=wp.vec3),closest:wp.array(dtype=wp.vec3),faces:wp.array(dtype=int)):
    i=wp.tid();query=wp.mesh_query_point_no_sign(mesh,points[i],10000.)
    faces[i]=-1
    if query.result:
        faces[i]=query.face;closest[i]=wp.mesh_eval_position(mesh,query.face,query.u,query.v)


def _closest_actual(vertices,faces,points):
    """Query the actual open surface at unit scale, retaining a double fallback.

    A missing float32 BVH hit is not proof of a hole or a valid support. Resolve
    only those failed queries against every actual triangle in bounded chunks.
    """
    used=vertices[np.unique(faces)];origin=(used.min(0)+used.max(0))/2
    scale=float(np.ptp(used,axis=0).max())
    if not np.isfinite(scale) or scale<=0:raise ValueError('Resolved actual support geometry required')
    v=(vertices-origin)/scale;p=(points-origin)/scale
    wp.init();device='cpu'
    mesh=wp.Mesh(points=wp.array(v.astype('float32'),dtype=wp.vec3,device=device),
                 indices=wp.array(faces.astype('int32').ravel(),dtype=int,device=device))
    output=wp.empty(len(points),dtype=wp.vec3,device=device);ids=wp.empty(len(points),dtype=int,device=device)
    wp.launch(_nearest,dim=len(points),inputs=[mesh.id,wp.array(p.astype('float32'),dtype=wp.vec3,device=device),output,ids],device=device)
    closest,index=output.numpy().astype(float),ids.numpy()
    missing=np.flatnonzero(index<0)
    if len(missing):
        top=trimesh.Trimesh(v,faces,process=False)
        for start in range(0,len(missing),64):
            selected=missing[start:start+64]
            found,_,triangle=trimesh.proximity.closest_point_naive(top,p[selected])
            closest[selected]=found;index[selected]=triangle
    if not np.isfinite(closest).all() or (index<0).any() or (index>=len(faces)).any():
        raise ValueError('Actual support query failed; cannot assume valid routing')
    return closest*scale+origin,index,{'normalized_query':True,'float64_fallback_points':len(missing),
        'actual_triangles':len(faces),'holes_filled':False}


def upward_patch(vertices,faces,up,tolerance,surface_region,*,return_report=False):
    mesh=trimesh.Trimesh(vertices,faces,process=False);selected=mesh.face_normals@up>.5
    report={'geometry_changed':False,'ignored_unresolved_planar_groups':0}
    if surface_region not in {'all_upward','lowest_upward_patch','highest_upward_patch'}:
        raise ValueError('Explicit supported upward surface region required')
    if surface_region!='all_upward':
        height=vertices[faces]@up
        flat=(mesh.face_normals@up>1-1e-6)&(np.ptp(height,axis=1)<=max(tolerance,1e-7))
        if not flat.any():raise ValueError('Declared planar support patch is absent')
        # A microscopic Boolean sliver can acquire an upward normal after
        # float32 rebaking. It is not a resolved supporting plane. Group by
        # height and require two in-plane dimensions at the declared geometric
        # resolution, independent of individual triangle size/tessellation.
        resolution=max(tolerance,1e-7);ids=np.flatnonzero(flat)
        level=height.mean(1);ids=ids[np.argsort(level[ids])];groups=[];start=0
        while start<len(ids):
            end=int(np.searchsorted(level[ids],level[ids[start]]+resolution,side='right'))
            group=ids[start:end];points=vertices[np.unique(faces[group])]
            points=points-points.mean(0);points-=np.outer(points@up,up)
            spread=np.linalg.svd(points,compute_uv=False)/np.sqrt(len(points))
            if len(spread)>=2 and spread[1]>resolution:groups.append(group)
            else:report['ignored_unresolved_planar_groups']+=1
            start=end
        if not groups:raise ValueError('No planar support patch resolved in two directions at the geometric tolerance')
        group=groups[0] if surface_region=='lowest_upward_patch' else groups[-1]
        selected=np.zeros(len(faces),bool);selected[group]=True
        report.update(plane_resolution_m=float(resolution),selected_plane_height_interval_m=[float(level[group].min()),float(level[group].max())])
    if not selected.any():raise ValueError('No actual upward support faces for this routing statement')
    return (selected,report) if return_report else selected


def validate_contact(request):
    if (not isinstance(request,dict) or set(request)!={'axis','end','fraction','maximum_gap_m','minimum_area_fraction'}
            or type(request['axis']) is not int or request['axis'] not in {0,1}
            or request['end'] not in {'minimum','maximum'}
            or not np.isfinite([request['fraction'],request['maximum_gap_m'],request['minimum_area_fraction']]).all()
            or not 0<request['fraction']<=.25 or not 0<request['maximum_gap_m']<=.01
            or not 0<request['minimum_area_fraction']<=1):
        raise ValueError('Explicit material-end region and bounded geometric support contact required')


def end_contact(points,rest,faces,vertices,support_faces,up,request,*,surface_region='lowest_upward_patch',tolerance=1e-6):
    """Measure actual end-band area near real upward triangles, including holes.

    Necessary geometry only. Above a footprint or a hole is insufficient;
    loaded force/contact and retention after an articulation opens stay native.
    """
    validate_contact(request)
    p,r,f,v,sf,up=map(np.asarray,[points,rest,faces,vertices,support_faces,up])
    if (p.shape!=r.shape or p.ndim!=2 or p.shape[1]!=3 or not np.isfinite([p,r]).all()
            or f.ndim!=2 or f.shape[1:]!=(3,) or f.dtype.kind not in 'iu' or not len(f) or f.min()<0 or f.max()>=len(p)
            or v.ndim!=2 or v.shape[1:]!=(3,) or not np.isfinite(v).all()
            or sf.ndim!=2 or sf.shape[1:]!=(3,) or sf.dtype.kind not in 'iu' or not len(sf) or sf.min()<0 or sf.max()>=len(v)
            or up.shape!=(3,) or not np.isfinite(up).all() or not np.isclose(np.linalg.norm(up),1.)
            or not np.isfinite(tolerance) or tolerance<0):
        raise ValueError('Matching finite actual and material coordinates required')
    centres=r[f].mean(1);axis=request['axis'];low,high=r[:,axis].min(),r[:,axis].max()
    if high<=low:raise ValueError('Resolved material-end axis required')
    selected=((centres[:,axis]-low)/(high-low)<=request['fraction'] if request['end']=='minimum'
              else (high-centres[:,axis])/(high-low)<=request['fraction'])
    if not selected.any():raise ValueError('Material end band is unresolved; refine the sheet')
    tri=r[f[selected]];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    if min(area)<=0:raise ValueError('Resolved positive material-end triangle area required')
    patch,selection=upward_patch(v,sf,up,tolerance,surface_region,return_report=True);top=trimesh.Trimesh(v,sf[patch],process=False)
    samples=p[f[selected]].mean(1)
    closest,ids,query=_closest_actual(v,sf[patch],samples)
    delta=samples-closest;distance=np.linalg.norm(delta,axis=1)
    height=np.einsum('ij,ij->i',delta,top.face_normals[ids])
    near=(distance<=request['maximum_gap_m'])&(height>=-tolerance)
    share=float(area[near].sum()/area.sum())
    return {'pass':share>=request['minimum_area_fraction'],'near_support_area_fraction':share,
            'material_region':request,'region_area_m2':float(area.sum()),'minimum_distance_m':float(distance.min()),
            'nearest_query':query,
            'support_region_selection':selection,
            'loaded_support_contact_proven':False,'native_retention_required':True,
            'scope':'Actual material-end triangle-centre area near the actual upward support; holes remain empty. No force or native stability approval.'}


def above_unthreaded(points,vertices,faces,up,*,tolerance=1e-6,surface_region='all_upward'):
    """Reject material below the support top, including through a real hole.

    A convex projected footprint scopes the routing statement, not collision
    geometry: holes remain open in simulation. Material outside the footprint
    can hang freely. Nearest upward faces supply actual local support planes.
    """
    p,v,f,up=np.asarray(points,float),np.asarray(vertices,float),np.asarray(faces),np.asarray(up,float)
    if (p.ndim!=2 or p.shape[1:]!=(3,) or not len(p) or v.ndim!=2 or v.shape[1:]!=(3,)
            or f.ndim!=2 or f.shape[1:]!=(3,) or f.dtype.kind not in 'iu' or not len(f)
            or f.min()<0 or f.max()>=len(v) or up.shape!=(3,) or not np.isfinite(up).all()
            or not np.isclose(np.linalg.norm(up),1.) or not np.isfinite(p).all() or not np.isfinite(v).all()
            or not np.isfinite(tolerance) or tolerance<0):
        raise ValueError('Finite sheet, support triangles, unit up and nonnegative tolerance required')
    selected,selection=upward_patch(v,f,up,tolerance,surface_region,return_report=True)
    top=trimesh.Trimesh(v,f[selected],process=False)
    seed=np.eye(3)[np.argmin(abs(up))];a=np.cross(up,seed);a/=np.linalg.norm(a);b=np.cross(up,a)
    basis=np.stack([a,b],1);projected=v[np.unique(f[selected])]@basis;hull=ConvexHull(projected)
    inside=np.max((p@basis)@hull.equations[:,:2].T+hull.equations[:,2],axis=1)<=tolerance
    if not inside.any():
        return {'pass':False,'interior_samples':0,'reason':'No sheet material above the declared support footprint'}
    # Open support surfaces need unsigned nearest-triangle queries. Use the
    # existing Warp dependency and CPU BVH, without an additional rtree package.
    closest,triangle,query=_closest_actual(v,f[selected],p[inside])
    heights=np.einsum('ij,ij->i',p[inside]-closest,top.face_normals[triangle])
    deficit=np.maximum(0.,-heights);bad=deficit>tolerance
    return {'pass':not bool(bad.any()),'interior_samples':int(inside.sum()),
            'below_support_samples':int(bad.sum()),'maximum_below_support_m':float(deficit.max()),
            'minimum_above_support_gap_m':float(heights.min()),
            'median_above_support_gap_m':float(np.median(heights)),
            'maximum_above_support_gap_m':float(heights.max()),
            'loaded_support_contact_proven':False,
            'surface_region':surface_region,'selected_support_faces':int(selected.sum()),
            'tolerance_m':float(tolerance),'collision_geometry_changed':False,
            'nearest_query':query,
            'support_region_selection':selection,
            'scope':'Initial above-support routing only. No contact, support force, retention, material or task approval.'}
