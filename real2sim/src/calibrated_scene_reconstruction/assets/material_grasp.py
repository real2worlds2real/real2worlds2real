"""Source-indexed contact regions for a pinch on one material component.

Two sides of a folded sheet can be far apart in intrinsic coordinates. Do not
require both fingers to touch one arbitrarily small material disk. Geometric
proximity and opposed normals are prerequisites, not friction or task proof.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation

from calibrated_scene_reconstruction.assets.surface_distance import triangle_samples


def contact_regions(points,rest,faces,pads,transforms,names,hull_rows,groups,*,contact_range_m,patch_radius,boundary_margin):
    points,rest,faces=np.asarray(points,float),np.asarray(rest,float),np.asarray(faces)
    pads,transforms=np.asarray(pads,float),np.asarray(transforms,float)
    if (points.shape!=rest.shape or points.ndim!=2 or points.shape[1:]!=(3,)
            or faces.ndim!=2 or faces.shape[1:]!=(3,) or faces.dtype.kind not in 'iu' or not len(faces)
            or faces.min()<0 or faces.max()>=len(points) or len(groups)!=2 or pads.shape!=(2,3)
            or transforms.shape!=(len(names),7) or not all(np.isfinite(x).all()for x in [points,rest,pads,transforms])
            or not np.isfinite([contact_range_m,patch_radius,boundary_margin]).all()
            or min(contact_range_m,patch_radius)<=0 or boundary_margin<0 or len(set(names))!=len(names)):
        raise ValueError('Finite same-sheet geometry, actual pad poses and explicit contact-region bounds required')
    axis=pads[1]-pads[0];length=np.linalg.norm(axis)
    if length<1e-8:raise ValueError('Resolved actual opposing-finger axis required')
    axis/=length
    edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
    unique,counts=np.unique(edges,axis=0,return_counts=True);boundary=np.unique(unique[counts==1])
    _,labels=connected_components(coo_matrix((np.ones(len(unique)),(unique[:,0],unique[:,1])),shape=(len(points),len(points))),directed=False)
    sampled=triangle_samples(points,faces);intrinsic=triangle_samples(rest,faces)
    components=np.r_[labels,np.tile(labels[faces[:,0]],4)]
    clearance=(cKDTree(rest[boundary]).query(intrinsic)[0] if len(boundary) else np.full(len(sampled),np.inf))
    candidates={}
    for finger,(role,links) in enumerate(groups.items()):
        found=[];direction=axis if finger==0 else -axis
        if not set(links)<=set(names):raise ValueError('Contact groups must identify actual native rigid bodies')
        for name,lo,hi,planes in hull_rows:
            if name not in links:continue
            pose=transforms[list(names).index(name)];r=Rotation.from_quat(pose[3:]).as_matrix()
            local=(sampled-pose[:3])@r
            ids=np.flatnonzero(((local>=lo-contact_range_m)&(local<=hi+contact_range_m)).all(1)&(clearance>boundary_margin))
            if not len(ids):continue
            signed=local[ids]@planes[:,:3].T+planes[:,3];active=signed.argmax(1);distance=signed[np.arange(len(ids)),active]
            alignment=(planes[active,:3]@r.T)@direction
            found.extend(ids[(abs(distance)<=contact_range_m)&(alignment>1e-9)].tolist())
        candidates[role]=np.unique(found).astype(int)
    common=None
    for ids in candidates.values():
        available=set(components[ids].tolist());common=available if common is None else common&available
    if not common:return {'pass':False,'reason':'No common material component with opposing interior contacts',
        'contact_sample_counts':{k:len(v)for k,v in candidates.items()},'task_approved':False}
    component=max(common,key=lambda c:min(np.count_nonzero(components[v]==c)for v in candidates.values()))
    regions={};all_nodes=[]
    for finger,(role,ids) in enumerate(candidates.items()):
        ids=ids[components[ids]==component];sample=int(ids[np.argmin(np.linalg.norm(sampled[ids]-pads[finger],axis=1))])
        anchor=intrinsic[sample];component_nodes=np.flatnonzero(labels==component)
        node=int(component_nodes[cKDTree(rest[component_nodes]).query(anchor)[1]])
        patch=np.flatnonzero((np.linalg.norm(rest-anchor,axis=1)<patch_radius)&(labels==component))
        if len(patch)<3:return {'pass':False,'reason':'Unresolved contacted material region','task_approved':False}
        regions[role]={'material_node':node,'material_patch':patch.tolist(),'contact_sample':sample,
            'material_anchor':anchor.tolist(),
            'contact_point_world':sampled[sample].tolist(),'source_contact_samples':int(len(ids)),
            'material_boundary_clearance_m':float(clearance[sample])};all_nodes.extend(patch.tolist())
    return {'pass':True,'component':int(component),'regions':regions,'material_patch':np.unique(all_nodes).tolist(),
        'contact_range_m':float(contact_range_m),'task_approved':False,
        'scope':'Actual opposing proximity regions on one connected material sheet; retain native penetration, retention and matched-friction checks.'}


def retained_regions(points,faces,regions,penetration,transforms,names,tolerance,contact_range_m,groups):
    """Track fixed source material indices at each corresponding finger."""
    from calibrated_scene_reconstruction.assets.hand_contact_isolation import material_samples
    if not regions.get('pass') or set(regions['regions'])!=set(groups):
        return {'pass':False,'reason':'No resolved per-finger source material regions'}
    checks={}
    for role,region in regions['regions'].items():
        report=penetration.check(material_samples(points,faces,region['material_patch']),transforms,names,tolerance,
                                 contact_range_m=contact_range_m)
        checks[role]={'pass':bool(report['pass'] and sum(report['near_surface_samples'].get(n,0)for n in groups[role])>0),
                      'contact':report}
    return {'pass':all(r['pass']for r in checks.values()),'regions':checks,'task_approved':False}
