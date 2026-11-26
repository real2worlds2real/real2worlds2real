"""Measure a real finite clamped sheet patch, independently of task success."""
import numpy as np


def measure(points,faces,indices,*,minimum_width_m,minimum_area_m2):
    v=np.asarray(points,float);f=np.asarray(faces,int);roots=np.asarray(indices,int)
    if min(minimum_width_m,minimum_area_m2)<=0 or not np.isfinite([minimum_width_m,minimum_area_m2]).all():
        raise ValueError('Positive physical clamp footprint bounds required')
    if roots.ndim!=1 or len(roots)<3 or len(np.unique(roots))!=len(roots) or roots.min()<0 or roots.max()>=len(v):
        raise ValueError('A clamped sheet needs at least three distinct actual material nodes')
    if (v.ndim!=2 or v.shape[1:]!=(3,) or not np.isfinite(v).all()
            or f.ndim!=2 or f.shape[1:]!=(3,) or not len(f)
            or f.min()<0 or f.max()>=len(v) or not np.isin(roots,np.unique(f)).all()):
        raise ValueError('Finite actual material triangles and their bound nodes required')
    # Blender's bundled Python has NumPy but not SciPy. Keep the exact
    # material connectivity check independent of optional solver packages.
    parents=list(range(len(v)));ranks=[0]*len(v)
    def find(index):
        index=int(index)
        while parents[index]!=index:
            parents[index]=parents[parents[index]];index=parents[index]
        return index
    for a,b,c in f.tolist():
        for left,right in [(a,b),(b,c)]:
            ra,rb=find(left),find(right)
            if ra==rb:continue
            if ranks[ra]<ranks[rb]:ra,rb=rb,ra
            parents[rb]=ra
            if ranks[ra]==ranks[rb]:ranks[ra]+=1
    owner=np.fromiter((find(i)for i in range(len(v))),dtype=np.int64,count=len(v))
    components=[]
    for component in np.unique(owner[f[:,0]]):
        local_faces=f[owner[f[:,0]]==component]
        local_roots=roots[owner[roots]==component]
        result=_patch(v,local_faces,local_roots,minimum_width_m,minimum_area_m2)
        components.append({'component':int(component),'surface_triangles':len(local_faces),
                           'bound_vertices':len(local_roots),**result})
    return {'pass':all(row['pass']for row in components),
            'fully_bonded_triangles':sum(row['fully_bonded_triangles']for row in components),
            'bonded_area_m2':sum(row['bonded_area_m2']for row in components),
            'minimum_in_plane_span_m':min(row['minimum_in_plane_span_m']for row in components),
            'minimum_required_area_m2':float(minimum_area_m2),'minimum_required_width_m':float(minimum_width_m),
            'components':components,'bounds_apply_to':'each connected material sheet',
            'width_measure':'Twice the second principal RMS spread within each bonded sheet patch; invariant to rigid frame, including repeated principal values',
            'scope':'Geometric clamp construction only; each disconnected sheet must independently satisfy its declared footprint. Actual solid contact, coupled response and native deformation remain mandatory'}


def _patch(v,f,roots,minimum_width_m,minimum_area_m2):
    bonded=f[np.isin(f,roots).all(1)]
    if not len(bonded):raise ValueError('A root boundary or isolated pins cannot substitute for a finite clamped material patch')
    q=v[bonded];area=float(np.linalg.norm(np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),axis=1).sum()/2)
    cloud=v[np.unique(bonded)];singular=np.linalg.svd(cloud-cloud.mean(0),compute_uv=False)
    width=float(2*singular[1]/np.sqrt(len(cloud)))
    return {'pass':bool(area>=minimum_area_m2 and width>=minimum_width_m),
            'fully_bonded_triangles':len(bonded),'bonded_area_m2':area,'minimum_in_plane_span_m':width,
            'minimum_required_area_m2':float(minimum_area_m2),'minimum_required_width_m':float(minimum_width_m)}
