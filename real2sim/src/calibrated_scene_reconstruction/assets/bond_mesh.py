"""Remove unresolved zero-area material inside an explicitly bonded region."""
import numpy as np


def trim_zero_area(points,faces,roots,*,area_tolerance_m2=1e-14):
    v=np.asarray(points,float);f=np.asarray(faces,int);r=np.asarray(roots,int)
    q=v[f];area=np.linalg.norm(np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),axis=1)/2
    bad=area<=area_tolerance_m2
    if np.any(bad&~np.isin(f,r).all(1)):raise ValueError('Unresolved free material cannot be deleted as bonded geometry')
    keep=f[~bad]
    if len(keep)<2:raise ValueError('No resolved physical surface remains')
    used=np.unique(keep);mapping=np.full(len(v),-1,int);mapping[used]=np.arange(len(used))
    return v[used],mapping[keep],mapping[r[mapping[r]>=0]].tolist(),{
        'removed_fully_bonded_zero_area_triangles':int(bad.sum()),'removed_area_m2':float(area[bad].sum()),
        'area_tolerance_m2':float(area_tolerance_m2),'free_material_removed':False,'vertex_positions_changed':False}
