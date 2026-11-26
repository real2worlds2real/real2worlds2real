"""Transfer discrete visual identities through an explicit rectangular material chart.

Geometry and physics are unchanged. Coverage errors are diagnostics: a coarse
triangle cannot represent every narrow stripe, so remapping alone is not approval.
This module intentionally uses only NumPy for Blender's bundled Python.
"""
import numpy as np


def from_grid(source_grid, source_labels, target_vertices, target_faces, target_uv):
    """Map each target face's material centre to the original a-b-c/a-c-d grid.

    ``source_labels`` is (rows, columns, 2), in the same triangle order as
    ``strip_sampling``. The returned labels and actual-area inventory expose
    missing or enlarged regions. No error tolerance or acceptance is inferred.
    """
    grid=np.asarray(source_grid,float);labels=np.asarray(source_labels)
    v=np.asarray(target_vertices,float);f=np.asarray(target_faces,int);uv=np.asarray(target_uv,float)
    if (grid.ndim!=3 or grid.shape[2:]!=(3,) or min(grid.shape[:2])<2
            or not np.isfinite(grid).all() or labels.shape!=(*((np.array(grid.shape[:2])-1).tolist()),2)
            or not np.issubdtype(labels.dtype,np.integer) or (labels<0).any()):
        raise ValueError('Finite rectangular source geometry and per-triangle material identities required')
    if (v.ndim!=2 or v.shape[1:]!=(3,) or not np.isfinite(v).all()
            or f.ndim!=2 or f.shape[1:]!=(3,) or not len(f) or f.min()<0 or f.max()>=len(v)
            or uv.shape!=(len(v),2) or not np.isfinite(uv).all()
            or (uv < -1e-10).any() or (uv > 1+1e-10).any()):
        raise ValueError('Target material coordinates must cover each actual vertex inside the source chart')
    shape=np.array(grid.shape[:2])-1
    q=np.clip(uv[f].mean(axis=1),0.,1.)*shape
    cell=np.minimum(np.floor(q).astype(int),shape-1);frac=q-cell
    target_labels=labels[cell[:,0],cell[:,1],(frac[:,1]<frac[:,0]).astype(int)]
    a,b,c,d=grid[:-1,:-1],grid[:-1,1:],grid[1:,1:],grid[1:,:-1]
    source_area=np.stack([np.linalg.norm(np.cross(b-a,c-a),axis=-1),
                          np.linalg.norm(np.cross(c-a,d-a),axis=-1)],axis=-1)/2
    tri=v[f];target_area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
    if (source_area<=1e-14).any() or (target_area<=1e-14).any():
        raise ValueError('Material-region transfer needs nondegenerate physical triangles')
    inventory=[]
    for material in np.union1d(labels,target_labels):
        before=float(source_area[labels==material].sum());after=float(target_area[target_labels==material].sum())
        source_share=before/float(source_area.sum());target_share=after/float(target_area.sum())
        inventory.append({'material':int(material),'source_area_m2':before,'target_area_m2':after,
                          'source_area_share':source_share,'target_area_share':target_share,
                          'area_share_error':target_share-source_share,
                          'lost_entire_region':bool(before>0 and after==0)})
    return target_labels,{'materials':inventory,'source_area_m2':float(source_area.sum()),
                          'target_area_m2':float(target_area.sum()),
                          'maximum_absolute_share_error':max(abs(r['area_share_error'])for r in inventory),
                          'scope':'Visual material-coordinate identity transfer and actual area audit; no geometry/native/source approval'}
