"""Static assembly pose proposals using the same region operator as validation."""
import numpy as np
from scipy.spatial.transform import Rotation

from .contract import require
from .material_projection import project_material


def axis_twist(vertices,faces,pose,axis,pivot,cameras,centers,extents,*,fit_indices,
               validation_indices,angles,opening=(3,3),center_sigma_px=8.,extent_sigma_px=8.):
    """Rotate an assembly about a supported axis; never animate its response.

    Uses actual triangle masks and declared morphology, rather than a convex
    hull whose filled gaps can misrepresent thin or separated parts. Flexible
    points may come from a previous native preparation, but fresh free native
    execution is still required after applying a selected initial pose.
    """
    vertices=np.asarray(vertices,float);pose=np.asarray(pose,float);axis=np.asarray(axis,float)
    pivot=np.asarray(pivot,float);centers=np.asarray(centers,float);extents=np.asarray(extents,float)
    fit=np.asarray(fit_indices,int);held=np.asarray(validation_indices,int);angles=np.asarray(angles,float)
    require(vertices.ndim==2 and vertices.shape[1]==3 and np.isfinite(vertices).all()
            and axis.shape==pivot.shape==(3,) and np.linalg.norm(axis)>0,'Finite native assembly and supported axis required')
    require(len(fit)>=3 and len(held)>=2 and not set(fit)&set(held)
            and centers.shape==extents.shape==(len(cameras),2),'Independent source region views required')
    require(angles.ndim==1 and 2<=len(angles)<=181 and np.isfinite(angles).all()
            and center_sigma_px>0 and extent_sigma_px>0,'Bounded angle candidates and observation uncertainties required')
    axis/=np.linalg.norm(axis)
    def evaluate(angle,indices):
        R=Rotation.from_rotvec(axis*angle).as_matrix();local=(vertices-pivot)@R.T+pivot
        world=local@pose[:3,:3].T+pose[:3,3];values=[project_material(world,faces,np.zeros(3),np.eye(3),cameras[i],opening=opening) for i in indices]
        c=np.array([v[0] for v in values]);e=np.array([v[1] for v in values])
        return {'center_rmse_px':float(np.sqrt(np.mean((c-centers[indices])**2))),
                'extent_rmse_px':float(np.sqrt(np.mean((e-extents[indices])**2)))}
    candidates=[]
    for angle in angles:
        m=evaluate(float(angle),fit);loss=(m['center_rmse_px']/center_sigma_px)**2+(m['extent_rmse_px']/extent_sigma_px)**2
        candidates.append({'angle_rad':float(angle),'fit_loss':loss,'fit':m})
    best=min(candidates,key=lambda row:(row['fit_loss'],abs(row['angle_rad'])))
    R=Rotation.from_rotvec(axis*best['angle_rad']).as_matrix();delta=np.eye(4);delta[:3,:3]=R;delta[:3,3]=pivot-R@pivot
    return {**best,'delta_local':delta,'validation':evaluate(best['angle_rad'],held),
            'candidates':candidates,'native_checked':False,'operator':'triangle_mask_with_declared_morphology'}
