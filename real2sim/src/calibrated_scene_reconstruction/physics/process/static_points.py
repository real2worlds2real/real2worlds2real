"""Source-only stationary feature depth checks before assigning rigid anchors."""
import cv2
import numpy as np
from scipy.optimize import least_squares

from .contract import require


def triangulate(pixels,camera,*,fit_indices,validation_indices,pixel_sigma=1.,min_parallax_deg=1.,max_rmse_px=1.5):
    """A declared stationary interval, not the moving response trajectory.

    Source fit views determine the point. Held-out views only report validity.
    Covariance covers local pixel noise; camera/scale/model bias remains outside
    it. Mesh intersection from one frame does not supply measured depth.
    """
    pixels=np.asarray(pixels,float);T=np.asarray(camera['T_world_cam']);K=np.asarray(camera['K']);D=np.asarray(camera['dist'])
    require(pixels.shape==(len(T),2) and pixel_sigma>0 and min_parallax_deg>0 and max_rmse_px>0,
            'Source pixel timebase and explicit observation uncertainty required')
    fit=np.array([i for i in fit_indices if np.isfinite(pixels[i]).all()],int)
    held=np.array([i for i in validation_indices if np.isfinite(pixels[i]).all()],int)
    require(len(fit)>=3 and len(held)>=2 and not set(fit)&set(held),'Disjoint stationary source views required')
    normalized=cv2.undistortPoints(pixels[fit].reshape(-1,1,2),K,D).reshape(-1,2);A=[]
    for i,uv in zip(fit,normalized):
        inverse=np.linalg.inv(T[i])[:3];A.extend([uv[0]*inverse[2]-inverse[0],uv[1]*inverse[2]-inverse[1]])
    _,_,V=np.linalg.svd(A,full_matrices=False);require(abs(V[-1,3])>1e-10,'Source views do not determine finite depth')
    seed=V[-1,:3]/V[-1,3]
    def project(point,ids):
        result=[]
        for i in ids:
            R=T[i,:3,:3].T
            result.append(cv2.projectPoints(np.asarray(point)[None],cv2.Rodrigues(R)[0],-R@T[i,:3,3],K,D)[0].reshape(2))
        return np.array(result)
    solved=least_squares(lambda p:(project(p,fit)-pixels[fit]).ravel(),seed,loss='soft_l1',f_scale=pixel_sigma,max_nfev=200)
    point=solved.x;rays=point-T[fit,:3,3];rays/=np.linalg.norm(rays,axis=1)[:,None]
    parallax=float(np.rad2deg(np.arccos(np.clip(np.min(rays@rays.T),-1,1))))
    # Unrobust local Jacobian at the final source fit, so reported covariance
    # does not mistake the robust loss's downweighting for extra information.
    eps=1e-5;J=np.column_stack([((project(point+np.eye(3)[j]*eps,fit)-project(point-np.eye(3)[j]*eps,fit))/(2*eps)).ravel() for j in range(3)])
    information=J.T@J;singular=np.linalg.svd(information,compute_uv=False)
    require(singular[-1]>1e-10,'Stationary depth is singular')
    covariance=np.linalg.inv(information)*pixel_sigma**2
    fit_rmse=float(np.sqrt(np.mean((project(point,fit)-pixels[fit])**2)))
    held_rmse=float(np.sqrt(np.mean((project(point,held)-pixels[held])**2)))
    positive=all(((point-T[i,:3,3])@T[i,:3,:3])[2]>0 for i in np.r_[fit,held])
    return {'point':point,'covariance':covariance,'max_sigma_m':float(np.sqrt(np.linalg.eigvalsh(covariance).max())),
        'parallax_deg':parallax,'fit_rmse_px':fit_rmse,'validation_rmse_px':held_rmse,
        'pass':bool(solved.success and positive and parallax>=min_parallax_deg and max(fit_rmse,held_rmse)<=max_rmse_px),
        'fit_indices':fit.tolist(),'validation_indices':held.tolist(),
        'scope':'Stationary source texture triangulation under camera/scale priors; local pixel covariance excludes camera, scale and incorrect identity bias. Not a native process result.'}


def placement_translation(model_points,source_points,*,max_translation_m,max_vertical_m,max_disagreement_m):
    """Bounded initial scene-placement hypothesis, not a physics parameter.

    All parts/supports must receive the same translation. Two anchors can test
    translation coherence but cannot establish an arbitrary rigid registration.
    Native image/contact verification remains mandatory.
    """
    model=np.asarray(model_points,float);source=np.asarray(source_points,float)
    require(model.shape==source.shape and model.ndim==2 and model.shape[1]==3 and len(model)>=2
            and np.isfinite(model).all() and np.isfinite(source).all(),'Two or more stationary material correspondences required')
    require(min(max_translation_m,max_vertical_m,max_disagreement_m)>0,'Declare initial placement uncertainty bounds')
    differences=source-model;translation=np.median(differences,axis=0);residuals=differences-translation
    require(np.linalg.norm(translation)<=max_translation_m and abs(translation[2])<=max_vertical_m,
            'Stationary source/model displacement exceeds the declared placement prior')
    require(np.max(np.linalg.norm(residuals,axis=1))<=max_disagreement_m,
            'Source anchors do not support one rigid translation; revise geometry, identity or camera')
    return {'translation':translation,'residuals':residuals,'native_simulation_run':False,'accepted':False,
        'scope':'Initial global placement proposal from stationary fit-view texture points only. Relative solid geometry and support contacts unchanged; source image agreement and free native contact/dynamics still require revalidation.'}


def refine_placement(model_points,pixels,camera,*,fit_indices,validation_indices,
                     translation_bounds,pixel_sigma):
    """Jointly refine one static scene translation from reviewed part points.

    Do not let a few texture anchors override the remaining initial geometry.
    Points belong to a declared stationary interval; a centroid of a changing
    silhouette is not a material point. Camera/scale and geometry stay fixed.
    Bounds and per-part uncertainty must be source/prior based, not adjusted
    against held-out errors. The result still needs native and image checks.
    """
    points=np.asarray(model_points,float);pixels=np.asarray(pixels,float)
    T=np.asarray(camera['T_world_cam']);K=np.asarray(camera['K']);D=np.asarray(camera['dist'])
    bounds=np.asarray(translation_bounds,float);sigma=np.asarray(pixel_sigma,float)
    require(points.ndim==2 and points.shape[1]==3 and len(points)>=2 and np.isfinite(points).all()
            and pixels.shape==(len(T),len(points),2),'Reviewed stationary part points and source camera timebase required')
    require(bounds.shape==(2,3) and np.isfinite(bounds).all() and np.all(bounds[0]<=0)
            and np.all(bounds[1]>=0) and np.all(bounds[0]<bounds[1]),'Finite translation bounds containing the seed required')
    require(sigma.shape==(len(points),) and np.isfinite(sigma).all() and np.all(sigma>0),
            'Explicit positive per-part pixel uncertainty required')
    fit=np.asarray(fit_indices,int);held=np.asarray(validation_indices,int)
    require(len(fit)>=3 and len(held)>=2 and not set(fit)&set(held),'Disjoint stationary source views required')
    require(np.all(np.isfinite(pixels).all(2)|np.isnan(pixels).all(2)),
            'Missing source points must have both coordinates missing')
    for ids in [fit,held]:
        require(np.all(np.isfinite(pixels[ids]).all(2).sum(0)>=2),'Each part needs independent fit and held-out source support')
    def predict(delta,ids):
        rows=[]
        for i in ids:
            R=T[i,:3,:3].T
            rows.append(cv2.projectPoints(points+delta,cv2.Rodrigues(R)[0],-R@T[i,:3,3],K,D)[0].reshape(-1,2))
        return np.array(rows)
    valid=np.isfinite(pixels[fit])
    solved=least_squares(lambda delta:((predict(delta,fit)-pixels[fit])/sigma[None,:,None])[valid],
                         np.zeros(3),bounds=bounds,max_nfev=100)
    require(solved.success and np.linalg.matrix_rank(solved.jac)==3,'Source views do not determine a finite translation refinement')
    require(all(np.all(((points+solved.x-T[i,:3,3])@T[i,:3,:3])[:,2]>0) for i in np.r_[fit,held]),
            'Placement puts reviewed points behind a source camera')
    metrics={}
    for name,ids in [('fit',fit),('validation',held)]:
        errors=predict(solved.x,ids)-pixels[ids]
        metrics[name+'_rmse_px']=np.sqrt(np.nanmean(errors**2,axis=(0,2)))
    return {'translation':solved.x,'metrics':metrics,'native_simulation_run':False,'accepted':False,
        'scope':'One bounded translation fitted jointly to stationary reviewed part points using fit views only. Relative geometry/contact and physical parameters unchanged. Held-out images and native physics still determine acceptance.'}
