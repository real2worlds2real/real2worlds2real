"""Infer an orthogonal structural frame from calibrated source line segments.

This is a source-only proposal for grids, rails and repeated planar structures.
It infers orientation, not metric pitch, plane offset, room bounds or support.
Independent source views and structural/physical checks remain required.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def line_planes(K, rotation_world_camera, segments):
    K=np.asarray(K,float);R=np.asarray(rotation_world_camera,float);p=np.asarray(segments,float)
    if (K.shape!=(3,3) or R.shape!=(3,3) or p.ndim!=3 or p.shape[1:]!=(2,2)
            or not all(np.isfinite(x).all() for x in [K,R,p]) or abs(np.linalg.det(K))<1e-9
            or not np.allclose(R.T@R,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(R),1.,atol=1e-6)):
        raise ValueError('Finite source segments, camera intrinsics and proper rotation required')
    homogeneous=np.concatenate([p,np.ones((*p.shape[:2],1))],axis=2)
    image_lines=np.cross(homogeneous[:,0],homogeneous[:,1])
    normals=image_lines@K@R.T
    lengths=np.linalg.norm(p[:,1]-p[:,0],axis=1)
    if np.any(lengths<1.) or np.any(np.linalg.norm(normals,axis=1)<1e-12):
        raise ValueError('Source line segments must span at least one pixel')
    return normals/np.linalg.norm(normals,axis=1,keepdims=True),lengths


def solve(normals,weights,*,up=(0.,0.,1.),maximum_tilt_deg=20.,inlier_degrees=1.,starts=12):
    """Fit two perpendicular in-plane axes, rejecting uninformative line sets."""
    n=np.asarray(normals,float);w=np.asarray(weights,float);up=np.asarray(up,float)
    if (n.ndim!=2 or n.shape[1:]!=(3,) or len(n)<12 or w.shape!=(len(n),)
            or up.shape!=(3,) or not all(np.isfinite(x).all() for x in [n,w,up])
            or (w<=0).any() or np.linalg.norm(up)<1e-12
            or not 0<maximum_tilt_deg<=35 or not 0<inlier_degrees<=3
            or type(starts) is not int or not 4<=starts<=48):
        raise ValueError('At least12 finite weighted source lines and bounded structural hypothesis required')
    lengths=np.linalg.norm(n,axis=1)
    if (lengths<1e-12).any():raise ValueError('Nonzero source line planes required')
    n=n/lengths[:,None];up=up/np.linalg.norm(up)
    seed=np.eye(3)[np.argmin(abs(up))];x=seed-up*(seed@up);x/=np.linalg.norm(x)
    base=np.column_stack([x,np.cross(up,x),up]);limit=np.radians(maximum_tilt_deg)
    weight=np.sqrt(np.minimum(w,np.median(w)*4)/np.mean(w));threshold=np.sin(np.radians(inlier_degrees))
    def axes(p):return base@Rotation.from_euler('xyz',p).as_matrix()
    def residual(p):
        r=n@axes(p)[:,:2];selected=np.argmin(abs(r),axis=1)
        return r[np.arange(len(n)),selected]*weight
    best=None
    for yaw in np.linspace(0,np.pi/2,starts,endpoint=False):
        fit=least_squares(residual,[0.,0.,yaw],bounds=([-limit,-limit,-np.pi],[limit,limit,np.pi]),
                          loss='soft_l1',f_scale=threshold/3,max_nfev=160)
        a=axes(fit.x);raw=n@a[:,:2];groups=np.argmin(abs(raw),axis=1);error=abs(raw[np.arange(len(n)),groups])
        valid=error<threshold;counts=[int(np.sum(valid&(groups==k))) for k in [0,1]]
        # Both line families must constrain the frame. One long family alone
        # cannot establish an observed plane normal.
        if min(counts)<4:continue
        score=float(w[valid].sum()/w.sum())-.05*float(np.median(error[valid])/threshold)
        if best is None or score>best[0]:best=(score,a,error,groups,valid,counts,fit)
    if best is None:raise ValueError('No two independently supported orthogonal line families')
    _,a,error,groups,valid,counts,fit=best
    # Robust-loss tails still exert torque on a weakly observed normal. Refit
    # the selected coherent families, then report residuals for every line.
    # Excluded lines remain explicit outliers rather than disappearing from QA.
    for _ in range(2):
        chosen=np.flatnonzero(valid);family=groups[chosen].copy()
        def inlier_residual(p):
            r=n[chosen]@axes(p)[:,:2]
            return r[np.arange(len(chosen)),family]*weight[chosen]
        fit=least_squares(inlier_residual,fit.x,
            bounds=([-limit,-limit,-np.pi],[limit,limit,np.pi]),loss='soft_l1',f_scale=threshold/8,max_nfev=160)
        a=axes(fit.x);raw=n@a[:,:2];groups=np.argmin(abs(raw),axis=1)
        error=abs(raw[np.arange(len(n)),groups]);valid=error<threshold
    counts=[int(np.sum(valid&(groups==k))) for k in [0,1]]
    singular=np.linalg.svd(fit.jac,compute_uv=False)
    tilt=float(np.degrees(np.arccos(np.clip(a[:,2]@up,-1,1))))
    return dict(accepted=False,orientation_proposal_pass=bool(fit.success and not fit.active_mask.any()
                and tilt<=maximum_tilt_deg and valid.mean()>=.5 and min(counts)>=4
                and singular[-1]>singular[0]*1e-5),axes_world=a.tolist(),tilt_deg=tilt,
                inlier_counts=counts,inlier_fraction=float(valid.mean()),
                jacobian_singular_values=singular.tolist(),
                median_inlier_plane_angle_deg=float(np.degrees(np.arcsin(np.median(error[valid])))),
                plane_angle_deg=np.degrees(np.arcsin(np.clip(error,0,1))).tolist(),
                family=groups.tolist(),inliers=valid.tolist(),bound_hit=bool(fit.active_mask.any()),
                metric_dimensions_inferred=False,scope=__doc__.strip())
