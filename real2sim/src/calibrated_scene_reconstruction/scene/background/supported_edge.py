"""Fit a horizontal furniture front edge; keep hidden depth an explicit prior.

Occluded rear corners must not be relabelled from nearby clutter. Two observed
endpoints across calibrated views determine a front edge. Optional corresponding
base endpoints bind that front face to an independently reconstructed support;
this is useful when an upper edge alone has weak depth parallax. A disclosed depth
prior completes the furniture but never counts as a measured rear surface.
"""
import numpy as np
from scipy.optimize import least_squares


def solve(cameras, observations, initial, *, depth_prior_m, floor_m=0., error_px=6.):
    names=list(map(str,cameras['names']));rows=[];seen=set()
    if (not np.isfinite([depth_prior_m,floor_m,error_px]).all() or depth_prior_m<=0
            or not 1<=error_px<=12):
        raise ValueError('Positive hidden-depth prior and bounded source uncertainty required')
    for row in observations:
        ids=np.asarray(row['endpoint_indices']);pixels=np.asarray(row['pixels'],float)
        if (row['frame'] not in names or row['frame'] in seen or ids.ndim!=1
                or len(ids)<1 or ids.dtype.kind not in 'iu' or len(set(ids))!=len(ids)
                or np.any((ids<0)|(ids>1)) or pixels.shape!=(len(ids),2)
                or not np.isfinite(pixels).all()):
            raise ValueError('Distinct frames and visible front-edge endpoint identities required')
        seen.add(row['frame']);i=names.index(row['frame']);w,h=cameras['size'][i]
        if np.any(pixels<0) or np.any(pixels[:,0]>=w) or np.any(pixels[:,1]>=h):
            raise ValueError('Observed endpoint outside source raster')
        if 'base_endpoint_indices' in row or 'base_pixels' in row:
            base_ids=np.asarray(row.get('base_endpoint_indices',[]))
            base_pixels=np.asarray(row.get('base_pixels',[]),float)
            if (base_ids.ndim!=1 or not len(base_ids) or base_ids.dtype.kind not in 'iu'
                    or len(set(base_ids))!=len(base_ids) or np.any((base_ids<0)|(base_ids>1))
                    or base_pixels.shape!=(len(base_ids),2) or not np.isfinite(base_pixels).all()
                    or np.any(base_pixels<0) or np.any(base_pixels[:,0]>=w) or np.any(base_pixels[:,1]>=h)):
                raise ValueError('Visible corresponding base endpoints must lie in the source raster')
            ids=np.r_[ids,base_ids+2]
            pixels=np.vstack([pixels,base_pixels])
        rows.append((i,ids,pixels))
    if (len(rows)<2 or any(sum(int(np.sum(ids==k))for _,ids,_ in rows)<2 for k in [0,1])
            or max(np.linalg.norm(cameras['centre'][a[0]]-cameras['centre'][b[0]])for a in rows for b in rows)<.05):
        raise ValueError('Each endpoint needs at least two independent calibrated observations')
    x0=np.array([*initial['front_xy'],np.radians(initial['yaw_deg']),
                 np.log(initial['width_m']),np.log(initial['height_m'])])
    if x0.shape!=(5,) or not np.isfinite(x0).all():raise ValueError('Finite positive edge hypothesis required')
    def vertices(x):
        direction=np.array([np.cos(x[2]),np.sin(x[2])]);xy=x[:2]+np.array([[-.5],[.5]])*np.exp(x[3])*direction
        return np.vstack([np.c_[xy,np.full(2,floor_m+np.exp(x[4]))],
                          np.c_[xy,np.full(2,floor_m)]])
    def project(x):
        v=vertices(x);pred=[]
        for i,ids,uv in rows:
            pc=(v[ids]-cameras['centre'][i])@cameras['R_world_cam'][i]
            if np.any(pc[:,2]<=.05):return None
            q=pc@cameras['K'][i].T;pred.append(q[:,:2]/q[:,2,None])
        return pred
    def residual(x):
        pred=project(x)
        if pred is None:return np.full(sum(len(p)*2 for _,_,p in rows),1e6)
        return np.concatenate([(q-p).ravel()for q,(_,_,p)in zip(pred,rows)])
    fit=least_squares(residual,x0,loss='soft_l1',f_scale=2.,max_nfev=800,
                      bounds=(x0+[-1.,-1.,-np.pi/2,np.log(.35),np.log(.35)],
                              x0+[1.,1.,np.pi/2,np.log(2.),np.log(2.)]))
    pred=project(fit.x);error=np.linalg.norm(residual(fit.x).reshape(-1,2),axis=1)
    singular=np.linalg.svd(fit.jac,compute_uv=False)
    direction=np.array([-np.sin(fit.x[2]),np.cos(fit.x[2])])
    centre=fit.x[:2]+depth_prior_m/2*direction
    return dict(source_edge_fit_pass=bool(fit.success and not np.any(fit.active_mask) and error.max()<=error_px
                                         and singular[-1]>singular[0]*1e-6),
        accepted=False,position=[*map(float,centre),float(floor_m)],
        size=[float(np.exp(fit.x[3])),float(depth_prior_m),float(np.exp(fit.x[4]))],
        yaw_deg=float(np.degrees(fit.x[2])),front_endpoints=vertices(fit.x)[:2].tolist(),
        base_endpoints=vertices(fit.x)[2:].tolist(),
        error_px=error.tolist(),jacobian_singular_values=singular.tolist(),bound_hit=bool(np.any(fit.active_mask)),
        source_checks=[dict(frame=row['frame'],landmark_indices=ids.tolist(),
                            observed=p.tolist(),projected=q.tolist())
                       for row,(_,ids,p),q in zip(observations,rows,pred)],
        inferred_dimensions=['depth'],scope=__doc__.strip())
