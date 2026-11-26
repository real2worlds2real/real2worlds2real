"""Bounded source refinement of an actual background mesh, never a box proxy.

This proposes placement and anisotropic dimensions. It neither changes the
source evidence nor accepts appearance/contact. A changed support height must
be reconciled with the parent furniture before rebuilding and reviewing.
"""
from types import SimpleNamespace

import numpy as np
from scipy.optimize import minimize

from calibrated_scene_reconstruction.core.geometry.raster import camera_triangle_mask
from .refine import rotation


def solve(vertices, faces, origin, yaw, views, bounds, *, budget=300, rasterizer=None):
    V,F,origin=np.asarray(vertices,float),np.asarray(faces,int),np.asarray(origin,float)
    bounds=np.asarray(bounds,float)
    if (V.ndim!=2 or V.shape[1]!=3 or F.ndim!=2 or F.shape[1]!=3
            or origin.shape!=(3,) or bounds.shape!=(7,2) or not 10<=budget<=1000
            or not np.isfinite(V).all() or not np.isfinite(origin).all()
            or not np.isfinite(bounds).all() or not np.isfinite(yaw)
            or np.any(bounds[:,0]>=0) or np.any(bounds[:,1]<=0)
            or F.min()<0 or F.max()>=len(V)):
        raise ValueError('Finite actual triangles and seven bounded translation/yaw/log-scale coordinates required')
    if len(views)<3 or max(np.linalg.norm(np.asarray(a['C'])-b['C']) for a in views for b in views)<.05:
        raise ValueError('At least three independent source views required')
    prepared=[]
    for view in views:
        mask=np.asarray(view['mask'],bool);domain=np.asarray(view['domain'],bool)
        if mask.shape!=domain.shape or mask.ndim!=2 or (mask&~domain).any() or mask.sum()<32:
            raise ValueError('Observed positive pixels and an explicit valid/occlusion domain required')
        prepared.append((view,mask,domain))
    local=(V-origin)@rotation(yaw)
    def geometry(x):return local*np.exp(x[4:])@rotation(yaw+x[3]).T+origin+x[:3]
    def measure(x,gpu=True):
        current=geometry(x);mesh=SimpleNamespace(vertices=current,faces=F);ious=[]
        for row,mask,domain in prepared:
            if gpu and rasterizer is not None:
                h,w=mask.shape
                prediction=rasterizer(mesh,row['K'],row['R'],row['C'],[0,0,w,h,1.],mask.shape)
            else:
                prediction=camera_triangle_mask((current-row['C'])@row['R'],F,row['K'],mask.shape)
            prediction&=domain
            ious.append(float((mask&prediction).sum()/max(1,(mask|prediction).sum())))
        return ious
    initial=np.zeros(7);best=[1-float(np.mean(measure(initial))),initial.copy()]
    def objective(x):
        loss=1-float(np.mean(measure(x)))
        if loss<best[0]:best[:]=[loss,x.copy()]
        return loss
    fit=minimize(objective,initial,method='Powell',bounds=bounds.tolist(),
                 options={'maxfev':budget,'xtol':1e-4,'ftol':1e-5})
    x=best[1]
    before=measure(initial,False);after=measure(x,False)
    return {'position':(origin+x[:3]).tolist(),'yaw_deg':float(np.degrees(yaw+x[3])),
            'size_multipliers':np.exp(x[4:]).tolist(),'delta':x.tolist(),
            'cpu_source_iou_before':before,'cpu_source_iou_after':after,
            'improved':bool(np.mean(after)>np.mean(before)),
            'bound_hit':bool(np.any(np.isclose(x,bounds[:,0],atol=1e-5)|np.isclose(x,bounds[:,1],atol=1e-5))),
            'evaluations':int(fit.nfev),'accepted':False,'scope':__doc__.strip()}
