"""Refine an initial supported assembly with actual projected material masks."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from .contract import require
from .initial_state import rigid_segment_pose,project,segment_distance,aperture_clearance
from .material_projection import project_material


def refine(bottom,top,radius,vertices,faces,initial_pose,cameras,centers,extents,source_lines,*,
           fit_indices,validation_indices,opening,floor_z,apertures,support_radius,other_segments=(),
           center_sigma_px=3.,extent_sigma_px=5.,line_sigma_px=3.,max_evaluations=700):
    """Source-static proposal only; native settling and all gates remain mandatory.

    A finite straight support section joins an arbitrary rigid/flexible region.
    Refine contact azimuth, slopes and axial orientation without changing its
    geometry. Derivative-free fitting respects the quantized region operator.
    """
    bottom,top=np.asarray(bottom,float),np.asarray(top,float);vertices=np.asarray(vertices,float)
    M=np.asarray(initial_pose,float);length=np.linalg.norm(top-bottom)
    fit=np.asarray(fit_indices,int);held=np.asarray(validation_indices,int)
    require(length>0 and radius>0 and len(fit)>=3 and len(held)>=2 and not set(fit)&set(held),
            'Reviewed finite support geometry and independent source views required')
    require(apertures and all(v>0 for v in [center_sigma_px,extent_sigma_px,line_sigma_px]),
            'Explicit contact openings and source observation uncertainties required')
    centers,extents,source_lines=map(lambda x:np.asarray(x,float),(centers,extents,source_lines))
    a,b=np.array([bottom,top])@M[:3,:3].T+M[:3,3];axis=(b-a)/length
    rim=apertures[-1];center=np.asarray(rim['center'],float);height=rim['height'];entry=a+(b-a)*(height-a[2])/(b[2]-a[2])
    zero_pose=rigid_segment_pose(bottom,top,a,b);twist=float(Rotation.from_matrix(M[:3,:3]@zero_pose[:3,:3].T).as_rotvec()@axis)
    seed=np.r_[axis[:2]/axis[2],np.arctan2(entry[1]-center[1],entry[0]-center[0]),twist]
    def geometry(x):
        direction=np.r_[x[:2],1.];direction/=np.linalg.norm(direction)
        z=floor_z+radius*np.linalg.norm(direction[:2])+1e-5
        r=rim['radius']-radius/direction[2]-2e-5
        contact=np.r_[center+r*np.array([np.cos(x[2]),np.sin(x[2])]),height]
        a=contact-direction*(height-z)/direction[2];b=a+direction*length
        return rigid_segment_pose(bottom,top,a,b,twist=x[3]),a,b
    def margins(x):
        _,a,b=geometry(x);values=[support_radius-radius-np.linalg.norm(a[:2]-center)]
        values.extend(aperture_clearance(a,b,radius,center=r['center'],height=r['height'],opening_radius=r['radius']) for r in apertures)
        values.extend(segment_distance(a,b,c,d)-radius-r-1e-5 for c,d,r in other_segments)
        return np.asarray(values)
    def measurements(x,indices):
        pose,_,_=geometry(x);world=vertices@pose[:3,:3].T+pose[:3,3]
        c=[];e=[];line=[];shaft=np.array([bottom,top])@pose[:3,:3].T+pose[:3,3]
        for i in indices:
            ci,ei=project_material(world,faces,np.zeros(3),np.eye(3),cameras[i],opening=opening);c.append(ci);e.append(ei)
            line.extend(project(shaft,cameras[i])@source_lines[i,:2]+source_lines[i,2])
        return np.array(c)-centers[indices],np.array(e)-extents[indices],np.asarray(line)
    cache={};feasible=[]
    def objective(x):
        key=tuple(np.round(x,12))
        if key in cache:return cache[key]
        c,e,l=measurements(x,fit)
        value=float(np.mean((c/center_sigma_px)**2)+np.mean((e/extent_sigma_px)**2)+np.mean((l/line_sigma_px)**2))
        cache[key]=value
        if margins(x).min()>=-1e-6:feasible.append((value,np.array(x,copy=True)))
        return value
    bounds=list(zip(seed- [.10,.10,.5,.8],seed+[.10,.10,.5,.8]))
    constraints=[{'type':'ineq','fun':lambda x:margins(x)/.001}]
    start=minimize(lambda x:float(np.sum((x-seed)**2)),seed,method='SLSQP',bounds=bounds,
                   constraints=constraints,options={'maxiter':100,'ftol':1e-11}).x
    require(margins(start).min()>=-1e-6,'No finite-contact initial pose within the source pose prior')
    # Infinitesimal pixel-mask derivatives vanish. Use symmetric physical steps
    # large enough to resolve the image operator, then project onto the smooth
    # finite-contact constraints. Preserve the best feasible iterate.
    steps=np.array([.002,.002,.01,.01])
    def gradient(x):
        return np.array([(objective(x+np.eye(4)[j]*step)-objective(x-np.eye(4)[j]*step))/(2*step)
                         for j,step in enumerate(steps)])
    objective(start)
    solved=minimize(objective,start,jac=gradient,method='SLSQP',bounds=bounds,constraints=constraints,
                    options={'maxiter':min(35,max(1,max_evaluations//12)),'ftol':1e-6})
    require(feasible,'Actual-mask pose proposal violates finite contact geometry')
    _,chosen=min(feasible,key=lambda row:row[0])
    def metrics(x,indices):
        c,e,l=measurements(x,indices)
        return {'head_center_rmse_px':float(np.sqrt(np.mean(c*c))),
                'head_extent_rmse_px':float(np.sqrt(np.mean(e*e))),
                'support_line_rmse_px':float(np.sqrt(np.mean(l*l)))}
    return {'pose':geometry(chosen)[0],'parameters':chosen,'seed_parameters':seed,
            'before_fit':metrics(start,fit),'fit':metrics(chosen,fit),
            'validation':metrics(chosen,held),'minimum_geometric_margin_m':float(margins(chosen).min()),
            'operator':'actual_triangle_masks_with_declared_morphology','native_checked':False,
            'optimizer_success':bool(solved.success),'evaluations':len(cache)}
