"""Source-reviewed parallel vertical lines constrain gravity independently of targets."""
import cv2
import numpy as np
from .contract import require


def estimate(camera,segments,*,up_prior=(0,0,1),tolerance_deg=1.5):
    K=np.asarray(camera['K']);D=np.asarray(camera['dist']);T=np.asarray(camera['T_world_cam'])
    fit=[row for row in segments if row['role']=='fit'];held=[row for row in segments if row['role']=='validation']
    require(len(fit)>=2 and len({row['part'] for row in fit})>=2 and len(held)>=2
            and {row['part'] for row in held}-{row['part'] for row in fit},
            'Use two independent fitting parts and a separately held-out vertical part')
    require(tolerance_deg>0,'Declare source vertical-line uncertainty')
    normals=[];weights=[];prepared=[]
    for row in segments:
        index=row['frame'];pixels=np.asarray(row['endpoints'],float)
        require(pixels.shape==(2,2) and np.isfinite(pixels).all() and 0<=index<len(T) and np.isfinite(T[index]).all(),
                'Finite source segments and camera poses required')
        pixels=cv2.undistortPoints(pixels[:,None],K,D,P=K).reshape(2,2)
        rays=np.c_[pixels,np.ones(2)]@np.linalg.inv(K).T;n=np.cross(*rays);length=np.linalg.norm(pixels[1]-pixels[0])
        require(length>=10 and np.linalg.norm(n)>1e-10,'Vertical segments must span enough source pixels')
        n=T[index,:3,:3]@(n/np.linalg.norm(n));prepared.append((row,pixels,n))
        if row['role']=='fit':normals.append(n);weights.append(length)
        else:require(row['role']=='validation','Explicit fit/validation source roles required')
    N=np.asarray(normals)*np.sqrt(np.asarray(weights)/np.mean(weights))[:,None]
    _,s,V=np.linalg.svd(N,full_matrices=True)
    require(s[1]/s[0]>.02,'Source line planes do not determine a gravity direction')
    direction=V[-1];prior=np.asarray(up_prior,float)
    require(prior.shape==(3,) and np.isfinite(prior).all() and np.linalg.norm(prior)>0,'Explicit gravity sign prior required')
    if direction@prior<0:direction=-direction
    errors=[]
    for row,pixels,_ in prepared:
        vanishing=K@T[row['frame'],:3,:3].T@direction
        predicted=vanishing[:2]-pixels.mean(0)*vanishing[2];observed=pixels[1]-pixels[0]
        cosine=abs(predicted@observed)/(np.linalg.norm(predicted)*np.linalg.norm(observed))
        error=float(np.degrees(np.arccos(np.clip(cosine,0,1))))
        errors.append({'id':row['id'],'part':row['part'],'role':row['role'],'angular_error_deg':error})
    return {'direction_world':direction.tolist(),'pass':all(row['angular_error_deg']<=tolerance_deg for row in errors),
        'tolerance_deg':tolerance_deg,'line_errors':errors,'line_plane_singular_values':s.tolist(),
        'scope':'Gravity inferred from reviewed parallel/upright structural-part priors, not IMU truth. Held-out lines do not fit direction; no target motion or measured dimensions used.'}
