"""Bridge a camera gap using a reviewed static plane and two independent anchors.

This is a conditional observation model, not target tracking or a pose
interpolator. Both tracking directions must agree and close at the independent
endpoints. The caller must review the plane and exclude all moving regions.
"""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .contract import require
from .vision import plane_intersections,pose


def bridge(images,cameras,start,end,allowed,*,normal,offset,min_inliers=30,
           position_tolerance_m=.01,angle_tolerance_deg=2.):
    """Images are source grayscale frames; allowed(frame) masks a static plane."""
    require(0<=start<end<len(images),'Two source camera anchor indices required')
    require(np.isfinite(cameras['T_world_cam'][[start,end]]).all(),'Independent endpoint cameras required')
    K,D=cameras['K'],cameras['dist'];n=len(images)
    def agree(a,b):
        return (np.isfinite(a).all() and np.isfinite(b).all()
                and np.linalg.norm(a[:3,3]-b[:3,3])<=position_tolerance_m
                and np.degrees(Rotation.from_matrix(a[:3,:3].T@b[:3,:3]).magnitude())<=angle_tolerance_deg)
    def walk(a,b,step):
        poses=np.full((n,4,4),np.nan);quality=np.full((n,3),np.nan)
        T=cameras['T_world_cam'][a];poses[a]=T
        pixels=None;xyz=None;previous=None
        for i in range(a,b+step,step):
            gray=images[i];mask=allowed(i)
            if previous is not None:
                seed=pixels.astype(np.float32).reshape(-1,1,2)
                moved,found,_=cv2.calcOpticalFlowPyrLK(previous,gray,seed,None,winSize=(31,31),maxLevel=4)
                back,returned,_=cv2.calcOpticalFlowPyrLK(gray,previous,moved,None,winSize=(31,31),maxLevel=4)
                points=moved.reshape(-1,2)
                valid=(found.ravel()>0)&(returned.ravel()>0)&(np.linalg.norm(seed-back,axis=(1,2))<.7)
                valid&=(points[:,0]>=0)&(points[:,0]<gray.shape[1])&(points[:,1]>=0)&(points[:,1]<gray.shape[0])
                uv=np.clip(points.astype(int),[0,0],[gray.shape[1]-1,gray.shape[0]-1])
                valid&=mask[uv[:,1],uv[:,0]]>0
                try:T,inside,error=pose(xyz[valid],points[valid],K,D,min_inliers=min_inliers)
                except ValueError:break
                xyz,pixels=xyz[valid][inside],points[valid][inside]
                poses[i]=T;quality[i]=[len(error),np.median(error),np.quantile(error,.95)]
            # Add only static-plane image features, using the current camera.
            # All newly triangulated points carry this explicit planar prior.
            if pixels is None or len(pixels)<250:
                available=mask.copy()
                if pixels is not None:
                    for p in pixels:cv2.circle(available,tuple(np.rint(p).astype(int)),10,0,-1)
                new=cv2.goodFeaturesToTrack(gray,800,qualityLevel=.01,minDistance=10,mask=available)
                if new is not None:
                    new=new.reshape(-1,2).astype(float)
                    positions=plane_intersections(new,T,K,D,normal,offset)
                    pixels=new if pixels is None else np.vstack([pixels,new])
                    xyz=positions if xyz is None else np.vstack([xyz,positions])
            if pixels is None or len(pixels)<min_inliers:break
            previous=gray
        return poses,quality
    forward,fq=walk(start,end,1);backward,bq=walk(end,start,-1)
    closed=agree(forward[end],cameras['T_world_cam'][end]) and agree(backward[start],cameras['T_world_cam'][start])
    output=np.full((n,4,4),np.nan);quality=np.full((n,3),np.nan);disagreement=[]
    if closed:
        for i in range(start,end+1):
            if not agree(forward[i],backward[i]):disagreement.append(i);continue
            # Endpoints remain the independent anchors; internal frames choose
            # the lower observed static-landmark reprojection error.
            if i in {start,end}:output[i]=cameras['T_world_cam'][i];quality[i]=cameras['quality'][i]
            elif fq[i,1]<=bq[i,1]:output[i]=forward[i];quality[i]=fq[i]
            else:output[i]=backward[i];quality[i]=bq[i]
    return {'T_world_cam':output,'quality':quality,'t':cameras['t'],'K':K,'dist':D,
            'closed':bool(closed),'disagreement':disagreement,'forward':forward,'backward':backward,
            'scope':'Bidirectional static-plane tracking; only accepted when independently anchored endpoints close. Plane prior and source mask require review.'}
