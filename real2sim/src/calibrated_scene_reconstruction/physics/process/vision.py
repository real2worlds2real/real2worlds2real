"""CPU observations for process videos localized against a reusable sparse scene.

The scene supplies metric scale; moving targets never become camera constraints.
This module estimates observations, not physical parameters or acceptance.
"""
import json
import sqlite3
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .contract import digest, require


def model_reference(database_path,model_path,image_names,*,max_per_identity=3):
    """Static reconstruction descriptors in their original reconstruction frame."""
    from calibrated_scene_reconstruction.core.adapters.colmap import image_tracks,point_records
    root=Path(model_path)
    observations=image_tracks(root/'images.bin',image_names)
    records=point_records(root/'points3D.bin',np.unique(np.concatenate([v[1] for v in observations.values()])))
    gathered={}
    with sqlite3.connect(f'file:{Path(database_path).resolve()}?mode=ro',uri=True) as database:
        ids={name:identity for identity,name in database.execute('select image_id,name from images')}
        for name in image_names:
            uv,point_ids=observations[name]
            row=database.execute('select rows,cols,data from keypoints where image_id=?',(ids[str(name)],)).fetchone()
            kp=np.frombuffer(row[2],np.float32).reshape(row[:2])[:,:2]
            row=database.execute('select rows,cols,data from descriptors where image_id=?',(ids[str(name)],)).fetchone()
            desc=np.frombuffer(row[2],np.uint8).reshape(row[:2])
            distance,index=cKDTree(kp).query(uv)
            for i,p in enumerate(point_ids):
                if distance[i]<.25 and records[int(p)][1]<2. and records[int(p)][2]>=3:
                    gathered.setdefault(int(p),[]).append(desc[index[i]])
    values=[];xyz=[];identities=[]
    for identity,descriptors in gathered.items():
        for i in np.unique(np.linspace(0,len(descriptors)-1,min(max_per_identity,len(descriptors))).astype(int)):
            values.append(descriptors[i]);xyz.append(records[identity][0]);identities.append(identity)
    require(len(values)>=100,'Insufficient static reconstruction feature identities')
    return {'descriptors':np.asarray(values,np.float32),'xyz':np.asarray(xyz),'point_ids':np.asarray(identities)}


def sparse_reference(session, *, views=40, per_view=2000, min_track=4, max_error=1., excluded_spheres=()):
    """Join original COLMAP descriptors to measured, metric static scene points."""
    from calibrated_scene_reconstruction.core.adapters.colmap import image_tracks, point_records
    session=Path(session)
    root=session/'splat/data'
    cameras=np.load(session/'parsing/posed_frames/cameras.npz')
    names=cameras['names'][np.linspace(0,len(cameras['names'])-1,views).astype(int)]
    observations=image_tracks(root/'sparse/full/images.bin',names)
    records=point_records(root/'sparse/full/points3D.bin',np.unique(np.concatenate([v[1] for v in observations.values()])))
    align=json.loads((session/'splat/align.json').read_text())['sim3']
    rotation=np.asarray(align['R']);translation=np.asarray(align['t']);scale=align['scale']
    descriptors,points,identities=[],[],[]
    with sqlite3.connect(f'file:{(root/"database.db").resolve()}?mode=ro',uri=True) as database:
        ids=dict((name,identity) for identity,name in database.execute('select image_id,name from images'))
        for name in names:
            uv,point_ids=observations[name]
            row=database.execute('select rows,cols,data from keypoints where image_id=?',(ids[str(name)],)).fetchone()
            kp=np.frombuffer(row[2],np.float32).reshape(row[:2])[:,:2]
            row=database.execute('select rows,cols,data from descriptors where image_id=?',(ids[str(name)],)).fetchone()
            desc=np.frombuffer(row[2],np.uint8).reshape(row[:2])
            distance,index=cKDTree(kp).query(uv)
            keep=[i for i,p in enumerate(point_ids) if distance[i]<.25 and records[int(p)][1]<max_error
                  and records[int(p)][2]>=min_track]
            keep=sorted(keep,key=lambda i:records[int(point_ids[i])][2],reverse=True)[:per_view]
            for i in keep:
                xyz=scale*(rotation@records[int(point_ids[i])][0])+translation
                if any(np.linalg.norm(xyz-np.asarray(center))<radius for center,radius in excluded_spheres):continue
                points.append(xyz);descriptors.append(desc[index[i]]);identities.append(int(point_ids[i]))
    require(len(points)>=100,'Not enough source-bound static scene descriptors')
    return {'xyz':np.asarray(points),'descriptors':np.asarray(descriptors,np.float32),'point_ids':np.asarray(identities),
            'sources':{str(p.resolve()):digest(p) for p in [root/'database.db',root/'sparse/full/images.bin',
                root/'sparse/full/points3D.bin',session/'splat/align.json',session/'parsing/posed_frames/cameras.npz']}}


def pose(xyz, pixels, K, dist, *, min_inliers=20, threshold=3.):
    """RANSAC camera pose from static landmarks, returning only verified inliers."""
    xyz,pixels=np.asarray(xyz,float),np.asarray(pixels,float)
    require(len(xyz)>=min_inliers,'Not enough static camera correspondences')
    centered=xyz-xyz.mean(0)
    _,singular,basis=np.linalg.svd(centered,full_matrices=False)
    if singular[-1]<max(1e-8,singular[0]*1e-6):
        flat=centered@basis[:2].T
        undistorted=cv2.undistortPoints(pixels.reshape(-1,1,2),K,dist,P=K).reshape(-1,2)
        _,inliers=cv2.findHomography(flat,undistorted,cv2.RANSAC,threshold,maxIters=3000,confidence=.999)
        require(inliers is not None,'Planar static camera localization failed')
        inliers=np.flatnonzero(inliers).reshape(-1,1)
        require(len(inliers)>=min_inliers,'Not enough planar camera inliers')
        ok,rotations,translations,_=cv2.solvePnPGeneric(xyz[inliers.ravel()],pixels[inliers.ravel()],K,dist,flags=cv2.SOLVEPNP_IPPE)
        require(bool(ok),'Planar pose ambiguity could not be resolved')
        rvec,tvec=min(zip(rotations,translations),key=lambda rt:np.mean((
            cv2.projectPoints(xyz[inliers.ravel()],*rt,K,dist)[0].reshape(-1,2)-pixels[inliers.ravel()])**2))
    else:
        ok,rvec,tvec,inliers=cv2.solvePnPRansac(xyz,pixels,K,dist,iterationsCount=2000,
            reprojectionError=threshold,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
    require(ok and inliers is not None and len(inliers)>=min_inliers,'Static-map camera localization failed')
    indices=inliers.ravel()
    rvec,tvec=cv2.solvePnPRefineLM(xyz[indices],pixels[indices],K,dist,rvec,tvec)
    predicted=cv2.projectPoints(xyz[indices],rvec,tvec,K,dist)[0].reshape(-1,2)
    error=np.linalg.norm(predicted-pixels[indices],axis=1)
    require(np.median(error)<=threshold,'Camera reprojection exceeds declared observation limit')
    R=cv2.Rodrigues(rvec)[0]
    T=np.eye(4);T[:3,:3]=R.T;T[:3,3]=-R.T@tvec.ravel()
    require(np.mean((xyz[indices]@R.T+tvec.ravel())[:,2]>0)>.99,'Camera landmarks behind the lens')
    return T,indices,error


class Localizer:
    def __init__(self, reference, *, min_inliers=20):
        self.reference=reference
        self.min_inliers=min_inliers
        self.neighbors=min(len(reference['point_ids']),max(8,1+int(np.unique(reference['point_ids'],return_counts=True)[1].max())))
        self.detector=cv2.SIFT_create(nfeatures=7000,contrastThreshold=.01)
        self.matcher=cv2.FlannBasedMatcher(dict(algorithm=1,trees=6),dict(checks=100))
        self.matcher.add([reference['descriptors']]);self.matcher.train()

    def __call__(self,gray,K,dist,mask=None):
        kp,desc=self.detector.detectAndCompute(gray,mask)
        require(desc is not None,'No static image features')
        # COLMAP's default L1_ROOT descriptors; OpenCV emits L2 SIFT instead.
        desc=np.sqrt(desc/np.maximum(desc.sum(axis=1,keepdims=True),1.))*512.
        matches=self.matcher.knnMatch(desc,k=self.neighbors)
        good=[]
        for row in matches:
            a=row[0]
            others=[b for b in row[1:] if self.reference['point_ids'][b.trainIdx]!=self.reference['point_ids'][a.trainIdx]]
            if others and a.distance<.8*others[0].distance:good.append(a)
        # One scene landmark may be visible in several reference views.
        unique={}
        for m in sorted(good,key=lambda m:m.distance):
            unique.setdefault(int(self.reference['point_ids'][m.trainIdx]),m)
        good=list(unique.values())
        xyz=self.reference['xyz'][[m.trainIdx for m in good]]
        pixels=np.asarray([kp[m.queryIdx].pt for m in good])
        T,indices,error=pose(xyz,pixels,K,dist,min_inliers=self.min_inliers)
        return T,xyz[indices],pixels[indices],error


def camera_track(video, reference, K, dist, *, exclusions=None, reanchor_every=30, min_inliers=20, allow_missing=False,direction='forward'):
    """Use static 3D landmarks, not a scene-wide homography, for translating cameras.

    Exclusions(frame, width, height) returns a uint8 *allowed* feature mask.
    Moving/occluded targets must be excluded from the static-map camera solve.
    """
    cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS)
    require(fps>0,'Cannot read video timebase')
    require(direction in {'forward','backward'},'Camera tracking direction must be forward or backward')
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    order=range(count) if direction=='forward' else range(count-1,-1,-1)
    localizer=Localizer(reference,min_inliers=min_inliers)
    poses,quality=[],[];previous=None;xyz=pixels=None
    for frame in order:
        if direction=='backward':cap.set(cv2.CAP_PROP_POS_FRAMES,frame)
        ok,image=cap.read();require(ok,'Incomplete video decode')
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        mask=exclusions(frame,image.shape[1],image.shape[0]) if exclusions else None
        anchored=False
        if previous is not None and len(pixels)>=min_inliers+5 and frame%reanchor_every:
            start=pixels.astype(np.float32).reshape(-1,1,2)
            end,found,_=cv2.calcOpticalFlowPyrLK(previous,gray,start,None,winSize=(25,25),maxLevel=4)
            back,returned,_=cv2.calcOpticalFlowPyrLK(gray,previous,end,None,winSize=(25,25),maxLevel=4)
            valid=(found.ravel()>0)&(returned.ravel()>0)&(np.linalg.norm(back-start,axis=(1,2))<1.)
            points=end.reshape(-1,2)
            if mask is not None:
                u=np.clip(points.astype(int),[0,0],[image.shape[1]-1,image.shape[0]-1])
                valid &= mask[u[:,1],u[:,0]]>0
            try:
                T,indices,error=pose(xyz[valid],points[valid],K,dist,min_inliers=min_inliers)
                xyz,pixels=xyz[valid][indices],points[valid][indices]
                anchored=True
            except ValueError:pass
        if not anchored:
            try:T,xyz,pixels,error=localizer(gray,K,dist,mask)
            except ValueError:
                if not allow_missing:raise
                poses.append(np.full((4,4),np.nan));quality.append([0,np.nan,np.nan])
                previous=None;xyz=pixels=None
                continue
        poses.append(T);quality.append([len(error),float(np.median(error)),float(np.quantile(error,.95))])
        previous=gray
        if frame%30==0:print(f'[camera] frame={frame} inliers={len(error)} median_px={np.median(error):.3f}',flush=True)
    cap.release()
    if direction=='backward':poses.reverse();quality.reverse()
    return {'t':np.arange(len(poses))/fps,'T_world_cam':np.asarray(poses),'quality':np.asarray(quality),
            'K':np.asarray(K),'dist':np.asarray(dist)}


def reconcile_cameras(forward,backward,*,position_tolerance_m,angle_tolerance_deg):
    """Fill independently localized camera gaps; reject conflicting valid poses.

    Both directions use static map landmarks. No response motion or temporal
    interpolation supplies missing camera states. Tolerances belong to the
    source camera observation plan, never the fitted dynamics.
    """
    from scipy.spatial.transform import Rotation
    require(position_tolerance_m>0 and angle_tolerance_deg>0,'Positive camera agreement tolerances required')
    require(np.array_equal(forward['t'],backward['t']) and np.array_equal(forward['K'],backward['K'])
            and np.array_equal(forward['dist'],backward['dist']),'Camera tracks must share time, intrinsics and coordinates')
    result={k:np.array(v,copy=True) for k,v in forward.items()}
    count=len(result['t']);source=np.zeros(count,int);conflicts=np.zeros(count,bool)
    for i,(a,b) in enumerate(zip(forward['T_world_cam'],backward['T_world_cam'])):
        fa,fb=np.isfinite(a).all(),np.isfinite(b).all()
        if fa and fb:
            distance=np.linalg.norm(a[:3,3]-b[:3,3])
            angle=np.degrees(Rotation.from_matrix(a[:3,:3].T@b[:3,:3]).magnitude())
            if distance>position_tolerance_m or angle>angle_tolerance_deg:
                result['T_world_cam'][i]=np.nan;result['quality'][i]=[0,np.nan,np.nan];conflicts[i]=True
                continue
        if fb and (not fa or backward['quality'][i,1]<forward['quality'][i,1]):
            result['T_world_cam'][i]=b;result['quality'][i]=backward['quality'][i];source[i]=2
        elif fa:source[i]=1
    result.update(direction_source=source,direction_conflict=conflicts)
    return result


def plane_intersections(pixels,T_world_cam,K,dist,normal,offset):
    """Intersect calibrated camera rays with n.X=offset; reject backward/horizon rays."""
    rays=cv2.undistortPoints(np.asarray(pixels,float).reshape(-1,1,2),K,dist).reshape(-1,2)
    rays=np.c_[rays,np.ones(len(rays))]@np.asarray(T_world_cam[:3,:3]).T
    center=T_world_cam[:3,3];normal=np.asarray(normal,float)
    denominator=rays@normal
    require(np.all(abs(denominator)>1e-8),'Observation rays too close to the plane horizon')
    depth=(offset-center@normal)/denominator
    require(np.all(depth>0),'Observation plane lies behind the camera')
    return center+depth[:,None]*rays


def plane_reference(image,T_world_cam,K,dist,normal,offset,mask):
    """Create a local static reference only from an explicitly reviewed planar region.

    Useful when a reused room has changed. A known initial object can establish
    the metric camera/plane, but target features must then be excluded from this
    camera reference, so subsequent target motion cannot move the camera estimate.
    """
    kp,desc=cv2.SIFT_create(nfeatures=8000,contrastThreshold=.015).detectAndCompute(image,mask)
    require(desc is not None and len(desc)>=30,'Insufficient features on the reviewed static plane')
    pixels=np.asarray([k.pt for k in kp])
    xyz=plane_intersections(pixels,T_world_cam,K,dist,normal,offset)
    desc=np.sqrt(desc/np.maximum(desc.sum(1,keepdims=True),1.))*512.
    return {'xyz':xyz,'descriptors':desc,'point_ids':np.arange(len(xyz))}


def track_roi(video,initial_box,*,search_margin=180):
    """Source-image template proposals; scores are observations, never physics proof."""
    cap=cv2.VideoCapture(str(video));ok,image=cap.read();require(ok,'Missing source video')
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    x1,y1,x2,y2=initial_box;template=gray[y1:y2,x1:x2]
    require(template.size>100,'A textured initial ROI is required')
    variants=[]
    for scale in [.75,.9,1.,1.1,1.25,1.4]:
        patch=cv2.resize(template,None,fx=scale,fy=scale)
        h,w=patch.shape
        for angle in [-25,-12,0,12,25]:
            rotation=cv2.getRotationMatrix2D((w/2,h/2),angle,1.)
            variants.append(cv2.warpAffine(patch,rotation,(w,h),borderMode=cv2.BORDER_REPLICATE))
    results=[[*initial_box,1.]];box=np.array(initial_box)
    for frame in range(1,int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ok,image=cap.read();require(ok,'Incomplete source video')
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        lo=np.maximum(box[:2]-search_margin,[0,0]);hi=np.minimum(box[2:]+search_margin,[gray.shape[1],gray.shape[0]])
        search=gray[lo[1]:hi[1],lo[0]:hi[0]];best=None
        for patch in variants:
            if min(search.shape[0]-patch.shape[0],search.shape[1]-patch.shape[1])<0:continue
            response=cv2.matchTemplate(search,patch,cv2.TM_CCOEFF_NORMED)
            _,score,_,location=cv2.minMaxLoc(response)
            if best is None or score>best[0]:best=(score,np.array(location)+lo,patch.shape)
        require(best is not None,'No feasible ROI proposal')
        score,origin,shape=best;box=np.r_[origin,origin+[shape[1],shape[0]]]
        results.append([*box,float(score)])
    cap.release()
    return np.asarray(results)


def material_regions(video,regions,*,hsv_lower,hsv_upper,min_area=30,min_aspect=1.1,max_aspect=4.,opening=3):
    """Observe a source-reviewed material region when texture tracking is ambiguous.

    All color/shape settings belong to the observation plan. No object category,
    dimensions, expected physical trajectory or fitted parameter is consulted.
    Missing/ambiguous regions are observations to review, never synthesized poses.
    """
    cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS)
    kernel_shape=(opening,opening) if isinstance(opening,int) else tuple(opening)
    require(len(kernel_shape)==2 and min(kernel_shape)>0,'Positive observation morphology dimensions required')
    result=[];masks=[]
    for frame in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ok,image=cap.read();require(ok,'Incomplete video')
        mask=regions(frame,image.shape[1],image.shape[0])
        threshold=cv2.inRange(cv2.cvtColor(image,cv2.COLOR_BGR2HSV),np.asarray(hsv_lower),np.asarray(hsv_upper))
        threshold=cv2.bitwise_and(threshold,mask)
        threshold=cv2.morphologyEx(threshold,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        threshold=cv2.morphologyEx(threshold,cv2.MORPH_OPEN,np.ones(kernel_shape,np.uint8))
        interior=cv2.erode(mask,np.ones(tuple(k+4 for k in kernel_shape),np.uint8))>0
        count,labels,stats,centroids=cv2.connectedComponentsWithStats(threshold)
        candidates=[]
        for i in range(1,count):
            x,y,w,h,area=stats[i]
            if area<min_area or not min_aspect<=w/max(h,1)<=max_aspect:continue
            component=(labels[max(0,y-1):y+h+1,max(0,x-1):x+w+1]==i).astype(np.uint8)
            boundary=cv2.dilate(component,np.ones((3,3),np.uint8))>0
            allowed=interior[max(0,y-1):y+h+1,max(0,x-1):x+w+1]
            if np.any(boundary&~allowed):continue  # A clipped background strip is not a complete target region.
            candidates.append((area,i))
        if not candidates:
            result.append([np.nan]*5);masks.append(np.zeros((32,32),np.uint8));continue
        _,i=max(candidates)
        x,y,w,h,area=stats[i];center=centroids[i]
        result.append([*center,float(area),float(w),float(h)])
        # Full masks are not retained by default; keep a normalized crop for review.
        masks.append(cv2.resize((labels[y:y+h,x:x+w]==i).astype(np.uint8)*255,(32,32),interpolation=cv2.INTER_NEAREST))
    cap.release()
    return {'t':np.arange(len(result))/fps,'features':np.asarray(result),'mask_crops':np.asarray(masks)}
