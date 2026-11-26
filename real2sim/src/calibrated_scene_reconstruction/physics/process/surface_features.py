"""Source texture landmarks on an explicitly reviewed opaque rigid surface.

The mesh supplies initial depth, not target motion. Raw source correspondences
and missing frames remain visible; neither native trajectories nor synthesized
images are used for feature matching.
"""
import cv2
import numpy as np

from .contract import require


def surface_anchors(pixels,vertices,faces,world_pose,camera):
    import trimesh
    K,D,T=camera;B=np.asarray(world_pose,float);T=np.asarray(T,float)
    normalized=cv2.undistortPoints(np.asarray(pixels,float).reshape(-1,1,2),K,D).reshape(-1,2)
    directions=np.c_[normalized,np.ones(len(normalized))]@T[:3,:3].T@B[:3,:3]
    origin=(T[:3,3]-B[:3,3])@B[:3,:3]
    mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
    locations,rays,_=mesh.ray.intersects_location(np.tile(origin,(len(pixels),1)),directions,multiple_hits=False)
    result=np.full((len(pixels),3),np.nan)
    if len(rays):result[rays]=locations
    return result


def rendered_anchors(pixels,vertices,faces,world_pose,camera,validation_views,*,limit_px):
    """Bind an asset's visible material identities, independently of real pose.

    Real-image rays alone provide depth hypotheses, not semantic asset
    correspondences. Here both annotations come from static native renders;
    held-out views validate the asset-local points before any motion fit.
    Transparent or refracted features must be excluded by the caller's review.
    """
    require(limit_px>0 and validation_views,'Independent rendered feature views required')
    points=surface_anchors(pixels,vertices,faces,world_pose,camera)
    require(np.isfinite(points).all(),'Rendered features miss the reviewed opaque surface')
    B=np.asarray(world_pose,float);world=points@B[:3,:3].T+B[:3,3];errors=[]
    for held_camera,observed in validation_views:
        K,D,T=held_camera;T=np.asarray(T,float);R=T[:3,:3].T
        predicted=cv2.projectPoints(world,cv2.Rodrigues(R)[0],-R@T[:3,3],K,D)[0].reshape(-1,2)
        observed=np.asarray(observed,float)
        require(observed.shape==predicted.shape and np.isfinite(observed).all(),'Reviewed native identities must match across views')
        errors.append(np.linalg.norm(predicted-observed,axis=1))
    errors=np.asarray(errors)
    require(float(errors.max())<=limit_px,'Native semantic feature binding fails independent rendered views')
    return {'points':points,'validation_errors_px':errors,'maximum_error_px':float(errors.max())}


def validate_rendered_binding(receipt,landmarks):
    """Reject stale evidence or a model using different material anchor points."""
    from .contract import digest
    require(receipt.get('pass') is True and receipt.get('reason') and receipt.get('source_sha256'),
            'Reviewed native material binding evidence required')
    require(all(digest(path)==value for path,value in receipt['source_sha256'].items()),
            'Native material correspondence evidence is stale')
    require(len(receipt['ids'])==len(set(receipt['ids']))==len(receipt['local_points']),
            'Native material binding identities must be unique')
    for name,point in zip(receipt['ids'],receipt['local_points']):
        require(name in landmarks and landmarks[name]['feature']=='point'
                and np.asarray(landmarks[name]['points']).shape==(1,3)
                and np.allclose(landmarks[name]['points'][0],point,rtol=0,atol=1e-9),
                'Model material landmark differs from native semantic correspondence: '+name)
    return True


def mutual_surface_matches(reference_descriptors,keypoints,descriptors,reference_pixels,*,ratio=.75,threshold=2.5):
    require(0<ratio<1 and threshold>0,'Declare source matching uncertainty')
    if descriptors is None or len(descriptors)<4 or len(reference_descriptors)<4:return {}
    matcher=cv2.BFMatcher();forward=matcher.knnMatch(reference_descriptors,descriptors,k=2)
    backward=matcher.knnMatch(descriptors,reference_descriptors,k=2)
    reverse={r[0].queryIdx:r[0].trainIdx for r in backward if len(r)==2 and r[0].distance<ratio*r[1].distance}
    matches=[r[0] for r in forward if len(r)==2 and r[0].distance<ratio*r[1].distance
             and reverse.get(r[0].trainIdx)==r[0].queryIdx]
    if len(matches)<4:return {}
    source=np.array([reference_pixels[m.queryIdx] for m in matches],float)
    target=np.array([keypoints[m.trainIdx].pt for m in matches],float)
    _,inliers=cv2.findHomography(source,target,cv2.RANSAC,threshold,maxIters=3000,confidence=.999)
    if inliers is None:return {}
    return {m.queryIdx:target[i] for i,m in enumerate(matches) if inliers[i,0]}


def annotated_tracks(images,pixels,*,end_pixels=None,max_fb_px=1.,max_consistency_px=2.,max_lk_error=40.):
    """Track source-labelled material points; no temporal gap filling.

    An optional independently annotated endpoint supplies a second tracking
    direction. Disagreement becomes missing, rather than blending trajectories.
    """
    require(2<=len(images)<=10000 and 1<=len(pixels)<=256,'Bounded source interval/points required')
    def direction(frames,seed):
        points=np.asarray(seed,np.float32).reshape(-1,1,2);result=[points[:,0].copy()]
        alive=np.ones(len(points),bool)
        for previous,current in zip(frames,frames[1:]):
            end,found,error=cv2.calcOpticalFlowPyrLK(previous,current,points,None,winSize=(21,21),maxLevel=4)
            back,returned,_=cv2.calcOpticalFlowPyrLK(current,previous,end,None,winSize=(21,21),maxLevel=4)
            alive &= (found[:,0]>0)&(returned[:,0]>0)&(np.linalg.norm(back-points,axis=(1,2))<=max_fb_px)&(error[:,0]<=max_lk_error)
            good=end[:,0].copy();good[~alive]=np.nan;result.append(good);points=end
        return np.array(result)
    forward=direction(images,pixels)
    if end_pixels is None:return forward
    require(np.asarray(end_pixels).shape==np.asarray(pixels).shape,'Independent endpoint point IDs must match')
    backward=direction(images[::-1],end_pixels)[::-1]
    agree=np.isfinite(forward).all(2)&np.isfinite(backward).all(2)&(np.linalg.norm(forward-backward,axis=2)<=max_consistency_px)
    forward[~agree]=np.nan
    return forward
