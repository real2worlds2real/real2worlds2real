"""Source-only calibrated RGB stereo proposals for object reconstruction.

This produces observed surface points, not an accepted mesh, topology, material
inventory or physics model. Callers retain each point's originating image and
camera provenance; projecting fused rear surfaces into a foreground mask does
not establish visibility. Final source/geometry/semantic gates remain separate.
"""
import cv2
import numpy as np


def reconstruct_pair(im1,im2,K1,R1,c1,K2,R2,c2,mask1,mask2,disparities=512):
    """Return world points, source RGB, diagnostics and rectified BGR views.

    Camera R matrices map camera coordinates to world coordinates. Input images
    are OpenCV BGR and must share shape with their masks. Distortion must already be represented in their
    calibration/preprocessing. ``disparities`` is a numerical search budget,
    never a metric size prior. Empty/ambiguous correspondence is not approval.
    """
    im1, im2 = np.asarray(im1), np.asarray(im2)
    mask1, mask2 = np.asarray(mask1, bool), np.asarray(mask2, bool)
    if (im1.shape != im2.shape or im1.ndim != 3 or im1.shape[2] != 3
            or im1.dtype != np.uint8 or im2.dtype != np.uint8
            or mask1.shape != im1.shape[:2] or mask2.shape != im1.shape[:2]):
        raise ValueError('Matched uint8 BGR image shapes and masks required')
    if not mask1.any() or not mask2.any():
        raise ValueError('Source stereo requires nonempty independently declared masks')
    if type(disparities) is not int or disparities <= 0 or disparities % 16:
        raise ValueError('Disparity search budget must be a positive multiple of 16')
    K1, K2, R1, R2, c1, c2 = map(np.asarray, (K1, K2, R1, R2, c1, c2))
    for K, R, c in [(K1, R1, c1), (K2, R2, c2)]:
        if (K.shape != (3, 3) or R.shape != (3, 3) or c.shape != (3,)
                or not all(np.isfinite(x).all() for x in [K, R, c])
                or K[0, 0] <= 0 or K[1, 1] <= 0
                or not np.allclose(K[2], [0, 0, 1])
                or not np.allclose(R.T @ R, np.eye(3), atol=1e-6)
                or np.linalg.det(R) <= 0):
            raise ValueError('Finite pinhole cameras with proper world rotations required')
    if np.linalg.norm(c1-c2) < 1e-8:
        raise ValueError('Two distinct camera centres are required for stereo geometry')
    rel=R2.T@R1;trans=R2.T@(c1-c2);size=(im1.shape[1],im1.shape[0])
    rr1,rr2,p1,p2,Q,_,_=cv2.stereoRectify(K1,None,K2,None,size,rel,trans.reshape(3,1),
            flags=cv2.CALIB_ZERO_DISPARITY,alpha=-1)
    axis=int(abs(p2[1,3])>abs(p2[0,3]));sign=-np.sign(p2[axis,3])
    # Full-image alpha=0 can zoom into unrelated background when wide camera
    # frusta intersect obliquely. Keep the complete declared source masks on a
    # shared rectified canvas instead of cropping away the observed object.
    focal=float((K1[0,0]+K2[0,0])/2);bounds=[]
    for K,rr,m in [(K1,rr1,mask1),(K2,rr2,mask2)]:
        y,x=np.nonzero(m);rays=np.c_[x,y,np.ones(len(x))]@np.linalg.inv(K).T@rr.T
        if np.any(rays[:,2]<=0):raise ValueError('Source mask crosses the rectified camera horizon')
        q=rays[:,:2]/rays[:,2,None];bounds.append(np.array([q.min(0),q.max(0)]))
    low=np.minimum(bounds[0][0],bounds[1][0]);high=np.maximum(bounds[0][1],bounds[1][1])
    focal=min(focal,3000/max(high-low));corner=np.floor(low*focal)-16
    size=tuple(np.ceil(high*focal-corner+17).astype(int))
    Krect=np.array([[focal,0,-corner[0]],[0,focal,-corner[1]],[0,0,1.]])
    p1=np.c_[Krect,np.zeros(3)];p2=np.c_[Krect,Krect@(rr2@trans)]
    maps=[cv2.initUndistortRectifyMap(K,None,rr,P,size,cv2.CV_32FC1)
          for K,rr,P in [(K1,rr1,p1),(K2,rr2,p2)]]
    rect=[cv2.remap(im,*mp,cv2.INTER_LINEAR)for im,mp in zip([im1,im2],maps)]
    masks=[cv2.remap(m.astype('uint8'),*mp,cv2.INTER_NEAREST).astype(bool)for m,mp in zip([mask1,mask2],maps)]
    if axis:rect=[x.swapaxes(0,1).copy()for x in rect];masks=[x.T.copy()for x in masks]
    # SGBM discards a search-width band even where the true correspondence is
    # valid. Empty padding avoids censoring the source ROI; masks still reject
    # any invented padded correspondence and original pixel coordinates return
    # unchanged after the bidirectional check.
    rect=[np.pad(x,((0,0),(disparities,disparities),(0,0)))for x in rect]
    masks=[np.pad(x,((0,0),(disparities,disparities)))for x in masks]
    mini=0 if sign>0 else -disparities
    def match(a,b,lo):
        matcher=cv2.StereoSGBM_create(minDisparity=lo,numDisparities=disparities,
            blockSize=5,P1=8*25,P2=32*25,disp12MaxDiff=1,preFilterCap=31,
            uniquenessRatio=10,speckleWindowSize=50,speckleRange=1,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        return matcher.compute(cv2.cvtColor(a,cv2.COLOR_BGR2GRAY),cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)).astype('float32')/16
    d1=match(*rect,mini);d2=match(*rect[::-1],-mini-disparities)
    y,x=np.indices(d1.shape);rx=(x-d1).astype('float32');fy=y.astype('float32')
    back=cv2.remap(d2,rx,fy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=-1e5)
    targetmask=cv2.remap(masks[1].astype('uint8'),rx,fy,cv2.INTER_NEAREST).astype(bool)
    # Bidirectional agreement can still retain accidental matches in unrelated
    # texture. Independently test the observed patches at each proposed shift.
    left=cv2.cvtColor(rect[0],cv2.COLOR_BGR2GRAY).astype('float32')
    right=cv2.cvtColor(rect[1],cv2.COLOR_BGR2GRAY).astype('float32')
    sx=np.zeros(d1.shape,np.float32);sy=sx.copy();sxx=sx.copy();syy=sx.copy();sxy=sx.copy()
    for dy in range(-2,3):
        for dx in range(-2,3):
            aa=np.roll(left,(-dy,-dx),(0,1));bb=cv2.remap(right,rx+dx,fy+dy,cv2.INTER_LINEAR)
            sx+=aa;sy+=bb;sxx+=aa*aa;syy+=bb*bb;sxy+=aa*bb
    vx=np.maximum(sxx-sx*sx/25,0);vy=np.maximum(syy-sy*sy/25,0)
    ncc=(sxy-sx*sy/25)/np.sqrt(np.maximum(vx*vy,1e-9))
    photo=(ncc>=.7)&(vx>=25*3**2)&(vy>=25*3**2)
    raw=masks[0]&targetmask&(sign*d1>1)&(abs(d1+back)<=.75)&(x%2==0)&(y%2==0)
    raw_count=int(raw.sum());keep=raw&photo
    d1=d1[:,disparities:-disparities];keep=keep[:,disparities:-disparities]
    rect=[v[:,disparities:-disparities]for v in rect]
    disparity=d1.T if axis else d1;valid=keep.T if axis else keep
    yy,xx=np.indices(disparity.shape);depth=np.divide(-focal*(rr2@trans)[axis],disparity,
        out=np.full(disparity.shape,np.nan),where=disparity!=0)
    camera_rect=np.stack([(xx-Krect[0,2])/focal*depth,(yy-Krect[1,2])/focal*depth,depth],-1)
    camera=camera_rect@rr1
    world=camera@R1.T+c1
    valid&=np.isfinite(world).all(2)&(camera[:,:,2]>.05)
    output=world[valid];col= (rect[0].swapaxes(0,1)if axis else rect[0])[valid,::-1]
    return output,col,dict(source_mask_samples=int(masks[0].sum()/4),bidirectional_points=raw_count,matched_points=len(output),independent_ncc_min=.7,patch_std_min=3.,rectification_axis=axis,rectified_size=list(map(int,size)),rectified_focal_px=focal,disparity_direction=float(sign),median_disparity=float(np.median(disparity[valid]))if valid.any()else None),rect
