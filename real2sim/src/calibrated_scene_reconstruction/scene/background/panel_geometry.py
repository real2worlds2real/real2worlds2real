"""Unapproved upright-panel geometry from source edges and capture-GS depth.

Screen/partition edges constrain orientation better than a PCA of streaked GS.
Depth is still uncertain; a source-view fit never approves a novel view.
"""
import numpy as np


def warp_capture(image,quad,width,height,valid=None):
    """Rectify photographed appearance, without inventing missing pixels."""
    import cv2
    q=np.asarray(quad,np.float32);h,w=image.shape[:2]
    if (q.shape!=(4,2) or not np.isfinite(q).all() or np.any(q<0) or np.any(q[:,0]>=w) or np.any(q[:,1]>=h)
            or not cv2.isContourConvex(q) or cv2.contourArea(q)<64 or not 32<=width<=2048 or not 32<=height<=2048):
        raise ValueError('Bounded convex source quad and texture resolution required')
    target=np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],np.float32)
    H=cv2.getPerspectiveTransform(q,target)
    result=cv2.warpPerspective(image,H,(width,height),flags=cv2.INTER_CUBIC)
    if valid is not None:
        if valid.shape!=(h,w):raise ValueError('Source visibility mask must align with capture')
        observed=cv2.warpPerspective(valid.astype('uint8'),H,(width,height),flags=cv2.INTER_NEAREST)>0
        # Unknown colours remain inferred neutral material. Never bake another
        # foreground object into the hidden part of this panel.
        if not observed.any():raise ValueError('No visible panel appearance')
        fallback=np.median(result[observed],axis=0)
        for y in range(height):
            row=result[y,observed[y]];result[y,~observed[y]]=np.median(row,axis=0) if len(row) else fallback
    return result


def solve(K,R,C,quad,anchor):
    q=np.asarray(quad,float);K=np.asarray(K,float);R=np.asarray(R,float);C=np.asarray(C,float);anchor=np.asarray(anchor,float)
    if q.shape!=(4,2) or not np.isfinite(q).all() or anchor.shape!=(3,) or not np.isfinite(anchor).all():
        raise ValueError('Ordered TL/TR/BR/BL source corners and capture depth anchor required')
    h=np.c_[q,np.ones(4)];vanish=np.cross(np.cross(h[0],h[1]),np.cross(h[3],h[2]))
    direction=R@np.linalg.inv(K)@vanish
    if np.linalg.norm(direction[:2])<1e-8:raise ValueError('Panel horizontal direction is degenerate')
    axis=direction.copy();axis[2]=0;axis/=np.linalg.norm(axis)
    normal=np.cross(axis,[0,0,1]);rays=h@np.linalg.inv(K).T@R.T;denominator=rays@normal
    if np.any(abs(denominator)/np.linalg.norm(rays,axis=1)<np.sin(np.radians(5))):raise ValueError('Panel rays are too grazing')
    distance=((anchor-C)@normal)/denominator
    if np.any(distance<=0):raise ValueError('Panel lies behind the source camera')
    points=C+rays*distance[:,None];horizontal=points@axis
    if horizontal[1]<horizontal[0]:axis=-axis;horizontal=-horizontal
    left,right=float(horizontal[[0,3]].mean()),float(horizontal[[1,2]].mean())
    bottom,top=float(points[[2,3],2].mean()),float(points[[0,1],2].mean())
    if right<=left or top<=bottom:raise ValueError('Positive panel width/height required')
    centre=points.mean(0);centre+=axis*((left+right)/2-centre@axis);centre[2]=(top+bottom)/2
    # Upright PCA is only an initializer. A real display may be tilted, and a
    # depth streak must not determine its normal. Refine a rigid rectangle from
    # its source edges, retaining the uncertain capture-depth anchor.
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    rotation=np.stack([axis,np.cross([0,0,1],axis),[0,0,1]],axis=1)
    initial=np.r_[centre,Rotation.from_matrix(rotation).as_rotvec(),np.log([right-left,top-bottom])]
    depth=float((anchor-C)@R[:,2]);corners=np.array([[-.5,.5],[.5,.5],[.5,-.5],[-.5,-.5]])
    def rectangle_at(x):
        axes=Rotation.from_rotvec(x[3:6]).as_matrix();return x[:3]+corners[:,0,None]*np.exp(x[6])*axes[:,0]+corners[:,1,None]*np.exp(x[7])*axes[:,2]
    def residual(x):
        pc=(rectangle_at(x)-C)@R
        if np.any(pc[:,2]<=.05):return np.full(9,1e6)
        uv=pc@K.T
        return np.r_[(uv[:,:2]/uv[:,2,None]-q).ravel(),((x[:3]-C)@R[:,2]-depth)*1000]
    fitted=least_squares(residual,initial,max_nfev=300,loss='linear')
    centre=fitted.x[:3];rotation=Rotation.from_rotvec(fitted.x[3:6]).as_matrix()
    width,height=np.exp(fitted.x[6:8]);rectangle=rectangle_at(fitted.x)
    pc=(rectangle-C)@R;projected=pc@K.T;error=np.linalg.norm(projected[:,:2]/projected[:,2,None]-q,axis=1)
    return {'centre':centre.tolist(),'width':float(width),'height':float(height),'rotation':rotation.tolist(),
            'tilt_deg':float(np.degrees(np.arccos(np.clip(rotation[2,2],-1,1)))),'yaw_deg':float(np.degrees(np.arctan2(rotation[1,0],rotation[0,0]))),
            'source_quad':q.tolist(),'rectangle_world':rectangle.tolist(),'reprojection_error_px':error.tolist(),'approved':False,
            'scope':'Source-edge orientation and GS-depth hypothesis; require other views, support/contact and full-scene acceptance.'}
