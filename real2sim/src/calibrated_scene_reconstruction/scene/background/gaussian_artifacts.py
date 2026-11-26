"""Propose broad GS kernels that degrade completed native appearance.

Rows need bad-pixel masks from actual hybrid/native-only/source comparisons
and reviewed centre masks for components the native candidate can replace.
Two independent rays must agree. Other centres and compact detail kernels are
preserved; this cannot approve a scene or infer a missing native component.
"""
import cv2
import numpy as np


def select(points,world_axes,cameras,rows,max_sigma_px=12.):
    P=np.asarray(points,float);axes=np.asarray(world_axes,float)
    if (P.shape!=(len(P),3) or axes.shape!=(len(P),3,3) or not np.isfinite(P).all()
            or not np.isfinite(axes).all() or not 2<=max_sigma_px<=32):
        raise ValueError('Finite captured Gaussian geometry and bounded projected size required')
    names=list(map(str,cameras['names']));votes=np.zeros(len(P),np.uint16);last=np.full_like(P,np.nan);seen=set()
    for row in rows:
        frame=row['frame']
        if frame in seen or frame not in names:raise ValueError('Distinct registered artifact views required')
        seen.add(frame);i=names.index(frame);bad=np.asarray(row['bad'],bool);replaceable=np.asarray(row['replaceable'],bool)
        if bad.shape!=replaceable.shape or bad.ndim!=2:raise ValueError('Aligned artifact and completed-component masks required')
        h,w=bad.shape;K=np.array(cameras['K'][i],float);K[0]*=w/cameras['size'][i][0];K[1]*=h/cameras['size'][i][1]
        centre=cameras['centre'][i];R=cameras['R_world_cam'][i];pc=(P-centre)@R;uv=pc@K.T;uv=uv[:,:2]/np.maximum(pc[:,2,None],1e-9)
        visible=(pc[:,2]>.05)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h);ids=np.flatnonzero(visible);xy=uv[ids].astype(int)
        ids=ids[replaceable[xy[:,1],xy[:,0]]]
        if not len(ids):continue
        z=pc[ids,2];J=np.zeros((len(ids),2,3));J[:,0,0]=K[0,0]/z;J[:,0,2]=-K[0,0]*pc[ids,0]/z**2
        J[:,1,1]=K[1,1]/z;J[:,1,2]=-K[1,1]*pc[ids,1]/z**2
        projected=J@(R.T@axes[ids]);xx=np.square(projected[:,0]).sum(1);yy=np.square(projected[:,1]).sum(1);cross=(projected[:,0]*projected[:,1]).sum(1)
        largest=np.sqrt(np.maximum(0,.5*(xx+yy+np.sqrt((xx-yy)**2+4*cross**2))))
        radius=3*np.sqrt(np.c_[xx,yy]);lo=np.maximum(np.floor(uv[ids]-radius),[0,0]).astype(int)
        hi=np.minimum(np.ceil(uv[ids]+radius)+1,[w,h]).astype(int);area=np.prod(np.maximum(0,hi-lo),axis=1)
        integral=cv2.integral(bad.astype('uint8'))
        count=integral[hi[:,1],hi[:,0]]-integral[lo[:,1],hi[:,0]]-integral[hi[:,1],lo[:,0]]+integral[lo[:,1],lo[:,0]]
        candidates=(largest>=max_sigma_px)&(area>0)&(count>=np.maximum(8,np.minimum(64,area*.05)))
        ids=ids[candidates]
        before=P[ids]-last[ids];after=P[ids]-centre
        cosine=np.einsum('ij,ij->i',before,after)/np.maximum(1e-12,np.linalg.norm(before,axis=1)*np.linalg.norm(after,axis=1))
        independent=(votes[ids]==0)|((np.linalg.norm(last[ids]-centre,axis=1)>.05)&(np.degrees(np.arccos(np.clip(cosine,-1,1)))>=2.))
        chosen=ids[independent];votes[chosen]+=1;last[chosen]=centre
    if len(seen)<2:raise ValueError('Two counterfactual artifact views required')
    return votes>=2
