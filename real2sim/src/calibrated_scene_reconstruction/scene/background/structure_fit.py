"""Observed structural-member proposals from calibrated RGB/GS, never size truth."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from calibrated_scene_reconstruction.scene.background.splat.config import sim3
from calibrated_scene_reconstruction.scene.review.environment import digest
from calibrated_scene_reconstruction.scene.session import Session


def vertical_planes(points, max_planes=8, thickness=.04, min_points=200):
    remaining=np.asarray(points,float).copy();result=[]
    for _ in range(max_planes):
        if len(remaining)<min_points:break
        best=None
        for angle in np.linspace(0,np.pi,180,endpoint=False):
            normal=np.array([np.cos(angle),np.sin(angle)])
            distance=remaining[:,:2]@normal
            bins=np.floor(distance/thickness).astype(int)
            values,counts=np.unique(bins,return_counts=True);j=counts.argmax()
            if best is None or counts[j]>best[0]:best=(int(counts[j]),normal,(values[j]+.5)*thickness)
        count,normal,offset=best
        selected=np.abs(remaining[:,:2]@normal-offset)<thickness
        subset=remaining[selected]
        if len(subset)<min_points:break
        tangent=np.array([-normal[1],normal[0]])
        along=subset[:,:2]@tangent;left,right=np.percentile(along,[2,98]);bottom,top=np.percentile(subset[:,2],[2,98])
        if right-left>.35 and top-bottom>.35:
            centre=normal*np.median(subset[:,:2]@normal)+tangent*(left+right)/2
            result.append({'centre_xy':centre.tolist(),'yaw_deg':float(np.degrees(np.arctan2(tangent[1],tangent[0]))),
                'width':float(right-left),'z_range':[float(bottom),float(top)],'normal_xy':normal.tolist(),'points':len(subset)})
        remaining=remaining[~selected]
    return result


def triangulate(cameras, frames, pixels, max_error_px=6., min_angle_deg=2.):
    """Triangulate agent-matched capture landmarks with explicit consistency gates."""
    pixels=np.asarray(pixels,float);names=list(map(str,cameras['names']))
    if len(frames)<2 or len(set(frames))!=len(frames) or pixels.ndim!=3 or pixels.shape[0]!=len(frames) or pixels.shape[1]<1 or pixels.shape[2]!=2 or not np.isfinite(pixels).all():
        raise ValueError('Distinct calibrated views with matched finite2D landmarks required')
    if not 0<max_error_px<=20 or not 0<min_angle_deg<=90:raise ValueError('Finite small pixel uncertainty and a useful triangulation baseline required')
    indices=[names.index(frame) for frame in frames];matrices=[]
    for i in indices:
        R=cameras['R_world_cam'][i].T;matrices.append(cameras['K'][i]@np.c_[R,-R@cameras['centre'][i]])
    result=[]
    for k in range(pixels.shape[1]):
        A=np.array([row for P,uv in zip(matrices,pixels[:,k]) for row in [uv[0]*P[2]-P[0],uv[1]*P[2]-P[1]]])
        _,_,V=np.linalg.svd(A);point=V[-1]
        if abs(point[3])<1e-10:raise ValueError('Landmark rays do not triangulate a finite point')
        result.append(point[:3]/point[3])
    result=np.asarray(result);errors=[];rays=[]
    for i,P,observed in zip(indices,matrices,pixels):
        projected=np.c_[result,np.ones(len(result))]@P.T
        if np.any(projected[:,2]<=0):raise ValueError('Triangulated landmark is behind a capture camera')
        errors.append(np.linalg.norm(projected[:,:2]/projected[:,2:]-observed,axis=1))
        direction=result-cameras['centre'][i];rays.append(direction/np.linalg.norm(direction,axis=1,keepdims=True))
    angles=np.max([np.degrees(np.arccos(np.clip(np.einsum('ij,ij->i',rays[i],rays[j]),-1,1))) for i in range(len(rays)) for j in range(i)],axis=0)
    if np.max(errors)>max_error_px:raise ValueError('Capture landmark reprojection error exceeds the allowed pixel uncertainty')
    if np.any(angles<min_angle_deg):raise ValueError('Landmark triangulation baseline is too narrow')
    return result,{'reprojection_error_px':np.asarray(errors).tolist(),'ray_angle_deg':angles.tolist()}


def visible_indices(points, K, R, centre, size, opacity, cell=4):
    """A mask may claim only front-most geometry, never a hidden room surface."""
    w,h=map(int,size);pc=(points-centre)@R;uv=pc@K.T
    uv=uv[:,:2]/np.maximum(pc[:,2,None],1e-9)
    good=(pc[:,2]>.05)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    ids=np.flatnonzero(good);xy=(uv[ids]//cell).astype(int)
    depth=np.full((h//cell+1,w//cell+1),np.inf)
    opaque=opacity[ids]>.2
    np.minimum.at(depth,(xy[opaque,1],xy[opaque,0]),pc[ids[opaque],2])
    front=pc[ids,2]<=depth[xy[:,1],xy[:,0]]*1.03+.03
    return ids[front],uv[ids[front]].astype(int)


def vertical_members(points, cell=.025, min_height=.35):
    points=np.asarray(points,float)
    if len(points)<20:return []
    xy=np.floor(points[:,:2]/cell).astype(int);lo=xy.min(0);xy-=lo
    shape=xy.max(0)+1
    lower=np.full(shape,np.inf);upper=np.full(shape,-np.inf);count=np.zeros(shape,int)
    np.minimum.at(lower,tuple(xy.T),points[:,2]);np.maximum.at(upper,tuple(xy.T),points[:,2]);np.add.at(count,tuple(xy.T),1)
    low=ndimage.minimum_filter(lower,3);high=ndimage.maximum_filter(upper,3)
    valid=(high-low>min_height)&(ndimage.uniform_filter(count.astype(float),3)*9>10)
    labels,n=ndimage.label(valid,np.ones((3,3)))
    result=[]
    for k in range(1,n+1):
        subset=points[labels[tuple(xy.T)]==k]
        if len(subset)<20:continue
        a,b=np.percentile(subset,[2,98],axis=0)
        if max(b[:2]-a[:2])>.2:continue
        result.append({'xy':np.median(subset[:,:2],axis=0).tolist(),'z_range':[float(a[2]),float(b[2])],
            'width_xy':(b[:2]-a[:2]).tolist(),'points':len(subset)})
    return result


def fit_top_profile(along, height, width):
    samples=[]
    edges=np.linspace(-width*.48,width*.48,16)
    for a,b in zip(edges[:-1],edges[1:]):
        values=height[(along>a)&(along<b)]
        if len(values)>20:samples.append([(a+b)/2,np.percentile(values,99)])
    if len(samples)<6:return None
    x,y=np.asarray(samples).T;slope,offset=np.polyfit(x,y,1)
    return {'left_z':float(offset-slope*width/2),'right_z':float(offset+slope*width/2),
            'residual_rms_m':float(np.std(y-(slope*x+offset))),'bins':len(samples),
            'method':'Upper source-mask contour rays intersected with observed plane; not a preset shape.'}


def mask_top_profile(plane, records, cameras, names):
    normal=np.r_[plane['normal_xy'],0.];centre=np.r_[plane['centre_xy'],0.]
    angle=np.radians(plane['yaw_deg']);tangent=np.array([np.cos(angle),np.sin(angle),0.]);all_points=[]
    for row in records:
        mask=cv2.imread(row['mask'],0);yy,xx=np.nonzero(mask);step=max(1,len(xx)//30000)
        i=names.index(row['frame']);rays=np.c_[xx[::step],yy[::step],np.ones(len(xx[::step]))]@np.linalg.inv(cameras['K'][i]).T@cameras['R_world_cam'][i].T
        denominator=rays@normal;distance=np.divide((centre-cameras['centre'][i])@normal,denominator,
            out=np.full(len(rays),np.nan),where=np.abs(denominator)>1e-6)
        valid=np.isfinite(distance)&(distance>0)&(distance<30)
        all_points.append(cameras['centre'][i]+rays[valid]*distance[valid,None])
    if not all_points or sum(map(len,all_points))<100:return None
    points=np.concatenate(all_points)
    return fit_top_profile((points-centre)@tangent,points[:,2],plane['width'])


def run(session, source_session, config, output):
    from plyfile import PlyData
    if (session/'qa/independent_freeze.json').exists():raise ValueError('Frozen delivery: use a derivative')
    out=Path(output).resolve()
    if not out.is_relative_to(session.dir.resolve()):raise ValueError('Structure proposals belong in this session')
    out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(Path(config).read_text());camera=session/'parsing/posed_frames/cameras.npz';z=np.load(camera);names=list(map(str,z['names']))
    if digest(camera)!=digest(source_session/'parsing/posed_frames/cameras.npz'):
        raise ValueError('Structure source must share the exact calibrated capture cameras')
    observations=json.loads(Path(cfg['observations']).read_text())['records']
    cloud=source_session/cfg.get('cloud','splat/export/full_colmap.ply')
    vertices=PlyData.read(str(cloud))['vertex'].data
    scale,R,t=sim3(source_session)
    points=scale*np.c_[vertices['x'],vertices['y'],vertices['z']].astype(float)@R.T+t
    colours=np.clip(.5+.2820947918*np.c_[vertices['f_dc_0'],vertices['f_dc_1'],vertices['f_dc_2']],0,1)
    opacity=1/(1+np.exp(-vertices['opacity'].astype(float)))
    inputs={str(p.resolve()):digest(p) for p in [Path(config),Path(cfg['observations']),cloud,camera,source_session/'splat/align.json',Path(__file__)]}
    results=[];visible_cache={}
    for request in cfg['requests']:
        lo,hi=np.asarray(request['roi_world'],float)
        selected=np.all((points>=lo)&(points<=hi),axis=1)&(opacity>.1)
        ids=np.flatnonzero(selected);P=points[ids];votes=np.zeros(len(P),int);used=[]
        records=list({r['frame']:r for r in observations if r['entity']==request['entity'] and r.get('mask')}.values())
        for row in records:
            i=names.index(row['frame']);mask=cv2.imread(row['mask'],0)
            if digest(row['mask'])!=row['mask_sha256']:raise ValueError('Observed mask changed')
            if digest(row['source'])!=row['source_sha256']:raise ValueError('Observed source image changed')
            inputs[str(Path(row['source']).resolve())]=row['source_sha256']
            inputs[str(Path(row['mask']).resolve())]=digest(row['mask'])
            mask=cv2.dilate(mask,np.ones((7,7),'uint8'))
            if i not in visible_cache:
                visible_cache[i]=visible_indices(points,z['K'][i],z['R_world_cam'][i],z['centre'][i],z['size'][i],opacity)
            front,pixels=visible_cache[i]
            claimed=front[mask[pixels[:,1],pixels[:,0]]>128]
            votes+=np.isin(ids,claimed).astype(int)
            used.append(row['frame'])
        valid=votes>=request.get('min_views',2)
        if request.get('colour')=='dark':valid&=colours[ids].max(1)<.3
        if request.get('colour')=='white':valid&=(colours[ids].max(1)-colours[ids].min(1)<.25)&(colours[ids].mean(1)>.5)
        if 'colour_axis' in request:
            valid&=colours[ids]@np.asarray(request['colour_axis'],float)>float(request.get('colour_min',0.))
        P=P[valid];colour=colours[ids][valid]
        np.savez_compressed(out/(request['entity']+'_points.npz'),points=P,colours=colour)
        result={'entity':request['entity'],'frames':used,'points':len(P),'approved':False}
        if request['kind']=='frame':
            band=P[(P[:,2]>request['base_z']+.2)&(P[:,2]<request.get('upper_z',3.))]
            result['vertical_members']=vertical_members(band,min_height=request.get('min_visible_height',.35))
        else:
            result['planes']=vertical_planes(P,request.get('max_planes',2),min_points=request.get('min_plane_points',200))
            if request.get('top_profile'):
                for plane in result['planes']:plane['top_profile']=mask_top_profile(plane,records,z,names)
        results.append(result);print('[structure.fit]',request['entity'],len(P),flush=True)
    for path,expected in inputs.items():
        if digest(path)!=expected:raise ValueError('Structure observation input changed: '+path)
    (out/'structures.json').write_text(json.dumps({'results':results,'input_sha256':inputs,
        'scope':'Unapproved source-constrained structure hypotheses; no true dimensions or task/robot modification.'},indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session');parser.add_argument('source_session');parser.add_argument('config');parser.add_argument('--output',required=True)
    a=parser.parse_args();run(Session(a.session),Session(a.source_session),a.config,a.output)


if __name__=='__main__':main()
