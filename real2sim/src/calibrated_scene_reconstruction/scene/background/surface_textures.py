"""Bake observed surface pixels into PBR texture hypotheses; never invent writing."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from calibrated_scene_reconstruction.core.paths import resolve
from calibrated_scene_reconstruction.scene.background.blender_env import label
from calibrated_scene_reconstruction.scene.capture.valid_pixels import ValidPixels
from calibrated_scene_reconstruction.scene.review.environment import digest
from calibrated_scene_reconstruction.scene.session import Session


def linear(rgb):
    rgb=np.asarray(rgb,float)
    return np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)


def srgb(rgb):
    rgb=np.clip(rgb,0,1)
    return np.where(rgb<=.0031308,12.92*rgb,1.055*rgb**(1/2.4)-.055)


def enclosed_mask_holes(mask,maximum):
    """Optional photographic inclusion of small fully enclosed mounted details.

    This is an explicit appearance hypothesis, not surface-depth evidence.
    Foreground exclusions connected to the image/mask exterior remain excluded.
    """
    if type(maximum)is not int or not 0<=maximum<=20000:raise ValueError('Bounded enclosed appearance-hole area required')
    result=np.asarray(mask,bool).copy()
    if not maximum:return result
    count,labels,stats,_=cv2.connectedComponentsWithStats((~result).astype('uint8'),8)
    boundary=set(np.r_[labels[0],labels[-1],labels[:,0],labels[:,-1]])
    for k in range(1,count):
        if k not in boundary and stats[k,cv2.CC_STAT_AREA]<=maximum:result[labels==k]=True
    return result


def relief_depth(points,opacity,K,R,centre,shape,cell=4):
    """Conservative front-most captured context depth for photographic relief."""
    h,w=shape;pc=(points-centre)@R;q=pc@K.T;uv=q[:,:2]/np.maximum(pc[:,2,None],1e-9)
    valid=(opacity>.3)&(pc[:,2]>.05)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    xy=(uv[valid]/cell).astype(int);depth=np.full(((h+cell-1)//cell,(w+cell-1)//cell),np.inf,np.float32)
    np.minimum.at(depth,(xy[:,1],xy[:,0]),pc[valid,2]);return depth


def quad_points(quad, size):
    q=np.asarray(quad,float);w,h=map(int,size)
    if q.shape!=(4,3) or not np.isfinite(q).all() or min(w,h)<8 or max(w,h)>4096:
        raise ValueError('Finite ordered bottom-left/right, top-right/left surface quad required')
    u=(np.arange(w)+.5)/w;v=1-(np.arange(h)+.5)/h
    u,v=np.meshgrid(u,v)
    return ((1-u)*(1-v))[...,None]*q[0]+(u*(1-v))[...,None]*q[1]+(u*v)[...,None]*q[2]+((1-u)*v)[...,None]*q[3]


def tiled_fallback(path, tile_m, quad, shape):
    """Tile a declared material hypothesis; these pixels remain unobserved."""
    tile=np.asarray(tile_m,float);q=np.asarray(quad,float);h,w=shape
    if tile.shape!=(2,) or not np.isfinite(tile).all() or np.any(tile<.02) or np.any(tile>5):
        raise ValueError('Finite metric fallback tile size required')
    texture=np.asarray(Image.open(path).convert('RGB'))/255.
    span=np.array([np.linalg.norm(q[1]-q[0]),np.linalg.norm(q[3]-q[0])])
    u=(np.arange(w)+.5)/w*span[0]/tile[0];v=(1-(np.arange(h)+.5)/h)*span[1]/tile[1]
    x,y=np.meshgrid((u%1*texture.shape[1]).astype('float32'),((1-v%1)*texture.shape[0]).astype('float32'))
    return cv2.remap(linear(texture).astype('float32'),x,y,cv2.INTER_LINEAR,borderMode=cv2.BORDER_WRAP).astype(float)


def run(session, config, output):
    if (session/'qa/independent_freeze.json').exists():raise ValueError('Frozen delivery: use a derivative')
    out=Path(output).resolve()
    if not out.is_relative_to(session.dir.resolve()):raise ValueError('Surface evidence belongs in this session')
    out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(Path(config).read_text());camera=session/'parsing/posed_frames/cameras.npz'
    z=np.load(camera);names=list(map(str,z['names']));validity=ValidPixels(session)
    observation=Path(cfg['observations']);rows=json.loads(observation.read_text())['records']
    inputs={str(p.resolve()):digest(p) for p in [Path(config),camera,observation,Path(__file__)]};results=[]
    for request in cfg['surfaces']:
        eid=label(request['id']);kind=request['kind'];fallback=linear(np.asarray(request.get('fallback_srgb',[.87,.87,.87])))
        relief=request.get('captured_relief');cloud=None
        if relief:
            band=float(relief['band_m'])
            if not 0<band<=.3 or relief.get('camera_sha256')!=digest(camera):raise ValueError('Small photographic relief band in the exact capture frame required')
            path=Path(relief['cloud']);cloud=np.load(path);inputs[str(path.resolve())]=digest(path)
        if kind=='planar':
            size=request['size_px'];P=quad_points(request['quad_world'],size);h,w=P.shape[:2]
            colour=np.broadcast_to(fallback,(h,w,3)).copy();covered=np.zeros((h,w),bool);quality=np.zeros((h,w))
            if request.get('fallback_texture'):
                fallback_path=resolve(request['fallback_texture']).resolve();inputs[str(fallback_path)]=digest(fallback_path)
                colour=tiled_fallback(fallback_path,request['fallback_tile_m'],request['quad_world'],(h,w))
            for frame in request['frames']:
                row=next(r for r in rows if r['entity']==request.get('observation_entity',request['entity']) and r['frame']==frame and r.get('mask'))
                source=Path(row['source']);mask_path=Path(row['mask'])
                for p,expected in [(source,row['source_sha256']),(mask_path,row['mask_sha256'])]:
                    if digest(p)!=expected:raise ValueError('Observation evidence changed: '+str(p))
                    inputs[str(p.resolve())]=expected
                raw=np.asarray(Image.open(source).convert('RGB'))/255.;mask=np.asarray(Image.open(mask_path))>128
                i=names.index(frame)
                if list(map(int,z['size'][i]))!=[raw.shape[1],raw.shape[0]]:raise ValueError('Registered source size mismatch')
                valid_frame=validity.get(frame,mask.shape)
                mask &= valid_frame
                mask=enclosed_mask_holes(mask,request.get('enclosed_appearance_holes_max_px',0))
                pc=(P-z['centre'][i])@z['R_world_cam'][i];uv=pc@z['K'][i].T;uv=uv[...,:2]/np.maximum(pc[...,2,None],1e-9)
                maps=uv.astype('float32')
                # Every bilinear RGB contributor must belong to the observed
                # surface; nearest-mask lookup leaks an occluder's edge colour.
                visible=cv2.remap(mask.astype('float32'),maps[...,0],maps[...,1],cv2.INTER_LINEAR)>=1.-1e-6
                if cloud is not None:
                    depth=relief_depth(cloud['X'],cloud['opacity'],z['K'][i],z['R_world_cam'][i],z['centre'][i],mask.shape)
                    first=cv2.remap(depth,maps[...,0]/4,maps[...,1]/4,cv2.INTER_NEAREST,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
                    # Include only actual captured surfaces close to this plane.
                    # Closer foreground objects and unknown depth remain excluded.
                    visible |= np.isfinite(first)&(abs(first-pc[...,2])<=band)
                visible &= cv2.remap(valid_frame.astype('float32'),maps[...,0],maps[...,1],cv2.INTER_LINEAR)>=1.-1e-6
                visible &= pc[...,2]>.05
                sampled=cv2.remap(raw.astype('float32'),maps[...,0],maps[...,1],cv2.INTER_LINEAR)
                # High-resolution nearby views win; never paint an occluder over
                # the masked observed surface. Bright reference removes a global
                # illumination hypothesis, not calibrated inverse rendering.
                detail=z['K'][i,0,0]/np.maximum(pc[...,2],.05)
                good=visible&(detail>quality)
                source_linear=linear(sampled)
                if request.get('white_reference_albedo') is not None:
                    reference=np.percentile(source_linear[visible],90,axis=0) if visible.any() else np.ones(3)
                    source_linear=np.clip(source_linear/np.maximum(reference,.05)*float(request['white_reference_albedo']),0,1)
                colour[good]=source_linear[good];quality[good]=detail[good];covered|=visible
        elif kind=='patch':
            source=session/'splat/data/images'/request['frame'];raw=np.asarray(Image.open(source).convert('RGB'))/255.
            inputs[str(source.resolve())]=digest(source)
            x0,y0,x1,y1=map(int,request['crop']);crop=raw[y0:y1,x0:x1]
            if crop.size==0 or crop.shape[:2]!=(y1-y0,x1-x0):raise ValueError('Patch crop outside observed source')
            colour=linear(crop);covered=np.ones(colour.shape[:2],bool)
            if request.get('mean_albedo') is not None:
                colour=np.clip(colour/np.maximum(np.median(colour,axis=(0,1)),.02)*np.asarray(request['mean_albedo']),0,1)
            # Remove only broad lighting variation; retain captured fiber/grain.
            if request.get('flatten_illumination',False):
                smooth=cv2.GaussianBlur(colour,(0,0),max(2,min(colour.shape[:2])/6))
                colour=np.clip(colour/np.maximum(smooth,.02)*np.median(smooth,axis=(0,1)),0,1)
        else:raise ValueError('Use planar observed face or captured material patch')
        target=out/(eid+'.png');Image.fromarray(np.round(srgb(colour)*255).astype('uint8')).save(target)
        coverage=out/(eid+'_observed.png');Image.fromarray(covered.astype('uint8')*255).save(coverage)
        results.append({'id':eid,'texture':str(target),'sha256':digest(target),'observed_fraction':float(covered.mean()),
                        'coverage':str(coverage),'unseen_fill':'Declared plain fallback; no generated text or hidden detail',
                        'approved':False})
        if kind=='planar':
            results[-1].update(entity=request['entity'],quad_world=request['quad_world'],
                               observation_entity=request.get('observation_entity',request['entity']))
            if request.get('fallback_texture'):
                results[-1]['unseen_fill']='Declared tiled material hypothesis; excluded from observed coverage; no recovered unseen detail.'
        if relief:results[-1]['captured_relief']='Depth-supported captured pixels within declared band; mounted details are photographic, not reconstructed solid relief.'
        if 'emission_gain' in request:
            gain=float(request['emission_gain'])
            if not np.isfinite(gain) or not 0<gain<=10:raise ValueError('Finite bounded photographic display gain required')
            # Float RGBE avoids8-bit quantization at small radiance scales.
            # This preserves captured lighting, not recovered surface albedo.
            emission=out/(eid+'_emission.hdr')
            if not cv2.imwrite(str(emission),(colour[...,::-1]*gain).astype('float32')):raise ValueError('Cannot write captured radiance texture')
            results[-1].update(emission_texture=str(emission),emission_gain=gain,emission_sha256=digest(emission),
                               appearance_mode='source-lit; not calibrated relightable reflectance')
    inputs.update({str(resolve(p).resolve()):expected for p,expected in validity.inputs.items()})
    for path,expected in inputs.items():
        if digest(path)!=expected:raise ValueError('Surface input changed: '+path)
    from calibrated_scene_reconstruction.scene.review.environment import protected_state
    result={'surfaces':results,'input_sha256':inputs,'protected_task_state':protected_state(session),
            'scope':'Capture-pixel texture/albedo hypotheses. White-reference normalization and patch tiling are inferred; no inverse-rendering or photorealism certification.'}
    (out/'textures.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('session');parser.add_argument('config');parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(run(Session(args.session),args.config,args.output)['surfaces'],indent=2))


if __name__=='__main__':main()
