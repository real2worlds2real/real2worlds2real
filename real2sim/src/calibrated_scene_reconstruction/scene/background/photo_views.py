"""Prioritise clear source photographs and fill only their unobserved texels.

These derived masks describe appearance, never a whole-object silhouette.
No text is generated and no colours are enhanced. Existing calibrated baking,
world/UV backing proof and actual source/free-view review remain mandatory.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from calibrated_scene_reconstruction.scene.capture.valid_pixels import ValidPixels
from calibrated_scene_reconstruction.scene.review.environment import digest,protected_state


def quality(image,mask):
    rgb=np.asarray(image);valid=np.asarray(mask,bool)
    if rgb.ndim!=3 or rgb.shape[2]!=3 or rgb.shape[:2]!=valid.shape:raise ValueError('Aligned RGB and source-owned mask required')
    inside=cv2.erode(valid.astype('uint8'),np.ones((5,5),'uint8'))>0
    if inside.sum()<64:return {'score':0.,'coverage':float(valid.mean()),'detail':0.,'unclipped_detail':0.}
    gray=cv2.cvtColor(rgb.astype('float32')/255,cv2.COLOR_RGB2GRAY)
    signal=cv2.GaussianBlur(abs(gray-cv2.GaussianBlur(gray,(0,0),1.1)),(0,0),1.)
    threshold=max(.003,float(np.quantile(signal[inside],.85)))
    details=inside&(signal>=threshold)
    detail=float(signal[details].mean()) if details.any() else 0.
    unclipped=(rgb.max(2)<250)&(rgb.min(2)>5)
    fraction=float(unclipped[details].mean()) if details.any() else 0.
    return {'score':float(np.sqrt(valid.mean())*detail*(.25+.75*fraction)),
            'coverage':float(valid.mean()),'detail':detail,'unclipped_detail':fraction}


def disjoint_coverage(masks):
    if not masks:raise ValueError('Source coverage masks required')
    covered=np.zeros_like(masks[0],bool);result=[]
    for mask in masks:
        mask=np.asarray(mask,bool)
        if mask.shape!=covered.shape:raise ValueError('Same photographic plane grid required')
        result.append(mask&~covered);covered|=mask
    return result


def prepare(session,config,output):
    config=Path(config).resolve();out=Path(output).resolve()
    if out.exists() or not out.is_relative_to(session.dir.resolve()) or (session/'qa/independent_freeze.json').exists():raise ValueError('New unfrozen appearance worklist required')
    cfg=json.loads(config.read_text());observation=Path(cfg['observations']).resolve();annotations=json.loads(observation.read_text())
    camera=session/'parsing/posed_frames/cameras.npz';C=np.load(camera);names=list(map(str,C['names']));validity=ValidPixels(session)
    inputs={str(p):digest(p) for p in [config,observation,camera.resolve(),Path(__file__).resolve()]}
    out.mkdir(parents=True);requests=copy.deepcopy(cfg);records=[];reports=[]
    for request in requests['surfaces']:
        if request.get('kind')!='planar':raise ValueError('Rank planar photographic surfaces separately from material patches')
        quad=np.asarray(request['quad_world'],float);size=np.asarray(request['size_px'],int)
        if quad.shape!=(4,3) or not np.isfinite(quad).all() or size.shape!=(2,) or min(size)<8:raise ValueError('Finite plane and texture dimensions required')
        scale=min(1.,512/max(size));w,h=np.maximum(16,np.rint(size*scale).astype(int));target=np.array([[0,h-1],[w-1,h-1],[w-1,0],[0,0]],np.float32)
        candidates=[];owner=request.get('observation_entity',request['entity'])
        for frame in request['frames']:
            rows=[r for r in annotations['records'] if r['entity']==owner and r['frame']==frame and r.get('mask')]
            if len(rows)!=1 or frame not in names:raise ValueError('One reviewed photographic owner mask per registered frame required')
            row=rows[0];i=names.index(frame);paths=[Path(row[k]).resolve() for k in ['source','mask']]
            for path,key in zip(paths,['source','mask']):
                if digest(path)!=row[key+'_sha256']:raise ValueError('Photographic annotation changed')
                inputs[str(path)]=digest(path)
            if paths[0]!=(session/'splat/data/images'/frame).resolve():raise ValueError('Use original registered source pixels')
            image=cv2.cvtColor(cv2.imread(str(paths[0])),cv2.COLOR_BGR2RGB);mask=cv2.imread(str(paths[1]),0)>128
            if image.shape[:2]!=mask.shape or list(C['size'][i])!=[image.shape[1],image.shape[0]]:raise ValueError('Photographic raster differs from calibrated source')
            mask &=validity.get(frame,mask.shape);pc=(quad-C['centre'][i])@C['R_world_cam'][i]
            if np.any(pc[:,2]<=.05):continue
            projected=pc@C['K'][i].T;projected=(projected[:,:2]/projected[:,2,None]).astype('float32')
            if not cv2.isContourConvex(projected) or cv2.contourArea(projected)<64:continue
            H=cv2.getPerspectiveTransform(projected,target)
            coverage=cv2.warpPerspective(mask.astype('float32'),H,(int(w),int(h)),flags=cv2.INTER_LINEAR)>=1.-1e-6
            warped=cv2.warpPerspective(image,H,(int(w),int(h)),flags=cv2.INTER_LINEAR)
            candidates.append({'frame':frame,'row':row,'image_mask':mask,'H':H,'coverage':coverage,'quality':quality(warped,coverage)})
        if not candidates:raise ValueError('No visible photographed plane: '+request['id'])
        candidates.sort(key=lambda r:r['quality']['score'],reverse=True)
        selected=disjoint_coverage([r['coverage'] for r in candidates]);alias='photo_scope_'+hashlib.sha256(request['id'].encode()).hexdigest()[:16]
        for candidate,allowed in zip(candidates,selected):
            mask=candidate['image_mask'];H=candidate['H']
            raster=cv2.warpPerspective(allowed.astype('uint8'),np.linalg.inv(H),(mask.shape[1],mask.shape[0]),flags=cv2.INTER_NEAREST)>0
            raster &=mask;path=out/(request['id']+'_'+Path(candidate['frame']).stem+'.png');cv2.imwrite(str(path),raster.astype('uint8')*255)
            row=copy.deepcopy(candidate['row']);row.update(entity=alias,source_owner=owner,mask=str(path),mask_sha256=digest(path),
                original_mask=candidate['row']['mask'],original_mask_sha256=candidate['row']['mask_sha256'],
                geometry_fit_scope='surface_part',geometry_fit_review='Photographic source priority subset only; never a geometric silhouette.',approved=False)
            records.append(row);inputs[str(path)]=digest(path)
        request['observation_entity']=alias;request['frames']=[r['frame'] for r in candidates]
        reports.append({'surface':request['id'],'source_owner':owner,'frames':[{'frame':r['frame'],**r['quality'],'selected_texels':int(m.sum())} for r,m in zip(candidates,selected)],'approved':False})
    from calibrated_scene_reconstruction.core.paths import resolve
    inputs.update({str(resolve(p).resolve()):sha for p,sha in validity.inputs.items()})
    if any(digest(p)!=sha for p,sha in inputs.items()):raise ValueError('Photographic priority inputs changed')
    derived=out/'observations.json';derived.write_text(json.dumps({'records':records,'input_sha256':inputs,'protected_task_state':protected_state(session),'scope':__doc__.strip()},indent=2)+'\n')
    requests['observations']=str(derived);(out/'surfaces.json').write_text(json.dumps(requests,indent=2)+'\n')
    result={'approved':False,'surfaces':reports,'input_sha256':{**inputs,str(derived):digest(derived)},'protected_task_state':protected_state(session),'scope':__doc__.strip()}
    (out/'ranking.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session');p.add_argument('config');p.add_argument('--output',required=True);a=p.parse_args()
    q=prepare(Session(a.session),a.config,a.output);print('Unapproved photographic source priorities',len(q['surfaces']))


if __name__=='__main__':main()
