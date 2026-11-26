"""Image-backed proposals for sharp background GS lacking an instance label.

SAM's object vocabulary is not a test of whether captured appearance exists.
Keep only repeatedly visible, sharp background pixels outside known furniture
and the fixed task exclusion. Unknown/unseen pixels never gain support here.
All proposals still require actual composed source/free-view acceptance.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from calibrated_scene_reconstruction.scene.background.gs_regions import sharp_mask
from calibrated_scene_reconstruction.scene.background.observed_context import parameter_digest
from calibrated_scene_reconstruction.scene.review.environment import digest,protected_state


def prepare(session,review,geometry_job,entities,observations,output,claim_scope='unassigned'):
    out=Path(output).resolve()
    if claim_scope not in ('unassigned','outside_known_furniture'):raise ValueError('Explicit captured background scope required')
    if out.exists() or not out.is_relative_to(session.dir.resolve()) or (session/'qa/independent_freeze.json').exists():
        raise ValueError('Use a new unfrozen environment proposal')
    camera=session/'parsing/posed_frames/cameras.npz';z=np.load(camera);names=list(map(str,z['names']))
    render=json.loads(Path(review).read_text());cfg=session.config('environment.json')
    if render.get('diagnostic')!='context_source':raise ValueError('Use original captured-GS source/free-view diagnostics')
    source=render.get('source_context_override');context=render.get('environment_context') or {}
    if source!=context.get('source_asset') or digest(source)!=context.get('source_asset_sha256'):
        raise ValueError('Bind the original same-capture field')
    scope=json.loads(context['source_background_scope_json']);job=json.loads(Path(geometry_job).read_text());mesh=np.load(job['geometry'])
    available={r['entity'] for r in job['parts'] if not r['external']}
    if not set(entities)<=available:raise ValueError('Explicit known background exclusion entities required')
    vertices={e:np.concatenate([mesh[f'v{i}'] for i,r in enumerate(job['parts']) if r['entity']==e]) for e in entities}
    paths=[camera,Path(review),Path(geometry_job),Path(job['geometry']),Path(source),Path(__file__)]
    from calibrated_scene_reconstruction.scene.background import gs_regions
    paths.append(Path(gs_regions.__file__));inputs={str(p.resolve()):digest(p) for p in paths};masks={}
    for p in observations:
        p=Path(p);inputs[str(p.resolve())]=digest(p)
        for row in json.loads(p.read_text())['records']:
            if row.get('mask'):masks.setdefault(row['frame'],[]).append(row)
    out.mkdir(parents=True);rows=[]
    for view in render['views']:
        frame=Path(view['source']).name;i=names.index(frame);raw=cv2.imread(view['source']);image=cv2.imread(view['renders'][0]);h,w=image.shape[:2]
        background=np.ones(raw.shape[:2],bool)
        for p,sha in scope[frame].items():
            if digest(p)!=sha:raise ValueError('Fixed background/task exclusion changed')
            background&=cv2.imread(p,0)>128;inputs[p]=sha
        background=cv2.resize(background.astype('uint8'),(w,h),interpolation=cv2.INTER_NEAREST)>0
        # Exclude known furniture conservatively. These are appearance-routing
        # masks, not a claim that a coarse proxy has the true shape.
        K=z['K'][i].copy();K[0]*=w/raw.shape[1];K[1]*=h/raw.shape[0]
        for P in vertices.values():
            pc=(P-z['centre'][i])@z['R_world_cam'][i];pc=pc[pc[:,2]>.15]
            if len(pc)<3:continue
            uv=pc@K.T;uv=uv[:,:2]/uv[:,2,None]
            uv=np.clip(uv,[-w*2,-h*2],[w*3,h*3]);hull=cv2.convexHull(uv.astype('float32')).round().astype('int32')
            mask=np.zeros((h,w),'uint8');cv2.fillConvexPoly(mask,hull,1);mask=cv2.dilate(mask,np.ones((13,13),'uint8'))
            background&=mask==0
        for row in masks.get(frame,[]):
            if digest(row['mask'])!=row['mask_sha256']:raise ValueError('Source object mask changed')
            inputs[row['mask']]=row['mask_sha256'];mask=cv2.resize(cv2.imread(row['mask'],0),(w,h),interpolation=cv2.INTER_NEAREST)>128
            # Floor and wall masks may be included as positive observations;
            # only known furniture masks are used as exclusions.
            if row['entity'] in entities:background&=~cv2.dilate(mask.astype('uint8'),np.ones((9,9),'uint8')).astype(bool)
        source_image=cv2.resize(raw,(w,h),interpolation=cv2.INTER_AREA);good=sharp_mask(source_image,image,background)
        path=out/(Path(frame).stem+'.png');cv2.imwrite(str(path),cv2.resize(good.astype('uint8')*255,(raw.shape[1],raw.shape[0]),interpolation=cv2.INTER_NEAREST))
        for p in [view['source'],*view['renders'],str(path)]:inputs[str(Path(p).resolve())]=digest(p)
        rows.append({'entity':'unassigned_capture' if claim_scope=='unassigned' else 'background_capture','claim':-1 if claim_scope=='unassigned' else 0,'frame':frame,'mask':str(path),'accepted_pixels':int(good.sum()),
                     'source':view['source'],'captured':view['renders'][0],'novel':view['renders'][1]})
    result={'version':1,'regions':rows,'input_sha256':inputs,'protected_task_state':protected_state(session),
            'parameters_sha256':parameter_digest(cfg['parameters']),'source_asset':source,'source_asset_sha256':digest(source),
            'excluded_known_entities':entities,'status':'unapproved_candidate',
            'claim_scope':claim_scope,
            'scope':'Sharp observed background appearance without an instance name; two separated front-visible views and ordinary GS/mesh/source/physics gates remain required.'}
    if any(digest(p)!=sha for p,sha in inputs.items()):raise ValueError('Capture support inputs changed')
    (out/'regions.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session');p.add_argument('review');p.add_argument('geometry_job')
    p.add_argument('--exclude-entities',nargs='+',required=True);p.add_argument('--observations',nargs='*',default=[]);p.add_argument('--output',required=True)
    p.add_argument('--claim-scope',choices=['unassigned','outside_known_furniture'],default='unassigned')
    a=p.parse_args();q=prepare(Session(a.session),a.review,a.geometry_job,a.exclude_entities,a.observations,a.output,a.claim_scope)
    print('Unapproved background support',len(q['regions']),'views')


if __name__=='__main__':main()
