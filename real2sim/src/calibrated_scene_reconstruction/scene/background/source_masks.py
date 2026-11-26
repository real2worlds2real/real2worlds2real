"""Record agent-drawn background polygons when a model misses a visible part.

These are explicit source annotations, not automatic semantic/geometry approval.
The original registered images and fixed task exclusion remain unchanged.
"""
import argparse
import json
import re
from pathlib import Path
import cv2
import numpy as np
from calibrated_scene_reconstruction.scene.review.environment import digest,protected_state


def prepare(session,config,output):
    out=Path(output).resolve();config=Path(config).resolve()
    if out.exists() or not out.is_relative_to(session.dir.resolve()) or (session/'qa/independent_freeze.json').exists():raise ValueError('Use an unfrozen environment derivative')
    camera=session/'parsing/posed_frames/cameras.npz';C=np.load(camera);names=list(map(str,C['names']));cfg=json.loads(config.read_text());rows=[]
    inputs={str(p.resolve()):digest(p) for p in [camera,config,Path(__file__)]};out.mkdir(parents=True);seen=set()
    for request in cfg['requests']:
        frame=request['frame'];entity=request['entity']
        if frame not in names or Path(frame).name!=frame or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,127}',entity):raise ValueError('Registered frame and simple background entity required')
        if (entity,frame)in seen:raise ValueError('Combine polygons into one source observation per entity/frame')
        seen.add((entity,frame))
        source=session/'splat/data/images'/frame
        if digest(source)!=cfg['source_sha256'][frame]:raise ValueError('Annotated source pixels changed')
        w,h=C['size'][names.index(frame)];mask=np.zeros((h,w),'uint8')
        for polygon in request['polygons']:
            p=np.asarray(polygon,float)
            if p.ndim!=2 or p.shape[1]!=2 or len(p)<3 or not np.isfinite(p).all() or np.any(p<0) or np.any(p[:,0]>=w) or np.any(p[:,1]>=h):raise ValueError('Bounded source polygon required')
            cv2.fillPoly(mask,[np.round(p).astype('int32')],255)
        inputs[str(source.resolve())]=digest(source)
        for suffix in ['_zone.png','_sam.png','_mask.png']:
            p=source.with_name(source.stem+suffix)
            if p.exists():
                valid=cv2.imread(str(p),0)
                if valid.shape!=mask.shape:raise ValueError('Fixed background exclusion raster mismatch')
                mask[valid<=128]=0;inputs[str(p.resolve())]=digest(p)
        folder=out/entity;folder.mkdir(exist_ok=True);path=folder/(Path(frame).stem+'_mask.png');cv2.imwrite(str(path),mask)
        rows.append({'entity':entity,'frame':frame,'source':str(source.resolve()),'source_sha256':digest(source),
                     'geometry_fit_scope':request.get('geometry_fit_scope','unknown'),'geometry_fit_review':request.get('geometry_fit_review',''),
                     'mask':str(path),'mask_sha256':digest(path),'approved':False,'source_method':'agent_polygon','basis':request['basis']})
    if any(digest(p)!=sha for p,sha in inputs.items()):raise ValueError('Source annotation input changed')
    record={'version':1,'records':rows,'input_sha256':inputs,'protected_task_state':protected_state(session),'scope':'Agent source-pixel annotations only; no true dimensions or automatic semantic/shape acceptance.'}
    (out/'observations.json').write_text(json.dumps(record,indent=2)+'\n');return record


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session');p.add_argument('config');p.add_argument('--output',required=True);a=p.parse_args()
    print('Source annotations',len(prepare(Session(a.session),a.config,a.output)['records']))


if __name__=='__main__':main()
