"""Observe every matching instance near an independently labelled container.

A source-only bounding rectangle associates candidates; it never fills pixels,
fixes a member count, merges physical bodies or approves identities. Preserve
all proposals so the agent can review instances and occlusion separately.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def select(candidates, context, padding, overlap):
    context=np.asarray(context)
    if (context.ndim!=2 or context.dtype.kind!='b' or not context.any()
            or not np.isfinite([padding,overlap]).all() or not 0<=padding<=1 or not 0<overlap<=1):
        raise ValueError('Observed boolean context and bounded association fractions required')
    yy,xx=np.nonzero(context);h,w=context.shape
    dx=(xx.max()-xx.min()+1)*padding;dy=(yy.max()-yy.min()+1)*padding
    box=[max(0,int(np.floor(xx.min()-dx))),max(0,int(np.floor(yy.min()-dy))),
         min(w,int(np.ceil(xx.max()+1+dx))),min(h,int(np.ceil(yy.max()+1+dy)))]
    region=np.zeros_like(context);x0,y0,x1,y1=box;region[y0:y1,x0:x1]=True
    union=np.zeros_like(context);rows=[]
    for j,(mask,confidence) in enumerate(candidates):
        mask=np.asarray(mask)
        if mask.shape!=context.shape or mask.dtype.kind!='b' or not np.isfinite(confidence):
            raise ValueError('Finite source confidence and matching boolean instance masks required')
        fraction=float((mask&region).sum()/max(1,mask.sum()));keep=bool(mask.any() and fraction>=overlap)
        if keep:union|=mask
        rows.append({'index':j,'confidence':float(confidence),'pixels':int(mask.sum()),
                     'context_overlap_fraction':fraction,'selected':keep})
    return union,{'instances':rows,'context_box':box,'padding_fraction':padding,
                  'minimum_overlap_fraction':overlap,'identities_approved':False,
                  'physical_attachment_inferred':False}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_context_cache(session, archive, request):
    if not request.get('collection_context'):return {}
    if 'context_inputs' not in archive:
        raise ValueError('Context collection requires immutable source provenance')
    inputs=json.loads(str(archive['context_inputs']))
    if not inputs:raise ValueError('Empty context evidence')
    expected=f"splat/object_fit/{request['collection_context']['object']}.npz"
    if expected not in inputs:
        raise ValueError('Collection provenance must name the declared independent context object')
    for relative,sha in inputs.items():
        path=(session.dir/relative).resolve()
        if not path.is_relative_to(session.dir.resolve()) or not path.is_file() or digest(path)!=sha:
            raise ValueError('Collection context source changed; preserve and reacquire observations')
    return inputs


def run(session,obj):
    from .blender.evidence import load
    from .observation_history import semantic_policy,color_inputs
    from .object_hull import CropSam3
    from .object_fit import placement
    request=session.scene_graph['dynamic_objects'][obj]['fit'];policy=semantic_policy(request)
    cfg=policy['collection_context'];context_obj=cfg['object']
    if context_obj==obj:raise ValueError('Collection context must be an independent observed object')
    z,frames,contexts,windows=load(session,context_obj)
    path=session/'splat/object_fit'/f'{obj}.npz'
    if path.exists():raise ValueError('Preserve an existing observation revision before collection acquisition')
    prompts=request['prompt'].split('|');sam=CropSam3(threshold=policy.get('instance_confidence_min',.2))
    context_file=session/'splat/object_fit'/f'{context_obj}.npz'
    inputs={str(p.relative_to(session.dir)):digest(p) for p in [context_file,session/'parsing/posed_frames/cameras.npz']}
    inputs.update(z.get('_object_domain_inputs',{}))
    used=[];masks=[];out_windows=[];raw_masks=[];raw_parts=[];all_proposals={};reviews=[]
    source=Path(str(z['source']))/'splat/data/images'
    for frame,context,window in zip(frames,contexts,windows):
        name=str(z['names'][frame]);image=cv2.imread(str(source/name));image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        x,y,_,_,scale=window;transform=np.array([[1/scale,0,x],[0,1/scale,y]])
        full=cv2.warpAffine(context.astype('uint8'),transform,image.shape[1::-1],flags=cv2.INTER_NEAREST)>0
        yy,xx=np.nonzero(full);cx,cy=float((xx.min()+xx.max())/2),float((yy.min()+yy.max())/2)
        crop=int(min(min(image.shape[:2]),max(640,3*max(np.ptp(xx),np.ptp(yy)))))
        parts=[];frame_review=[]
        all_proposals[f'view_{frame}_context']=full
        for j,prompt in enumerate(prompts):
            candidates=sam.instances(image,prompt,cx,cy,crop);mask,review=select(candidates,full,cfg['padding_fraction'],cfg['min_overlap_fraction'])
            parts.append(mask);prefix=f'view_{frame}_part_{j}'
            for k,(proposal,_) in enumerate(candidates):all_proposals[prefix+f'_proposal_{k}']=proposal
            frame_review.append({'frame':name,'prompt':prompt,'prefix':prefix,**review})
        reviews.extend(frame_review)
        merged=np.logical_or.reduce(parts)
        optional=set(policy.get('optional_part_queries',[]))
        if not all(m.any() for prompt,m in zip(prompts,parts) if prompt not in optional):continue
        yy,xx=np.nonzero(merged);side=int(np.clip(1.6*max(np.ptp(xx),np.ptp(yy)),160,min(image.shape[:2])))
        x0=int(np.clip((xx.min()+xx.max())/2-side/2,0,image.shape[1]-side));y0=int(np.clip((yy.min()+yy.max())/2-side/2,0,image.shape[0]-side));scale=min(1.,480/side)
        m=merged[y0:y0+side,x0:x0+side]
        if scale<1:m=cv2.resize(m.astype('uint8'),(round(side*scale),)*2,interpolation=cv2.INTER_NEAREST)>0
        used.append(int(frame));masks.append(m);out_windows.append([x0,y0,side,side,scale]);raw_masks.append(merged);raw_parts.append(parts)
        print('[collection]',name,[sum(r['selected']for r in row['instances'])for row in frame_review],flush=True)
    if len(used)<8:raise ValueError('Fewer than eight complete source collection observations')
    payload=dict(frames=np.array(used),masks=np.array(masks,dtype=object),windows=np.array(out_windows),
                 prompt=request['prompt'],semantic_query_policy=json.dumps(policy,sort_keys=True),
                 mask_component_policy=json.dumps(None),source_color_inputs=json.dumps(color_inputs(session,z,used),sort_keys=True),
                 context_inputs=json.dumps(inputs,sort_keys=True))
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**payload);np.savez_compressed(path.with_suffix('.acquired.npz'),**payload)
    np.savez_compressed(path.with_suffix('.components.acquired.npz'),names=z['names'][used],masks=np.array(raw_masks),part_masks=np.array(raw_parts),prompts=prompts)
    np.savez_compressed(path.with_suffix('.instances.acquired.npz'),**all_proposals)
    path.with_suffix('.instances.json').write_text(json.dumps({'reviews':reviews,'context_inputs':inputs,'accepted':False,'scope':__doc__},indent=2)+'\n')
    start=json.loads(context_file.with_suffix('.start.json').read_text());start['placement']=placement(session)
    start['basis']='Observed context location is a proposal seed, not the collection height or physical attachment.'
    path.with_suffix('.start.json').write_text(json.dumps(start,indent=2)+'\n')
    return {'frames':len(used),'accepted':False,'path':str(path)}


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session');p.add_argument('object');a=p.parse_args()
    print(run(Session(a.session),a.object))


if __name__=='__main__':main()
