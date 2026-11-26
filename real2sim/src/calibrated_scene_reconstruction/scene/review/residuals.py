"""Source/render crops for the next agent environment iteration.

This triages missing instances, parts and appearance. Thresholds are not an
acceptance score; the independent full-scene quality gates remain mandatory.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
from calibrated_scene_reconstruction.scene.review.environment import digest


def regions(raw,captured,candidate,background,maximum=8):
    if raw.shape!=captured.shape or raw.shape!=candidate.shape or background.shape!=raw.shape[:2]:raise ValueError('Aligned source/reference/candidate/background required')
    gray=cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY).astype(float)/255
    detail=cv2.GaussianBlur(abs(gray-cv2.GaussianBlur(gray,(0,0),1.2)),(0,0),2.)
    source_error=abs(raw.astype(float)-captured).mean(2)/255;candidate_error=abs(raw.astype(float)-candidate).mean(2)/255
    masks={'restore_or_refine':background&(source_error<.10)&(candidate_error>source_error+.06)&(detail>.008),
           'mesh_repair':background&(source_error>=.10)&(candidate_error>.12)&(detail>.008)}
    rows=[]
    for action,mask in masks.items():
        mask=cv2.morphologyEx(mask.astype('uint8'),cv2.MORPH_CLOSE,np.ones((9,9),'uint8'))
        mask&=background.astype('uint8')
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
        for i in range(1,count):
            x,y,w,h,area=map(int,stats[i])
            if area<max(96,raw.shape[0]*raw.shape[1]*.0002):continue
            selected=labels==i;priority=float(candidate_error[selected].mean()*np.sqrt(area))
            rows.append({'action':action,'box':[x,y,x+w,y+h],'pixels':area,'priority':priority,'reviewed':False})
    return sorted(rows,key=lambda r:r['priority'],reverse=True)[:maximum]


def propose(composite,reference,output):
    composite,reference=Path(composite).resolve(),Path(reference).resolve();out=Path(output).resolve()
    if out.exists():raise ValueError('Use a new appearance worklist')
    full,original=[json.loads(p.read_text()) for p in [composite,reference]]
    if full.get('diagnostic') or original.get('diagnostic')!='context_source':raise ValueError('Use actual composite and original captured-GS comparison')
    if any(full.get(k)!=original.get(k) for k in ['usd','camera_manifest','environment_context','effective_photometry','context_background_override']):raise ValueError('Same scene, cameras and photometry required')
    scope=json.loads(full['environment_context']['source_background_scope_json']);references={r['source']:r for r in original['views']}
    inputs={str(p):digest(p) for p in [composite,reference,Path(__file__).resolve()]};rows=[];out.mkdir(parents=True)
    for view in full['views']:
        src=view['source'];ref=references[src];paths=[src,ref['renders'][0],view['renders'][0]];raw,captured,candidate=[cv2.imread(p) for p in paths]
        h,w=candidate.shape[:2];source_h,source_w=raw.shape[:2];mask=np.ones(raw.shape[:2],bool)
        for p,sha in scope[Path(src).name].items():
            if digest(p)!=sha:raise ValueError('Fixed background scope changed')
            mask&=cv2.imread(p,0)>128;inputs[p]=sha
        mask=cv2.resize(mask.astype('uint8'),(w,h),interpolation=cv2.INTER_NEAREST)>0;raw=cv2.resize(raw,(w,h),interpolation=cv2.INTER_AREA)
        for p in paths:inputs[p]=digest(p)
        found=regions(raw,captured,candidate,mask)
        for i,row in enumerate(found):
            x0,y0,x1,y1=row['box'];pad=16;x0,y0=max(0,x0-pad),max(0,y0-pad);x1,y1=min(w,x1+pad),min(h,y1+pad)
            panel=Image.new('RGB',(960,270),(25,25,25));draw=ImageDraw.Draw(panel)
            for j,(label,im)in enumerate([('source',raw),('render',candidate),('original GS',captured)]):
                crop=Image.fromarray(cv2.cvtColor(im[y0:y1,x0:x1],cv2.COLOR_BGR2RGB));crop.thumbnail((320,240));panel.paste(crop,(j*320+(320-crop.width)//2,25));draw.text((j*320+8,6),label,fill='white')
            path=out/(Path(src).stem+f'_{i}.jpg');panel.save(path,quality=92)
            row.update(source=src,candidate=view['renders'][0],crop=str(path),source_box=[round(x0*source_w/w),round(y0*source_h/h),round(x1*source_w/w),round(y1*source_h/h)])
        rows.extend(found)
    rows.sort(key=lambda r:r['priority'],reverse=True)
    result={'version':1,'regions':rows,'input_sha256':inputs,'pass':False,'scope':'Agent worklist for instance count/parts/pose/material or GS restoration. Pixel residuals are triage only; no task sizes, CAD or automatic photorealism pass.'}
    if any(digest(p)!=sha for p,sha in inputs.items()):raise ValueError('Appearance worklist inputs changed')
    (out/'worklist.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('composite');p.add_argument('reference');p.add_argument('--output',required=True);a=p.parse_args();q=propose(a.composite,a.reference,a.output);print('Open appearance regions',len(q['regions']))


if __name__=='__main__':main()
