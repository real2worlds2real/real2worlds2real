"""Verify visible GS contribution in the actual hybrid composite.

Loaded field bytes or matching photographic mesh textures do not demonstrate
visible GS. Compare the same scene with GS disabled while native meshes remain.
This is a necessary visibility check, never a photorealism approval.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from calibrated_scene_reconstruction.scene.background.gs_regions import sharp_mask
from calibrated_scene_reconstruction.scene.review.environment import digest


def measure(source,full,field,empty,disabled,background,paired=True):
    contribution=np.max(abs(field.astype(float)-empty),axis=2)>16
    affected=np.max(abs(full.astype(float)-disabled),axis=2)>16
    eligible=background&contribution
    if paired:
        eligible=sharp_mask(source,field,eligible)
        affected &= np.mean(abs(full.astype(float)-source),axis=2)<=.18*255
    visible=cv2.erode((eligible&affected).astype('uint8'),np.ones((3,3),'uint8'))>0
    return visible,{'visible_gs_pixels':int(visible.sum()),'eligible_pixels':int(eligible.sum()),
                    'visible_fraction':float(visible.sum()/max(1,eligible.sum()))}


def audit(full_review,field_review,empty_review,disabled_review,output=None):
    paths=[Path(p).resolve() for p in [full_review,field_review,empty_review,disabled_review]]
    full,field,empty,disabled=[json.loads(p.read_text()) for p in paths]
    if [r.get('diagnostic') for r in [full,field,empty,disabled]]!=[None,'context_only','context_empty','context_disabled']:
        raise ValueError('Delivered composite, actual field, empty and native-present GS-disabled diagnostics required')
    keys=['usd','camera_manifest','environment','environment_context','effective_photometry','context_background_override']
    if any(any(r.get(k)!=full.get(k) for k in keys) or r.get('static_scene_preserved')is not True for r in [full,field,empty,disabled]):
        raise ValueError('GS visibility needs the same static scene, cameras and photometry')
    inputs={str(p):digest(p) for p in [*paths,Path(__file__).resolve()]}
    for r in [full,field,empty,disabled]:
        if any(r.get(k) for k in ['background_field_override','background_transform_override','render_settings','source_context_override']):raise ValueError('GS visibility must use delivered field and settings')
        for p,sha in r['input_sha256'].items():
            if p in inputs and inputs[p]!=sha:raise ValueError('Conflicting GS visibility inputs')
            inputs[p]=sha
    others=[{r['source']:r for r in record['views']} for record in [field,empty,disabled]]
    scope=json.loads(full['environment_context']['source_background_scope_json']);rows=[]
    for row in full['views']:
        source=row['source'];references=[r[source] for r in others];raw=cv2.imread(source)
        background=np.ones(raw.shape[:2],bool)
        for p,sha in scope[Path(source).name].items():
            if digest(p)!=sha:raise ValueError('Frozen foreground exclusion changed')
            inputs[p]=sha;background &= cv2.imread(p,0)>128
        scores=[]
        for offset in [0,1]:
            files=[row['renders'][offset],*[r['renders'][offset] for r in references]]
            images=[cv2.imread(p) for p in files];h,w=images[0].shape[:2]
            expected=cv2.resize(raw,(w,h),interpolation=cv2.INTER_AREA)
            mask=cv2.resize(background.astype('uint8'),(w,h),interpolation=cv2.INTER_NEAREST)>0 if offset==0 else np.ones((h,w),bool)
            visible,score=measure(expected,*images,mask,paired=offset==0);scores.append(score)
            for p in [source,*files]:inputs[str(Path(p).resolve())]=digest(p)
        rows.append({'source':source,'matched':scores[0],'novel':scores[1]})
    usable=[r for r in rows if r['matched']['visible_gs_pixels']>=64 and r['novel']['visible_gs_pixels']>=64]
    result={'version':1,'pass':len({r['source'] for r in usable})>=2,'full_review':str(paths[0]),'field_review':str(paths[1]),
            'empty_review':str(paths[2]),'disabled_review':str(paths[3]),'views':rows,'input_sha256':inputs,
            'scope':'Actual visible contribution in at least two captured/free views; no minimum field count or visibility percentage certifies quality. Whole-scene appearance review remains mandatory.'}
    if any(digest(p)!=sha for p,sha in inputs.items()):raise ValueError('GS visibility evidence changed')
    if output:Path(output).write_text(json.dumps(result,indent=2)+'\n')
    return result


def validate(path,full_review):
    path=Path(path).resolve();r=json.loads(path.read_text())
    if Path(r['full_review']).resolve()!=Path(full_review).resolve():raise ValueError('GS visibility belongs to another composite')
    if any(digest(p)!=sha for p,sha in r['input_sha256'].items()):raise ValueError('GS visibility evidence changed')
    actual=audit(r['full_review'],r['field_review'],r['empty_review'],r['disabled_review'])
    if actual!=r or r.get('pass') is not True:raise ValueError('GS has no verified visible contribution in the delivered composite')
    return {str(path):digest(path),**r['input_sha256']}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['composite','field','empty','disabled']:p.add_argument(name)
    p.add_argument('--output',required=True);a=p.parse_args();r=audit(a.composite,a.field,a.empty,a.disabled,a.output)
    print('Actual GS visibility','PASS' if r['pass'] else 'FAIL')


if __name__=='__main__':main()
