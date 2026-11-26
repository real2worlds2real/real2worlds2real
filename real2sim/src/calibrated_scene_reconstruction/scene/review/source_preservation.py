"""Guard captured GS appearance against both filtering and mesh overpainting."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from calibrated_scene_reconstruction.scene.review.environment import digest
from calibrated_scene_reconstruction.scene.review.hybrid_composition import measure


def audit(composite,reference,empty,output=None,frozen_exclusion=None):
    paths=[Path(p).resolve() for p in (composite,reference,empty)]
    full,source,background=[json.loads(p.read_text()) for p in paths]
    if full.get('diagnostic') or source.get('diagnostic')!='context_source' or background.get('diagnostic')!='context_empty':
        raise ValueError('Delivered composite, original source GS and empty diagnostic required')
    for r in (full,source,background):
        if any(r.get(k) for k in ('background_field_override','background_transform_override','render_settings')):
            raise ValueError('Source preservation requires the delivered render profile')
        if r.get('static_scene_preserved') is not True:raise ValueError('Static scene preservation required')
    keys=('usd','camera_manifest','environment','environment_context','effective_photometry','context_background_override')
    if any(any(full.get(k)!=r.get(k) for k in keys) for r in (source,background)):
        raise ValueError('Source comparison must bind the same scene, cameras and photometry')
    context=full['environment_context'];asset=Path(source['source_context_override']).resolve()
    scope=json.loads(context.get('source_background_scope_json','{}'))
    if not scope:raise ValueError('Fixed source background masks required; do not score frozen task/robot appearance')
    if str(asset)!=context.get('source_asset') or digest(asset)!=context.get('source_asset_sha256'):
        raise ValueError('Reference is not the original same-capture GS bound by the candidate')
    for p in (full['usd'],full['camera_manifest']):
        if any(r['input_sha256'].get(p)!=digest(p) for r in (full,source,background)):
            raise ValueError('Source comparison must bind actual scene and camera bytes')
    inputs={str(p):digest(p) for p in paths};inputs[str(Path(__file__).resolve())]=digest(__file__)
    inputs[str(Path(__file__).with_name('hybrid_composition.py').resolve())]=digest(Path(__file__).with_name('hybrid_composition.py'))
    frozen={}
    if frozen_exclusion:
        from calibrated_scene_reconstruction.scene.review.frozen_scope import validate as frozen_scope
        frozen,new_inputs=frozen_scope(frozen_exclusion,full);inputs.update(new_inputs)
    for r in (full,source,background):
        for p,expected in r['input_sha256'].items():
            if p in inputs and inputs[p]!=expected:raise ValueError('Shared source comparison input changed: '+p)
            inputs[p]=expected
    if inputs.get(str(asset))!=context['source_asset_sha256']:raise ValueError('Reference payload hash missing')
    refs={r['source']:r for r in source['views']};empties={r['source']:r for r in background['views']};scores=[]
    for r in full['views']:
        ref=refs[r['source']];clear=empties[r['source']]
        images=[Path(p).resolve() for p in (r['source'],r['renders'][0],ref['renders'][0],clear['renders'][0])]
        for p in images:inputs[str(p)]=digest(p)
        raw,composed,captured,zero=[cv2.imread(str(p)) for p in images]
        mask=np.ones(raw.shape[:2],bool);masks=scope.get(images[0].name)
        if not masks:raise ValueError('Source background scope missing for '+images[0].name)
        for path,expected in masks.items():
            if digest(path)!=expected:raise ValueError('Original background scope changed: '+path)
            region=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            if region is None or region.shape!=mask.shape:raise ValueError('Original background mask resolution mismatch')
            mask &= region>128;inputs[path]=expected
        foreground=frozen.get(images[0].name)
        if frozen_exclusion and foreground is None:raise ValueError('Frozen scoring scope missing source view')
        if foreground:
            keep=cv2.imread(foreground['mask'],0)
            if keep is None or keep.shape!=mask.shape:raise ValueError('Frozen exclusion raster mismatch')
            mask &=keep>128
        mask=cv2.resize(mask.astype('uint8'),(composed.shape[1],composed.shape[0]),interpolation=cv2.INTER_NEAREST)
        mask=cv2.erode(mask,np.ones((5,5),'uint8'))>0
        raw=cv2.resize(raw,(composed.shape[1],composed.shape[0]),interpolation=cv2.INTER_AREA)
        score=measure(raw,captured,composed,mask,zero)
        score['scored']=score['reliable_pixels']>=1024
        score['applicable']=not (foreground and score['masked_pixels']<1024 and foreground['excluded_pixels']>=.9*foreground['pixels'])
        score['pass']=(score['scored'] and score['overpainted_fraction']<=.05) if score['applicable'] else None
        if not score['applicable']:score['reason']='No scorable background outside audited frozen silhouettes'
        scores.append({'source':r['source'],'background_scope_fraction':float(mask.mean()),**score})
    applicable=[r for r in scores if r['applicable']]
    result={'version':1,'pass':len(applicable)>=2 and all(r['pass'] for r in applicable),
            'composite':str(paths[0]),'reference':str(paths[1]),'empty':str(paths[2]),'scores':scores,'input_sha256':inputs,
            'frozen_exclusion':str(Path(frozen_exclusion).resolve()) if frozen_exclusion else None,
            'scope':'Original captured GS preservation in the fixed background masks (task-zone projection, semantic exclusion and valid pixels). Frozen task/robot appearance is excluded. Detects filtering losses and native overpainting; not novel-view or photorealism approval.'}
    for p,expected in inputs.items():
        if digest(p)!=expected:raise ValueError('Source preservation input changed: '+p)
    if output is not None:
        out=Path(output)
        if out.exists():raise ValueError('Use a new source-preservation output')
        out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    return result


def validate(path,composite):
    path=Path(path).resolve();result=json.loads(path.read_text())
    if Path(result['composite']).resolve()!=Path(composite).resolve():raise ValueError('Source preservation belongs to another render')
    if result!=audit(result['composite'],result['reference'],result['empty'],frozen_exclusion=result.get('frozen_exclusion')) or result.get('pass') is not True:
        raise ValueError('Original captured GS preservation has not passed')
    return {str(path):digest(path),**result['input_sha256']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('composite');p.add_argument('reference');p.add_argument('empty');p.add_argument('--output',required=True);p.add_argument('--frozen-exclusion')
    a=p.parse_args();r=audit(a.composite,a.reference,a.empty,a.output,a.frozen_exclusion);print('Original GS preservation','PASS' if r['pass'] else 'FAIL')


if __name__=='__main__':main()
