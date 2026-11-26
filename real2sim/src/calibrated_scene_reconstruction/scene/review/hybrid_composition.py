"""Detect native meshes painting over reliable captured GS appearance.

Compare the delivered composite with a read-only context-only ablation at the
same cameras. Only source-supported pixels in declared GS-owned instance masks
are scored. This regression gate cannot certify novel geometry or photorealism.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from calibrated_scene_reconstruction.scene.review.environment import digest


def measure(source, context, composite, mask, empty):
    arrays=[np.asarray(a) for a in (source,context,composite,empty)]
    if any(a.shape!=arrays[0].shape for a in arrays) or arrays[0].ndim!=3 or arrays[0].shape[2]!=3:
        raise ValueError('Aligned RGB composition images required')
    mask=np.asarray(mask,bool)
    if mask.shape!=arrays[0].shape[:2]:raise ValueError('Aligned instance mask required')
    original,baseline,full,background=[a.astype(float)/255. for a in arrays]
    base_error=np.abs(baseline-original).mean(2);error=np.abs(full-original).mean(2)
    # A dark source pixel matching an empty background is not observed GS.
    # Require the field to actually contribute in a separate empty ablation.
    supported=np.max(abs(baseline-background),axis=2)>8/255.
    reliable=mask&(base_error<=.10)&supported
    lost=reliable&(error>base_error+.08)&(error>.18)
    count=int(reliable.sum());fraction=float(lost.sum()/count) if count else None
    return {'masked_pixels':int(mask.sum()),'reliable_pixels':count,'overpainted_pixels':int(lost.sum()),
            'overpainted_fraction':fraction,'scored':count>=64,
            'pass':count>=64 and fraction<=.05}


def audit(composite_review,context_review,observations,ownership,output,empty_review,frozen_exclusion=None):
    """Bind both renders and observation masks; absent evidence fails closed."""
    paths=[Path(composite_review).resolve(),Path(context_review).resolve(),Path(empty_review).resolve(),*[Path(p).resolve() for p in observations]]
    full,base,empty=[json.loads(p.read_text()) for p in paths[:3]]
    if full.get('diagnostic') or base.get('diagnostic')!='context_only' or empty.get('diagnostic')!='context_empty':
        raise ValueError('Delivered composite and explicit context-only diagnostic required')
    if any(r.get(k) for r in (full,base,empty) for k in ('background_field_override','background_transform_override','render_settings')):
        raise ValueError('Composition inputs must use the delivered payload and photometry')
    if any(any(full.get(k)!=r.get(k) for k in ('camera_manifest','environment','environment_context','effective_photometry','input_sha256','context_background_override')) for r in (base,empty)):
        raise ValueError('Composition ablation must bind the same scene, cameras and render profile')
    inputs={str(p):digest(p) for p in paths};inputs.update(full['input_sha256']);inputs[str(Path(__file__).resolve())]=digest(__file__)
    frozen={}
    if frozen_exclusion:
        from calibrated_scene_reconstruction.scene.review.frozen_scope import validate as frozen_scope
        frozen,new_inputs=frozen_scope(frozen_exclusion,full);inputs.update(new_inputs)
    fields={row['source']:row for row in base['views']}
    empty_fields={row['source']:row for row in empty['views']}
    rows=[r for p in paths[3:] for r in json.loads(p.read_text())['records']]
    entities={k for k,v in ownership.items() if v['role']=='gs' or v.get('preserve_gs') is True}
    identities={k:k for k in entities}
    for entity in entities:
        for alias in ownership[entity].get('observation_entities',[]):
            if alias in identities and identities[alias]!=entity:raise ValueError('Ambiguous source instance association')
            identities[alias]=entity
    scores=[];seen=set()
    for view in full['views']:
        source=Path(view['source']).resolve();other=fields.get(view['source'])
        if other is None or view['source'] not in empty_fields:raise ValueError('Context diagnostic is missing a captured camera')
        composite=Path(view['renders'][0]);context=Path(other['renders'][0])
        background=Path(empty_fields[view['source']]['renders'][0])
        for path in (source,composite,context,background):inputs[str(path.resolve())]=digest(path)
        rgb=cv2.imread(str(composite));baseline=cv2.imread(str(context))
        empty_rgb=cv2.imread(str(background))
        raw=cv2.imread(str(source));raw=cv2.resize(raw,(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_AREA)
        for row in rows:
            entity=identities.get(row['entity'])
            if entity not in entities or Path(row['source']).resolve()!=source or not row.get('mask'):continue
            key=(entity,str(source))
            if key in seen:continue
            seen.add(key)
            for key_path,expected in [(source,row['source_sha256']),(Path(row['mask']).resolve(),row['mask_sha256'])]:
                if digest(key_path)!=expected:raise ValueError('Composition observation changed: '+str(key_path))
                inputs[str(key_path)]=expected
            mask=cv2.imread(row['mask'],cv2.IMREAD_GRAYSCALE)
            mask=cv2.resize(mask,(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_NEAREST)>128
            if frozen_exclusion:
                if source.name not in frozen:raise ValueError('Frozen scoring scope missing source view')
                keep=cv2.imread(frozen[source.name]['mask'],0)
                mask &=cv2.resize(keep,(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_NEAREST)>128
            mask=cv2.erode(mask.astype('uint8'),np.ones((5,5),'uint8'))>0
            result=measure(raw,baseline,rgb,mask,empty_rgb)
            scores.append({'entity':entity,'observation_entity':row['entity'],'source':str(source),'composite':str(composite),'context':str(context),**result})
    scored=[r for r in scores if r['scored']];observed={r['entity'] for r in scored}
    result={'version':1,'pass':len({r['source'] for r in scored})>=2 and observed==entities and all(r['pass'] for r in scored),
            'composite_review':str(paths[0]),'context_review':str(paths[1]),'empty_review':str(paths[2]),'observations':[str(p) for p in paths[3:]],'gs_entities':sorted(entities),
            'unscored_entities':sorted(entities-observed),'scores':scores,'input_sha256':inputs,
            'frozen_exclusion':str(Path(frozen_exclusion).resolve()) if frozen_exclusion else None,
            'scope':'Captured GS preservation only; independent object completeness, free-view floaters, physics and photorealism reviews still required.'}
    for p,expected in inputs.items():
        if digest(p)!=expected:raise ValueError('Composition input changed: '+p)
    if output is not None:
        out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
        if out.exists():raise ValueError('Use a new composition audit output')
        out.write_text(json.dumps(result,indent=2)+'\n')
    return result


def validate(path,composite_review,ownership):
    path=Path(path).resolve();result=json.loads(path.read_text())
    if result.get('version')!=1 or Path(result['composite_review']).resolve()!=Path(composite_review).resolve():
        raise ValueError('Composition audit belongs to another render')
    for p,expected in result['input_sha256'].items():
        if digest(p)!=expected:raise ValueError('Composition evidence changed: '+p)
    actual=audit(result['composite_review'],result['context_review'],result['observations'],ownership,None,result['empty_review'],result.get('frozen_exclusion'))
    if actual!=result or actual.get('pass') is not True:
        raise ValueError('GS preservation has failed or lacks source-supported evidence')
    return {str(path):digest(path),**result['input_sha256']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('composite');p.add_argument('context');p.add_argument('manifest')
    p.add_argument('--observations',nargs='+',required=True);p.add_argument('--output',required=True);p.add_argument('--empty',required=True);p.add_argument('--frozen-exclusion')
    a=p.parse_args();manifest=json.loads(Path(a.manifest).read_text())
    ownership=(manifest.get('splat_context') or manifest)['appearance_ownership']
    result=audit(a.composite,a.context,a.observations,ownership,a.output,a.empty,a.frozen_exclusion)
    print('GS preservation', 'PASS' if result['pass'] else 'FAIL', 'unscored entities:',result['unscored_entities'])


if __name__=='__main__':main()
