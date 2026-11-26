"""Propose sharp photographed subregions for hybrid appearance ownership.

The original instance annotation remains the semantic evidence. This helper
only narrows its appearance mask; it cannot establish whole-object geometry or
approve GS reliability, unseen surfaces, or the composed scene.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from calibrated_scene_reconstruction.scene.background.gs_regions import sharp_mask
from calibrated_scene_reconstruction.scene.review.environment import digest, protected_state


def prepare(session, review, observations, entities, output):
    out=Path(output).resolve();review=Path(review).resolve();render=json.loads(review.read_text())
    if (out.exists() or not out.is_relative_to(session.dir.resolve())
            or (session/'qa/independent_freeze.json').exists()):
        raise ValueError('New unfrozen appearance proposal required')
    if render.get('diagnostic')!='context_source':
        raise ValueError('Use original captured-GS source/free-view diagnostics')
    context=render.get('environment_context') or {};original=render.get('source_context_override')
    if original!=context.get('source_asset') or digest(original)!=context.get('source_asset_sha256'):
        raise ValueError('Bind the original same-capture field')
    if not entities or len(set(entities))!=len(entities):
        raise ValueError('Explicit distinct appearance entities required')
    from calibrated_scene_reconstruction.scene.background.blender_env import label
    for entity in entities:label(entity)
    views={Path(v['source']).name:v for v in render['views']};inputs={str(review):digest(review),str(Path(__file__).resolve()):digest(__file__)}
    inputs.update(render['input_sha256']);rows=[];seen=set();out.mkdir(parents=True)
    for observation in observations:
        observation=Path(observation).resolve();inputs[str(observation)]=digest(observation)
        for row in json.loads(observation.read_text())['records']:
            frame=row['frame'];key=(row['entity'],frame)
            if row['entity'] not in entities or frame not in views or not row.get('mask') or key in seen:continue
            seen.add(key)
            if Path(frame).name!=frame or Path(views[frame]['source']).resolve()!=Path(row['source']).resolve():
                raise ValueError('Appearance annotation must use the exact rendered source image')
            for name in ['source','mask']:
                if digest(row[name])!=row[name+'_sha256']:raise ValueError('Source appearance annotation changed')
                inputs[str(Path(row[name]).resolve())]=row[name+'_sha256']
            v=views[frame];image=cv2.imread(v['renders'][0]);h,w=image.shape[:2]
            raw=cv2.imread(row['source']);mask=cv2.imread(row['mask'],0)
            source=cv2.resize(raw,(w,h),interpolation=cv2.INTER_AREA)
            region=cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST)>128
            good=sharp_mask(source,image,region)
            # These are partial surface observations, never a fitted silhouette.
            record=dict(row);target=out/(row['entity']+'_'+Path(frame).stem+'.png')
            cv2.imwrite(str(target),cv2.resize(good.astype('uint8')*255,(mask.shape[1],mask.shape[0]),interpolation=cv2.INTER_NEAREST))
            record.update(mask=str(target),mask_sha256=digest(target),approved=False,
                          source_method='source_supported_appearance_subset',geometry_fit_scope='surface_part',
                          geometry_fit_review='Only sharp source-supported appearance pixels; never a whole-object silhouette.',
                          original_mask=row['mask'],original_mask_sha256=row['mask_sha256'],
                          retained_fraction=float(good.sum()/max(1,region.sum())))
            rows.append(record)
            for p in v['renders']:inputs[str(Path(p).resolve())]=digest(p)
    if any(digest(p)!=sha for p,sha in inputs.items()):raise ValueError('Appearance proposal input changed')
    result={'version':1,'records':rows,'input_sha256':inputs,'protected_task_state':protected_state(session),'scope':__doc__.strip()}
    (out/'observations.json').write_text(json.dumps(result,indent=2)+'\n');return result
