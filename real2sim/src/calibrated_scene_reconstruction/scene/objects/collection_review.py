"""Source-only identity review of preserved collection instances.

Spatial association is only a proposal: a label on a nearby package is not a
fruit. Keep or reject complete acquired instances against RGB, retaining every
camera and negative background. No rendered model enters these decisions.
"""
import json
from collections import ChainMap
from pathlib import Path

import cv2
import numpy as np

from .collection_observations import digest


def prepare(session, obj, *, source_annotations=None):
    base=session/'splat/object_fit'/obj
    archive=base.with_suffix('.instances.acquired.npz')
    metadata=base.with_suffix('.instances.json')
    observations=base.with_suffix('.npz')
    cameras=session/'parsing/posed_frames/cameras.npz'
    with np.load(cameras) as z:names=list(map(str,z['names']))
    with np.load(observations,allow_pickle=True) as z:
        frames=z['frames'];windows=z['windows']
    records=json.loads(metadata.read_text())['reviews']
    root=session/'qa/source_collections'/obj;root.mkdir(parents=True,exist_ok=True)
    number=max([int(p.name[6:]) for p in root.glob('round_*') if p.name[6:].isdigit()],default=0)+1
    out=root/f'round_{number}';out.mkdir()
    inputs={str(p.relative_to(session.dir)):digest(p) for p in [archive,metadata,observations,cameras,Path(__file__)]
            if p.is_relative_to(session.dir)}
    # Capture the review implementation even though it lives outside the session.
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    inputs[str((out/'source.py').relative_to(session.dir))]=digest(out/'source.py')
    additional={}
    if source_annotations is not None:
        # Positive source boundaries repair a missed semantic part. They are
        # not an error-dependent exclusion and cannot erase negative pixels.
        from calibrated_scene_reconstruction.core.session_code import source_path
        from calibrated_scene_reconstruction.scene.capture.annotation_masks import rasterize
        path=source_path(session.config_dir,source_annotations)
        annotations=json.loads(path.read_text())
        expected={str(names[i]) for i in frames}
        if (annotations.get('basis')!='source_image_only' or annotations.get('object')!=obj
                or not annotations.get('frames') or not set(annotations['frames'])<=expected):
            raise ValueError('Additional boundaries need this object and current source frames')
        (out/'source_annotations.json').write_bytes(path.read_bytes())
        for frame,row in annotations['frames'].items():
            mask=rasterize(session/'splat/data/images'/frame,row)==0
            if not mask.any():raise ValueError('Additional source boundary cannot be empty')
            additional[frame]=mask
        np.savez_compressed(out/'additional.npz',**additional)
        for file in [out/'source_annotations.json',out/'additional.npz']:
            inputs[str(file.relative_to(session.dir))]=digest(file)
    rows=[];tiles=[];decisions={}
    with np.load(archive) as raw:
        for frame,window in zip(frames,windows):
            name=names[frame];rgb=session/'splat/data/images'/name
            inputs[str(rgb.relative_to(session.dir))]=digest(rgb)
            image=cv2.imread(str(rgb));overlay=image.copy();items=[]
            for record in [r for r in records if r['frame']==name]:
                for instance in record['instances']:
                    if not instance['selected']:continue
                    key=record['prefix']+f"_proposal_{instance['index']}"
                    mask=raw[key];yy,xx=np.nonzero(mask)
                    identifier=len(items)
                    colour=tuple(map(int,cv2.applyColorMap(np.array([[30+identifier*31%220]],'uint8'),cv2.COLORMAP_TURBO)[0,0]))
                    contour,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(overlay,contour,-1,colour,2)
                    cv2.putText(overlay,str(identifier),(int(xx.mean()),int(yy.mean())),0,.7,colour,2)
                    items.append({'id':identifier,'key':key,'prompt':record['prompt'],'pixels':int(mask.sum())})
            if name in additional:
                mask=additional[name];identifier=len(items)
                contour,_=cv2.findContours(mask.astype('uint8'),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay,contour,-1,(0,255,255),2)
                yy,xx=np.nonzero(mask)
                cv2.putText(overlay,str(identifier),(int(xx.mean()),int(yy.mean())),0,.7,(0,255,255),2)
                items.append({'id':identifier,'key':name,'prompt':'Source-annotated missing part',
                              'archive':'additional','pixels':int(mask.sum())})
            x,y,w,h,_=window;x,y,w,h=map(int,[x,y,w,h]);views=[]
            for image in [image,overlay]:
                tile=cv2.resize(image[y:y+h,x:x+w],(380,380))
                cv2.putText(tile,name,(6,24),0,.65,(0,255,255),2);views.append(tile)
            tiles.append(np.hstack(views));rows.append({'frame':name,'instances':items})
            decisions[name]={str(r['id']):{'decision':'pending','basis':''}for r in items}
    for k in range(0,len(tiles),5):cv2.imwrite(str(out/f'review_{k//5}.jpg'),np.vstack(tiles[k:k+5]))
    manifest={'object':obj,'inputs':inputs,'rows':rows,'basis':'source_image_only','accepted':False}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'review.json').write_text(json.dumps({'manifest_sha256':digest(out/'manifest.json'),'basis':'source_image_only','frames':decisions},indent=2)+'\n')
    return out


def reviewed_union(raw, row, decisions):
    items=row['instances']
    if set(decisions)!={str(r['id'])for r in items}:
        raise ValueError('Review every associated source instance exactly once')
    union=None
    for item in items:
        decision=decisions[str(item['id'])]
        if (set(decision)!={'decision','basis'} or decision['decision'] not in {'keep','reject'}
                or not isinstance(decision['basis'],str) or not decision['basis'].strip()):
            raise ValueError('Each source instance requires resolved identity and an RGB basis')
        mask=np.asarray(raw[item['key']],bool)
        if union is None:union=np.zeros_like(mask)
        if decision['decision']=='keep':union|=mask
    if union is None or not union.any():raise ValueError('An entire source view cannot be deleted by identity review')
    return union


def load(session,obj,z,frames,masks,windows):
    relative=session.scene_graph['dynamic_objects'][obj]['fit'].get('collection_review')
    if relative is None:return masks,{}
    if not isinstance(relative,str):raise ValueError('Collection review requires a session-relative path')
    out=(session.dir/relative).resolve();root=(session/'qa/source_collections'/obj).resolve()
    if not out.is_relative_to(root):raise ValueError('Collection review escaped object scope')
    manifest_path=out/'manifest.json';review_path=out/'review.json'
    manifest=json.loads(manifest_path.read_text());review=json.loads(review_path.read_text())
    if (manifest['object']!=obj or manifest['basis']!='source_image_only' or review['basis']!='source_image_only'
            or review['manifest_sha256']!=digest(manifest_path)):
        raise ValueError('Collection review manifest or source basis changed')
    inputs=manifest['inputs']
    required={f'splat/object_fit/{obj}{suffix}'for suffix in ['.npz','.instances.acquired.npz','.instances.json']}
    required.add('parsing/posed_frames/cameras.npz')
    required.update('splat/data/images/'+str(z['names'][i])for i in frames)
    if not required<=set(inputs):raise ValueError('Collection review is missing source provenance')
    for relative,sha in inputs.items():
        path=(session.dir/relative).resolve()
        if not path.is_relative_to(session.dir.resolve()) or not path.is_file() or digest(path)!=sha:
            raise ValueError('Collection source changed; review current RGB and instances again')
    expected=[str(z['names'][i])for i in frames]
    if [r['frame']for r in manifest['rows']]!=expected or set(review['frames'])!=set(expected):
        raise ValueError('Collection review must preserve every observed frame in order')
    result=[]
    supplemental={}
    if (out/'additional.npz').exists():
        for file in ['additional.npz','source_annotations.json']:
            if str((out/file).relative_to(session.dir.resolve())) not in inputs:
                raise ValueError('Additional source boundary is missing its provenance')
        with np.load(out/'additional.npz') as data:supplemental=dict(data)
    with np.load(session/'splat/object_fit'/f'{obj}.instances.acquired.npz') as raw:
        for row,window,mask in zip(manifest['rows'],windows,masks):
            for item in row['instances']:
                if item.get('archive')=='additional' and item['key'] not in supplemental:
                    raise ValueError('Additional source instance is missing')
            full=reviewed_union(ChainMap(supplemental,raw),row,review['frames'][row['frame']]);x,y,w,h,_=window
            x,y,w,h=map(int,[x,y,w,h]);crop=full[y:y+h,x:x+w]
            result.append(cv2.resize(crop.astype('uint8'),mask.shape[::-1],interpolation=cv2.INTER_NEAREST)>0)
    return result,{**inputs,**{str(p.relative_to(session.dir.resolve())):digest(p)for p in [manifest_path,review_path]}}
