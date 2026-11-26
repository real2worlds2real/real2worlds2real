"""Exclude captured foreground silhouettes from background GS support votes.

A frozen foreground mesh crop misses photographic ghosts at the wrong depth.
Remove those source pixels from evidence, without changing cameras, foreground
geometry, poses or original capture/loss masks. Other independent views can
still support true background behind the foreground.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from calibrated_scene_reconstruction.scene.review.environment import digest, protected_state


def subtract(background, foreground, dilation=7):
    if background.shape!=foreground.shape:raise ValueError('Aligned source masks required')
    if type(dilation)is not int or not 1<=dilation<=31 or dilation%2!=1:raise ValueError('Bounded odd foreground dilation required')
    return np.asarray(background,bool)&~cv2.dilate(np.asarray(foreground,'uint8'),np.ones((dilation,dilation),'uint8')).astype(bool)


def prepare(session, regions, observations, entities, output, dilation=7):
    source=Path(regions).resolve();out=Path(output).resolve();record=json.loads(source.read_text())
    if out.exists() or not out.is_relative_to(session.dir.resolve()) or (session/'qa/independent_freeze.json').exists():raise ValueError('New unfrozen background support proposal required')
    if not entities or record['protected_task_state']!=protected_state(session):raise ValueError('Explicit source exclusions and matching protected state required')
    camera=session/'parsing/posed_frames/cameras.npz';C=np.load(camera);names=list(map(str,C['names']))
    inputs={str(source):digest(source),**record['input_sha256'],str(Path(__file__).resolve()):digest(__file__),str(camera):digest(camera)}
    masks={};frames=set()
    for p in observations:
        p=Path(p).resolve();inputs[str(p)]=digest(p)
        for row in json.loads(p.read_text())['records']:
            if row['entity']not in entities:continue
            frame=row['frame']
            if frame not in names:raise ValueError('Registered foreground source required')
            frames.add(frame)
            if not row.get('mask'):continue
            for key in ['mask','source']:
                path=str(Path(row[key]).resolve())
                if digest(path)!=row[key+'_sha256']:raise ValueError('Foreground source/mask changed')
                inputs[path]=row[key+'_sha256']
            expected=session/'splat/data/images'/frame
            if digest(expected)!=row['source_sha256']:raise ValueError('Foreground belongs to another capture')
            mask=cv2.imread(row['mask'],0)>128
            if mask.shape!=tuple(C['size'][names.index(frame)][::-1]):raise ValueError('Calibrated foreground raster required')
            masks[frame]=masks.get(frame,np.zeros_like(mask))|mask
    if any(digest(p)!=sha for p,sha in inputs.items()):raise ValueError('Source evidence changed')
    out.mkdir(parents=True);rows=[];removed=0
    for i,row in enumerate(record['regions']):
        frame=row['frame']
        if frame not in frames:raise ValueError('Every support view needs an explicit foreground observation, including abstentions: '+frame)
        original=cv2.imread(row['mask'],0)>128
        if frame in masks:
            fg=cv2.resize(masks[frame].astype('uint8'),original.shape[::-1],interpolation=cv2.INTER_NEAREST)>0
            clean=subtract(original,fg,dilation)
        else:clean=original
        removed+=int((original&~clean).sum());path=out/f'{i}_{Path(frame).stem}.png';cv2.imwrite(str(path),clean.astype('uint8')*255)
        row={**row,'mask':str(path),'accepted_pixels':int(clean.sum())};row.pop('mask_sha256',None);inputs[str(path)]=digest(path);rows.append(row)
    record.update(regions=rows,input_sha256=inputs,source_exclusions={'entities':entities,'removed_pixels':removed,'dilation_px':dilation,'pass':False,
        'scope':'Source foreground evidence excluded; abstentions remain explicit and require visual review. No frozen geometry/camera or original masks changed.'})
    if any(digest(p)!=sha for p,sha in inputs.items()):raise ValueError('Source exclusion inputs changed')
    (out/'regions.json').write_text(json.dumps(record,indent=2)+'\n');return record


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session');p.add_argument('regions');p.add_argument('--observations',nargs='+',required=True);p.add_argument('--entities',nargs='+',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=prepare(Session(a.session),a.regions,a.observations,a.entities,a.output);print('Excluded foreground support pixels',r['source_exclusions']['removed_pixels'])


if __name__=='__main__':main()
