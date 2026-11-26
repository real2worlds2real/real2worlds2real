"""Metric visible-surface samples from reviewed instance masks and SfM tracks.

These are source points, not complete object boxes. Occluded backfaces, a
featureless surface or a cropped image cannot establish an object's full size.
No Gaussian field, inferred mesh, class-size prior or actual dimensions is read.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from calibrated_scene_reconstruction.scene.capture.colmap_points import image_tracks,point_records
from calibrated_scene_reconstruction.scene.capture.valid_pixels import ValidPixels
from calibrated_scene_reconstruction.scene.review.environment import digest


def select_tracks(pixels,identities,mask,valid):
    pixels=np.asarray(pixels,float);identities=np.asarray(identities)
    if pixels.shape!=(len(identities),2) or mask.shape!=valid.shape or not np.isfinite(pixels).all():
        raise ValueError('Aligned source feature coordinates and camera-valid instance mask required')
    ij=np.rint(pixels).astype(int);h,w=mask.shape
    inside=(ij[:,0]>=0)&(ij[:,0]<w)&(ij[:,1]>=0)&(ij[:,1]<h)
    selected=np.flatnonzero(inside)
    selected=selected[mask[ij[selected,1],ij[selected,0]]&valid[ij[selected,1],ij[selected,0]]]
    return identities[selected]


def prepare(session,observations,output,*,minimum_views=2,max_error_px=1.5):
    if type(minimum_views)is not int or minimum_views<2 or not 0<max_error_px<=4:
        raise ValueError('At least two independently observed views and a finite source residual bound required')
    source=Path(observations).resolve();out=Path(output).resolve()
    if not out.is_relative_to(session.dir.resolve()):raise ValueError('Source measurements belong in this session')
    out.mkdir(parents=True,exist_ok=False)
    camera=session/'parsing/posed_frames/cameras.npz';align=session/'splat/align.json'
    image_file=session/'splat/data/sparse/full/images.bin';point_file=image_file.with_name('points3D.bin')
    inputs={str(p.resolve()):digest(p)for p in [source,camera,align,image_file,point_file,Path(__file__)]}
    rows=[r for r in json.loads(source.read_text())['records']if r.get('approved')is True]
    if not rows:raise ValueError('Review source instance identity before metric association')
    z=np.load(camera);names=list(map(str,z['names']));frames={r['frame']for r in rows}
    if not frames<=set(names):raise ValueError('Every source view must have a registered current camera')
    tracks=image_tracks(image_file,frames);validity=ValidPixels(session)
    votes=defaultdict(lambda:defaultdict(set));seen=set()
    for row in rows:
        key=(row['entity'],row['frame'])
        if key in seen:raise ValueError('A duplicate view cannot supply independent evidence')
        seen.add(key)
        if not row.get('geometry_fit_review'):raise ValueError('Source identity review basis required')
        mask_file=Path(row['mask']);rgb_file=Path(row['source'])
        for file,expected in [(mask_file,row['mask_sha256']),(rgb_file,row['source_sha256'])]:
            if digest(file)!=expected:raise ValueError('Reviewed source changed: '+str(file))
            inputs[str(file.resolve())]=expected
        mask=cv2.imread(str(mask_file),cv2.IMREAD_GRAYSCALE)
        i=names.index(row['frame']);shape=tuple(map(int,z['size'][i][::-1]))
        if mask is None or mask.shape!=shape:raise ValueError('Registered source mask dimensions required')
        mask=cv2.erode((mask>128).astype('uint8'),np.ones((3,3),'uint8'))>0
        uv,ids=tracks[row['frame']]
        for point in select_tracks(uv,ids,mask,validity.get(row['frame'],shape)):
            votes[row['entity']][int(point)].add(row['frame'])
    from calibrated_scene_reconstruction.core.paths import resolve
    inputs.update({str(resolve(p)):v for p,v in validity.inputs.items()})
    all_ids={p for group in votes.values()for p in group};points=point_records(point_file,all_ids)
    sim=json.loads(align.read_text())['sim3'];entities={}
    for entity,group in votes.items():
        ids=[p for p,views in group.items()if len(views)>=minimum_views and points[p][1]<=max_error_px]
        if ids:
            xyz=np.array([points[p][0]for p in ids]);xyz=float(sim['scale'])*xyz@np.asarray(sim['R']).T+sim['t']
            file=out/(entity+'.npz');np.savez_compressed(file,point_ids=ids,world=xyz,
                error_px=[points[p][1]for p in ids],observed_views=[len(group[p])for p in ids])
            statistics={'samples':str(file),'sha256':digest(file),'visible_point_bounds':np.quantile(xyz,[.01,.99],axis=0).tolist()}
        else:statistics={}
        entities[entity]={'count':len(ids),'source_view_counts':{str(p):len(group[p])for p in ids},
            'source_frames':sorted({v for p in ids for v in group[p]}),**statistics,
            'complete_extent_measured':False,'approved':False}
    if any(digest(p)!=v for p,v in inputs.items()):raise ValueError('Source inputs changed during metric extraction')
    result={'entities':entities,'inputs':inputs,'minimum_views':minimum_views,'max_error_px':max_error_px,
        'scope':__doc__.strip(),'task_approved':False}
    (out/'measurements.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    from calibrated_scene_reconstruction.scene.session import Session
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('session');parser.add_argument('observations');parser.add_argument('output')
    args=parser.parse_args();result=prepare(Session(args.session),args.observations,args.output)
    print({name:row['count']for name,row in result['entities'].items()})
