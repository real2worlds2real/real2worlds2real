"""Track a moving camera against a metric static map, preserving missing poses."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from calibrated_scene_reconstruction.physics.process import vision
from calibrated_scene_reconstruction.physics.process.contract import digest,require,write


def track(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(Path(args.config).read_text())['videos'][args.case]
    reference=dict(np.load(args.reference,allow_pickle=False))
    camera=dict(np.load(args.camera,allow_pickle=False))
    regions=cfg.get('moving_regions',[])
    def exclusions(frame,width,height):
        mask=np.full((height,width),255,np.uint8)
        for region in regions:
            keys=sorted((int(k),box) for k,box in region['keyframes'].items())
            if not keys[0][0]<=frame<=keys[-1][0]:continue
            boxes=np.asarray([b for _,b in keys]);clock=[t for t,_ in keys]
            x1,y1,x2,y2=[int(np.interp(frame,clock,boxes[:,j])) for j in range(4)]
            x1,x2=np.clip([x1,x2],0,width);y1,y2=np.clip([y1,y2],0,height)
            mask[y1:y2,x1:x2]=0
        return mask
    directions=['forward','backward'] if args.direction=='both' else [args.direction]
    results={}
    for direction in directions:
        result=vision.camera_track(cfg['path'],reference,camera['K'],camera['dist'],exclusions=exclusions,
            min_inliers=args.min_inliers,allow_missing=True,direction=direction)
        np.savez_compressed(output/(direction+'.npz'),**result);results[direction]=result
    if args.direction=='both':
        result=vision.reconcile_cameras(results['forward'],results['backward'],
            position_tolerance_m=args.agreement_m,angle_tolerance_deg=args.agreement_deg)
    else:result=results[args.direction]
    np.savez_compressed(output/'camera_track.npz',**result)
    valid=np.isfinite(result['T_world_cam']).all(axis=(1,2))
    require(valid.any(),'No camera states localized against the declared reference')
    report={'complete':bool(valid.all()),'valid_frames':int(valid.sum()),'frames':len(valid),
        'missing_frames':np.flatnonzero(~valid).tolist(),
        'direction_conflicts':int(result.get('direction_conflict',np.zeros(len(valid))).sum()),
        'median_reprojection_px':float(np.median(result['quality'][valid,1])),
        'inputs':{str(Path(p).resolve()):digest(p) for p in [args.config,args.reference,args.camera,cfg['path'],__file__,vision.__file__]},
        'scope':'Static-map localization only; no temporal interpolation or target-motion camera constraints. Metric-map and observation review still required.'}
    write(output/'coverage.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    for name in ['config','case','reference','camera','output']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--direction',choices=['forward','backward','both'],default='both')
    parser.add_argument('--min-inliers',type=int,default=20)
    parser.add_argument('--agreement-m',type=float,default=.01)
    parser.add_argument('--agreement-deg',type=float,default=2.)
    track(parser.parse_args())
