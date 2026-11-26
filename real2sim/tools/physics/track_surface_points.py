"""Source-labelled rigid material point tracks, with explicit occlusion gaps."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from calibrated_scene_reconstruction.physics.process.contract import read,write,digest,require
from calibrated_scene_reconstruction.physics.process.surface_features import annotated_tracks,surface_anchors


def run(args):
    plan=read(args.plan);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    require(plan.get('opaque_reference') is True and plan.get('reason') and len(set(plan['ids']))==len(plan['ids']),
            'Explicit source identities and an opaque material review required')
    cap=cv2.VideoCapture(plan['video']);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
    cam=np.load(plan['camera'],allow_pickle=False);mesh=np.load(plan['mesh'],allow_pickle=False)
    uv=np.full((count,len(plan['ids']),2),np.nan);covered=set()
    for interval in plan['intervals']:
        start,end=interval['start'],interval['end'];require(0<=start<end<count and not covered.intersection(range(start,end+1)),
            'Explicit disjoint source intervals; do not fill an occluded gap')
        covered.update(range(start,end+1));images=[];cap.set(cv2.CAP_PROP_POS_FRAMES,start)
        for i in range(start,end+1):
            ok,image=cap.read();require(ok,'Missing source frame');images.append(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
        values=annotated_tracks(images,interval['pixels'],end_pixels=interval.get('end_pixels'),
            max_fb_px=plan['max_fb_px'],max_consistency_px=plan['max_consistency_px'],max_lk_error=plan['max_lk_error'])
        require(values.shape[1]==len(plan['ids']),'Source identity count mismatch');uv[start:end+1]=values
    ref=plan['reference_frame'];require(np.isfinite(uv[ref]).all(),'Reference annotations missing after source tracking checks')
    xyz=surface_anchors(uv[ref],mesh['V'],mesh['F'],plan['reference_world_pose'],(cam['K'],cam['dist'],cam['T_world_cam'][ref]))
    require(np.isfinite(xyz).all(),'Source points do not intersect the reviewed surface')
    data={'t':cam['t'],'local_points':xyz,'uv':uv}
    for j,name in enumerate(plan['ids']):data[name]=uv[:,j]
    np.savez_compressed(out/'observations.npz',**data)
    regions=np.load(plan['regions'])['features']
    for i in sorted(covered):
        cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,image=cap.read();require(ok,'Missing source review frame')
        cx,cy=regions[i,:2];unit=regions[i,plan['unit_feature_index']]
        x1,y1,x2,y2=np.rint(np.array([cx,cy,cx,cy])+unit*np.asarray(plan['crop_offsets'])).astype(int)
        for j,pixel in enumerate(uv[i]):
            if np.isfinite(pixel).all():
                cv2.circle(image,tuple(np.rint(pixel).astype(int)),2,(0,255,0),1)
                cv2.putText(image,str(j),tuple(np.rint(pixel+[4,0]).astype(int)),0,.35,(0,255,0),1)
        cv2.imwrite(str(out/f'source_{i:05d}.jpg'),image[max(0,y1):y2,max(0,x1):x2])
    cap.release();paths=[args.plan,plan['video'],plan['camera'],plan['mesh'],plan['regions'],__file__,
        'src/calibrated_scene_reconstruction/physics/process/surface_features.py']
    write(out/'proposals.json',{'source_sha256':{str(Path(p).resolve()):digest(p) for p in paths},
        'ids':plan['ids'],'valid_frames':np.isfinite(uv).all(2).sum(0).tolist(),
        'scope':'Source-only annotated opaque material identities, LK forward/back checks and independently annotated end closure where supplied. Gaps are missing. Initial surface depth is a geometry prior; no runtime response poses are prescribed.'})
    print({name:int(np.isfinite(uv[:,j]).all(1).sum()) for j,name in enumerate(plan['ids'])})


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--plan',required=True);p.add_argument('--output',required=True);run(p.parse_args())
