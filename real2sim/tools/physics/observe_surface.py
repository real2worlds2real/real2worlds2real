"""Propose source-only rigid opaque-surface keypoints; review before calibration."""
import argparse
from pathlib import Path

import cv2
import numpy as np

from calibrated_scene_reconstruction.physics.process.contract import read,write,digest,require
from calibrated_scene_reconstruction.physics.process.surface_features import surface_anchors,mutual_surface_matches
from calibrated_scene_reconstruction.physics.process.vision import pose


def run(args):
    plan=read(args.plan);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    require(plan.get('opaque_reference') is True and plan.get('reason'),
            'Review an opaque direct-material reference; transmitted glass pixels cannot be anchored to its front mesh')
    cap=cv2.VideoCapture(plan['video']);count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS)
    camera=dict(np.load(plan['camera'],allow_pickle=False));mesh=np.load(plan['mesh'],allow_pickle=False)
    regions=np.load(plan['regions'],allow_pickle=False)['features'];ref=plan['reference_frame']
    require(len(camera['t'])==count==len(regions) and np.allclose(camera['t'],np.arange(count)/fps,atol=1e-8),'Source camera/video/region timebase mismatch')
    cap.set(cv2.CAP_PROP_POS_FRAMES,ref);ok,image=cap.read();require(ok,'Reference frame missing')
    mask=np.zeros(image.shape[:2],np.uint8);cv2.fillPoly(mask,[np.rint(plan['reference_polygon']).astype(np.int32)],255)
    sift=cv2.SIFT_create(nfeatures=2000,contrastThreshold=.008)
    kp,desc=sift.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),mask)
    require(desc is not None and len(kp)>=8,'Not enough source texture on the reviewed opaque surface')
    pixels=np.array([k.pt for k in kp]);xyz=surface_anchors(pixels,mesh['V'],mesh['F'],plan['reference_world_pose'],
        (camera['K'],camera['dist'],camera['T_world_cam'][ref]))
    keep=np.isfinite(xyz).all(1);xyz,desc,pixels=xyz[keep],desc[keep],pixels[keep]
    require(len(xyz)>=8,'Opaque reference points do not intersect the reviewed geometry')
    uv=np.full((count,len(xyz),2),np.nan);poses=np.full((count,4,4),np.nan);records=[]
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    for i in range(count):
        ok,image=cap.read();require(ok,'Source decode incomplete');cx,cy=regions[i,:2];unit=regions[i,plan['unit_feature_index']]
        bounds=np.rint(np.array([cx,cy,cx,cy])+unit*np.array(plan['crop_offsets'])).astype(int)
        x1,y1,x2,y2=np.clip(bounds,[0,0,0,0],[image.shape[1],image.shape[0],image.shape[1],image.shape[0]])
        mask[:]=0;mask[y1:y2,x1:x2]=255
        current,values=sift.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),mask)
        matches=mutual_surface_matches(desc,current,values,pixels,ratio=plan['ratio'],threshold=plan['homography_threshold_px'])
        for index,point in matches.items():uv[i,index]=point
        ids=np.flatnonzero(np.isfinite(uv[i]).all(1));record={'frame':i,'matched_points':len(ids)}
        if len(ids)>=plan['min_pose_inliers']:
            try:
                local_cam,inliers,error=pose(xyz[ids],uv[i,ids],camera['K'],camera['dist'],
                    min_inliers=plan['min_pose_inliers'],threshold=plan['pose_threshold_px'])
                poses[i]=camera['T_world_cam'][i]@np.linalg.inv(local_cam)
                record.update(pose_inliers=len(inliers),pose_median_error_px=float(np.median(error)))
            except ValueError as exc:record['pose_missing_reason']=str(exc)
        crop=image[y1:y2,x1:x2].copy()
        for index in ids:
            point=tuple(np.rint(uv[i,index]-[x1,y1]).astype(int));cv2.circle(crop,point,1,(0,255,0),1)
        cv2.imwrite(str(out/f'source_{i:05d}.jpg'),crop)
        records.append(record)
    cap.release();np.savez_compressed(out/'surface_tracks.npz',t=camera['t'],uv=uv,local_points=xyz,carrier_pose=poses)
    paths=[args.plan,plan['video'],plan['camera'],plan['regions'],plan['mesh'],__file__,
           'src/calibrated_scene_reconstruction/physics/process/surface_features.py','src/calibrated_scene_reconstruction/physics/process/vision.py']
    write(out/'proposals.json',{'frames':records,'source_sha256':{str(Path(p).resolve()):digest(p) for p in paths},
        'scope':'Unreviewed source-only rigid texture correspondences on a declared opaque reference surface. Mesh-derived initial depth is a geometry prior. Missing matches/poses are never interpolated; carrier poses are observation diagnostics, never runtime drivers.'})
    print({'reference_points':len(xyz),'frames_with_pose':int(np.isfinite(poses).all((1,2)).sum()),'frames':count})


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--plan',required=True);p.add_argument('--output',required=True);run(p.parse_args())
