"""Source-only straight-part proposals for agent observation review.

No scene, simulation, CAD dimensions or fitted physical response are inputs.
Proposals may be wrong; source overlays and selected observations need review.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from calibrated_scene_reconstruction.physics.process.contract import read,write,digest,require


def observe(args):
    plan=read(args.plan);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    features=np.load(plan['regions'],allow_pickle=False)['features']
    cap=cv2.VideoCapture(plan['video']);fps=cap.get(cv2.CAP_PROP_FPS)
    detector=cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD);records=[]
    for frame in plan['frames']:
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,image=cap.read();require(ok,'Missing source frame')
        cx,cy=features[frame,:2];unit=features[frame,plan['unit_feature_index']]
        bounds=np.asarray(plan['crop_offsets'])*unit+[cx,cy,cx,cy]
        x1,y1,x2,y2=np.clip(np.rint(bounds).astype(int),[0,0,0,0],[image.shape[1],image.shape[0],image.shape[1],image.shape[0]])
        crop=image[y1:y2,x1:x2].copy();gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        excluded=None
        if plan.get('exclusion_mask_dir'):
            excluded=cv2.imread(str(Path(plan['exclusion_mask_dir'])/f'{frame:05d}.png'),0)
            require(excluded is not None,'Missing declared source exclusion mask')
        enhanced=np.rint((gray.astype(float)/255)**plan.get('gamma',1.)*255).astype(np.uint8)
        detected=detector.detect(enhanced)[0];rows=[]
        if detected is not None:
            for endpoints in np.asarray(detected).reshape(-1,2,2):
                p=endpoints+np.array([x1,y1]);p=p[np.argsort(p[:,1])];delta=p[1]-p[0];length=np.linalg.norm(delta)
                if length<unit*plan['min_length_units'] or delta[1]/length<plan['min_vertical_fraction']:continue
                height=(cy-p[0,1])/unit
                if not plan['tip_height_units'][0]<=height<=plan['tip_height_units'][1]:continue
                normal=np.array([delta[1],-delta[0]])/length
                distance=abs(normal@(np.array([cx,cy])-p.mean(0)))
                if distance>unit*plan['max_anchor_distance_units']:continue
                # Thin dark stripes should contrast with their immediate flanks.
                sample=p[0]+np.linspace(.1,.9,25)[:,None]*delta
                def intensity(offset):
                    points=sample+normal*offset-[x1,y1]
                    return cv2.remap(gray,points[:,0].astype(np.float32)[None],points[:,1].astype(np.float32)[None],cv2.INTER_LINEAR)[0]
                middle=np.minimum(intensity(-1),intensity(1));flanks=(intensity(-4).astype(float)+intensity(4))/2
                contrast=float(np.median(flanks-middle))
                if contrast<plan.get('min_dark_contrast',0):continue
                if np.median(middle)>plan.get('max_center_gray',255):continue
                if excluded is not None:
                    xy=np.rint(sample).astype(int)
                    if (excluded[xy[:,1],xy[:,0]]>0).mean()>.2:continue
                rows.append({'endpoints':p.tolist(),'length_px':float(length),'anchor_distance_px':float(distance),'contrast':contrast,
                             'direction':(delta/length).tolist()})
        # Collinear duplicate edge proposals are retained for explicit review;
        # silently merging them can confuse a cable with a nearby target shaft.
        rows.sort(key=lambda r:(r['endpoints'][0][0],r['endpoints'][0][1]))
        overlay=crop.copy()
        for i,row in enumerate(rows):
            row['id']=i;a,b=np.rint(np.asarray(row['endpoints'])-[x1,y1]).astype(int)
            color=(int((i*83)%180+60),int((i*47)%180+60),int((i*127)%180+60))
            cv2.line(overlay,tuple(a),tuple(b),color,1,cv2.LINE_AA);cv2.circle(overlay,tuple(a),3,color,1)
            cv2.putText(overlay,str(i),tuple(a+[3,-3]),0,.35,color,1,cv2.LINE_AA)
        cv2.imwrite(str(out/f'source_{frame:05d}.jpg'),crop)
        cv2.imwrite(str(out/f'proposals_{frame:05d}.jpg'),overlay)
        records.append({'frame':frame,'t':frame/fps,'crop':[int(x1),int(y1),int(x2),int(y2)],'lines':rows})
    paths=[args.plan,plan['video'],plan['regions'],__file__]
    if plan.get('exclusion_mask_dir'):paths += [str(Path(plan['exclusion_mask_dir'])/f'{i:05d}.png') for i in plan['frames']]
    cap.release();write(out/'proposals.json',{'source_sha256':{str(Path(p).resolve()):digest(p) for p in paths},
        'frames':records,'scope':'Source-only line proposals. Unreviewed; no native simulation, geometry or physical response used.'})
    print({row['frame']:len(row['lines']) for row in records})


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--plan',required=True);parser.add_argument('--output',required=True)
    observe(parser.parse_args())
