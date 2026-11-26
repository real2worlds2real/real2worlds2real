"""Source-only per-part video masks for real/simulation contour comparisons."""
import argparse
import json
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image

from calibrated_scene_reconstruction.physics.process.contract import digest,read,require,write


def segment(args):
    import torch
    from calibrated_scene_reconstruction.core.perception.sam import load_sam3
    from calibrated_scene_reconstruction.physics.process.resources import watch
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    plan=read(args.plan);regions=np.load(plan['regions'],allow_pickle=False)
    features=regions['features'];cap=cv2.VideoCapture(plan['video']);fps=cap.get(cv2.CAP_PROP_FPS)
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));require(len(features)==total,'Source regions must cover each video frame')
    free=int(subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
    require(free>=8192+4608,'Insufficient shared headroom for source segmentation')
    os.environ['SCENE_RECONSTRUCTION_SHARED_GPU_ACTIVE']='1';os.environ['SCENE_RECONSTRUCTION_SHARED_GPU_RESERVE_MIB']='4608'
    os.environ['SCENE_RECONSTRUCTION_PROCESS_ATTEMPT']='1';monitor=watch(out)
    torch.set_num_threads(8)
    model,processor=load_sam3('cuda',local_files_only=True,dtype=torch.bfloat16)
    source_hashes={str(Path(p).resolve()):digest(p) for p in [args.plan,plan['video'],plan['regions'],__file__]}
    frames=args.frames if args.frames else list(range(total));records=[]
    (out/'union').mkdir();(out/'overlays').mkdir()
    for part in plan['parts']:(out/part['id']).mkdir()
    for number,frame in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,bgr=cap.read();require(ok,'Source video frame missing')
        image=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);H,W=image.shape[:2]
        cx,cy=features[frame,:2];unit=features[frame,plan['unit_feature_index']]
        require(np.isfinite([cx,cy,unit]).all() and unit>0,'Missing source ROI observation')
        union=np.zeros((H,W),np.uint8);rows=[]
        for part in plan['parts']:
            offsets=np.asarray(part['crop_offsets'])*unit+np.array([cx,cy,cx,cy])
            x1,y1,x2,y2=np.clip(np.rint(offsets).astype(int),[0,0,0,0],[W,H,W,H])
            require(x2-x1>20 and y2-y1>20,'Invalid source part crop')
            crop=Image.fromarray(image[y1:y2,x1:x2]);inputs=processor(images=crop,text=part['prompt'],return_tensors='pt').to('cuda')
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):result=model(**inputs)
            result=processor.post_process_instance_segmentation(result,threshold=part['threshold'],mask_threshold=.5,
                target_sizes=inputs['original_sizes'].tolist())[0]
            seed=np.array([cx,cy])+np.asarray(part['seed_offset'])*unit-[x1,y1]
            anchor=np.array([cx,cy])+np.asarray(part.get('anchor_offset',[0,0]))*unit-[x1,y1]
            candidates=[]
            for mask,score in zip(result['masks'],result['scores']):
                mask=mask.cpu().numpy().astype(bool);ys,xs=np.nonzero(mask)
                if len(xs)<12:continue
                distance=float(np.linalg.norm(np.array([xs.mean(),ys.mean()])-seed)/unit)
                if distance>2.5:continue
                if 'max_anchor_distance' in part and np.min(np.hypot(xs-anchor[0],ys-anchor[1]))>part['max_anchor_distance']*unit:continue
                candidates.append((float(score),distance,mask))
            target=np.zeros((y2-y1,x2-x1),bool)
            if candidates:
                if part['selection']=='one':target=max(candidates,key=lambda r:r[0]-.12*r[1])[2]
                else:
                    for _,_,mask in candidates:target|=mask
            full=np.zeros((H,W),np.uint8);full[y1:y2,x1:x2]=target.astype(np.uint8)*255
            cv2.imwrite(str(out/part['id']/f'{frame:05d}.png'),full);union|=full
            rows.append({'id':part['id'],'candidates':len(candidates),'pixels':int(target.sum()),
                         'scores':[s for s,_,_ in candidates],'crop':[int(x1),int(y1),int(x2),int(y2)]})
        cv2.imwrite(str(out/'union'/f'{frame:05d}.png'),union)
        overlay=bgr.copy();contours,_=cv2.findContours(union,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay,contours,-1,(30,240,60),2)
        if number%15==0 or args.frames:cv2.imwrite(str(out/'overlays'/f'{frame:05d}.jpg'),overlay)
        records.append({'frame':frame,'t':frame/fps,'parts':rows})
        if number%10==0:print(f'[source.masks] {number+1}/{len(frames)} frame={frame}',flush=True)
    cap.release();monitor.set()
    write(out/'masks.json',{'fps':fps,'frames':records,'source_sha256':source_hashes,
        'model':'facebook/sam3','local_cached_model':True,
        'scope':'Source-image segmentation proposals, not ground truth. No simulated silhouette or calibrated physical response is used. Occlusion and motion blur may leave uncertain contours.'})
    print(str(out),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    p.add_argument('--frames',type=int,nargs='+');segment(p.parse_args())
