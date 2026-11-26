"""Associate support instances in current images using the observed task-plane position.

A nearby desk of the same class is a different instance. The position chooses an
instance; it never clips that instance to the candidate model or its silhouette.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from calibrated_scene_reconstruction.core.paths import rel
from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.objects.object_hull import CropSam3, project
from .blender import require_mutable, sha


def select_instance(candidates, pixels, ignored):
    pixels=np.asarray(pixels,float)
    h,w=ignored.shape
    finite=np.isfinite(pixels).all(1)
    inside=finite&(pixels[:,0]>=0)&(pixels[:,0]<w)&(pixels[:,1]>=0)&(pixels[:,1]<h)
    xy=pixels[inside].astype(int)
    xy=xy[~ignored[xy[:,1],xy[:,0]]]
    if len(xy)<10:
        raise ValueError('Too little visible positional evidence identifies the task support')
    values=[]
    for mask,confidence in candidates:
        hits=int(mask[xy[:,1],xy[:,0]].sum())
        values.append((hits,float(confidence),mask))
    values.sort(key=lambda item:(item[0],item[1]),reverse=True)
    if not values or values[0][0]<max(10,.5*len(xy)):
        raise ValueError('No support instance covers the observed task-plane position')
    if len(values)>1 and values[1][0]>=.9*values[0][0]:
        intersection=(values[0][2]&values[1][2]).sum()
        union=(values[0][2]|values[1][2]).sum()
        if intersection/max(1,union)<.8:
            raise ValueError('Several different support instances have ambiguous position evidence')
    return values[0][2],{'position_votes':values[0][0],'visible_probes':len(xy),
                        'confidence':values[0][1],'candidate_count':len(values)}



def attached_parts(candidates, body, K, R, C, footprint, selected=None):
    """Positive attached-part evidence, independent of the candidate mesh."""
    h,w=body.shape;distance=cv2.distanceTransform((~body).astype('uint8'),cv2.DIST_L2,5)
    union=body.copy();evidence=[]
    lo,hi=np.asarray(footprint,float);margin=min(.15,.1*np.linalg.norm(hi-lo))
    for mask,confidence in candidates:
        yy,xx=np.nonzero(mask)
        if len(xx)<20:continue
        upper=yy<=np.percentile(yy,5)
        if np.min(distance[yy[upper],xx[upper]])>max(4,.025*max(h,w)):continue
        lower=yy>=np.percentile(yy,98);pixel=[float(np.median(xx[lower])),float(np.median(yy[lower])),1.]
        ray=R@np.linalg.solve(K,pixel)
        if ray[2]>=-1e-6:continue
        foot=C-C[2]/ray[2]*ray
        boundary_margin=max(2,round(.02*h))
        clipped=int(yy.max())>=h-boundary_margin
        if not clipped and not ((foot[:2]>=lo-margin)&(foot[:2]<=hi+margin)).all():continue
        union|=mask
        record={'confidence':float(confidence),'floor_intersection_xy':None if clipped else foot[:2].tolist(),
                'foot_cropped':clipped,'foot_boundary_margin_px':boundary_margin,'pixels':len(xx),
                'floor_contact_status':'unobserved at frame boundary' if clipped else 'mask endpoint ray; contact unconfirmed'}
        evidence.append(record)
        if selected is not None:selected.append((mask,record))
    return union,evidence

def run(session):
    require_mutable(session)
    camera_path=session/'parsing/posed_frames/cameras.npz'
    camera=dict(np.load(camera_path))
    graph=session.scene_graph;name=graph['task']['support'];support=graph['furniture'][name]
    carve_path=session/'splat/support_carve.json';carve=json.loads(carve_path.read_text())
    from .mask_inputs import settings as mask_settings
    masks = mask_settings(session)
    revisions={key:masks[key] for key in ('support_suffix','occluder_suffix')}
    if carve.get('mask_revisions', revisions) != revisions:
        raise ValueError('Support observation revisions changed; repeat carving first')
    if carve['support']!=name:raise ValueError('Observed plane belongs to a different support')
    lo,hi=np.array(carve['footprint'])
    # Interior probes identify the observed support, without imposing its edge.
    xx,yy=np.meshgrid(np.linspace(lo[0]+.2*(hi[0]-lo[0]),hi[0]-.2*(hi[0]-lo[0]),16),
                      np.linspace(lo[1]+.2*(hi[1]-lo[1]),hi[1]-.2*(hi[1]-lo[1]),12))
    points=np.c_[xx.ravel(),yy.ravel(),np.full(xx.size,carve['top_z'])]
    part_prompts=(support.get('fit') or {}).get('observation_parts',[])
    prompt=support.get('_label') or session.config('task.json',{}).get('support')
    if not prompt:raise ValueError('Support identity needs a current-capture semantic label')
    model=CropSam3(threshold=.3)
    directory=session/'splat/data/images';root=session/'support/observations'
    numbers=[int(p.name[6:]) for p in root.glob('round_*') if p.name[6:].isdigit()]
    output=root/f'round_{max(numbers,default=0)+1}';output.mkdir(parents=True)
    inputs={rel(camera_path):sha(camera_path),rel(carve_path):sha(carve_path)}
    accepted,rejected,tiles=[],[],[]
    for i,frame in enumerate(camera['names']):
        frame=str(frame);stem=Path(frame).stem;source=directory/frame;occlusion=directory/(stem+masks['occluder_suffix']+'.png')
        if not occlusion.exists():continue
        ignored=cv2.imread(str(occlusion),0)<128
        u,v,depth=project(points,camera['K'][i],camera['R_world_cam'][i],camera['centre'][i])
        pixels=np.c_[u[depth>.05],v[depth>.05]]
        inputs.update({rel(source):sha(source),rel(occlusion):sha(occlusion)})
        image=cv2.imread(str(source));h,w=image.shape[:2]
        candidates=model.instances(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),prompt,w/2,h/2,max(w,h))
        try:
            mask,evidence=select_instance(candidates,pixels,ignored)
        except ValueError as exc:
            rejected.append({'frame':frame,'reason':str(exc)});continue
        part_evidence=[]
        for part_index,part_prompt in enumerate(part_prompts):
            parts=model.instances(cv2.cvtColor(image,cv2.COLOR_BGR2RGB),part_prompt,w/2,h/2,max(w,h))
            selected=[]
            mask,details=attached_parts(parts,mask,camera['K'][i],camera['R_world_cam'][i],camera['centre'][i],carve['footprint'],selected)
            for part_number,(part_mask,record) in enumerate(selected):
                part_path=output/f'{stem}_part{part_index}_{part_number}.png'
                cv2.imwrite(str(part_path),np.where(part_mask,0,255).astype('uint8'))
                record.update(path=rel(part_path),sha256=sha(part_path))
            part_evidence.append({'prompt':part_prompt,'selected':details})
        evidence['attached_parts']=part_evidence
        path=output/(stem+'.png');cv2.imwrite(str(path),np.where(mask,0,255).astype('uint8'))
        accepted.append({'frame':frame,'path':rel(path),'sha256':sha(path),**evidence})
        if len(tiles)<16:
            vis=(image*.3).astype('uint8');vis[mask]=image[mask]
            cv2.drawContours(vis,cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(0,220,0),2)
            cv2.putText(vis,frame,(5,24),0,.6,(0,255,255),1);tiles.append(cv2.resize(vis,(320,480)))
        if len(accepted)%20==0:print(f'[support.observations] {len(accepted)} accepted, {len(rejected)} ambiguous',flush=True)
    if tiles:
        while len(tiles)%4:tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(output/'review.jpg'),np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)]))
    result={'support':name,'prompt':prompt,'dilation_pixels':0,'inputs':inputs,'frames':accepted,
            'rejected':rejected,'task_or_support_truth_read':False,
            'selection':'observed task-plane interior identifies an instance; full selected mask retained'}
    path=output/'observations.json';path.write_text(json.dumps(result,indent=2)+'\n')
    if len(accepted)<8:raise ValueError('Fewer than8 independently associated support observations')
    session.write_json('support/observations.json',{'path':rel(path),'sha256':sha(path)})
    print(f'[support.observations] {len(accepted)} accepted, {len(rejected)} rejected: {path}',flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('session')
    args=parser.parse_args();run(Session(args.session))


if __name__=='__main__':main()
