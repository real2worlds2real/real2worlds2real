"""Propose extra existing capture views for local hybrid appearance evidence.

This changes neither input video sampling nor reconstruction/calibration. The
requests remain unapproved SAM observations and require source/free-view review.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.scene.review.environment import digest,protected_state


def choose_views(points,cameras,count=4,minimum_baseline=.15):
    if type(count)is not int or not 2<=count<=12 or not .05<=minimum_baseline<=.5:
        raise ValueError('Use2..12 capture reviews and a bounded independent baseline')
    P=np.asarray(points,float)
    if P.shape!=(len(P),3) or not len(P) or not np.isfinite(P).all():raise ValueError('Finite source-owned points required')
    P=P[::max(1,len(P)//1500)];ranked=[]
    for i in range(len(cameras['names'])):
        pc=(P-cameras['centre'][i])@cameras['R_world_cam'][i];uv=pc@cameras['K'][i].T
        uv=uv[:,:2]/np.maximum(pc[:,2,None],1e-9);w,h=cameras['size'][i]
        inside=(pc[:,2]>.15)&(uv[:,0]>8)&(uv[:,0]<w-8)&(uv[:,1]>8)&(uv[:,1]<h-8)
        if inside.mean()<.65:continue
        lo,hi=np.percentile(uv[inside],[2,98],axis=0);area=float(np.prod(hi-lo))
        if area<1500:continue
        margin=(hi-lo)*.08+8;roi=np.r_[np.maximum(lo-margin,0),np.minimum(hi+margin,[w,h])].round().astype(int)
        ranked.append((area*inside.mean(),i,roi.tolist()))
    selected=[]
    for score,i,roi in sorted(ranked,reverse=True):
        if any(np.linalg.norm(cameras['centre'][i]-cameras['centre'][r['index']])<minimum_baseline for r in selected):continue
        selected.append({'frame':str(cameras['names'][i]),'index':i,'roi':roi,'score':score})
        if len(selected)==count:break
    return selected


def prepare(session,source,entities,output,count=4):
    out=Path(output).resolve()
    if out.exists() or not out.is_relative_to(session.dir.resolve()) or (session/'qa/independent_freeze.json').exists():
        raise ValueError('Use new output in an unfrozen environment derivative')
    camera=session/'parsing/posed_frames/cameras.npz';scamera=source/'parsing/posed_frames/cameras.npz'
    if digest(camera)!=digest(scamera):raise ValueError('Source inventory must use this exact calibrated capture')
    from calibrated_scene_reconstruction.scene.background.splat.config import sim3
    from plyfile import PlyData
    cfg=session.config('environment.json');specs={e['id']:e for e in cfg['parameters']['entities']}
    selection=session.config('observed_context_owned.json');seg=Path(selection['segmentation']);world=Path(selection['segmentation_world'])
    cloud=source/'splat/export/background.ply';v=PlyData.read(str(cloud))['vertex'].data
    scale,R,t=sim3(source);P=scale*np.c_[v['x'],v['y'],v['z']].astype(float)@R.T+t
    if np.load(world)['X'].shape!=P.shape or not np.allclose(np.load(world)['X'],P,atol=2e-6):raise ValueError('Source semantic rows do not match this field')
    z=np.load(camera);labels=np.load(seg);inventory=seg.parent/'inventory.json';inv=json.loads(inventory.read_text());classes={r['id']:r['class'] for r in inv['things']}
    inputs={str(p.resolve()):digest(p) for p in [camera,scamera,seg,world,cloud,inventory,source/'splat/align.json',session.config_dir/'environment.json',Path(__file__)]}
    requests=[];coverage={}
    for entity in entities:
        e=specs[entity];claim=e.get('source_instance');stuff=e.get('source_stuff')
        if claim is not None:mask=labels['thing']==claim;prompt=e.get('capture_prompt',classes[claim])
        elif stuff is not None:mask=(labels['thing']<0)&(labels['stuff']==stuff);prompt=e.get('capture_prompt',inv['stuff'][stuff])
        else:raise ValueError('Explicit semantic source owner required')
        rows=choose_views(P[mask],z,count);coverage[entity]=rows
        for row in rows:
            frame=row['frame'];image=session/'splat/data/images'/frame;source_image=source/'splat/data/images'/frame
            if digest(image)!=digest(source_image):raise ValueError('Review source pixels differ from reconstruction capture')
            inputs.update({str(p.resolve()):digest(p) for p in [image,source_image]})
            requests.append({'entity':entity,'frame':frame,'prompt':prompt,'roi':row['roi'],'allow_large':stuff is not None})
    if any(digest(p)!=h for p,h in inputs.items()):raise ValueError('View proposal inputs changed')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'version':1,'requests':requests,'coverage':coverage,
        'input_sha256':inputs,'protected_task_state':protected_state(session),'scope':'Additional review/mask requests only. No input downsampling, task-size truth or automatic GS acceptance.'},indent=2)+'\n')
    return requests


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session');p.add_argument('source_session');p.add_argument('--entities',nargs='+',required=True)
    p.add_argument('--count',type=int,default=4);p.add_argument('--output',required=True);a=p.parse_args()
    print('Unapproved observation requests',len(prepare(Session(a.session),Session(a.source_session),a.entities,a.output,a.count)))


if __name__=='__main__':main()
