"""Source-only calibrated motion diagnostic on the existing observation cache.

No candidate mesh chooses correspondences or removes views. Transmitted pixels
are conservatively excluded from tracking, since refraction is not rigid motion.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .blender.store import digest


def fundamental(k0,r0,c0,k1,r1,c1):
    rotation=np.asarray(r1).T@np.asarray(r0)
    t=np.asarray(r1).T@(np.asarray(c0)-np.asarray(c1))
    if np.linalg.norm(t)<1e-8:raise ValueError('Camera pair has no measurable translational baseline')
    x,y,z=t;skew=np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
    result=np.linalg.inv(k1).T@skew@rotation@np.linalg.inv(k0)
    return result/np.linalg.norm(result)


def epipolar_distance(first,second,matrix):
    first=np.c_[np.asarray(first),np.ones(len(first))];second=np.c_[np.asarray(second),np.ones(len(second))]
    a=first@matrix.T;b=second@matrix
    return abs(np.sum(second*a,axis=1))/np.sqrt(np.maximum(1e-20,np.sum(a[:,:2]**2+b[:,:2]**2,axis=1)))


def correspondences(first,second,first_mask,second_mask):
    mask=cv2.erode(np.asarray(first_mask,'uint8'),np.ones((9,9),'uint8'))*255
    points=cv2.goodFeaturesToTrack(first,400,.005,3,mask=mask)
    if points is None:return np.empty((0,2)),np.empty((0,2)),np.empty(0,bool)
    moved,ok,_=cv2.calcOpticalFlowPyrLK(first,second,points,None,winSize=(25,25),maxLevel=4)
    back,back_ok,_=cv2.calcOpticalFlowPyrLK(second,first,moved,None,winSize=(25,25),maxLevel=4)
    a,b=points[:,0],moved[:,0];xy=np.rint(b).astype(int)
    h,w=second.shape;inside=(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)
    target=np.zeros(len(a),bool);target[inside]=np.asarray(second_mask,bool)[xy[inside,1],xy[inside,0]]
    valid=ok.ravel().astype(bool)&back_ok.ravel().astype(bool)&inside&target
    valid&=np.linalg.norm(back[:,0]-a,axis=1)<=.5
    # Photometric consistency is independent of calibrated epipolar error.
    # No epipolar outlier filtering is used to make a moving part look static.
    for i in np.flatnonzero(valid):
        p=cv2.getRectSubPix(first,(15,15),tuple(map(float,a[i]))).astype(float).ravel()
        q=cv2.getRectSubPix(second,(15,15),tuple(map(float,b[i]))).astype(float).ravel()
        p-=p.mean();q-=q.mean();denominator=np.linalg.norm(p)*np.linalg.norm(q)
        valid[i]=denominator>1e-8 and np.dot(p,q)/denominator>=.8
    return a,b,valid


def full_mask(mask,window,shape):
    x,y,w,h,_=window;x,y,w,h=map(int,[x,y,w,h])
    result=np.zeros(shape,bool);result[y:y+h,x:x+w]=cv2.resize(mask.astype('uint8'),(w,h),interpolation=cv2.INTER_NEAREST)>0
    return result


def run(s,obj,part_prompt,reference_prompt,output,*,exclude_prompt=None):
    from .physical_observations import observations
    z,frames,masks,windows=observations(s,obj,part_prompt,exterior_to=exclude_prompt)
    acquired=s/'splat/object_fit'/f'{obj}.acquired.npz';raw=s/'splat/object_fit'/f'{obj}.components.acquired.npz'
    with np.load(acquired,allow_pickle=True) as data:prompts=str(data['prompt']).split('|')
    if reference_prompt not in prompts or exclude_prompt and exclude_prompt not in prompts:
        raise ValueError('Use independently acquired rigid-reference and optical-occluder prompts')
    with np.load(raw) as data:names=list(map(str,data['names']));labels=data['part_masks']
    output=Path(output).resolve()
    if not output.is_relative_to((s/'qa').resolve()):raise ValueError('Source-motion evidence belongs inside this capture QA')
    output.mkdir(parents=True,exist_ok=False);inputs={str(raw.relative_to(s.dir)):digest(raw),
        str(acquired.relative_to(s.dir)):digest(acquired),
        'parsing/posed_frames/cameras.npz':digest(s/'parsing/posed_frames/cameras.npz')}
    inputs[f'splat/object_fit/{obj}.npz']=digest(s/'splat/object_fit'/f'{obj}.npz')
    inputs.update(z.get('_object_domain_inputs',{}))
    from .blender.part_evidence import requirements
    for item in requirements(s,obj):
        if item['part_prompt']==part_prompt:
            inputs.update(item.get('evidence_files',{}))
            if item.get('part_review'):
                folder=s/item['part_review']
                inputs.update({str(path.relative_to(s.dir)):digest(path) for path in folder.iterdir() if path.is_file()})
    (output/'source.py').write_bytes(Path(__file__).read_bytes())
    inputs[str((output/'source.py').relative_to(s.dir.resolve()))]=digest(output/'source.py')
    views=[]
    for k,i in enumerate(frames):
        name=str(z['names'][i]);path=s/'splat/data/images'/name;image=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if image is None:raise ValueError('Missing source RGB: '+name)
        inputs[str(path.relative_to(s.dir))]=digest(path)
        part=full_mask(masks[k],windows[k],image.shape)
        domain=full_mask(z['_object_domains'][k],windows[k],image.shape) if '_object_domains' in z else np.ones(image.shape,bool)
        reference=labels[names.index(name),prompts.index(reference_prompt)].astype(bool)&domain&~part
        if exclude_prompt:
            optical=labels[names.index(name),prompts.index(exclude_prompt)].astype(bool)
            part&=~optical;reference&=~optical
        views.append({'i':int(i),'name':name,'image':image,'part':part,'reference':reference})
    views.sort(key=lambda row:row['i']);rows=[];traces={};tiles=[]
    for number,(a,b) in enumerate(zip(views,views[1:])):
        row={'first':a['name'],'second':b['name'],'regions':{}}
        try:F=fundamental(z['K'][a['i']],z['R_world_cam'][a['i']],z['centre'][a['i']],
                          z['K'][b['i']],z['R_world_cam'][b['i']],z['centre'][b['i']])
        except ValueError as error:row['unobserved_reason']=str(error);rows.append(row);continue
        tile=cv2.cvtColor(b['image'],cv2.COLOR_GRAY2BGR)
        for region,colour in [('part',(60,200,40)),('reference',(220,180,40))]:
            p,q,valid=correspondences(a['image'],b['image'],a[region],b[region]);errors=epipolar_distance(p[valid],q[valid],F)
            traces[f'{number}_{region}_first']=p;traces[f'{number}_{region}_second']=q;traces[f'{number}_{region}_valid']=valid
            row['regions'][region]={'proposed_matches':len(p),'retained_matches':int(valid.sum()),
                'median_epipolar_px':float(np.median(errors)) if len(errors) else None,
                'p95_epipolar_px':float(np.percentile(errors,95)) if len(errors) else None,
                'correspondence_filter':'Source mask, forward/back optical flow and local normalized correlation only; no model or epipolar rejection'}
            for x,y in q[valid]:cv2.circle(tile,(round(float(x)),round(float(y))),3,colour,-1)
        cv2.putText(tile,a['name']+' -> '+b['name'],(16,30),0,.8,(0,255,255),2)
        tiles.append(cv2.resize(tile,(640,360)));rows.append(row)
    for start in range(0,len(tiles),8):cv2.imwrite(str(output/f'tracks_{start//8+1}.jpg'),np.vstack(tiles[start:start+8]))
    np.savez_compressed(output/'source_tracks.npz',**traces)
    reliable=[r for r in rows if all(r.get('regions',{}).get(k,{}).get('retained_matches',0)>=12 for k in ['part','reference'])]
    summary={}
    for region in ['part','reference']:
        values=[r['regions'][region]['p95_epipolar_px'] for r in reliable]
        summary[region]={'reliable_pairs':len(values),'median_pair_p95_epipolar_px':float(np.median(values)) if values else None}
    category='insufficient_source_correspondence'
    if len(reliable)>=8:
        part,reference=[summary[k]['median_pair_p95_epipolar_px'] for k in ['part','reference']]
        category=('camera_or_reference_motion_requires_review' if reference>2. else
                  'source_motion_or_correspondence_requires_review' if part>max(2.,reference+1.5) else
                  'consistent_with_static_epipolar_projection')
    result={'source_only':True,'candidate_geometry_used':False,'source_frames_removed':False,
        'source_views':len(views),'pairs':rows,'summary':summary,'diagnosis':category,
        'physical_parameters_identified':False,'task_approved':False,'inputs_sha256':inputs,
        'scope':'Existing cache only. Conservative direct-material source tracking versus an independently labelled rigid reference. Epipolar consistency does not prove absence of motion along epipolar lines; inconsistent tracking does not by itself identify deformation. Agent RGB review required; never remove bad-fit views or alter a source-state contract automatically.'}
    (output/'diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    from calibrated_scene_reconstruction.scene.session import Session
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session');p.add_argument('object');p.add_argument('part_prompt');p.add_argument('reference_prompt')
    p.add_argument('--exclude-prompt');p.add_argument('--output',required=True)
    a=p.parse_args();r=run(Session(a.session),a.object,a.part_prompt,a.reference_prompt,Path(a.output),exclude_prompt=a.exclude_prompt)
    print(json.dumps({'diagnosis':r['diagnosis'],'summary':r['summary']},indent=2))
