"""Source-only identity review of semantic part components, before fitting.

White packaging, labels and a flexible target can share one text query. Agents
review connected components against original pixels, without a candidate mesh.
The review may retain several visible components across real occlusions.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .blender.store import digest


def propagated_box(first,second,box):
    """Source optical-flow box proposal; final masks still require RGB review."""
    first,second=np.asarray(first),np.asarray(second)
    if first.shape!=second.shape or first.ndim!=3 or first.shape[2]!=3:
        raise ValueError('Same-resolution source RGB frames required for annotation propagation')
    a,b=[cv2.cvtColor(image,cv2.COLOR_RGB2GRAY) for image in [first,second]]
    x0,y0,x1,y1=np.rint(box).astype(int);mask=np.zeros(a.shape,'uint8')
    mask[max(0,y0-12):min(a.shape[0],y1+12),max(0,x0-12):min(a.shape[1],x1+12)]=255
    points=cv2.goodFeaturesToTrack(a,200,.005,3,mask=mask)
    if points is None or len(points)<6:raise ValueError('Too few source features; annotate this view directly')
    moved,ok,_=cv2.calcOpticalFlowPyrLK(a,b,points,None,winSize=(31,31),maxLevel=4)
    back,back_ok,_=cv2.calcOpticalFlowPyrLK(b,a,moved,None,winSize=(31,31),maxLevel=4)
    valid=ok.ravel().astype(bool)&back_ok.ravel().astype(bool)&(np.linalg.norm(back-points,axis=2).ravel()<.75)
    if valid.sum()<6:raise ValueError('Source optical flow is ambiguous; annotate this view directly')
    affine,inliers=cv2.estimateAffinePartial2D(points[valid],moved[valid],method=cv2.RANSAC,ransacReprojThreshold=2.)
    if affine is None or inliers.sum()<6:raise ValueError('Source annotation propagation lacks measured inliers')
    corners=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]],float)@affine[:,:2].T+affine[:,2]
    low=corners.min(0);high=corners.max(0);bounds=np.r_[low,high]
    bounds[[0,2]]=np.clip(bounds[[0,2]],0,a.shape[1]);bounds[[1,3]]=np.clip(bounds[[1,3]],0,a.shape[0])
    return bounds.tolist(),{'inliers':int(inliers.sum()),'source_features':len(points),
                          'scope':'Source-only proposal, not an accepted mask or physical measurement'}


def relative_box(first_mask,second_mask,box):
    """Coarse source-detection box proposal when pixel tracking is ambiguous."""
    bounds=[]
    for mask in [first_mask,second_mask]:
        y,x=np.nonzero(mask)
        if len(x)<16:raise ValueError('Source detection lacks enough pixels for a box proposal')
        bounds.append((np.array([x.min(),y.min()]),np.array([x.max()+1,y.max()+1])))
    (a,b),(c,d)=bounds
    result=(np.asarray(box,float).reshape(2,2)-a)/(b-a)*(d-c)+c
    h,w=np.asarray(second_mask).shape
    result[:,0]=np.clip(result[:,0],0,w);result[:,1]=np.clip(result[:,1],0,h)
    return result.ravel().tolist()


def components(mask):
    count,labels,stats,centres=cv2.connectedComponentsWithStats(np.asarray(mask,'uint8'),8)
    return labels,[{'id':i,'pixels':int(stats[i,cv2.CC_STAT_AREA]),'centre':centres[i].tolist()}
                   for i in range(1,count)]


def prepare(s,obj,prompt,*,alternative_prompt=None,source_boxes=None,label_proposal=None):
    from .blender.evidence import load
    z,frames,masks,windows=load(s,obj)
    raw=s/'splat/object_fit'/f'{obj}.components.acquired.npz'
    acquired=s/'splat/object_fit'/f'{obj}.acquired.npz'
    with np.load(acquired,allow_pickle=True) as data:prompts=str(data['prompt']).split('|')
    if prompts.count(prompt)!=1:raise ValueError('Review one original semantic part query')
    with np.load(raw) as data:names=list(map(str,data['names']));labels=data['part_masks'][:,prompts.index(prompt)]
    root=s/'qa/source_parts'/obj;root.mkdir(parents=True,exist_ok=True)
    n=max([int(p.name[6:]) for p in root.glob('round_*') if p.name[6:].isdigit()],default=0)+1
    out=root/f'round_{n}';out.mkdir();rows=[];tiles=[]
    parent_inputs={}
    if label_proposal:
        if alternative_prompt or source_boxes:raise ValueError('Reuse one source proposal or acquire a new query')
        parent=(s.dir/label_proposal).resolve()
        if not parent.is_relative_to(root.resolve()):raise ValueError('Source label proposal escaped object review scope')
        metadata=json.loads((parent/'manifest.json').read_text())
        if (metadata['object']!=obj or metadata['prompt']!=prompt or metadata['raw_labels_sha256']!=digest(raw)
                or metadata['proposed_labels_sha256']!=digest(parent/'labels.npz')):
            raise ValueError('Source label proposal changed or belongs to another object')
        with np.load(parent/'labels.npz') as data:
            if list(map(str,data['names']))!=names:raise ValueError('Source proposal frame ordering changed')
            labels=data['masks']
        parent_inputs={str(path.relative_to(s.dir.resolve())):digest(path) for path in [parent/'manifest.json',parent/'labels.npz']}
    annotations=None
    if source_boxes:
        from calibrated_scene_reconstruction.core.session_code import source_path
        path=source_path(s.config_dir,source_boxes);annotations=json.loads(path.read_text())
        if set(annotations)!={'version','object','frames'} or annotations['version']!=1 or annotations['object']!=obj:
            raise ValueError('Source box annotations need version1, object and named frames')
        (out/'source_annotations.json').write_bytes(path.read_bytes())
    if alternative_prompt or annotations:
        from .object_hull import CropSam3
        segmenter=CropSam3(crop=640)
        proposed=np.zeros_like(labels)
        for k,frame in enumerate(frames):
            name=str(z['names'][frame]);image=cv2.imread(str(s/'splat/data/images'/name))[:,:,::-1]
            x,y,w,h,_=windows[k];x,y,w,h=map(int,[x,y,w,h])
            if annotations:
                from PIL import Image
                import torch
                record=annotations['frames'].get(name,{})
                if set(record)!={'rgb_sha256','boxes'} or record['rgb_sha256']!=digest(s/'splat/data/images'/name):
                    raise ValueError('Annotate each current source frame and bind its actual RGB')
                boxes=[];classes=[];target=np.zeros((h,w),bool)
                for box in record['boxes']:
                    if (set(box)!={'xyxy','label','basis'} or type(box['label']) is not int or box['label'] not in (0,1)
                            or not isinstance(box['basis'],str) or not box['basis'].strip()):
                        raise ValueError('Source boxes need explicit positive/negative identity and source basis')
                    bounds=np.asarray(box['xyxy'],float)
                    if (bounds.shape!=(4,) or not np.isfinite(bounds).all() or not 0<=bounds[0]<bounds[2]<=image.shape[1]
                            or not 0<=bounds[1]<bounds[3]<=image.shape[0]):raise ValueError('Invalid source pixel box')
                    bounds-=np.array([x,y,x,y]);bounds[[0,2]]=np.clip(bounds[[0,2]],0,w);bounds[[1,3]]=np.clip(bounds[[1,3]],0,h)
                    if bounds[2]<=bounds[0] or bounds[3]<=bounds[1]:raise ValueError('Source annotation lies outside this observation crop')
                    boxes.append(bounds.tolist());classes.append(box['label'])
                    if box['label']:
                        a,b,c,d=np.rint(bounds).astype(int);target[b:d,a:c]=True
                if not target.any():raise ValueError('A positive source part box is required')
                inputs=segmenter.proc(images=Image.fromarray(image[y:y+h,x:x+w]),text=alternative_prompt or prompt,
                    input_boxes=[boxes],input_boxes_labels=[classes],return_tensors='pt').to(segmenter.dev)
                with torch.no_grad():prediction=segmenter.model(**inputs)
                prediction=segmenter.proc.post_process_instance_segmentation(prediction,threshold=.2,mask_threshold=.5,
                    target_sizes=inputs['original_sizes'].tolist())[0]
                candidates=[m.cpu().numpy().astype(bool) for m in prediction['masks']]
                if candidates:
                    selected=max(candidates,key=lambda m:(m&target).sum()/max(1,(m|target).sum()))
                    proposed[names.index(name),y:y+h,x:x+w]=selected
                count=len(candidates)
            else:
                instances=segmenter.instances(image,alternative_prompt,x+w/2,y+h/2,int(max(w,h)))
                for mask,score in instances:proposed[names.index(name)]|=mask
                count=len(instances)
            print('SOURCE_PART_QUERY',name,count,flush=True)
        labels=proposed
    # Strip independently known occluders before component identity review.
    # This does not select source pixels using a candidate render. A one-pixel
    # conservative dilation avoids discarding valid pixels through resampling;
    # actual fitting still applies the original unchanged visibility domain.
    if '_object_domains' in z:
        labels=labels.copy()
        for k,frame in enumerate(frames):
            name=str(z['names'][frame]);x,y,w,h,_=windows[k];x,y,w,h=map(int,[x,y,w,h])
            known=cv2.dilate(z['_object_domains'][k].astype('uint8'),np.ones((3,3),'uint8'))
            known=cv2.resize(known,(w,h),interpolation=cv2.INTER_NEAREST)>0
            visible=np.zeros(labels.shape[-2:],bool);visible[y:y+h,x:x+w]=known
            labels[names.index(name)] &= visible
    np.savez_compressed(out/'labels.npz',names=names,masks=labels)
    for k,frame in enumerate(frames):
        name=str(z['names'][frame]);mask=labels[names.index(name)];cc,items=components(mask)
        image=cv2.imread(str(s/'splat/data/images'/name));overlay=image.copy()
        for item in items:
            component=cc==item['id'];colour=tuple(int(v) for v in cv2.applyColorMap(
                np.array([[20+(item['id']*37)%215]],'uint8'),cv2.COLORMAP_TURBO)[0,0])
            contours,_=cv2.findContours(component.astype('uint8'),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay,contours,-1,colour,2)
            if item['pixels']>=20:
                x,y=np.rint(item['centre']).astype(int);cv2.putText(overlay,str(item['id']),(x,y),0,.65,colour,2)
        x,y,w,h,_=windows[k];x,y,w,h=map(int,[x,y,w,h]);tile=[]
        for source in [image,overlay]:
            view=cv2.resize(source[y:y+h,x:x+w],(384,384));cv2.putText(view,name,(5,22),0,.6,(0,255,255),1);tile.append(view)
        tiles.append(np.hstack(tile))
        from .source_label_detail import render as source_detail
        detail, detail_record = source_detail(image[:, :, ::-1], mask)
        if detail is not None:
            detail_path = out/f'source_detail_{k+1:03d}.png'
            cv2.imwrite(str(detail_path), detail[:, :, ::-1])
            detail_record.update(path=detail_path.name, sha256=digest(detail_path))
        rows.append({'frame':name,'rgb_sha256':digest(s/'splat/data/images'/name),
                     'components':items,'source_detail':detail_record})
    for start in range(0,len(tiles),8):cv2.imwrite(str(out/f'source_components_{start//8+1}.jpg'),np.vstack(tiles[start:start+8]))
    manifest={'object':obj,'prompt':prompt,'raw_labels_sha256':digest(raw),'acquisition_sha256':digest(acquired),
              'proposed_labels_sha256':digest(out/'labels.npz'),'alternative_prompt':alternative_prompt,
              'source_annotations_sha256':digest(out/'source_annotations.json') if annotations else None,
              'parent_inputs':parent_inputs,'source_domain_inputs':z.get('_object_domain_inputs',{}),
              'whole_observations_sha256':digest(s/'splat/object_fit'/f'{obj}.npz'),
              'cameras_sha256':digest(s/'parsing/posed_frames/cameras.npz'),'rows':rows,'connectivity':8,
              'scope':'Original source RGB and raw semantic masks only; no candidate geometry or true dimensions'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'review.json').write_text(json.dumps({'manifest_sha256':digest(out/'manifest.json'),
        'frames':{row['frame']:{'status':'pending','keep_components':[],'basis':''} for row in rows},
        'instruction':'Inspect both whole-context sheets and source_detail native-pixel crops, especially for narrow parts and false continuations across occluders. Verify connected components belonging to this physical material in source RGB, preserving every verified component across real occlusions. Distinct packaging is not the target. Default visibility is direct. If material is also seen through a source-verified transmitting surface, declare direct_or_transmitted and a separate optical contract; identity alone does not establish exterior depth. Hidden or uncertain identity remains unobserved. Verified material identity overrides a broad enclosing-part query only on independently reviewed pixels. Never use rendered errors to select components. No boundaries are fitted to a model.'},indent=2)+'\n')
    return out


def load(s,obj,prompt,names,masks,expected_frames):
    from .mechanisms import declaration
    plan=declaration(s)
    requests=[item for item in (plan or {}).get('observations',[])
              if item.get('part_prompt')==prompt and item['link'].split('/')[0]==obj and item.get('part_review')]
    from .blender.surface_observations import requirements as surface_regions
    requests += [item for item in surface_regions(s,obj) if item['part_prompt']==prompt]
    if not requests:return masks,set()
    paths={item['part_review'] for item in requests}
    if len(paths)!=1:raise ValueError('A source part needs one consistent identity review')
    out=(s.dir/next(iter(paths))).resolve()
    if not out.is_relative_to((s/'qa/source_parts'/obj).resolve()):raise ValueError('Part review escaped source object scope')
    manifest=json.loads((out/'manifest.json').read_text());review=json.loads((out/'review.json').read_text())
    for path in [out/'manifest.json',out/'review.json']:
        relative=str(path.relative_to(s.dir.resolve()))
        if any(item['evidence_files'].get(relative)!=digest(path) for item in requests):
            raise ValueError('Source component review must be bound to the physical observation contract')
    raw=s/'splat/object_fit'/f'{obj}.components.acquired.npz';acquired=s/'splat/object_fit'/f'{obj}.acquired.npz'
    if (review['manifest_sha256']!=digest(out/'manifest.json') or manifest['object']!=obj or manifest['prompt']!=prompt
            or manifest['connectivity']!=8 or manifest['raw_labels_sha256']!=digest(raw)
            or manifest['acquisition_sha256']!=digest(acquired)
            or manifest['cameras_sha256']!=digest(s/'parsing/posed_frames/cameras.npz')):
        raise ValueError('Source part labels or cameras changed; review again')
    if manifest['proposed_labels_sha256']!=digest(out/'labels.npz'):raise ValueError('Source part proposal labels changed')
    if manifest.get('source_annotations_sha256') and manifest['source_annotations_sha256']!=digest(out/'source_annotations.json'):
        raise ValueError('Source annotation proposal changed')
    if manifest.get('whole_observations_sha256') and manifest['whole_observations_sha256']!=digest(s/'splat/object_fit'/f'{obj}.npz'):
        raise ValueError('Source observation windows changed')
    for name,expected in {**manifest.get('parent_inputs',{}),**manifest.get('source_domain_inputs',{})}.items():
        path=(s.dir/name).resolve()
        if not path.is_relative_to(s.dir.resolve()) or digest(path)!=expected:
            raise ValueError('Source visibility or parent proposal changed')
    with np.load(out/'labels.npz') as data:
        if list(map(str,data['names']))!=names or data['masks'].shape!=np.asarray(masks).shape:
            raise ValueError('Source label proposal topology/frame mismatch')
        masks=data['masks']
    result=np.asarray(masks).copy();seen=set();verified=set()
    for row in manifest['rows']:
        name=row['frame'];decision=review['frames'].get(name,{})
        if name in seen or name not in names:raise ValueError('Part review source frame mismatch')
        seen.add(name)
        if digest(s/'splat/data/images'/name)!=row['rgb_sha256']:raise ValueError('Reviewed source RGB changed')
        cc,items=components(masks[names.index(name)]);keep=decision.get('keep_components')
        if items!=row['components']:raise ValueError('Source connected components changed')
        if (decision.get('status') not in {'verified','unobserved'} or not str(decision.get('basis','')).strip()
                or not isinstance(keep,list) or any(type(i) is not int for i in keep)
                or len(set(keep))!=len(keep) or not set(keep)<={item['id'] for item in items}
                or (decision['status']=='verified')!=bool(keep)):
            raise ValueError('Review each source part identity and visible components before fitting')
        visibility=decision.get('visibility','direct')
        if visibility not in {'direct','direct_or_transmitted'} or (visibility=='direct_or_transmitted'
                and any(not item.get('transmitting_links') for item in requests)):
            raise ValueError('Transmitted source identities require a separate source-backed optical visibility contract')
        result[names.index(name)]=np.isin(cc,keep)
        if decision['status']=='verified':verified.add(name)
    if seen!=set(expected_frames):raise ValueError('Review all currently used source part views')
    return result,verified


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session');p.add_argument('object');p.add_argument('prompt')
    p.add_argument('--alternative-prompt')
    p.add_argument('--source-boxes',help='Config-relative source pixel annotations; final masks require independent RGB review')
    p.add_argument('--label-proposal',help='Session-relative existing source proposal, refined using current independent visibility')
    a=p.parse_args()
    from calibrated_scene_reconstruction.scene.session import Session
    print(prepare(Session(a.session),a.object,a.prompt,alternative_prompt=a.alternative_prompt,source_boxes=a.source_boxes,label_proposal=a.label_proposal))
