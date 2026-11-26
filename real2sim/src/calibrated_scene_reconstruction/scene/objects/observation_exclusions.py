"""Reviewed source masks for observation-only items on a measured support.

A source-labelled planar item can touch the target in image space. Its known
identity must not turn it into target geometry. Project the source label using
measured cameras/support, then require per-view RGB review before use. These
pixels become unknown, never negative evidence; no candidate model is read.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from calibrated_scene_reconstruction.core.session_code import source_path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def declaration(s,obj):
    if not hasattr(s,'config_dir'):return []
    path=Path(s.config_dir)/'agent/observation_exclusions.json'
    plan=(s.config('agent/observation_exclusions.json',{}) if hasattr(s,'config') else
          json.loads(path.read_text()) if path.is_file() else {})
    if not plan:return []
    if plan.get('version')!=1 or set(plan)!={'version','regions'} or not isinstance(plan['regions'],list):
        raise ValueError('Observation exclusions need a version1 source-region plan')
    regions=[r for r in plan['regions'] if r.get('object')==obj]
    for r in regions:
        required={'id','object','anchor_frame','source_image','source_sha256','mask','mask_sha256','support','margin_pixels','basis','protected_prompts'}
        if (set(r)!=required or any(not isinstance(r[k],str) or not r[k].strip() for k in required-{'margin_pixels','protected_prompts'})
                or type(r['margin_pixels']) is not int or not 0<=r['margin_pixels']<=8
                or not isinstance(r['protected_prompts'],list) or not r['protected_prompts']
                or any(not isinstance(p,str) or not p.strip() for p in r['protected_prompts'])
                or not r['id'].replace('_','').isalnum()):
            raise ValueError('Declare an identified source mask, frame, measured support and bounded pixel margin')
    if len({r['id'] for r in regions})!=len(regions):raise ValueError('Duplicate observation-only identity')
    return regions


def source_inputs(s,obj,cameras):
    regions=declaration(s,obj);items=[]
    for r in regions:
        names=list(map(str,cameras['names']))
        if r['anchor_frame'] not in names:raise ValueError('Exclusion anchor has no current measured camera')
        index=names.index(r['anchor_frame']);source=(s.dir/r['source_image']).resolve()
        if not source.is_relative_to(s.dir.resolve()):raise ValueError('Source evidence must remain in this capture')
        registered=s/'splat/data/images'/r['anchor_frame']
        if source!=registered.resolve():
            key=json.loads((s/'capture/keyframes.json').read_text())
            if source!=(s/'capture/rgb.png').resolve() or r['anchor_frame']!=f'0_{key["photo"]:05d}.jpg':
                raise ValueError('Source label must match its registered camera pixels')
        mask=source_path(s.config_dir,r['mask'])
        if sha(source)!=r['source_sha256'] or sha(mask)!=r['mask_sha256']:
            raise ValueError('Observation-only source RGB/mask changed')
        pixels=cv2.imread(str(mask),cv2.IMREAD_GRAYSCALE)
        image=cv2.imread(str(source))
        if pixels is None or image is None or pixels.shape!=image.shape[:2] or pixels.shape!=tuple(cameras['size'][index][::-1]):
            raise ValueError('Source exclusion mask must use actual registered camera resolution')
        if not (pixels>127).any():raise ValueError('Empty observation-only source mask')
        support=s.scene_graph['furniture'].get(r['support'],{})
        if not (support.get('fix') or {}).get('model'):raise ValueError('Exclusion plane needs a verified modeled support')
        from calibrated_scene_reconstruction.scene.support.blender import verify
        verify(s,r['support'])
        items.append({'region':r,'anchor':index,'mask':pixels>127,'top_z':float(support['box'][1][2]),
                      'support_fix':support['fix']})
    return items


def plane_camera(K,R,centre,height):
    camera=np.asarray(R,float).T
    return np.asarray(K)@np.column_stack([camera[:,0],camera[:,1],camera@(np.array([0.,0.,height])-centre)])


def projected_mask(mask,H_source,H_target,size,margin=0):
    homography=H_target@np.linalg.inv(H_source)
    result=cv2.warpPerspective(np.asarray(mask,'uint8'),homography,tuple(map(int,size)),flags=cv2.INTER_NEAREST)>0
    if margin:result=cv2.dilate(result.astype('uint8'),np.ones((2*margin+1,)*2,'uint8'))>0
    return result


def prepare(s,obj):
    cameras=dict(np.load(s/'parsing/posed_frames/cameras.npz'))
    items=source_inputs(s,obj,cameras)
    if not items:raise ValueError('No observation-only source regions declared')
    with np.load(s/'splat/object_fit'/f'{obj}.npz',allow_pickle=True) as data:frames=data['frames']
    component_path=(s/'splat/object_fit'/obj).with_suffix('.components.acquired.npz')
    acquisition_path=(s/'splat/object_fit'/obj).with_suffix('.acquired.npz')
    with np.load(component_path) as data:
        part_names=list(map(str,data['names']));part_masks=data['part_masks']
    with np.load(acquisition_path,allow_pickle=True) as data:prompts=str(data['prompt']).split('|')
    root=s/'qa/object_exclusions'/obj;root.mkdir(parents=True,exist_ok=True)
    number=max([int(p.name[6:]) for p in root.glob('round_*') if p.name[6:].isdigit()],default=0)+1
    out=root/f'round_{number}';out.mkdir();rows=[];tiles=[]
    for i in frames:
        name=str(cameras['names'][i]);size=cameras['size'][i];combined=np.zeros(tuple(size[::-1]),bool)
        for item in items:
            r=item['region'];a=item['anchor'];h=item['top_z']
            anchor=plane_camera(cameras['K'][a],cameras['R_world_cam'][a],cameras['centre'][a],h)
            target=plane_camera(cameras['K'][i],cameras['R_world_cam'][i],cameras['centre'][i],h)
            combined|=projected_mask(item['mask'],anchor,target,size,r['margin_pixels'])
        # A plane footprint behind the bottle/robot is not a visible source
        # item. Protect independently segmented target links before review.
        protected=np.zeros_like(combined)
        for item in items:
            for prompt in item['region']['protected_prompts']:
                if prompts.count(prompt)!=1:raise ValueError('Protected source part must name one acquired prompt')
                protected|=part_masks[part_names.index(name),prompts.index(prompt)]
        # Conservative image-space foreground protection. Semantic labels
        # often leave white/printed holes within one object; those holes are
        # not evidence that a reference plane is visible through the object.
        # This hull only restricts exclusions, never adds target pixels or
        # supplies a model shape. Concave cases still require RGB review.
        yy,xx=np.nonzero(protected)
        if len(xx)>=3:
            safe=np.zeros_like(protected,dtype='uint8')
            cv2.fillConvexPoly(safe,cv2.convexHull(np.c_[xx,yy].astype('int32')),1)
            protected=safe>0
        combined &= cv2.dilate(protected.astype('uint8'),np.ones((5,5),'uint8'))==0
        filename=Path(name).stem+'.png';cv2.imwrite(str(out/filename),combined.astype('uint8')*255)
        source=s/'splat/data/images'/name;image=cv2.imread(str(source));overlay=image.copy()
        contours,_=cv2.findContours(combined.astype('uint8'),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay,contours,-1,(0,0,255),3)
        cv2.putText(overlay,name,(12,32),0,1.,(0,0,255),2)
        tile=np.hstack([cv2.resize(image,(480,270)),cv2.resize(overlay,(480,270))]);tiles.append(tile)
        rows.append({'frame':name,'mask':filename,'mask_sha256':sha(out/filename),'rgb_sha256':sha(source)})
    for start in range(0,len(tiles),8):cv2.imwrite(str(out/f'source_review_{start//8+1}.jpg'),np.vstack(tiles[start:start+8]))
    manifest={'regions':declaration(s,obj),'support_inputs':[{k:v for k,v in item.items() if k not in {'mask','anchor'}} for item in items],
              'cameras_sha256':sha(s/'parsing/posed_frames/cameras.npz'),'rows':rows,
              'source_components_sha256':sha(component_path),'source_acquisition_sha256':sha(acquisition_path),
              'scope':'Source identity exclusion only; no task dimensions or rendered candidate geometry'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'review.json').write_text(json.dumps({'manifest_sha256':sha(out/'manifest.json'),
        'frames':{row['frame']:{'status':'pending','basis':''} for row in rows},
        'instruction':'Inspect each actual RGB overlay. Only exclude the observation-only item. If foreground overlap, nonplanarity or identity is ambiguous, correct source labels; do not approve the projected mask.'},indent=2)+'\n')
    (root/'current.json').write_text(json.dumps({'directory':str(out.relative_to(s.dir)),'manifest_sha256':sha(out/'manifest.json')},indent=2)+'\n')
    return out


def load(s,obj,cameras,frames):
    if not declaration(s,obj):return None,{}
    items=source_inputs(s,obj,cameras);pointer=s/'qa/object_exclusions'/obj/'current.json'
    info=json.loads(pointer.read_text());out=(s.dir/info['directory']).resolve()
    if not out.is_relative_to((s/'qa/object_exclusions'/obj).resolve()):raise ValueError('Exclusion evidence escaped its object scope')
    manifest=json.loads((out/'manifest.json').read_text());review=json.loads((out/'review.json').read_text())
    expected=[{k:v for k,v in item.items() if k not in {'mask','anchor'}} for item in items]
    if (sha(out/'manifest.json')!=info['manifest_sha256'] or review.get('manifest_sha256')!=info['manifest_sha256']
            or manifest['regions']!=declaration(s,obj) or manifest['support_inputs']!=expected
            or manifest['source_components_sha256']!=sha((s/'splat/object_fit'/obj).with_suffix('.components.acquired.npz'))
            or manifest['source_acquisition_sha256']!=sha((s/'splat/object_fit'/obj).with_suffix('.acquired.npz'))
            or manifest['cameras_sha256']!=sha(s/'parsing/posed_frames/cameras.npz')):
        raise ValueError('Source exclusion inputs changed; prepare and review again')
    rows={row['frame']:row for row in manifest['rows']};files={}
    def bind(path):files[str(path.relative_to(s.dir.resolve()))]=sha(path)
    for path in [pointer,out/'manifest.json',out/'review.json']:bind(path.resolve())
    result=[]
    for i in frames:
        name=str(cameras['names'][i]);row=rows.get(name);decision=review.get('frames',{}).get(name,{})
        if row is None or decision.get('status')!='verified' or not str(decision.get('basis','')).strip():
            raise ValueError('Review observation-only identity and foreground visibility in every source view')
        path=(out/row['mask']).resolve();source=s/'splat/data/images'/name
        if not path.is_relative_to(out) or sha(path)!=row['mask_sha256'] or sha(source)!=row['rgb_sha256']:
            raise ValueError('Reviewed exclusion RGB or mask changed')
        mask=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape!=tuple(cameras['size'][i][::-1]):raise ValueError('Invalid source exclusion mask')
        result.append(mask>127);bind(path)
    return result,files


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('session');parser.add_argument('object')
    args=parser.parse_args()
    from calibrated_scene_reconstruction.scene.session import Session
    print(prepare(Session(args.session),args.object))
