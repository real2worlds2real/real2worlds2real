"""Keep capture-supported sharp GS regions inside an explicit appearance plan.

Region masks select existing rows only. They neither synthesize GS nor certify
novel views. Native gap filling, source fidelity and whole-scene review remain
mandatory. No task sizes, CAD or task/video-specific rules are used.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from calibrated_scene_reconstruction.scene.review.environment import digest, protected_state


def sharp_mask(source, rendered, instance):
    if source.shape != rendered.shape or source.shape[:2] != instance.shape:
        raise ValueError('Aligned RGB and instance mask required')
    def signal(image):
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype('float32')/255
        return cv2.GaussianBlur(abs(gray-cv2.GaussianBlur(gray,(0,0),1.2)),(0,0),2.)
    expected,actual=signal(source),signal(rendered)
    blurry=(expected>.012)&(actual<expected*.6)
    error=abs(source.astype(float)-rendered).mean(2)/255
    good=instance&~blurry&(error<=.12)
    # Exclude mixed pixels at semantic/blur boundaries; the mesh supplies gaps.
    return cv2.erode(good.astype('uint8'),np.ones((3,3),'uint8'))>0


def prepare(session, review, observations, entities, output):
    out=Path(output).resolve()
    if not out.is_relative_to(session.dir.resolve()) or (session/'qa/independent_freeze.json').exists():
        raise ValueError('Use an unfrozen environment derivative')
    render=json.loads(Path(review).read_text())
    if render.get('diagnostic')!='context_source':raise ValueError('Use original captured-GS diagnostics')
    views={Path(v['source']).name:v for v in render['views']}
    specs={e['id']:e for e in session.config('environment.json')['parameters']['entities']}
    if not entities or not set(entities)<=set(specs):raise ValueError('Explicit background owners required')
    out.mkdir(parents=True,exist_ok=False)
    camera=session/'parsing/posed_frames/cameras.npz'
    inputs={str(p.resolve()):digest(p) for p in (Path(review),camera,Path(__file__))}
    rows=[];seen=set()
    for observation in observations:
        path=Path(observation);inputs[str(path.resolve())]=digest(path)
        for row in json.loads(path.read_text())['records']:
            entity,frame=row['entity'],row['frame']
            if entity not in entities or frame not in views or not row.get('mask') or (entity,frame) in seen:continue
            spec=specs[entity];claim=spec.get('source_instance');stuff=spec.get('source_stuff')
            if type(claim)is int and claim>=0:label_kind='thing'
            elif type(stuff)is int and stuff>=0:claim=-2-stuff;label_kind='stuff'
            else:raise ValueError('Regional GS owner needs explicit source-instance/surface association')
            for key in ('source','mask'):
                if digest(row[key])!=row[key+'_sha256']:raise ValueError('Capture observation changed')
                inputs[str(Path(row[key]).resolve())]=row[key+'_sha256']
            raw=cv2.imread(row['source']);matched=views[frame]['renders'][0]
            image=cv2.imread(matched);h,w=image.shape[:2]
            source=cv2.resize(raw,(w,h),interpolation=cv2.INTER_AREA)
            instance=cv2.resize(cv2.imread(row['mask'],0),(w,h),interpolation=cv2.INTER_NEAREST)>128
            good=sharp_mask(source,image,instance)
            mask=out/(entity+'_'+Path(frame).stem+'.png')
            cv2.imwrite(str(mask),cv2.resize(good.astype('uint8')*255,(raw.shape[1],raw.shape[0]),interpolation=cv2.INTER_NEAREST))
            for p in views[frame]['renders']:inputs[str(Path(p).resolve())]=digest(p)
            inputs[str(mask)]=digest(mask)
            rows.append({'entity':entity,'claim':claim,'label_kind':label_kind,'frame':frame,'mask':str(mask),'accepted_pixels':int(good.sum()),
                         'instance_pixels':int(instance.sum()),'source':row['source'],'captured':matched,'novel':views[frame]['renders'][1]})
            seen.add((entity,frame))
    for entity in entities:
        if len({r['frame'] for r in rows if r['entity']==entity and r['accepted_pixels']>=32})<2:
            raise ValueError('Two usable source regions required: '+entity)
    result={'version':1,'regions':rows,'input_sha256':inputs,'protected_task_state':protected_state(session),
            'scope':'Sharp source-supported GS-row proposal. Requires semantic ownership, independent views, native completion and novel-view review.'}
    for p,sha in inputs.items():
        if digest(p)!=sha:raise ValueError('Regional selection evidence changed: '+p)
    (out/'regions.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def select(session, path, points, opacity, claims, cameras, owners,world_axes=None):
    """Require two separated front-visible source regions for each retained row."""
    path=Path(path).resolve();record=json.loads(path.read_text())
    if record.get('version')!=1 or record.get('protected_task_state')!=protected_state(session):
        raise ValueError('GS regions belong to another capture/task state')
    inputs={str(path):digest(path),**record['input_sha256']}
    for p,sha in inputs.items():
        if digest(p)!=sha:raise ValueError('GS region evidence changed: '+p)
    camera=session/'parsing/posed_frames/cameras.npz'
    if inputs.get(str(camera.resolve()))!=digest(camera):raise ValueError('Regions need current calibrated cameras')
    if not set(r['entity'] for r in record['regions'])==set(owners):
        raise ValueError('Regions must cover exactly the retained GS/hybrid owners')
    names=list(map(str,cameras['names']));points=np.asarray(points,float);claims=np.asarray(claims)
    votes=np.zeros(len(points),np.uint16);last=np.full((len(points),3),np.nan);visited=set()
    for row in record['regions']:
        key=(row['entity'],row['frame'])
        if key in visited:raise ValueError('Duplicate source masks do not add independent GS evidence')
        visited.add(key)
        if owners[row['entity']]!=row['claim']:raise ValueError('Regional semantic association changed')
        if row['mask'] not in inputs:raise ValueError('Region mask must be hash-bound')
        i=names.index(row['frame']);mask=cv2.imread(row['mask'],0)
        if mask is None or list(cameras['size'][i])!=[mask.shape[1],mask.shape[0]]:raise ValueError('Region raster differs from calibration')
        C=cameras['centre'][i];pc=(points-C)@cameras['R_world_cam'][i];uv=pc@cameras['K'][i].T
        uv=uv[:,:2]/np.maximum(pc[:,2,None],1e-9);h,w=mask.shape
        ids=np.flatnonzero((pc[:,2]>.1)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h))
        xy=uv[ids].astype(int);cells=xy//4;zbuffer=np.full((h//4+1,w//4+1),np.inf)
        opaque=np.asarray(opacity)[ids]>.2
        np.minimum.at(zbuffer,(cells[opaque,1],cells[opaque,0]),pc[ids[opaque],2])
        front=pc[ids,2]<=zbuffer[cells[:,1],cells[:,0]]*1.03+.03
        good=front&(claims[ids]==row['claim'])&(mask[xy[:,1],xy[:,0]]>128)
        if world_axes is not None:
            # A centre pixel cannot certify a broad Gaussian: its footprint
            # must stay in the trusted region rather than blur into neighbours.
            basis=cameras['R_world_cam'][i].T@np.asarray(world_axes)[ids];depth=pc[ids,2,None]
            jx=cameras['K'][i,0,0]/depth*(basis[:,0,:]-pc[ids,0,None]/depth*basis[:,2,:])
            jy=cameras['K'][i,1,1]/depth*(basis[:,1,:]-pc[ids,1,None]/depth*basis[:,2,:])
            radius=3*np.sqrt(np.c_[(jx*jx).sum(1),(jy*jy).sum(1)])
            radius=np.clip(radius,1,1e6);lo=np.floor(uv[ids]-radius).astype(int);hi=np.ceil(uv[ids]+radius+1).astype(int)
            inside=(lo[:,0]>=0)&(lo[:,1]>=0)&(hi[:,0]<=w)&(hi[:,1]<=h)
            low=np.clip(lo,[0,0],[w,h]);high=np.clip(hi,[0,0],[w,h]);integral=cv2.integral((mask>128).astype('uint8'))
            area=(high[:,0]-low[:,0])*(high[:,1]-low[:,1]);covered=integral[high[:,1],high[:,0]]-integral[low[:,1],high[:,0]]-integral[high[:,1],low[:,0]]+integral[low[:,1],low[:,0]]
            good&=inside&(covered>=.98*np.maximum(area,1))
        before=points[ids]-last[ids];after=points[ids]-C
        cosine=np.einsum('ij,ij->i',before,after)/np.maximum(1e-12,np.linalg.norm(before,axis=1)*np.linalg.norm(after,axis=1))
        angle=np.degrees(np.arccos(np.clip(cosine,-1,1)))
        separated=(votes[ids]==0)|((np.linalg.norm(last[ids]-C,axis=1)>.05)&(angle>=2.))
        take=ids[good&separated];votes[take]+=1;last[take]=C
    keep=votes>=2
    return keep,{'retained':int(keep.sum()),'owners':{e:int((keep&(claims==claim)).sum()) for e,claim in owners.items()},
                 'minimum_separated_views':2,'minimum_ray_angle_deg':2.,'projected_sigma_support':3. if world_axes is not None else None,
                 'scope':'Frontmost source pixels with resolved detail and useful parallax; not a novel-view or photorealism pass.'},inputs


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session');p.add_argument('review')
    p.add_argument('--observations',nargs='+',required=True);p.add_argument('--entities',nargs='+',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();prepare(Session(a.session),a.review,a.observations,a.entities,a.output)


if __name__=='__main__':main()
