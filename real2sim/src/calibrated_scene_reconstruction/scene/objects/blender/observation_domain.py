"""Independent visibility/negative labels for task-object accuracy checks.

No model silhouette selects unknown pixels. Unlabelled texture/part interiors
are unknown; independently labelled scene background remains negative evidence.
All cached views remain, with at least eight informative object observations.
"""
import hashlib
import json
import re
import cv2
import numpy as np


def policy(s, obj):
    if not hasattr(s, 'scene_graph'):
        return None  # Lightweight legacy mesh clients have no scene-label policy.
    value=(s.scene_graph['dynamic_objects'][obj].get('fit') or {}).get('observation_domain')
    if value is None:return None
    if (not isinstance(value,dict) or set(value) not in ({'occluder_suffix'},{'background_suffix','occluder_suffix'})
            or any(not isinstance(v,str) or not re.fullmatch(r'_[A-Za-z0-9_]+',v) for v in value.values())
            or value.get('background_suffix')==value['occluder_suffix']):
        raise ValueError('Object domains need distinct local background/occluder label revisions')
    return value


def resolve_labels(positive, background, occluder):
    positive,background,occluder=(np.asarray(x,bool) for x in (positive,background,occluder))
    conflict=positive&background&~occluder
    if conflict.sum()>.25*max(1,int((positive&~occluder).sum())):
        raise ValueError('Object/background labels disagree on over25% of visible foreground')
    known=(positive|background)&~occluder&~conflict
    if positive.any() and not (positive&known).any():
        raise ValueError('Independent occluder labels erased the entire target; review source identity or genuine visibility before fitting')
    return positive&known,known


def _semantic_domains(s,obj,cameras,frames,masks,windows):
    spec=policy(s,obj)
    from calibrated_scene_reconstruction.scene.objects.observation_exclusions import load as exclusions
    excluded,exclusion_files=exclusions(s,obj,cameras,frames)
    if spec is None and excluded is None:return masks,None,{}
    if spec is None:
        domains=[];positives=[]
        for mask,window,region in zip(masks,windows,excluded):
            x,y,w,h,_=window
            known=~(cv2.resize(region[int(y):int(y+h),int(x):int(x+w)].astype('uint8'),mask.shape[::-1],interpolation=cv2.INTER_NEAREST)>0)
            domains.append(known);positives.append(mask&known)
        return positives,domains,exclusion_files
    files={};metadata={}
    def bind(path):
        if not path.is_file():raise ValueError('Missing independent object observation: '+str(path))
        files[str(path.relative_to(s.dir))]=hashlib.sha256(path.read_bytes()).hexdigest()
    for suffix in spec.values():
        path=s/'splat'/('masks'+suffix+'.json');bind(path);metadata[suffix]=json.loads(path.read_text())
    def crop(image,window,shape):
        x,y,w,h,_=window
        return cv2.resize(image[int(y):int(y+h),int(x):int(x+w)].astype('uint8'),shape[::-1],interpolation=cv2.INTER_NEAREST)>0
    positives,domains=[],[]
    files.update(exclusion_files)
    for index,(i,mask,window) in enumerate(zip(frames,masks,windows)):
        labels={}
        for role,suffix in spec.items():
            path=s/'splat/data/images'/(str(cameras['names'][i]).rsplit('.',1)[0]+suffix+'.png');bind(path)
            raw=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
            if raw is None:raise ValueError('Unreadable object observation')
            if raw.shape != tuple(np.asarray(cameras['size'][i], int)[::-1]):
                raise ValueError('Object observation dimensions differ from their registered camera')
            native=raw<128
            if role=='background_suffix':
                margin=int(metadata[suffix].get('dilation_pixels',0))
                if not 0<=margin<=32:raise ValueError('Invalid independent label dilation')
                if margin:native=cv2.erode(native.astype('uint8'),np.ones((2*margin+1,)*2,'uint8'))>0
            labels[role]=crop(native,window,mask.shape)
        # Occlusion-only policy retains the legacy full visible image domain:
        # independently segmented foreground blockers become unknown, not
        # missing target material. A separate background policy remains useful
        # when the target's observed semantic interior itself is incomplete.
        background=labels.get('background_suffix',~mask)
        try:
            positive,known=resolve_labels(mask,background,labels['occluder_suffix'])
        except ValueError as exc:
            raise ValueError(f'{obj} {cameras["names"][i]}: {exc}') from exc
        if excluded is not None:
            known &= ~crop(excluded[index],window,mask.shape)
            positive &= known
        positives.append(positive);domains.append(known)
    if sum(int(m.sum())>=50 and int((d&~m).sum())>=50 for m,d in zip(positives,domains))<8:
        raise ValueError('Object domains require8 informative foreground/background views')
    return positives,domains,files


def load(s,obj,cameras,frames,masks,windows):
    """Intersect semantic evidence with pixels actually captured by the camera.

    Undistortion padding is unknown even when no semantic occlusion policy was
    requested. Its location comes from calibration, never image colour or a
    candidate silhouette. A real black object/background remains observed.
    """
    from calibrated_scene_reconstruction.scene.capture.valid_pixels import ValidPixels, resize_valid
    from calibrated_scene_reconstruction.core.paths import resolve
    positives,domains,files=_semantic_domains(s,obj,cameras,frames,masks,windows)
    validity=ValidPixels(s);observed=[];known=[];changed=False
    for k,(frame,mask,window) in enumerate(zip(frames,positives,windows)):
        size=tuple(np.asarray(cameras['size'][frame],int)[::-1])
        full=validity.get(str(cameras['names'][frame]),size)
        xywh=np.asarray(window[:4],float)
        if not np.allclose(xywh,np.rint(xywh)):
            raise ValueError('Camera validity needs integral source crop coordinates')
        x,y,w,h=np.rint(xywh).astype(int)
        if x<0 or y<0 or min(w,h)<=0 or x+w>size[1] or y+h>size[0]:
            raise ValueError('Object observation window lies outside its camera image')
        valid=resize_valid(full[y:y+h,x:x+w],mask.shape[::-1])
        domain=valid if domains is None else valid&domains[k]
        positive=mask&domain
        if mask.any() and not positive.any():
            raise ValueError('Object has no actually captured pixels; review its source observation')
        observed.append(positive);known.append(domain);changed|=not valid.all()
    files=dict(files)
    # The scene manifest may live in configs/, so preserve absolute paths for
    # inputs outside the session; Path joining in the round store supports both.
    for path,value in validity.inputs.items():
        source=resolve(path)
        files[str(source.relative_to(s.dir.resolve())) if source.is_relative_to(s.dir.resolve()) else str(source)]=value
    if domains is None and not changed and not validity.inputs:return positives,None,files
    return observed,known,files
