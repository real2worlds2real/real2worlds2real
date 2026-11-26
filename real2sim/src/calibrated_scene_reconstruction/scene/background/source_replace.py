"""Select captured kernels owned by a reviewed source appearance region.

Useful when textureless-surface GS has unreliable depth: a narrow native
distance band misses its floating paint. Two separated source rays must agree
on the same owner. Unselected kernels, including other materials or contents,
stay intact. This is an unapproved replacement proposal, never a quality pass.
"""
from pathlib import Path
import json

import cv2
import numpy as np

from calibrated_scene_reconstruction.scene.review.environment import digest


def material_subset(rgb,mask,tolerance=.10):
    """Narrow a reviewed material mask by its dominant chroma, not brightness.

    An optional agent tool for a uniform coloured material. It cannot establish
    object identity or separate materials that share the same chroma.
    """
    rgb=np.asarray(rgb,float)/255.;mask=np.asarray(mask,bool)
    if rgb.shape!=(*mask.shape,3) or not np.isfinite(rgb).all() or not .02<=tolerance<=.2:
        raise ValueError('Aligned source RGB and bounded material tolerance required')
    visible=mask&(rgb.sum(2)>.1)&(rgb.max(2)<.98)
    if visible.sum()<64:raise ValueError('Sufficient unclipped material pixels required')
    chroma=rgb/np.maximum(rgb.sum(2,keepdims=True),1e-9);reference=np.median(chroma[visible],axis=0)
    selected=mask&(np.linalg.norm(chroma-reference,axis=2)<=tolerance)&(rgb.sum(2)>.1)
    return selected,reference.tolist()


def select(points,cameras,rows):
    P=np.asarray(points,float);names=list(map(str,cameras['names']))
    if P.shape!=(len(P),3) or not np.isfinite(P).all():raise ValueError('Finite captured world points required')
    votes=np.zeros(len(P),np.uint16);last=np.full_like(P,np.nan);seen=set()
    for row in rows:
        frame=row['frame']
        if frame in seen or frame not in names:raise ValueError('Distinct registered source replacement frames required')
        seen.add(frame);i=names.index(frame);mask=np.asarray(row['pixels'],bool);w,h=cameras['size'][i]
        if mask.shape!=(h,w):raise ValueError('Source-owned mask must match calibrated image')
        # Do not classify mixed semantic boundary pixels as a material surface.
        mask=cv2.erode(mask.astype('uint8'),np.ones((3,3),'uint8'))>0
        centre=cameras['centre'][i];pc=(P-centre)@cameras['R_world_cam'][i];uv=pc@cameras['K'][i].T
        uv=uv[:,:2]/np.maximum(pc[:,2,None],1e-9)
        valid=(pc[:,2]>.05)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h);ids=np.flatnonzero(valid);xy=uv[ids].astype(int)
        ids=ids[mask[xy[:,1],xy[:,0]]]
        before=P[ids]-last[ids];after=P[ids]-centre
        cosine=np.einsum('ij,ij->i',before,after)/np.maximum(1e-12,np.linalg.norm(before,axis=1)*np.linalg.norm(after,axis=1))
        independent=(votes[ids]==0)|((np.linalg.norm(last[ids]-centre,axis=1)>.05)&(np.degrees(np.arccos(np.clip(cosine,-1,1)))>=2.))
        chosen=ids[independent];votes[chosen]+=1;last[chosen]=centre
    if len(seen)<2:raise ValueError('Two source-owned views required')
    return votes>=2


def prepare(session,points,observations,owners):
    if not isinstance(observations,dict) or not set(observations)<=set(owners):raise ValueError('Source replacement must stay within audited photographed owners')
    camera=session/'parsing/posed_frames/cameras.npz';C=np.load(camera)
    inputs={str(camera.resolve()):digest(camera),str(Path(__file__).resolve()):digest(__file__)};removed=np.zeros(len(points),bool);reports=[]
    for entity,paths in observations.items():
        if not isinstance(paths,list) or not paths:raise ValueError('Explicit source annotation files required')
        rows=[];seen=set()
        for path in paths:
            path=Path(path).resolve();data=json.loads(path.read_text());inputs[str(path)]=digest(path)
            for row in data['records']:
                if row['entity']!=entity or not row.get('mask') or row['frame'] in seen:continue
                if not isinstance(row.get('appearance_owner_review'),str) or not row['appearance_owner_review'].strip():
                    raise ValueError('Review the same source material/part owner before replacing its GS: '+entity)
                if Path(row['frame']).name!=row['frame'] or Path(row['source']).resolve()!=(session/'splat/data/images'/row['frame']).resolve():raise ValueError('Use original registered source pixels')
                for key in ['source','mask']:
                    if digest(row[key])!=row[key+'_sha256']:raise ValueError('Source replacement annotation changed')
                    inputs[str(Path(row[key]).resolve())]=row[key+'_sha256']
                rows.append({'frame':row['frame'],'pixels':cv2.imread(row['mask'],0)>128});seen.add(row['frame'])
        selected=select(points,C,rows);removed|=selected
        reports.append({'entity':entity,'source_frames':sorted(seen),'selected_rows':int(selected.sum()),'approved':False})
    if any(digest(p)!=sha for p,sha in inputs.items()):raise ValueError('Source replacement inputs changed')
    return removed,inputs,reports
