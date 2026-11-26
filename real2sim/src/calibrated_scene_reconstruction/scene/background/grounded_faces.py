"""Upright face proposals from photographed floor contacts and upper corners.

The support height comes from capture calibration/current support geometry.
Only a visible frontal envelope is fitted; hidden depth and construction stay
agent hypotheses. A part mask is never mistaken for a whole-object silhouette.
"""
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from calibrated_scene_reconstruction.scene.review.environment import digest,protected_state


def solve(K,R,C,pixels,floor,max_error_px=6.):
    K,R,C,q=map(lambda x:np.asarray(x,float),[K,R,C,pixels])
    if (q.shape!=(4,2) or not np.isfinite(q).all() or not np.isfinite(floor)
            or not cv2.isContourConvex(q.astype('float32')) or cv2.contourArea(q.astype('float32'))<64
            or not 0<max_error_px<=12):raise ValueError('Ordered TL/TR/BR/BL face corners and known support plane required')
    rays=np.c_[q,np.ones(4)]@np.linalg.inv(K).T@R.T
    if np.any(rays[2:,2]>=-.02):raise ValueError('Observed floor contacts need downward, nongrazing rays')
    bottom=C+rays[2:]*((floor-C[2])/rays[2:,2])[:,None]
    left,right=bottom[1],bottom[0];axis=right-left;axis[2]=0;width=np.linalg.norm(axis)
    if width<.02:raise ValueError('Distinct photographed floor contacts required')
    axis/=width;normal=np.cross(axis,[0,0,1]);centre=(left+right)/2;den=rays[:2]@normal
    if np.any(abs(den)<.05):raise ValueError('Upper corners are too grazing for this face')
    top=C+rays[:2]*(((centre-C)@normal)/den)[:,None];height=float(np.mean(top[:,2])-floor)
    if height<=.02:raise ValueError('Upper corners must be above photographed floor contacts')
    initial=np.r_[centre[:2],np.arctan2(axis[1],axis[0]),np.log([width,height])]
    def rectangle(x):
        tangent=np.array([np.cos(x[2]),np.sin(x[2]),0]);base=np.r_[x[:2],floor];w,h=np.exp(x[3:])
        return base+np.array([-1,1,1,-1])[:,None]*w/2*tangent+np.array([1,1,0,0])[:,None]*h*[0,0,1]
    def residual(x):
        pc=(rectangle(x)-C)@R
        if np.any(pc[:,2]<=.05):return np.full(8,1e6)
        uv=pc@K.T;return (uv[:,:2]/uv[:,2,None]-q).ravel()
    span=max(width,height);fit=least_squares(residual,initial,bounds=(initial-np.r_[span,span,.35,1.1,1.1],initial+np.r_[span,span,.35,1.1,1.1]),loss='soft_l1',f_scale=2.,max_nfev=250)
    error=np.linalg.norm(residual(fit.x).reshape(4,2),axis=1)
    if error.max()>max_error_px:raise ValueError('Source corners do not support an upright floor-contact face')
    return {'front_floor_centre':[float(fit.x[0]),float(fit.x[1]),float(floor)],'yaw_deg':float(np.degrees(fit.x[2])),
            'width':float(np.exp(fit.x[3])),'height':float(np.exp(fit.x[4])),'rectangle_world':rectangle(fit.x).tolist(),
            'reprojection_error_px':error.tolist(),'approved':False,
            'scope':'Visible upright face anchored to reconstructed support. Not whole-object depth, actual dimensions or scene acceptance.'}


def run(session,config,output):
    path=Path(config).resolve();out=Path(output).resolve();q=json.loads(path.read_text());camera=session/'parsing/posed_frames/cameras.npz'
    if out.exists() or not out.is_relative_to(session.dir.resolve()) or (session/'qa/independent_freeze.json').exists():raise ValueError('New unfrozen face proposal required')
    if not q.get('assumption','').strip() or len(q.get('held_out',[]))<2:raise ValueError('Explicit upright/floor-contact hypothesis and two held-out source views required')
    C=np.load(camera);names=list(map(str,C['names']));frames={q['frame'],*[r['frame'] for r in q['held_out']]}
    if q['camera_sha256']!=digest(camera) or set(q['source_sha256'])!=frames:raise ValueError('Bind all annotated source cameras/images')
    i=names.index(q['frame']);floor=session.config('environment.json')['physics']['ground_height'];r=solve(C['K'][i],C['R_world_cam'][i],C['centre'][i],q['pixels'],floor)
    points=np.array(r['rectangle_world']);checks=[];seen={q['frame']}
    for row in q['held_out']:
        if row['frame'] in seen:raise ValueError('Distinct held-out source frames required')
        seen.add(row['frame']);j=names.index(row['frame']);pc=(points-C['centre'][j])@C['R_world_cam'][j];uv=pc@C['K'][j].T
        error=np.linalg.norm(uv[:,:2]/uv[:,2,None]-np.array(row['pixels']),axis=1)
        if np.any(pc[:,2]<=0) or error.max()>6:raise ValueError('Held-out source corners disagree with grounded face')
        checks.append({'frame':row['frame'],'reprojection_error_px':error.tolist()})
    inputs={str(p.resolve()):digest(p) for p in [path,camera,Path(__file__).resolve()]}
    for frame in frames:
        image=session/'splat/data/images'/frame
        if digest(image)!=q['source_sha256'][frame]:raise ValueError('Annotated face source changed')
        inputs[str(image.resolve())]=digest(image)
    r.update(entity=q['entity'],assumption=q['assumption'],held_out=checks,input_sha256=inputs,protected_task_state=protected_state(session))
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(r,indent=2)+'\n');return r
