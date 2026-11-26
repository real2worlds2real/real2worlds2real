"""Rigid panel proposals from visible source corners, without supplied dimensions.

Three corresponding corners determine a rectangular hypothesis. Occluded
corners stay inferred, and two additional source frames must check the fit.
No source-view fit approves materials, support stability or free views.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from calibrated_scene_reconstruction.scene.background.structure_fit import triangulate
from calibrated_scene_reconstruction.scene.review.environment import digest,protected_state

QUAD=np.array([[-.5,.5],[.5,.5],[.5,-.5],[-.5,-.5]])


def corner_ids(values):
    if (not isinstance(values,list) or not 3<=len(values)<=4 or len(set(values))!=len(values)
            or any(type(i)is not int or i not in range(4) for i in values)):
        raise ValueError('Three or four distinct visible rectangle corner indices required')
    return values


def solve(cameras,frames,pixels,indices,held_out,error_px=6.):
    indices=corner_ids(indices);pixels=np.asarray(pixels,float)
    if len(held_out)<2:raise ValueError('Two distinct held-out source frames required')
    if pixels.shape!=(len(frames),len(indices),2):raise ValueError('Matched visible-corner observations required')
    points,initial_report=triangulate(cameras,frames,pixels,max_error_px=error_px)
    names=list(map(str,cameras['names']));seen=set(frames)
    for row in held_out:
        if row['frame'] in seen or row['frame'] not in names:raise ValueError('Distinct registered held-out frames required')
        seen.add(row['frame']);ids=corner_ids(row['corner_indices']);p=np.asarray(row['pixels'],float)
        if p.shape!=(len(ids),2) or not np.isfinite(p).all():raise ValueError('Finite held-out visible corners required')
    for frame,p in [*zip(frames,pixels),*[(r['frame'],r['pixels']) for r in held_out]]:
        w,h=cameras['size'][names.index(frame)];p=np.asarray(p)
        if np.any(p<0) or np.any(p[:,0]>=w) or np.any(p[:,1]>=h):raise ValueError('Visible corners must lie inside the source raster')
    affine=np.linalg.lstsq(np.c_[QUAD[indices],np.ones(len(indices))],points,rcond=None)[0]
    a,b,centre=affine
    if np.linalg.norm(a)<1e-8:raise ValueError('Noncollinear visible panel corners required')
    x=a/np.linalg.norm(a);z=b-b@x*x
    if np.linalg.norm(z)<1e-8:raise ValueError('Noncollinear visible panel corners required')
    z/=np.linalg.norm(z);rotation=np.stack([x,np.cross(z,x),z],axis=1)
    initial=np.r_[centre,Rotation.from_matrix(rotation).as_rotvec(),np.log([np.linalg.norm(a),np.linalg.norm(b)])]
    def rectangle(v):
        axes=Rotation.from_rotvec(v[3:6]).as_matrix()
        return v[:3]+QUAD[:,0,None]*np.exp(v[6])*axes[:,0]+QUAD[:,1,None]*np.exp(v[7])*axes[:,2]
    def project(vertices,frame):
        i=names.index(frame);pc=(vertices-cameras['centre'][i])@cameras['R_world_cam'][i]
        if np.any(pc[:,2]<=.05):raise ValueError('Panel corner behind a source camera')
        uv=pc@cameras['K'][i].T;return uv[:,:2]/uv[:,2,None]
    def residual(v):
        vertices=rectangle(v)
        try:return np.concatenate([(project(vertices,frame)[indices]-p).ravel() for frame,p in zip(frames,pixels)])
        except ValueError:return np.full(pixels.size,1e6)
    fit=least_squares(residual,initial,max_nfev=1000,xtol=1e-11,ftol=1e-11,gtol=1e-11)
    vertices=rectangle(fit.x);errors=np.linalg.norm(residual(fit.x).reshape(len(frames),len(indices),2),axis=2)
    checks=[]
    for row in held_out:
        error=np.linalg.norm(project(vertices,row['frame'])[row['corner_indices']]-row['pixels'],axis=1)
        checks.append({'frame':row['frame'],'corner_indices':row['corner_indices'],'error_px':error.tolist(),
                       'observed':row['pixels'],'projected':project(vertices,row['frame'])[row['corner_indices']].tolist()})
    training_pass=bool(np.isfinite(fit.x).all() and errors.max()<=error_px)
    held_pass=all(max(row['error_px'])<=error_px for row in checks)
    result={'centre':fit.x[:3].tolist(),'rotation':Rotation.from_rotvec(fit.x[3:6]).as_matrix().tolist(),
            'width':float(np.exp(fit.x[6])),'height':float(np.exp(fit.x[7])),'quad_world':vertices.tolist(),
            'fit_error_px':errors.tolist(),'triangulation':initial_report,'held_out':checks,
            'inferred_corners':sorted(set(range(4))-set(indices)),'source_fit_pass':training_pass and held_pass,
            'approved':False,'scope':__doc__.strip()}
    if not training_pass or not held_pass:
        failure=ValueError('Rigid rectangle exceeds source pixel uncertainty' if not training_pass else 'Held-out panel corners exceed pixel uncertainty')
        failure.proposal=result
        raise failure
    return result


def run(session,config,output):
    config=Path(config).resolve();out=Path(output).resolve();q=json.loads(config.read_text())
    if out.exists() or not out.is_relative_to(session.dir.resolve()) or (session/'qa/independent_freeze.json').exists():
        raise ValueError('New unfrozen session-local panel proposal required')
    definitions={e['id'] for e in session.config('environment.json')['parameters']['entities']}
    if q['entity'] not in definitions:raise ValueError('Explicit background panel required')
    graph=session.scene_graph
    if q['entity'] in {graph['task']['support'],*graph.get('dynamic_objects',{})}:raise ValueError('Cannot fit a frozen task/support')
    camera=session/'parsing/posed_frames/cameras.npz';C=np.load(camera)
    frames={*q['frames'],*[r['frame'] for r in q['held_out']]}
    if q['camera_sha256']!=digest(camera) or set(q['source_sha256'])!=frames:raise ValueError('Bind every source camera and image')
    from calibrated_scene_reconstruction.scene.background import structure_fit
    out.parent.mkdir(parents=True,exist_ok=True);implementation=out.with_name(out.stem+'_solver.py');dependency=out.with_name(out.stem+'_triangulation.py')
    if implementation.exists() or dependency.exists():raise ValueError('Use new immutable implementation snapshots')
    implementation.write_bytes(Path(__file__).read_bytes());dependency.write_bytes(Path(structure_fit.__file__).read_bytes())
    inputs={str(p.resolve()):digest(p) for p in [config,camera,implementation,dependency]}
    for frame in frames:
        source=session/'splat/data/images'/frame
        if Path(frame).name!=frame or digest(source)!=q['source_sha256'][frame]:raise ValueError('Registered source image changed')
        inputs[str(source.resolve())]=digest(source)
    failure=None
    try:result=solve(C,q['frames'],q['pixels'],q['corner_indices'],q['held_out'],q.get('error_px',6.))
    except ValueError as error:
        if not hasattr(error,'proposal'):raise
        failure=error;result=error.proposal
    result.update(entity=q['entity'],part=q.get('part'),input_sha256=inputs,protected_task_state=protected_state(session),
                  implementation_snapshot=str(implementation),triangulation_snapshot=str(dependency))
    if any(digest(p)!=sha for p,sha in inputs.items()):raise ValueError('Panel source evidence changed')
    out.write_text(json.dumps(result,indent=2)+'\n')
    if failure:raise failure
    return result


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['session','config']:p.add_argument(name)
    p.add_argument('--output',required=True);a=p.parse_args();r=run(Session(a.session),a.config,a.output)
    print('Unapproved multi-view panel:',r['entity'],r['width'],r['height'])


if __name__=='__main__':main()
