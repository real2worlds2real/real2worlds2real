"""Fit a floor-supported box to corresponding visible source corners.

Silhouettes alone can fit a nearly square box with the wrong face orientation,
which then projects two photographed faces onto one native face. Corner IDs
are source annotations, never supplied dimensions; unseen corners remain an
explicit rectangular hypothesis. This does not approve physical or appearance
quality.
"""
import argparse
import itertools
import inspect
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from calibrated_scene_reconstruction.scene.background.refine import rotation
from calibrated_scene_reconstruction.scene.review.environment import digest,protected_state

CORNERS=np.array(list(itertools.product([-.5,.5],[-.5,.5],[0.,1.])))


def solve(cameras,observations,initial,floor,error_px=6.):
    names=list(map(str,cameras['names']));data=[];seen=set()
    if not np.isfinite(floor) or not 1<=error_px<=12:raise ValueError('Finite support height and explicit bounded pixel uncertainty required')
    for row in observations:
        ids=row['corner_indices'];pixels=np.asarray(row['pixels'],float)
        if (not isinstance(ids,list) or not ids or len(set(ids))!=len(ids)
                or any(type(i)is not int or not 0<=i<8 for i in ids)
                or pixels.shape!=(len(ids),2) or not np.isfinite(pixels).all()
                or row['frame'] in seen or row['frame'] not in names):
            raise ValueError('Distinct source frames and corresponding visible box corners required')
        seen.add(row['frame']);i=names.index(row['frame']);w,h=cameras['size'][i]
        if np.any(pixels<0) or np.any(pixels[:,0]>=w) or np.any(pixels[:,1]>=h):raise ValueError('Visible source corners must be inside the raster')
        data.append((i,ids,pixels))
    if len(data)<2 or max(np.linalg.norm(cameras['centre'][a[0]]-cameras['centre'][b[0]]) for a in data for b in data)<.05:
        raise ValueError('Independent source views required')
    observed=sorted({k for _,ids,_ in data for k in ids});known=CORNERS[observed]
    if len(observed)<3 or np.linalg.matrix_rank(known[:,:2]-known[:,:2].mean(0))<2 or not np.any(known[:,2]>0):
        raise ValueError('Visible corners must constrain both body widths and its upper extent')
    origin=np.asarray(initial['position'],float);size=np.asarray(initial['size'],float);yaw=np.radians(initial['yaw_deg'])
    if origin.shape!=(3,) or size.shape!=(3,) or not np.isfinite(origin).all() or not np.isfinite(size).all() or not np.isfinite(yaw) or np.any(size<=0):raise ValueError('Finite positive box hypothesis required')
    def vertices(x):return CORNERS*(size*np.exp(x[3:]))@rotation(yaw+x[2]).T+[origin[0]+x[0],origin[1]+x[1],floor]
    def residual(x):
        V=vertices(x);values=[]
        for i,ids,pixels in data:
            pc=(V[ids]-cameras['centre'][i])@cameras['R_world_cam'][i]
            if np.any(pc[:,2]<=.05):return np.full(sum(len(ids)*2 for _,ids,_ in data),1e6)
            q=pc@cameras['K'][i].T;values.extend((q[:,:2]/q[:,2,None]-pixels).ravel())
        return values
    candidates=[]
    for seed in np.linspace(-np.pi,np.pi,8,endpoint=False):
        start=np.zeros(6);start[2]=seed
        lower=[-.7,-.7,seed-np.pi/4,*([np.log(.35)]*3)]
        upper=[.7,.7,seed+np.pi/4,*([np.log(1.8)]*3)]
        fit=least_squares(residual,start,bounds=(lower,upper),loss='soft_l1',f_scale=2.,max_nfev=600)
        candidates.append(fit)
    fit=min(candidates,key=lambda f:np.mean(np.square(residual(f.x))));error=np.linalg.norm(np.asarray(residual(fit.x)).reshape(-1,2),axis=1)
    checks=[]
    for row,(i,ids,pixels) in zip(observations,data):
        pc=(vertices(fit.x)[ids]-cameras['centre'][i])@cameras['R_world_cam'][i];q=pc@cameras['K'][i].T
        checks.append({'frame':row['frame'],'corner_indices':ids,'observed':pixels.tolist(),'projected':(q[:,:2]/q[:,2,None]).tolist()})
    result={'position':[float(origin[0]+fit.x[0]),float(origin[1]+fit.x[1]),float(floor)],
            'size':(size*np.exp(fit.x[3:])).tolist(),'yaw_deg':float(np.degrees(yaw+fit.x[2])),
            'corner_world':vertices(fit.x).tolist(),'error_px':error.tolist(),'bound_hit':bool(np.any(fit.active_mask)),
            'observed_corner_indices':observed,'inferred_corner_indices':sorted(set(range(8))-set(observed)),
            'source_checks':checks,'source_fit_pass':bool(np.isfinite(fit.x).all() and error.max()<=error_px and not np.any(fit.active_mask)),
            'approved':False,'scope':__doc__.strip()}
    if not result['source_fit_pass']:
        failure=ValueError('Box corners exceed pixel uncertainty or fitting bounds: errors_px='+str(error.round(2).tolist())+'; active_bounds='+str(fit.active_mask.tolist()))
        failure.proposal=result
        raise failure
    return result


def run(session,config,output):
    p=Path(config).resolve();out=Path(output).resolve();cfg=json.loads(p.read_text())
    if out.exists() or not out.is_relative_to(session.dir.resolve()) or (session/'qa/independent_freeze.json').exists():raise ValueError('New unfrozen source proposal required')
    environment=session.config('environment.json');matches=[e for e in environment['parameters']['entities'] if e['id']==cfg['entity'] and e['kind'] in ('box','open_box')]
    graph=session.scene_graph
    if len(matches)!=1 or cfg['entity'] in {graph['task']['support'],*graph.get('dynamic_objects',{})}:raise ValueError('Existing background box required')
    if abs(matches[0]['position'][2]-environment['physics']['ground_height'])>.03:raise ValueError('This fit requires an existing floor-supported box; do not move a supported box down to the floor')
    snapshot=out.with_name(out.stem+'_environment.json')
    if snapshot.exists():raise ValueError('Use a new immutable hypothesis snapshot')
    snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes((session.config_dir/'environment.json').read_bytes())
    implementation=out.with_name(out.stem+'_solver.py');dependency=out.with_name(out.stem+'_rotation.py')
    if implementation.exists() or dependency.exists():raise ValueError('Use new immutable implementation snapshots')
    implementation.write_bytes(Path(__file__).read_bytes());dependency.write_text('import numpy as np\n\n'+inspect.getsource(rotation))
    camera=session/'parsing/posed_frames/cameras.npz';inputs={str(path):digest(path) for path in [p,camera.resolve(),snapshot,implementation,dependency]}
    if cfg['camera_sha256']!=digest(camera):raise ValueError('Exact source cameras required')
    for row in cfg['observations']:
        source=session/'splat/data/images'/row['frame']
        if Path(row['frame']).name!=row['frame'] or digest(source)!=cfg['source_sha256'].get(row['frame']):raise ValueError('Source annotation image changed')
        inputs[str(source.resolve())]=digest(source)
    failure=None
    try:result=solve(np.load(camera),cfg['observations'],matches[0],environment['physics']['ground_height'],cfg.get('error_px',6.))
    except ValueError as error:
        if not hasattr(error,'proposal'):raise
        failure=error;result=error.proposal
    result.update(entity=cfg['entity'],input_sha256=inputs,protected_task_state=protected_state(session),
                  implementation_snapshot=str(implementation),rotation_snapshot=str(dependency))
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
    if failure:raise failure
    return result


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session');p.add_argument('config');p.add_argument('--output',required=True);a=p.parse_args()
    q=run(Session(a.session),a.config,a.output);print('Unapproved source-corner box fit; maximum error',max(q['error_px']))


if __name__=='__main__':main()
