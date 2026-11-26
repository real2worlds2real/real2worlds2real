"""Vertical wall proposals from matched source landmarks and visible edges.

No wall dimensions are supplied. Verticality, device-to-wall parallelism and
unseen extensions remain agent hypotheses. These proposals cannot approve a
scene or replace physical and source/free-view checks.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.scene.background.structure_fit import triangulate
from calibrated_scene_reconstruction.scene.review.environment import digest,protected_state
from calibrated_scene_reconstruction.scene.session import Session


def solve(cameras,frames,pixels,boundary_frame,boundary_pixels,held_out,fit_error_px=3.,held_out_error_px=6.):
    points,report=triangulate(cameras,frames,pixels,max_error_px=fit_error_px)
    if len(points)<4:raise ValueError('At least four matched surface landmarks required')
    centre=points.mean(0);_,singular,V=np.linalg.svd(points-centre,full_matrices=False)
    if singular[0]<=1e-8 or singular[1]<singular[0]*.02:raise ValueError('Surface landmarks are nearly collinear')
    normal=V[-1]
    if abs(normal[2])>np.sin(np.radians(10)):raise ValueError('Observed feature plane is not close to vertical')
    normal[2]=0;normal/=np.linalg.norm(normal)
    names=list(map(str,cameras['names']));indices=[names.index(f) for f in frames]
    if (cameras['centre'][indices].mean(0)-centre)@normal<0:normal=-normal
    residual=float(np.sqrt(np.mean(((points-centre)@normal)**2)))
    if residual>.02:raise ValueError('Observed landmarks do not support one vertical plane')
    if not 0<held_out_error_px<=20:raise ValueError('Bounded held-out pixel error required')
    seen=set(frames);checks=[]
    for row in held_out:
        frame=row['frame'];observed=np.asarray(row['pixels'],float)
        if frame in seen or observed.shape!=(len(points),2) or not np.isfinite(observed).all():
            raise ValueError('Distinct held-out observations of the same landmarks required')
        seen.add(frame);i=names.index(frame);pc=(points-cameras['centre'][i])@cameras['R_world_cam'][i]
        if np.any(pc[:,2]<=0):raise ValueError('Held-out landmarks are behind the camera')
        uv=pc@cameras['K'][i].T;error=np.linalg.norm(uv[:,:2]/uv[:,2,None]-observed,axis=1)
        if error.max()>held_out_error_px:raise ValueError('Held-out landmark reprojection exceeds pixel uncertainty')
        checks.append({'frame':frame,'error_px':error.tolist()})
    edge=np.asarray(boundary_pixels,float)
    if edge.shape!=(2,2) or not np.isfinite(edge).all():raise ValueError('Two observed wall-edge pixels required')
    i=names.index(boundary_frame);C=cameras['centre'][i]
    rays=np.c_[edge,np.ones(2)]@np.linalg.inv(cameras['K'][i]).T@cameras['R_world_cam'][i].T
    denominator=rays@normal
    if np.any(abs(denominator)/np.linalg.norm(rays,axis=1)<np.sin(np.radians(5))):
        raise ValueError('Wall-edge rays are too grazing for a stable extent')
    distance=((centre-C)@normal)/denominator
    if np.any(distance<=0):raise ValueError('Wall edges intersect behind the source camera')
    edges=C+rays*distance[:,None];tangent=edges[1]-edges[0];tangent[2]=0;width=float(np.linalg.norm(tangent))
    if width<=1e-6:raise ValueError('Observed wall edges do not span a face')
    tangent/=width
    return {'centre_xy':edges.mean(0)[:2].tolist(),'width':width,
            'yaw_deg':float(np.degrees(np.arctan2(tangent[1],tangent[0]))),'normal_xy':normal[:2].tolist(),
            'landmark_points_world':points.tolist(),'edge_points_world':edges.tolist(),
            'fit':report,'held_out':checks,'vertical_plane_rms_m':residual,
            'fit_error_limit_px':fit_error_px,'held_out_error_limit_px':held_out_error_px,
            'approved':False,'scope':'Visible-edge extent and a vertical-plane hypothesis. Surface offset, wall height and unseen continuation are not measured.'}


def run(session,config,output):
    if (session/'qa/independent_freeze.json').exists():raise ValueError('Frozen delivery: use a derivative')
    output=Path(output).resolve()
    if output.exists() or not output.is_relative_to(session.dir.resolve()):raise ValueError('Use a new session-local plane proposal')
    config=Path(config).resolve();q=json.loads(config.read_text())
    if not q.get('entity') or not q.get('surface_relation','').strip() or len(q.get('held_out',[]))<2:
        raise ValueError('Name the entity, surface-relation hypothesis and at least two held-out views')
    camera=session/'parsing/posed_frames/cameras.npz';z=np.load(camera)
    from calibrated_scene_reconstruction.scene.background import structure_fit
    paths=[config,camera,Path(__file__),Path(structure_fit.__file__)]
    frames={*q['frames'],q['boundary_frame'],*[r['frame'] for r in q['held_out']]}
    if q.get('camera_sha256')!=digest(camera) or set(q.get('source_sha256',{}))!=frames:
        raise ValueError('Bind the calibrated cameras and all annotated source images')
    for frame in sorted(frames):
        if frame not in set(map(str,z['names'])):raise ValueError('Use registered source frames')
        image=session/'splat/data/images'/frame
        if digest(image)!=q['source_sha256'][frame]:raise ValueError('Annotated source image changed')
        paths.append(image)
    inputs={str(p.resolve()):digest(p) for p in paths}
    result=solve(z,q['frames'],q['pixels'],q['boundary_frame'],q['boundary_pixels'],q['held_out'])
    result.update(entity=q['entity'],surface_relation=q['surface_relation'],
                  input_sha256=inputs,protected_task_state=protected_state(session))
    if any(digest(p)!=h for p,h in inputs.items()):raise ValueError('Landmark plane input changed during fitting')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session');p.add_argument('config');p.add_argument('--output',required=True)
    a=p.parse_args();r=run(Session(a.session),a.config,a.output);print('Unapproved landmark wall:',r['entity'],r['centre_xy'],r['yaw_deg'])


if __name__=='__main__':main()
