"""Propose horizontal background supports from capture geometry and source masks.

Use front-visible thin GS surfaces and independently triangulated COLMAP
tracks. Neither Gaussian centres nor a sparse bounding-box percentile alone
establish a physical tabletop. Height and horizontal extent remain hypotheses;
source/free-view and support checks are required after applying a proposal.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData
from scipy.spatial.transform import Rotation

from calibrated_scene_reconstruction.scene.review.environment import digest,protected_state


def horizontal_mode(points,band=.03,minimum=10):
    p=np.asarray(points,float)
    if (p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all() or len(p)<minimum
            or not .005<=band<=.05):raise ValueError('Sufficient finite horizontal-surface samples required')
    z=p[:,2];order=np.argsort(z);values=z[order]
    ends=np.searchsorted(values,values+2*band,side='right');start=int(np.argmax(ends-np.arange(len(z))))
    selected=order[start:ends[start]]
    if len(selected)<minimum:raise ValueError('No coherent horizontal support level')
    centre=float(np.median(z[selected]));consensus=abs(z-centre)<=band
    if consensus.sum()<minimum or np.ptp(p[consensus,:2],axis=0).max()<.1:
        raise ValueError('Support samples lack spatial extent or consensus')
    other=z[~consensus]
    if len(other):
        other=np.sort(other);n=np.searchsorted(other,other+2*band,side='right')-np.arange(len(other))
        if n.max()>=consensus.sum()*.8:raise ValueError('Competing horizontal levels need distinct source patches')
    return centre,consensus


def vote(ids,points,centre,counts,last):
    before=points[ids]-last[ids];after=points[ids]-centre
    cosine=np.einsum('ij,ij->i',before,after)/np.maximum(1e-12,np.linalg.norm(before,axis=1)*np.linalg.norm(after,axis=1))
    independent=(counts[ids]==0)|((np.linalg.norm(last[ids]-centre,axis=1)>.05)
                 &(np.degrees(np.arccos(np.clip(cosine,-1,1)))>=2.))
    selected=ids[independent];counts[selected]+=1;last[selected]=centre


def run(session,context_path,observations,entity,output):
    from calibrated_scene_reconstruction.scene.background import structure_fit
    from calibrated_scene_reconstruction.scene.capture import colmap_points
    out=Path(output).resolve();context_path=Path(context_path).resolve()
    if out.exists() or not out.is_relative_to(session.dir.resolve()) or (session/'qa/independent_freeze.json').exists():
        raise ValueError('New unfrozen background support proposal required')
    native=session.config('environment.json');definitions={e['id']:e for e in native['parameters']['entities']}
    graph=session.scene_graph
    if (entity not in definitions or entity in {graph['task']['support'],*graph.get('dynamic_objects',{})}
            or definitions[entity]['kind'] not in ('desk','cabinet','cabinet_open','shelf')):
        raise ValueError('Explicit background horizontal support required')
    camera=session/'parsing/posed_frames/cameras.npz';C=np.load(camera);names=list(map(str,C['names']))
    context=json.loads(context_path.read_text());cloud=context_path.parent/'context.ply'
    if context['camera_sha256']!=digest(camera) or context['protected_task_state']!=protected_state(session):raise ValueError('Same capture and frozen foreground required')
    v=PlyData.read(str(cloud))['vertex'].data;M=np.asarray(context['matrix']);scale=np.cbrt(np.linalg.det(M[:3,:3]));R=M[:3,:3]/scale
    points=(np.c_[v['x'],v['y'],v['z'],np.ones(len(v))]@M.T)[:,:3]
    logs=np.c_[v['scale_0'],v['scale_1'],v['scale_2']]
    axes=R@Rotation.from_quat(np.c_[v['rot_1'],v['rot_2'],v['rot_3'],v['rot_0']]).as_matrix()
    normal=axes[np.arange(len(v)),:,logs.argmin(1)];ratio=np.exp(np.sort(logs,axis=1)[:,1]-logs.min(1))
    opacity=1/(1+np.exp(-np.clip(v['opacity'],-80,80)))
    thin=(abs(normal[:,2])>np.cos(np.radians(20)))&(ratio>2)&(opacity>.2)
    sparse,_,_=colmap_points.world_points(session);gs_votes=np.zeros(len(points),int);sparse_votes=np.zeros(len(sparse),int)
    gs_last=np.full_like(points,np.nan);sparse_last=np.full_like(sparse,np.nan)
    paths=[context_path,cloud,camera,session/'splat/align.json',session/'splat/data/sparse/full/points3D.bin',Path(__file__),Path(structure_fit.__file__),Path(colmap_points.__file__)]
    inputs={str(p.resolve()):digest(p) for p in paths};seen=set();centres=[]
    for path in observations:
        path=Path(path).resolve();inputs[str(path)]=digest(path)
        for row in json.loads(path.read_text())['records']:
            if row['entity']!=entity or row['frame'] in seen or not row.get('mask'):continue
            seen.add(row['frame']);i=names.index(row['frame']);centres.append(C['centre'][i])
            for key in ['source','mask']:
                if digest(row[key])!=row[key+'_sha256']:raise ValueError('Support source annotation changed')
                inputs[str(Path(row[key]).resolve())]=row[key+'_sha256']
            if Path(row['source']).resolve()!=(session/'splat/data/images'/row['frame']).resolve():raise ValueError('Use this registered capture')
            mask=cv2.imread(row['mask'],0)
            if mask.shape!=tuple(C['size'][i][::-1]):raise ValueError('Calibrated support mask required')
            ids,uv=structure_fit.visible_indices(points,C['K'][i],C['R_world_cam'][i],C['centre'][i],C['size'][i],opacity)
            vote(ids[mask[uv[:,1],uv[:,0]]>128],points,C['centre'][i],gs_votes,gs_last)
            pc=(sparse-C['centre'][i])@C['R_world_cam'][i];q=pc@C['K'][i].T;uv=q[:,:2]/np.maximum(pc[:,2,None],1e-9);w,h=C['size'][i]
            valid=(pc[:,2]>.05)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h);ids=np.flatnonzero(valid);xy=uv[ids].astype(int)
            vote(ids[mask[xy[:,1],xy[:,0]]>128],sparse,C['centre'][i],sparse_votes,sparse_last)
    if len(seen)<2 or np.linalg.norm(np.asarray(centres)[:,None]-centres,axis=2).max()<.05:raise ValueError('Independent observed support views required')
    candidates=points[thin&(gs_votes>=2)];height,consensus=horizontal_mode(candidates)
    reference=sparse[(sparse_votes>=2)&(abs(sparse[:,2]-height)<=.04)]
    if len(reference)<3:raise ValueError('Horizontal GS needs independent sparse-track support')
    sparse_height=float(np.median(reference[:,2]))
    if abs(sparse_height-height)>.03:raise ValueError('GS and sparse tracks disagree on support height')
    if any(digest(p)!=sha for p,sha in inputs.items()):raise ValueError('Support evidence changed')
    out.mkdir(parents=True);np.savez_compressed(out/'samples.npz',gs=candidates[consensus],sparse=reference)
    result={'entity':entity,'height_m':height,'gs_samples':int(consensus.sum()),'sparse_height_m':sparse_height,
            'sparse_samples':len(reference),'gs_height_mad_m':float(np.median(abs(candidates[consensus,2]-height))),
            'samples':str((out/'samples.npz').resolve()),'samples_sha256':digest(out/'samples.npz'),
            'frames':sorted(seen),'input_sha256':inputs,'protected_task_state':protected_state(session),
            'approved':False,'scope':__doc__.strip()}
    (out/'proposal.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['session','context','entity']:p.add_argument(name)
    p.add_argument('--observations',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    q=run(Session(a.session),a.context,a.observations,a.entity,a.output)
    print('Unapproved support height:',q['height_m'],'GS/sparse samples',q['gs_samples'],q['sparse_samples'])


if __name__=='__main__':main()
