"""Fit only observed colour channels from immutable background renderer probes.

A clipped blue channel must not erase valid red/green observations. Unobserved
channels keep unit gain and are explicitly unidentified, never recovered albedo.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from calibrated_scene_reconstruction.scene.background.material_fit import colour_scale
from calibrated_scene_reconstruction.scene.review.environment import digest, protected_state


def fit_channels(scales, curves, targets, usable):
    scales=np.asarray(scales,float);curves=np.asarray(curves,float)
    targets=np.asarray(targets,float);usable=np.asarray(usable,bool)
    if (scales.ndim!=1 or not 3<=len(scales)<=8 or not np.isfinite(scales).all()
            or np.any(scales<=0) or np.any(np.diff(scales)<=0)
            or curves.ndim!=3 or curves.shape[0]!=len(scales) or curves.shape[2]!=3
            or targets.shape!=curves.shape[1:] or usable.shape!=targets.shape
            or not np.isfinite(curves).all() or not np.isfinite(targets).all()):
        raise ValueError('Ordered probe scales and finite aligned per-channel observations required')
    gains=[];errors=[];counts=[];bracketed_counts=[];boundary=[];identified=[]
    for c in range(3):
        valid=usable[:,c] & (np.ptp(curves[:,:,c],axis=0)>.025)
        counts.append(int(valid.sum()))
        bracketed=valid & (targets[:,c]>=curves[:,:,c].min(0)) & (targets[:,c]<=curves[:,:,c].max(0))
        bracketed_counts.append(int(bracketed.sum()))
        if valid.sum()<2:
            gains.append(1.);errors.append(None);boundary.append(False);identified.append(False);continue
        # Reuse the calibrated monotone response inversion, independently for
        # each observed channel. No clipped channel supplies an equality target.
        response=np.repeat(curves[:,valid,c,None],3,axis=2)
        target=np.repeat(targets[valid,c,None],3,axis=1)
        gain,error=colour_scale(scales,response,target)
        hit=bool(gain[0]<=scales[0]*1.001 or gain[0]>=scales[-1]/1.001)
        okay=bracketed.sum()>=2 and not hit
        gains.append(gain[0] if okay else 1.);errors.append(error[0])
        boundary.append(hit);identified.append(bool(okay))
    return {'gain_rgb':gains,'mean_absolute_rgb_error':errors,
            'identified_rgb':identified,'usable_views_rgb':counts,
            'bracketed_views_rgb':bracketed_counts,'probe_boundary_hit_rgb':boundary}


def fit(session, folder, output, observations_override=None):
    folder=Path(folder).resolve();output=Path(output).resolve()
    if (not folder.is_relative_to(session.dir.resolve()) or not output.is_relative_to(session.dir.resolve())
            or output.exists() or (session/'qa/independent_freeze.json').exists()):
        raise ValueError('New analysis in an unfrozen background session required')
    path=folder/'job.json';job=json.loads(path.read_text());config=Path(job['config'])
    if job['protected_task_state']!=protected_state(session):raise ValueError('Protected task changed')
    inputs={**job['input_sha256'],str(path):digest(path),str(Path(__file__).resolve()):digest(__file__)}
    for p,sha in inputs.items():
        if digest(p)!=sha:raise ValueError('Renderer probe input changed: '+p)
    cfg=json.loads(config.read_text());renders=[];photometry=None
    for probe in job['probes']:
        p=Path(probe['output'])/'review.json';r=json.loads(p.read_text())
        if (r.get('diagnostic')!='material_probe' or r.get('static_scene_preserved') is not True
                or r.get('camera_manifest')!=job['camera_manifest'] or r.get('usd')!=probe['usd']
                or r['input_sha256'].get(probe['usd'])!=digest(probe['usd'])
                or r['input_sha256'].get(job['camera_manifest'])!=digest(job['camera_manifest'])):
            raise ValueError('Fixed actual renderer probes required')
        if renders and r.get('effective_photometry')!=photometry:raise ValueError('Probe photometry changed')
        photometry=r.get('effective_photometry');inputs[str(p)]=digest(p)
        inputs[probe['usd']]=digest(probe['usd'])
        renders.append({Path(v['source']).name:v for v in r['views']})
    observations=[]
    observation_paths=observations_override if observations_override is not None else cfg['observations']
    for name in observation_paths:
        p=Path(name).resolve();inputs[str(p)]=digest(p);observations+=json.loads(p.read_text())['records']
    results=[]
    for binding in cfg['bindings']:
        if not binding['material'].startswith('/World/Environment/'):
            raise ValueError('Only background materials may be fitted')
        curves=[[] for _ in renders];targets=[];usable=[];frames=[];pixels=[];seen=set()
        for row in observations:
            frame=row['frame']
            if row['entity']!=binding['source_entity'] or frame in seen or any(frame not in r for r in renders):continue
            if Path(row['source']).resolve()!=(session/'splat/data/images'/frame).resolve():
                raise ValueError('Source annotations must use the registered session pictures')
            for key in ['source','mask']:
                if digest(row[key])!=row[key+'_sha256']:raise ValueError('Source colour observation changed')
                inputs[row[key]]=row[key+'_sha256']
            images=[cv2.imread(r[frame]['renders'][0])[:,:,::-1]/255. for r in renders]
            h,w=images[0].shape[:2]
            if any(im.shape!=images[0].shape for im in images):raise ValueError('Aligned renderer probes required')
            mask=cv2.resize(cv2.imread(row['mask'],0),(w,h),interpolation=cv2.INTER_NEAREST)>128
            mask=cv2.erode(mask.astype('uint8'),np.ones((9,9),'uint8'))>0
            raw=cv2.resize(cv2.imread(row['source']),(w,h),interpolation=cv2.INTER_AREA)[:,:,::-1]
            valid=mask[:,:,None] & (raw>8) & (raw<250)
            count=valid.sum(axis=(0,1));good=count>=128
            targets.append([float(np.median(raw[:,:,c][valid[:,:,c]])/255) if good[c] else 0. for c in range(3)])
            for k,image in enumerate(images):
                curves[k].append([float(np.median(image[:,:,c][valid[:,:,c]])) if good[c] else 0. for c in range(3)])
                p=renders[k][frame]['renders'][0];inputs[p]=digest(p)
            seen.add(frame);frames.append(frame);usable.append(good);pixels.append(count.tolist())
        if not frames:
            results.append({'material':binding['material'],'approved':False,'reason':'No source views'});continue
        q=fit_channels([p['scale'] for p in job['probes']],curves,targets,usable)
        results.append({**q,'material':binding['material'],'parameter':binding['parameter'],
                        'frames':frames,'unclipped_pixels_rgb':pixels,'source_median_rgb':targets,'approved':False})
    if any(digest(p)!=sha for p,sha in inputs.items()):raise ValueError('Colour analysis input changed')
    result={'version':1,'results':results,'input_sha256':inputs,'source_observations':list(map(str,observation_paths)),
            'protected_task_state':protected_state(session),
            'scope':'Unapproved background colour-gain proposals. Two responsive, unclipped, bracketed source views per identified channel. Boundary or unidentified gains remain unit gain. No albedo recovery, geometry, scale or whole-scene appearance approval.'}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session');p.add_argument('probes');p.add_argument('--output',required=True)
    p.add_argument('--observations',nargs='+',help='New source annotations for the same immutable renderer probes; original probe inputs stay verified')
    a=p.parse_args();print(json.dumps(fit(Session(a.session),a.probes,a.output,a.observations)['results'],indent=2))


if __name__=='__main__':main()
