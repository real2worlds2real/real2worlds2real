"""Background-only PBR colour proposals measured with renderer probes.

Probe scenes never replace the delivered scene. A proposed reflectance/gain is
not measured albedo and still requires a new full-scene source/free-view render.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from calibrated_scene_reconstruction.scene.review.environment import digest,protected_state


def probe_values(cfg):
    mode = cfg.get('probe_mode', 'relative')
    if mode not in ('relative', 'absolute'):
        raise ValueError('Use relative multipliers or absolute constant-PBR colour probes')
    scales = cfg.get('scales', [.015625, .0625, .25, 1., 4.] if mode == 'relative' else [.005, .02, .08, .3, .9])
    maximum = 8. if mode == 'relative' else .95
    if (not 3 <= len(scales) <= 8 or any(isinstance(x, bool) or not isinstance(x, (int, float))
            or not .001 <= x <= maximum for x in scales) or np.any(np.diff(scales) <= 0)):
        raise ValueError('Use3..8 ordered bounded probe values; absolute PBR colours stay within0.001..0.95')
    bindings = cfg.get('bindings')
    if not isinstance(bindings, list) or not bindings or any(not row.get('parameter') or not row.get('source_entity')
            or not row.get('material', '').startswith('/World/Environment/') for row in bindings):
        raise ValueError('Each background colour binding needs its material, source entity and target parameter')
    return mode, scales


def colour_scale(scales,curves,targets):
    """Fit positive RGB multipliers to observed renderer response, not gamma."""
    scales=np.asarray(scales,float);curves=np.asarray(curves,float);targets=np.asarray(targets,float)
    if (len(scales)<3 or np.any(scales<=0) or np.any(np.diff(scales)<=0) or curves.ndim!=3
            or curves.shape[0]!=len(scales) or curves.shape[2]!=3 or targets.shape!=curves.shape[1:]
            or not np.isfinite(curves).all() or not np.isfinite(targets).all()):raise ValueError('Finite ordered probe scales and per-view RGB required')
    grid=np.linspace(np.log(scales[0]),np.log(scales[-1]),256);result=[];errors=[]
    for channel in range(3):
        predictions=np.array([np.interp(grid,np.log(scales),np.maximum.accumulate(curves[:,i,channel])) for i in range(curves.shape[1])])
        error=abs(predictions-targets[:,channel,None]).mean(0);i=error.argmin();result.append(float(np.exp(grid[i])));errors.append(float(error[i]))
    return result,errors


def prepare(session,usd,cameras,config,output):
    from pxr import Sdf,UsdShade
    from calibrated_scene_reconstruction.scene.review.ablation import copy_stage
    out=Path(output).resolve();config=Path(config).resolve();usd=Path(usd).resolve();cameras=Path(cameras).resolve();cfg=json.loads(config.read_text())
    if out.exists() or not out.is_relative_to(session.dir.resolve()) or (session/'qa/independent_freeze.json').exists():raise ValueError('Use an unfrozen background derivative')
    mode, scales = probe_values(cfg)
    out.mkdir(parents=True);probes=[];texture_inputs={}
    for index,scale in enumerate(scales):
        path=out/f'probe_{index}.usda';stage=copy_stage(usd,path,'material_probe')
        for row in cfg['bindings']:
            if not row['material'].startswith('/World/Environment/'):raise ValueError('Background material only; frozen materials are protected')
            mat=UsdShade.Material(stage.GetPrimAtPath(row['material']));shader=mat.ComputeSurfaceSource()[0]
            if not shader or shader.GetIdAttr().Get()!='UsdPreviewSurface':raise ValueError('Explicit PBR surface required')
            colour=shader.GetInput('diffuseColor');connection=colour.GetConnectedSource()
            if connection:
                if mode == 'absolute':raise ValueError('Absolute PBR colour probes require a constant material; preserve texture detail with relative probes')
                texture=UsdShade.Shader(connection[0].GetPrim())
                if texture.GetIdAttr().Get()!='UsdUVTexture':raise ValueError('Image-based PBR colour required')
                source=texture.GetInput('file').Get();source=Path(source.resolvedPath or source.path)
                if not source.is_file():raise ValueError('Unpacked local PBR image required for probe baking')
                image=cv2.imread(str(source),cv2.IMREAD_UNCHANGED)
                if image is None or image.dtype!=np.uint8 or image.ndim!=3:raise ValueError('sRGB8 PBR colour image required')
                rgb=image[:,:,:3].astype(float)/255;linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
                linear=np.clip(linear*scale,0,1);rgb=np.where(linear<=.0031308,12.92*linear,1.055*linear**(1/2.4)-.055)
                baked=out/(f'probe_{index}_'+Path(row['material']).name+'.png');cv2.imwrite(str(baked),np.round(rgb*255).astype('uint8'))
                texture.GetInput('file').Set(Sdf.AssetPath(str(baked)));texture_inputs[str(source)]=digest(source);texture_inputs[str(baked)]=digest(baked)
            else:colour.Set((float(scale),)*3 if mode == 'absolute' else tuple(np.clip(np.asarray(colour.Get())*scale,0,1).tolist()))
        stage.GetRootLayer().Save();probes.append({'scale':scale,'usd':str(path),'output':str(out/f'render_{index}')})
    record={'version':1,'probes':probes,'probe_mode':mode,'config':str(config),'camera_manifest':str(cameras),'protected_task_state':protected_state(session),
            'input_sha256':{**texture_inputs,**{str(p):digest(p) for p in [usd,cameras,config,Path(__file__).resolve()]}},'scope':'Diagnostic only; positive colour hypotheses from fixed scene lighting. No true sizes, geometry changes or robot operation.'}
    (out/'job.json').write_text(json.dumps(record,indent=2)+'\n');return record


def fit(session,folder):
    folder=Path(folder).resolve()
    if not folder.is_relative_to(session.dir.resolve()) or (session/'qa/independent_freeze.json').exists():raise ValueError('Use an unfrozen background derivative')
    job=json.loads((folder/'job.json').read_text());cfg=json.loads(Path(job['config']).read_text())
    if job['protected_task_state']!=protected_state(session):raise ValueError('Protected scene changed during material probes')
    inputs={**job['input_sha256'],str((folder/'job.json').resolve()):digest(folder/'job.json')};observations=[]
    for p in cfg['observations']:
        p=Path(p).resolve();inputs[str(p)]=digest(p);observations+=json.loads(p.read_text())['records']
    rendered=[]
    for probe in job['probes']:
        path=Path(probe['output'])/'review.json';q=json.loads(path.read_text())
        if (q.get('diagnostic')!='material_probe' or q.get('static_scene_preserved')is not True
                or q.get('camera_manifest')!=job['camera_manifest'] or q.get('usd')!=probe['usd']
                or q['input_sha256'].get(probe['usd'])!=digest(probe['usd'])
                or q['input_sha256'].get(job['camera_manifest'])!=digest(job['camera_manifest'])):raise ValueError('Fixed diagnostic material renders required')
        if rendered and q.get('effective_photometry')!=photometry:raise ValueError('Probe photometry changed')
        photometry=q.get('effective_photometry')
        inputs[str(path)]=digest(path);inputs[probe['usd']]=digest(probe['usd']);rendered.append({Path(r['source']).name:r for r in q['views']})
    results=[]
    for binding in cfg['bindings']:
        samples=[];frames=[];sources=[];curves=[[] for _ in rendered];seen=set()
        for row in observations:
            frame=row['frame']
            if row['entity']!=binding['source_entity'] or not row.get('mask') or frame in seen or any(frame not in r for r in rendered):continue
            for key in ['source','mask']:
                if digest(row[key])!=row[key+'_sha256']:raise ValueError('Material source observation changed')
                inputs[row[key]]=row[key+'_sha256']
            images=[cv2.imread(r[frame]['renders'][0]) for r in rendered];h,w=images[0].shape[:2]
            mask=cv2.resize(cv2.imread(row['mask'],0),(w,h),interpolation=cv2.INTER_NEAREST)>128;mask=cv2.erode(mask.astype('uint8'),np.ones((9,9),'uint8'))>0
            # Reject clipped photographic pixels and thin semantic boundaries;
            # source and candidate still need final geometry-aware visual QA.
            raw=cv2.resize(cv2.imread(row['source']),(w,h),interpolation=cv2.INTER_AREA);mask&=(raw.max(2)<250)&(raw.min(2)>8)
            if mask.sum()<128:continue
            seen.add(frame);frames.append(frame);sources.append(np.median(raw[mask][:,::-1],axis=0)/255);samples.append(int(mask.sum()))
            for k,(image,render) in enumerate(zip(images,rendered)):
                curves[k].append(np.median(image[mask][:,::-1],axis=0)/255);path=render[frame]['renders'][0];inputs[path]=digest(path)
        if frames:
            dynamic=np.ptp(np.asarray(curves),axis=0).mean(1)>.025
            frames=[f for f,ok in zip(frames,dynamic) if ok];sources=[v for v,ok in zip(sources,dynamic) if ok];samples=[v for v,ok in zip(samples,dynamic) if ok]
            curves=[[v for v,ok in zip(probe,dynamic) if ok] for probe in curves]
        if len(frames)<2:
            results.append({'material':binding['material'],'approved':False,'reason':'Need at least two usable source views','frames':frames});continue
        gain,error=colour_scale([p['scale'] for p in job['probes']],curves,sources)
        results.append({'material':binding['material'],'parameter':binding['parameter'],
                        ('reflectance_rgb' if job.get('probe_mode') == 'absolute' else 'gain_rgb'):gain,
                        'probe_boundary_hit':any(np.isclose(value, job['probes'][edge]['scale']) for value in gain for edge in [0, -1]),
                        'mean_absolute_rgb_error':error,'frames':frames,'pixels':samples,
                        'source_median_rgb':np.asarray(sources).tolist(),'probe_median_rgb':np.asarray(curves).tolist(),'approved':False})
    for p,sha in inputs.items():
        if digest(p)!=sha:raise ValueError('Material-fit evidence changed')
    q={'version':1,'results':results,'input_sha256':inputs,'protected_task_state':protected_state(session),
       'scope':'Unapproved colour/gain proposals; not measured albedo. Reject clipped reflectance and rerender the full hybrid before adoption.'}
    (folder/'proposals.json').write_text(json.dumps(q,indent=2)+'\n');return q


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session');sub=p.add_subparsers(dest='step',required=True)
    a=sub.add_parser('prepare');a.add_argument('usd');a.add_argument('cameras');a.add_argument('config');a.add_argument('--output',required=True)
    b=sub.add_parser('fit');b.add_argument('folder');args=p.parse_args();session=Session(args.session)
    if args.step=='prepare':q=prepare(session,args.usd,args.cameras,args.config,args.output);print('Material probes',len(q['probes']))
    else:q=fit(session,args.folder);print('Unapproved material proposals',len(q['results']))


if __name__=='__main__':main()
