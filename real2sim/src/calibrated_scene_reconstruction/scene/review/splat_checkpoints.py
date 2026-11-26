"""Read-only checkpoint comparison at fixed captured/novel review cameras.

Task geometry is unchanged. The same existing task-zone exclusion is applied
in memory to every checkpoint; all images are actual3DGUT renders, not generated
or sharpened pictures. Novel views have no ground-truth image.
"""
import argparse
import gc
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw
import torch

from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.review.cameras import view_name
from calibrated_scene_reconstruction.scene.background.splat.config import settings,task_zone
from calibrated_scene_reconstruction.scene.background.splat.geometry import world_of_model


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    from threedgrut.datasets.protocols import Batch
    from threedgrut.model.model import MixtureOfGaussians
    from threedgrut.utils.render import apply_background
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session')
    parser.add_argument('--output',default='qa/background_diagnostics/fixed_checkpoint_comparison')
    parser.add_argument('--candidates',nargs='+',required=True,help='Saved background round labels')
    parser.add_argument('--cameras',default='qa/background_diagnostics/cameras.json')
    args=parser.parse_args();s=Session(args.session);out=s/args.output
    if out.exists():raise ValueError('Preserve previous comparisons; choose a new output directory')
    if (s/'qa/independent_freeze.json').exists():raise ValueError('Use a background derivative for new review artifacts')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]+',label) for label in args.candidates):raise ValueError('Use simple saved round labels')
    cameras=s/args.cameras;manifest=json.loads(cameras.read_text())
    fingerprints={str(cameras):digest(cameras),**manifest.get('input_sha256',{})}
    for path in [s/'splat/align.json',s/'splat/train.json',s.config_dir/'splat.json']:
        if path.exists():fingerprints[str(path)]=digest(path)
    for path,expected in fingerprints.items():
        if digest(path)!=expected:raise ValueError('Review preparation input changed: '+path)
    for anchor in manifest['anchors']:
        if digest(anchor['source'])!=anchor['source_sha256']:raise ValueError('Fixed review source changed')
        fingerprints[anchor['source']]=anchor['source_sha256']
    offsets=[v['offset_m'] for v in manifest['anchors'][0]['views']]
    if offsets.count(0.)!=1 or any([v['offset_m'] for v in a['views']]!=offsets for a in manifest['anchors']):
        raise ValueError('Use the same unique captured/novel offsets for each anchor')
    out.mkdir(parents=True);W,H=manifest['width'],manifest['height']
    Wm,_=world_of_model(s/'splat',s);Mi=np.linalg.inv(Wm)
    scale=float(np.cbrt(np.linalg.det(Mi[:3,:3])));rotation=Mi[:3,:3]/scale
    inside,zone=task_zone(s,settings(s)['crop'])
    rows=[]
    for label in args.candidates:
        meta=s/'background_rounds'/label/'splat/train.json';record=json.loads(meta.read_text())
        checkpoint=s/record['checkpoint'];fingerprints[str(meta)]=digest(meta)
        fingerprints[str(checkpoint)]=digest(checkpoint)
        value=torch.load(checkpoint,map_location='cuda',weights_only=False);conf=value['config']
        method=conf.post_processing.method
        if method not in ('none',None):raise ValueError('Unsupported post-processing needs its exact checkpoint module: '+str(method))
        model=MixtureOfGaussians(conf);model.init_from_checkpoint(value,setup_optimizer=False)
        del value
        model.eval()
        with torch.no_grad():
            positions=model.positions.detach().cpu().numpy().astype(float)
            world=positions@Wm[:3,:3].T+Wm[:3,3]
            excluded=inside(world);model.density[torch.as_tensor(excluded,device='cuda')]=-100.
            model.build_acc()
            print(f'[background.compare] {label}: {model.num_gaussians} Gaussians, {excluded.sum()} task-zone excluded in memory',flush=True)
            directory=out/label;directory.mkdir();images=[]
            for anchor in manifest['anchors']:
                K=np.asarray(anchor['K'],float).copy();K[0]*=W/anchor['source_size'][0];K[1]*=H/anchor['source_size'][1]
                yy,xx=torch.meshgrid(torch.arange(H,device='cuda')+.5,torch.arange(W,device='cuda')+.5,indexing='ij')
                rays=torch.stack([(xx-K[0,2])/K[0,0],(yy-K[1,2])/K[1,1],torch.ones_like(xx)],-1)
                rays=(rays/rays.norm(dim=-1,keepdim=True))[None];origins=torch.zeros_like(rays)
                for view in anchor['views']:
                    T=np.eye(4);T[:3,:3]=rotation@np.asarray(view['R_world_cam']);T[:3,3]=Mi[:3,:3]@view['centre']+Mi[:3,3]
                    batch=Batch(rays_ori=origins,rays_dir=rays,T_to_world=torch.tensor(T,dtype=torch.float32,device='cuda')[None],intrinsics=[float(K[0,0]),float(K[1,1]),float(K[0,2]),float(K[1,2])])
                    rendered=model(batch);rendered=apply_background(model.background,rendered,batch)
                    rgb=rendered['pred_features'][0,...,:3].clamp(0,1).float().cpu().numpy()
                    alpha=rendered['pred_opacity'][0,...,0].float().cpu().numpy()
                    name=view_name(Path(anchor["name"]).stem,view["offset_m"])
                    path=directory/(name+'.png');Image.fromarray(np.rint(rgb*255).astype('uint8')).save(path)
                    np.savez_compressed(directory/(name+'_opacity.npz'),opacity=alpha)
                    images.append({'source':anchor['name'],'offset_m':view['offset_m'],'image':str(path)})
                    print(f'[background.compare] {label} {name}',flush=True)
            rows.append({'label':label,'checkpoint':str(checkpoint),'checkpoint_sha256':fingerprints[str(checkpoint)],
                         'iterations':record['settings']['iterations'],'gaussians':model.num_gaussians,'task_zone_excluded':int(excluded.sum()),'images':images})
        del model,rendered,batch;gc.collect();torch.cuda.empty_cache()
    # The original is paired only with offset0; novel columns remain unpaired.
    panel=Image.new('RGB',(640*(2+len(offsets)),36+len(manifest['anchors'])*390),(28,28,28));draw=ImageDraw.Draw(panel)
    first,last=rows[0],rows[-1]
    headers=['Original captured image',first['label']+' captured']+[last['label']+(' captured' if offset==0 else f' +{offset:.3g}m novel') for offset in offsets]
    for i,header in enumerate(headers):draw.text((i*640+8,8),header,fill='white')
    for j,anchor in enumerate(manifest['anchors']):
        stem=Path(anchor['name']).stem;y=36+j*390
        paths=[Path(anchor['source']),out/args.candidates[0]/(view_name(stem,0.)+'.png')]+[
               out/args.candidates[-1]/(view_name(stem,offset)+'.png') for offset in offsets]
        for i,path in enumerate(paths):panel.paste(Image.open(path).convert('RGB').resize((640,360)),(640*i,y))
        draw.text((8,y+364),anchor['name']+' | background-only rendering; task-zone meshes intentionally excluded | novel views have no GT',fill='white')
    panel.save(out/'original_captured_novel_comparison.jpg',quality=95)
    for path,expected in fingerprints.items():
        if digest(path)!=expected:raise ValueError('Comparison input changed: '+path)
    (out/'comparison.json').write_text(json.dumps({'scope':'read-only; identical existing task-zone crop; frozen task meshes unchanged',
        'novel_ground_truth':False,'input_sha256':fingerprints,'task_zone':zone,'candidates':rows},indent=2)+'\n')
    print(f'[background.compare] saved {out}',flush=True)


if __name__=='__main__':main()
