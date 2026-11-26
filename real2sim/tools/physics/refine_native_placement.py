"""Propose one bounded scene translation using source and native initial features."""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from calibrated_scene_reconstruction.physics.process.contract import read,write,digest,require
from calibrated_scene_reconstruction.physics.process.initial_alignment import refine_translation


def run(args):
    root=Path(args.round);p=read(root/'plan.json');cfg=read(p['inputs']['scene_setup']['path'])
    trial=root/'trials'/f'{args.trial:04d}';state=np.load(trial/'native_states.npz');prediction=np.load(trial/'predictions.npz')
    observed=np.load(p['inputs']['observations']['path']);camera=np.load(p['inputs']['process_camera']['path'])
    split=cfg['initial_split'];count=max(*split['fit'],*split['validation'])+1
    cameras=[(camera['K'],camera['dist'],T) for T in camera['T_world_cam'][:count]]
    landmarks={**cfg.get('initial_landmarks',{}),**cfg.get('rigid_landmarks',{})};features=[];skipped=[];used=[]
    def positions(name,points):
        values=[]
        for i in range(count):
            v=state[name+f'.{i:05d}'];R=Rotation.from_quat(v[[4,5,6,3]]).as_matrix()
            values.append(np.asarray(points)@R.T+v[:3])
        return np.asarray(values)
    group_refs={}
    for key,group in cfg.get('landmark_groups',{}).items():
        pieces=[np.array([state[name+f'.{i:05d}'] for i in range(count)]) for name in group['bodies']]
        for item in group.get('surfaces',[]):
            path=Path(p['inputs'][item['mesh']]['path']);mesh=np.load(path);pieces.append(positions(item['body'],mesh['V']));used.append(path)
        reference=np.concatenate(pieces,axis=1).mean(1)[:,None,:];group_refs[key]=('region_center',reference)
        if group.get('extent_output'):group_refs[group['extent_output']]=('region_extent',reference)
    for channel in p['channels']:
        key=channel['predicted'];source=observed[channel['observed']][:count]
        if np.isfinite(source).sum()<2:continue
        row={'id':channel['id'],'observed':source,'sigma':channel['tolerance']};feature_bodies=[]
        if key in landmarks:
            item=landmarks[key];require(item['feature'] in {'point','axis_angle'},'Unsupported rigid placement landmark')
            row.update(kind='point' if item['feature']=='point' else 'axis',xyz=positions(item['body'],item['points']))
            feature_bodies=[item['body']]
        elif key in cfg.get('feature_sets',{}) and len(cfg['feature_sets'][key]['members'])==1:
            item=cfg['feature_sets'][key];require(item['encoding']=='line_probes','Unsupported placement set encoding')
            lm=landmarks[item['members'][0]];path=Path(p['inputs'][item['probes']]['path']);used.append(path)
            row.update(kind='line_probes',xyz=positions(lm['body'],lm['points']),probe_y=np.load(path)['y'][:count],observed=source[:,0])
            feature_bodies=[lm['body']]
        elif key=='marker_uv':
            path=Path(p['inputs']['marker_mesh']['path']);mesh=np.load(path);used.append(path)
            row.update(kind='region_center',xyz=positions(cfg['target'],[mesh['V'].mean(0)]),baseline=prediction[key][:count])
            feature_bodies=[cfg['target']]
        elif key in group_refs:
            kind,reference=group_refs[key];row.update(kind=kind,xyz=reference,baseline=prediction[key][:count])
            group=next(g for name,g in cfg['landmark_groups'].items() if key==name or g.get('extent_output')==key)
            feature_bodies=[*group['bodies'],*[s['body'] for s in group.get('surfaces',[])]]
        else:
            skipped.append({'id':channel['id'],'reason':'No exact identity mapping or local region proxy available for proposal; final gate retained.'});continue
        row['bodies']=feature_bodies;features.append(row)
    bounds=[[-value*.001,value*.001] for value in args.bounds_mm]
    rotation=read(args.rotation) if args.rotation else None
    if rotation:
        members=set(rotation.pop('body_ids'));rotation['feature_ids']=[]
        for feature in features:
            bodies=set(feature['bodies'])
            require(not bodies&members or bodies<=members,'A projected feature cannot mix moving and stationary group members')
            if bodies<=members:rotation['feature_ids'].append(feature['id'])
    endpoints=None
    if args.endpoints:
        endpoint_plan=read(args.endpoints);endpoints={}
        for key,item in endpoint_plan.items():
            require(key in {f['id'] for f in features} and item['body'] in cfg['bodies'],
                    'Endpoint refinement must name an observed feature and native body')
            direction=np.asarray(item['direction_local'],float)
            directions=[]
            for i in range(count):
                v=state[item['body']+f'.{i:05d}'];directions.append((Rotation.from_quat(v[[4,5,6,3]]).as_matrix()@direction).tolist())
            endpoints[key]={'directions_world':directions,'bounds_m':item['bounds_m'],'reason':item['reason']}
    result=refine_translation(features,cameras,fit_indices=split['fit'],validation_indices=split['validation'],
                              bounds_m=bounds,strategy=args.strategy,rotation=rotation,endpoints=endpoints)
    result['translation_world_m']=result['translation_world_m'].tolist();result['skipped_channels']=skipped
    result['features']=[{'id':r['id'],'kind':r['kind'],'sigma':r['sigma']} for r in features]
    paths=[root/'plan.json',trial/'native_states.npz',trial/'predictions.npz',Path(p['inputs']['scene_setup']['path']),
           Path(p['inputs']['process_camera']['path']),Path(p['inputs']['observations']['path']),*used,
           Path(__file__),Path('src/calibrated_scene_reconstruction/physics/process/initial_alignment.py'),Path('src/calibrated_scene_reconstruction/physics/process/initial_state.py'),
           Path('src/calibrated_scene_reconstruction/physics/process/line_features.py')]
    if args.rotation:paths.append(Path(args.rotation))
    if args.endpoints:paths.append(Path(args.endpoints))
    result['source_sha256']={str(path.resolve()):digest(path) for path in paths}
    write(args.output,result);print(result,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--round',required=True);ap.add_argument('--trial',type=int,default=0)
    ap.add_argument('--output',required=True);ap.add_argument('--bounds-mm',type=float,nargs=3,default=[3.,3.,1.5])
    ap.add_argument('--strategy',choices=['balanced','minimax'],default='minimax')
    ap.add_argument('--rotation',help='Optional source-declared initial group axis/pivot/bounds/body_ids JSON; external contacts need new native verification')
    ap.add_argument('--endpoints',help='Optional observed endpoint feature -> body/local direction/length-delta bounds/reason JSON; requires coherent derived geometry updates')
    run(ap.parse_args())
