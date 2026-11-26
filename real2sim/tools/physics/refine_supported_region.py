"""Propose a supported assembly pose from a native initial shape and source masks."""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd

from calibrated_scene_reconstruction.physics.process.contract import read,write,digest
from calibrated_scene_reconstruction.physics.process.supported_region_pose import refine


def run(plan_path,output):
    proposal=read(plan_path);root=Path(proposal['round']);p=read(root/'plan.json')
    cfg=read(p['inputs']['scene_setup']['path']);trial=root/'trials'/f"{proposal.get('trial',0):04d}"
    bindings=read(trial/'bindings.json')['bodies'];states=np.load(trial/'native_states.npz');stage=Usd.Stage.Open(str(trial/'native_initial.usdc'))
    group=cfg['landmark_groups'][proposal['group']];body=proposal['body'];container=proposal['container']
    def pose(name):
        v=states[name+'.00000'];M=np.eye(4);M[:3,:3]=Rotation.from_quat(v[[4,5,6,3]]).as_matrix();M[:3,3]=v[:3]
        return M
    B,A=pose(container),pose(body);pieces=[];faces=[];offset=0;used=[]
    for name in group['bodies']:
        X=(states[name+'.00000']-A[:3,3])@A[:3,:3]
        mesh=stage.GetPrimAtPath(bindings[name]['path']);F=np.asarray(mesh.GetAttribute('faceVertexIndices').Get()).reshape(-1,3)
        pieces.append(X);faces.append(F+offset);offset+=len(X)
    for item in group.get('surfaces',[]):
        path=Path(p['inputs'][item['mesh']]['path']);mesh=np.load(path);T=np.linalg.inv(A)@pose(item['body'])
        X=mesh['V']@T[:3,:3].T+T[:3,3];pieces.append(X);faces.append(mesh['F']+offset);offset+=len(X);used.append(path)
    camera_path=Path(p['inputs'][proposal.get('camera','process_camera')]['path']);camera=np.load(camera_path)
    observations_path=Path(p['inputs']['observations']['path']);observations=np.load(observations_path)
    lines_path=Path(proposal['source_lines']);lines=np.load(lines_path)[proposal['source_line_key']]
    split=proposal['split'];count=max(*split['fit'],*split['validation'])+1
    valid=np.isfinite(lines[:count]).all(1);fit=[i for i in split['fit'] if valid[i]][::proposal.get('fit_stride',1)]
    held=[i for i in split['validation'] if valid[i]]
    cameras=[(camera['K'],camera['dist'],np.linalg.inv(B)@T) for T in camera['T_world_cam'][:count]]
    support=proposal['support'];others=[]
    for item in proposal.get('other_segments',[]):
        T=np.linalg.inv(B)@pose(item['body']);X=np.asarray(item['points'])@T[:3,:3].T+T[:3,3]
        others.append((*X,item['radius']))
    print('Source region fit views',fit,'held views',held,flush=True)
    result=refine(support['bottom'],support['top'],support['radius'],np.vstack(pieces),np.vstack(faces),
        np.linalg.inv(B)@A,cameras,observations[proposal['center_observation']][:count],
        observations[proposal['extent_observation']][:count],lines[:count],fit_indices=fit,validation_indices=held,
        opening=group['opening'],floor_z=proposal['floor_z'],apertures=proposal['apertures'],
        support_radius=proposal['support_radius'],other_segments=others,max_evaluations=proposal.get('max_evaluations',700))
    result={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in result.items()}
    paths=[Path(plan_path),root/'plan.json',trial/'native_initial.usdc',trial/'native_states.npz',trial/'bindings.json',
           camera_path,observations_path,lines_path,*used,Path(__file__),Path('src/calibrated_scene_reconstruction/physics/process/supported_region_pose.py'),
           Path('src/calibrated_scene_reconstruction/physics/process/scene_model.py'),Path('src/calibrated_scene_reconstruction/physics/process/mask_raster.py')]
    result['source_sha256']={str(path.resolve()):digest(path) for path in paths}
    result['scope']='Initial source-static geometry proposal from a recorded native shape. No runtime response targets or source observations changed. Fresh free native validation required.'
    write(output,result);print(result['before_fit'],result['fit'],result['validation'],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--plan',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();run(a.plan,a.output)
