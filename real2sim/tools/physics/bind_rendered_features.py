"""Bind reviewed native texture identities before real motion calibration."""
import argparse
from pathlib import Path

import numpy as np
from pxr import Usd,UsdGeom

from calibrated_scene_reconstruction.physics.process.contract import read,write,require,digest
from calibrated_scene_reconstruction.physics.process.surface_features import rendered_anchors


def run(plan_path,output):
    p=read(plan_path)
    require(p.get('opaque_features') is True and p.get('reason'),
            'Review visible opaque material features; exclude transmitted/reflected features')
    stage=Usd.Stage.Open(p['stage']);mesh=stage.GetPrimAtPath(p['mesh']);body=stage.GetPrimAtPath(p['body'])
    require(mesh.IsA(UsdGeom.Mesh) and body,'Explicit native body and mesh required')
    cache=UsdGeom.XformCache();B=np.asarray(cache.GetLocalToWorldTransform(body)).T
    M=np.linalg.inv(B)@np.asarray(cache.GetLocalToWorldTransform(mesh)).T
    V=np.asarray(mesh.GetAttribute('points').Get())@M[:3,:3].T+M[:3,3]
    counts=np.asarray(mesh.GetAttribute('faceVertexCounts').Get())
    require((counts==3).all(),'Triangulated reviewed feature surface required')
    F=np.asarray(mesh.GetAttribute('faceVertexIndices').Get()).reshape(-1,3)
    if p.get('face_subset'):
        subset=stage.GetPrimAtPath(p['face_subset']);F=F[np.asarray(subset.GetAttribute('indices').Get())]
    camera=np.load(p['camera'],allow_pickle=False)
    view=lambda frame:(camera['K'],camera['dist'],camera['T_world_cam'][frame])
    require(len(p['ids'])==len(set(p['ids']))==len(p['pixels']),'Unique material identities required')
    result=rendered_anchors(p['pixels'],V,F,B,view(p['frame']),
        [(view(row['frame']),row['pixels']) for row in p['validation']],limit_px=p['limit_px'])
    paths=[plan_path,p['stage'],p['camera'],p['image'],__file__,
           'src/calibrated_scene_reconstruction/physics/process/surface_features.py',*[r['image'] for r in p['validation']]]
    write(output,{'pass':True,'ids':p['ids'],'local_points':result['points'].tolist(),
        'validation_errors_px':result['validation_errors_px'].tolist(),
        'maximum_error_px':result['maximum_error_px'],
        'source_sha256':{str(Path(path).resolve()):digest(path) for path in paths},
        'reason':p['reason'],'scope':'Native asset semantic correspondence only. Does not fit real motion or certify dynamics.'})


if __name__=='__main__':
    ap=argparse.ArgumentParser(__doc__);ap.add_argument('--plan',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();run(a.plan,a.output)
