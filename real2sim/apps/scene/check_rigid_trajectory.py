"""Check actual closed surfaces at every native rigid diagnostic timestep.

This supplements contact reports; it cannot approve a full scene or task.
"""
import argparse
import hashlib
import json
from pathlib import Path
import inspect
import numpy as np

from calibrated_scene_reconstruction.assets.metric_mesh import asset
from calibrated_scene_reconstruction.assets.rigid_trajectory import measure
from calibrated_scene_reconstruction.core.paths import resolve


def run(directory, spacing_m, tolerance_m, device):
    directory=Path(directory).resolve()
    output=directory/'actual_surface_trajectory.json'
    if output.exists():raise ValueError('Preserve previous trajectory measurements')
    source=directory/'input.json';record=json.loads(source.read_text())
    paths=[source,directory/'physics_poses.npz',directory/'result.json',Path(__file__),Path(inspect.getfile(measure))]
    def check_mesh(fix,expected):
        path=resolve(fix['mesh'])
        if not expected or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('Native trajectory source mesh changed or is unbound')
        paths.append(path)
    check_mesh(record['fix'],record['mesh_sha256'])
    parts=asset(record['fix'])['parts']
    if any(p['kind']!='rigid' for p in parts):raise ValueError('Rigid trajectories only')
    if record.get('companion'):
        c=record['companion'];check_mesh(c['source_fix'],c['mesh_sha256'])
        companion=asset(c['source_fix'])['parts']
        if len(companion)!=1 or companion[0]['kind']!='rigid':raise ValueError('Single rigid companion required')
        part=companion[0];part['name']=c['name']
        if not c.get('placement_on_body_transform',False):
            # Historical diagnostics baked the initial placement into vertices.
            # Current trials preserve the delivered source collider coordinates.
            part['V']=part['V']@np.asarray(c['rotation']).T+np.asarray(c['translation_m'])
        parts.append(part)
    with np.load(directory/'physics_poses.npz') as data:
        if list(data['names'])!=[p['name'] for p in parts]:raise ValueError('Native pose/body order mismatch')
        poses,times=data['poses'],data['times_s']
    expected_dt=1/record['profile']['physics_hz']
    if not np.allclose(np.diff(times),expected_dt,atol=1e-10,rtol=1e-7):
        raise ValueError('Every native step is required; sparse visualization frames cannot qualify')
    if record.get('static_support'):
        from scipy.spatial.transform import Rotation
        support=record['static_support'];path=Path(support['mesh'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=support['mesh_sha256']:
            raise ValueError('Modeled support changed after native execution')
        paths.append(path)
        with np.load(path) as data:
            parts.append({'name':'support/'+support['name'],'V':data['V'],'F':data['F']})
        quaternion=Rotation.from_euler('z',support['yaw_deg'],degrees=True).as_quat()[[3,0,1,2]]
        fixed=np.r_[support['xy'],0.,quaternion]
        poses=np.concatenate([poses,np.broadcast_to(fixed,(len(times),1,7))],axis=1)
    report=measure(parts,poses,times,spacing_m=spacing_m,tolerance_m=tolerance_m,device=device,
                   progress=lambda i,n,m:print(f'{i}/{n} steps; peak {m:.8g} m',flush=True))
    report.update(task_approved=False,diagnostic_only=True,
                  evidence_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    output.write_text(json.dumps(report,indent=2)+'\n')
    return {k:v for k,v in report.items() if k!='per_frame_maximum_m'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory');parser.add_argument('--spacing-m',type=float,required=True)
    parser.add_argument('--tolerance-m',type=float,required=True)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    result=run(args.directory,args.spacing_m,args.tolerance_m,args.device)
    print(json.dumps(result,indent=2))
    if not result['pass']:raise SystemExit(2)
