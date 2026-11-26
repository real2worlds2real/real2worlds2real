"""Acquire a full task-support plane from the registered input capture."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.core.paths import rel,resolve
from .observed_plane import surface_plane


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(session):
    from .blender import require_mutable
    require_mutable(session)
    import pycolmap
    from calibrated_scene_reconstruction.scene.capture.placement_planes import from_photo
    sparse=session/'splat/data/sparse/full';rec=pycolmap.Reconstruction(str(sparse))
    _,points,evidence=from_photo(session,rec)
    alignment=session/'splat/align.json';sim=json.loads(alignment.read_text())['sim3']
    world=float(sim['scale'])*points@np.asarray(sim['R']).T+np.asarray(sim['t'])
    plane=surface_plane(world)
    files=[alignment,*[sparse/name for name in ['cameras.bin','images.bin','points3D.bin']],
           session/'parsing/instances.json',session/'parsing/masks.npz',session/'capture/keyframes.json',session.config_dir/'scene_graph.json']
    files+=list((session/'parsing/placement_plane_masks').glob('support__*.png'))
    # The graph identifies the support, but fitted geometry is not evidence
    # for its observed plane. Bind the actual capture inputs separately.
    files=[p for p in files if p!=session.config_dir/'scene_graph.json']
    result={'version':1,'support':session.scene_graph['task']['support'],'plane':plane,
            'capture_evidence':evidence,'input_sha256':{rel(p):sha(p) for p in files if p.is_file()},
            'engine_sha256':{rel(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('observed_plane.py'),
                Path(__file__).resolve().parents[1]/'capture/placement_planes.py',
                Path(__file__).resolve().parents[1]/'capture/plane_observations.py']},
            'scope':'Observed geometry only. No candidate model, task pass or true object/furniture dimensions establish the normal or offset.'}
    path=session/'support/surface_plane.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')
    return result


def read(session):
    path=session/'support/surface_plane.json'
    if not path.exists():return None
    record=json.loads(path.read_text())
    if record.get('version')!=1 or record.get('support')!=session.scene_graph['task']['support']:
        raise ValueError('Observed plane belongs to a different task support')
    for name,value in {**record['input_sha256'],**record['engine_sha256']}.items():
        if not resolve(name).is_file() or sha(resolve(name))!=value:raise ValueError('Observed support-plane input or engine changed: '+name)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('session');a=p.parse_args()
    r=run(Session(a.session));print(json.dumps(r['plane'],indent=2))


if __name__=='__main__':main()
