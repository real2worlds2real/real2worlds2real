"""Audit native background supports on actual exported geometry, without moving tasks."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from calibrated_scene_reconstruction.core.paths import ROOT
from calibrated_scene_reconstruction.scene.background.blender_env import label, verify
from calibrated_scene_reconstruction.scene.review.environment import digest
from calibrated_scene_reconstruction.scene.session import Session


def extract(stage, support, objects=()):
    from pxr import Usd, UsdGeom
    cache = UsdGeom.XformCache()
    rows, arrays = [], {}
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        external_entity = next((name for prefix, name in [('/World/Room/'+support, support),
                               *[('/World/Objects/'+name, name) for name in objects]]
                               if path.startswith(prefix) and path[len(prefix):][:1] in ('', '/', '_')), None)
        external = external_entity is not None
        if not prim.IsA(UsdGeom.Mesh) or not (path.startswith('/World/Environment/') or external):
            continue
        entity = external_entity
        ancestor = prim
        while entity is None and ancestor and str(ancestor.GetPath()).startswith('/World/Environment'):
            entity = ancestor.GetAttribute('userProperties:r2s2r_entity').Get()
            ancestor = ancestor.GetParent()
        if not entity:
            raise ValueError('Exported background mesh has no entity identity: '+path)
        mesh = UsdGeom.Mesh(prim)
        points = np.asarray(mesh.GetPointsAttr().Get(), float)
        transform = np.asarray(cache.GetLocalToWorldTransform(prim), float)
        points = (np.c_[points, np.ones(len(points))]@transform)[:, :3]
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), int)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), int)
        faces, offset = [], 0
        for count in counts:
            polygon = indices[offset:offset+count]
            faces += [[int(polygon[0]), int(polygon[j]), int(polygon[j+1])] for j in range(1, count-1)]
            offset += count
        if not len(points) or not faces or not np.isfinite(points).all():
            raise ValueError('Invalid exported physical geometry: '+path)
        index = len(rows)
        arrays[f'v{index}'] = points
        arrays[f'f{index}'] = np.asarray(faces, np.int32)
        rows.append({'entity': entity, 'part': str(prim.GetParent().GetName()), 'path': path, 'external': external})
    return rows, arrays


def run(session, usd, variant='twin', delivery=False):
    from pxr import Usd
    if (session/'qa/independent_freeze.json').exists():
        raise ValueError('Frozen session: create an environment derivative')
    built = verify(session, variant)
    usd = Path(usd).resolve()
    stage = Usd.Stage.Open(str(usd))
    meta = dict(stage.GetRootLayer().customLayerData).get('r2s2r_environment', {})
    if meta.get('asset_sha256') != built['output_sha256'][built['asset']] or meta.get('variant') != variant:
        raise ValueError('Audit the actual current variant export')
    root = session/'qa/environment_physics'/label(variant)
    root.mkdir(parents=True, exist_ok=True)
    rounds = [int(p.name[6:]) for p in root.glob('round_*') if p.name[6:].isdigit()]
    out = root/f'round_{max(rounds, default=0)+1}'
    out.mkdir()
    support = session.scene_graph['task']['support']
    objects = session.scene_graph.get('dynamic_objects', {})
    rows, arrays = extract(stage, support, objects)
    from calibrated_scene_reconstruction.scene.background.appearance_plan import required_parts
    required_parts(built.get('assembly_requirements',{}),rows,arrays)
    np.savez_compressed(out/'geometry.npz', **arrays)
    contract = built.get('physics', {})
    worker = Path(__file__).with_name('environment_physics_worker.py')
    passive_app = ROOT/'apps/scene/environment_settle.py'
    shutil.copyfile(worker, out/'worker.py')
    shutil.copyfile(passive_app, out/'passive_worker.py')
    inputs = {str(p.resolve()): digest(p) for p in [usd, worker, passive_app, Path(__file__),
              session/f'scene/environment/{variant}/current.json']}
    for path, expected in built['input_sha256'].items():
        inputs[path] = expected
    job = {'geometry': str((out/'geometry.npz').resolve()), 'parts': rows, 'contract': contract,
           'geometry_sha256':digest(out/'geometry.npz'),
           'native_manifest_sha256':digest(session/f'scene/environment/{variant}/current.json'),
           'output': str((out/'result.json').resolve()), 'external_entities': [support, *objects]}
    (out/'job.json').write_text(json.dumps(job, indent=2)+'\n')
    with (out/'audit.log').open('w') as log:
        subprocess.run(['tools/run_blender.sh', '--python', str((out/'worker.py').resolve()), '--',
                        str((out/'job.json').resolve())], check=True, stdout=log, stderr=subprocess.STDOUT)
    result = json.loads((out/'result.json').read_text())
    result['static_pass'] = result['pass']
    result['passive_stability_pass'] = False
    if result['static_pass']:
        with (out/'passive.log').open('w') as log:
            subprocess.run(['tools/run_isaac.sh', str((out/'passive_worker.py').resolve()), str((out/'job.json').resolve()),
                            '--output', str((out/'passive.json').resolve())],
                           check=True, stdout=log, stderr=subprocess.STDOUT)
        passive = json.loads((out/'passive.json').read_text())
        result['passive_stability_pass'] = passive['pass']
        result['dynamic_stability_verified'] = True
        result['passive_result'] = passive
    result['pass'] = result['static_pass'] and result['passive_stability_pass']
    result.update(version=1, variant=variant, delivery=delivery, usd=str(usd), input_sha256=inputs,
                  native_manifest_sha256=digest(session/f'scene/environment/{variant}/current.json'),
                  scope='Exported background geometry and read-only task support. No task/robot dynamics, actual dimensions or CAD evaluation.')
    for path, expected in inputs.items():
        if digest(path) != expected:
            raise ValueError('Physics input changed: '+path)
    evidence = ['geometry.npz', 'job.json', 'worker.py', 'passive_worker.py']
    if (out/'passive.json').exists():evidence.append('passive.json')
    for name in evidence:
        result['input_sha256'][str((out/name).resolve())] = digest(out/name)
    (out/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    pointer = {'path': str((out/'result.json').resolve()), 'sha256': digest(out/'result.json')}
    (root/('delivery_current.json' if delivery else 'current.json')).write_text(json.dumps(pointer, indent=2)+'\n')
    return result


def current(session, variant='twin', render=None, delivery=False):
    root = session/'qa/environment_physics'/label(variant)
    pointer = json.loads((root/('delivery_current.json' if delivery else 'current.json')).read_text())
    path = Path(pointer['path']).resolve()
    if not path.is_relative_to(root.resolve()) or digest(path) != pointer['sha256']:
        raise ValueError('Environment physics result changed')
    result = json.loads(path.read_text())
    if delivery:
        expected=(session/f'export/usd/{session.name}_twin_blender.usdz').resolve()
        if result.get('delivery') is not True or Path(result['usd']).resolve()!=expected:
            raise ValueError('S6 physics must check the actual delivered twin_blender USDZ')
    if result.get('pass') is not True or result.get('passive_stability_pass') is not True:
        raise ValueError('Environment support/connectivity/stability audit has not passed')
    if result['native_manifest_sha256'] != digest(session/f'scene/environment/{variant}/current.json'):
        raise ValueError('Environment changed since physical audit')
    verify(session, variant)
    for source, expected in result['input_sha256'].items():
        if digest(source) != expected:
            raise ValueError('Physics geometry/configuration changed: '+source)
    if render is not None and render['input_sha256'].get(result['usd']) != result['input_sha256'][result['usd']]:
        raise ValueError('Render and physical audit must bind the same exported scene')
    return result, pointer['sha256']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session')
    parser.add_argument('usd')
    parser.add_argument('--variant', default='twin')
    parser.add_argument('--delivery',action='store_true',help='Check the packaged S6 USDZ separately from S3 review evidence')
    args = parser.parse_args()
    result = run(Session(args.session), args.usd, args.variant,args.delivery)
    print('Environment physics', 'PASS' if result['pass'] else 'FAIL', len(result['entities']), 'entities')
    if result['pass'] is not True:raise SystemExit(1)


if __name__ == '__main__':
    main()
