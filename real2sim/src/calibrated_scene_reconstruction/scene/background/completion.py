"""Require an agent-declared, geometrically enclosed hybrid environment.

The agent infers a footprint, ceiling and hidden surfaces from the current
capture. This module checks that declaration and the actual exported triangles;
it never fabricates measurements or treats inferred structure as observed truth.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from calibrated_scene_reconstruction.scene.review.environment import digest, protected_state
from calibrated_scene_reconstruction.scene.session import Session


def specification(value, reserved=()):
    if not isinstance(value, dict) or value.get('version') != 1:
        raise ValueError('Hybrid environment requires completion version1 with an inferred enclosure')
    footprint = np.asarray(value.get('footprint'), float)
    if footprint.ndim != 2 or footprint.shape[1] != 2 or len(footprint) < 3 or not np.isfinite(footprint).all():
        raise ValueError('Completion needs a finite convex counterclockwise footprint')
    edges = np.roll(footprint, -1, axis=0)-footprint
    turn = edges[:, 0]*np.roll(edges, -1, axis=0)[:, 1]-edges[:, 1]*np.roll(edges, -1, axis=0)[:, 0]
    # Require every vertex on the interior side of every edge, also rejecting
    # self-crossing stars that a local turning test alone would admit.
    q = footprint[None, :, :]-footprint[:, None, :]
    side = edges[:, None, 0]*q[:, :, 1]-edges[:, None, 1]*q[:, :, 0]
    if np.any(turn <= 1e-8) or np.any(side < -1e-8):
        raise ValueError('Completion needs a convex counterclockwise footprint without crossings')
    floor, ceiling = value.get('floor_z'), value.get('ceiling_z')
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in (floor, ceiling)) or ceiling-floor < 2.:
        raise ValueError('Completion floor/ceiling need a finite habitable height hypothesis')
    surfaces = value.get('surfaces')
    if not isinstance(surfaces, list) or not surfaces:
        raise ValueError('Declare observed and inferred completion surfaces')
    entities = set()
    for row in surfaces:
        if (not isinstance(row, dict) or not isinstance(row.get('entity'), str) or not row['entity']
                or row['entity'] in entities or row['entity'] in reserved
                or row.get('role') not in ('floor', 'wall', 'ceiling')
                or row.get('provenance') not in ('observed', 'observed_and_inferred', 'inferred')
                or not isinstance(row.get('basis'), str) or not row['basis'].strip()):
            raise ValueError('Completion surfaces need distinct background identities, roles and explicit inference provenance')
        entities.add(row['entity'])
    if {row['role'] for row in surfaces} != {'floor', 'wall', 'ceiling'}:
        raise ValueError('Complete the floor, walls AND ceiling before hybrid delivery')
    return value


def inside(points, footprint, margin=0.):
    p = np.asarray(points, float); poly = np.asarray(footprint, float)
    edges = np.roll(poly, -1, axis=0)-poly
    q = p[..., None, :2]-poly
    cross = edges[:, 0]*q[..., 1]-edges[:, 1]*q[..., 0]
    return np.all(cross >= margin*np.linalg.norm(edges, axis=1), axis=-1)


def run(session, usd, variant='twin'):
    from pxr import Usd
    from calibrated_scene_reconstruction.scene.background.blender_env import label, verify
    from calibrated_scene_reconstruction.scene.background.environment_physics import extract
    built = verify(session, variant)
    cfg = built.get('completion')
    if cfg is None:
        if built.get('splat_context'):
            raise ValueError('Hybrid environment must declare completion before presentation/delivery')
        return None
    specification(cfg, {session.scene_graph['task']['support'], *session.scene_graph.get('dynamic_objects', {})})
    usd = Path(usd).resolve(); stage = Usd.Stage.Open(str(usd))
    meta = dict(stage.GetRootLayer().customLayerData).get('r2s2r_environment', {})
    if meta.get('asset_sha256') != built['output_sha256'][built['asset']] or meta.get('variant') != variant:
        raise ValueError('Completion must audit the actual current exported environment')
    # Only the enclosure's designated triangles can close a ray. Furniture or
    # a Gaussian cannot hide missing architecture from the completeness check.
    rows, arrays = extract(stage, session.scene_graph['task']['support'])
    entities = {row['entity'] for row in cfg['surfaces']}
    selected = [(i, row) for i, row in enumerate(rows) if row['entity'] in entities]
    if {row['entity'] for _, row in selected} != entities:
        raise ValueError('Completion surface missing from the actual composed USD')
    for _, row in selected:
        prim = stage.GetPrimAtPath(row['path']); ancestor = prim
        from pxr import UsdGeom
        if UsdGeom.Imageable(prim).ComputeVisibility() == 'invisible':
            raise ValueError('Completion geometry is invisible: '+row['entity'])
        while ancestor and str(ancestor.GetPath()).startswith('/World/Environment'):
            if ancestor.GetAttribute('primvars:hideForCamera').Get():
                raise ValueError('Completion geometry is hidden behind the GS: '+row['entity'])
            ancestor = ancestor.GetParent()
    root = session/'qa/environment_completion'/label(variant)
    root.mkdir(parents=True, exist_ok=True)
    rounds = [int(p.name[6:]) for p in root.glob('round_*') if p.name[6:].isdigit()]
    out = root/f'round_{max(rounds, default=0)+1}'; out.mkdir()
    np.savez_compressed(out/'geometry.npz', **{f'{k}{j}': arrays[f'{k}{i}'] for j, (i, _) in enumerate(selected) for k in ('v', 'f')})
    worker = Path(__file__).with_name('completion_worker.py'); shutil.copyfile(worker, out/'worker.py')
    camera = session/'parsing/posed_frames/cameras.npz'
    job = {'specification': cfg, 'geometry': str((out/'geometry.npz').resolve()),
           'parts': [row for _, row in selected], 'cameras': str(camera.resolve()), 'output': str((out/'result.json').resolve())}
    (out/'job.json').write_text(json.dumps(job, indent=2)+'\n')
    inputs = {str(p.resolve()): digest(p) for p in [usd, worker, Path(__file__), camera,
        session/f'scene/environment/{variant}/current.json', out/'geometry.npz', out/'job.json', out/'worker.py']}
    with (out/'audit.log').open('w') as log:
        subprocess.run(['tools/run_blender.sh', '--python', str(out/'worker.py'), '--', str(out/'job.json')],
                       check=True, stdout=log, stderr=subprocess.STDOUT)
    result = json.loads((out/'result.json').read_text())
    for p, sha in inputs.items():
        if digest(p) != sha: raise ValueError('Completion input changed: '+p)
    result.update(version=1, variant=variant, usd=str(usd), input_sha256=inputs,
                  protected_task_state=protected_state(session), native_manifest_sha256=digest(session/f'scene/environment/{variant}/current.json'))
    result['scope'] = 'Enclosure coverage of actual exported geometry, including inferred hidden structure; not measured truth or photorealism acceptance.'
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    (root/'current.json').write_text(json.dumps({'path': str((out/'result.json').resolve()), 'sha256': digest(out/'result.json')}, indent=2)+'\n')
    if not result['pass']:
        raise ValueError('Environment enclosure incomplete: '+str(out/'result.json'))
    return result


def current(session, variant='twin', usd=None):
    from calibrated_scene_reconstruction.scene.background.blender_env import verify
    built = verify(session, variant)
    if not built.get('completion') and not built.get('splat_context'):return None
    root = session/'qa/environment_completion'/variant
    pointer = json.loads((root/'current.json').read_text()); path = Path(pointer['path']).resolve()
    if not path.is_relative_to(root.resolve()) or digest(path) != pointer['sha256']:
        raise ValueError('Completion evidence changed')
    result = json.loads(path.read_text())
    if (result.get('pass') is not True or result['native_manifest_sha256'] != digest(session/f'scene/environment/{variant}/current.json')
            or result['protected_task_state'] != protected_state(session)
            or (usd is not None and Path(result['usd']).resolve() != Path(usd).resolve())):
        raise ValueError('Complete the current exported environment before presentation/delivery')
    for p, sha in result['input_sha256'].items():
        if digest(p) != sha:raise ValueError('Completion geometry/evidence changed: '+p)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('session'); p.add_argument('usd'); p.add_argument('--variant', default='twin')
    a = p.parse_args(); result = run(Session(a.session), a.usd, a.variant)
    print('Environment completion PASS' if result else 'Pure native environment without completion declaration')


if __name__ == '__main__': main()
