"""Propose source-owned GS replacement around a reviewed native assembly.

Uniform native materials need not invent a photographic plane as permission to
replace distorted GS. Require a GS on/off material control, unchanged selected
native geometry, closed named background parts and two independent source-owner
rays. Normally intersect a bounded triangle-distance band. Explicit source-only
components handle rejected GS with unreliable depth; require a separate depth
diagnosis and whole-field review. Keep all other kernel bytes.
The result remains an unapproved field candidate requiring fresh measurement.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from plyfile import PlyData, PlyElement

from calibrated_scene_reconstruction.assets.topology import closed_oriented_surface
from calibrated_scene_reconstruction.core.paths import ROOT
from calibrated_scene_reconstruction.scene.background.context_refresh import backing
from calibrated_scene_reconstruction.scene.background.observed_context import parameter_digest
from calibrated_scene_reconstruction.scene.background import source_replace
from calibrated_scene_reconstruction.scene.review.environment import digest, protected_state


def triangles(vertices, faces):
    result = []
    for triangle in np.rint(np.asarray(vertices)[np.asarray(faces)]*1e6).astype('int64'):
        row = list(map(tuple, triangle)); first = min(range(3), key=lambda i: row[i])
        result.append(tuple(row[first:]+row[:first]))
    return sorted(result)


def validate_targets(request, ownership, parts):
    targets = request.get('owners')
    if not isinstance(targets, dict) or not 1 <= len(targets) <= 8:
        raise ValueError('Explicit bounded native replacement owners required')
    for owner, row in targets.items():
        available = {p['part'] for p in parts if p['entity'] == owner and not p['external']}
        if (ownership.get(owner, {}).get('role') != 'native_replacement' or not row.get('parts')
                or len(set(row['parts'])) != len(row['parts']) or not set(row['parts']) <= available
                or not row.get('free_view_rejection', '').strip() or not row.get('observations')):
            raise ValueError('Only explicitly rejected GS on named native background parts may be replaced')
    margin = request.get('surface_margin_m', .035)
    if isinstance(margin, bool) or not isinstance(margin, (float, int)) or not .01 <= margin <= .10:
        raise ValueError('Bounded source-owned surface distance required')
    mode = request.get('selection_mode', 'surface_and_source')
    if mode not in ('surface_and_source', 'source_component'):
        raise ValueError('Explicit supported replacement selection mode required')
    if mode == 'source_component' and not request.get('bad_depth_review', '').strip():
        raise ValueError('Source-only replacement needs an explicit bad-depth diagnosis and component review')
    return targets, float(margin)


def prepare(session, source_path, request_path, output):
    from pxr import Usd, UsdGeom
    from calibrated_scene_reconstruction.scene.background.blender_env import verify
    from calibrated_scene_reconstruction.scene.background.environment_physics import current
    source_path, request_path, out = [Path(p).resolve() for p in (source_path, request_path, output)]
    if out.exists() or not out.is_relative_to(session.dir.resolve()) or (session/'qa/independent_freeze.json').exists():
        raise ValueError('New unfrozen source-owned field candidate required')
    native = verify(session); physical, _ = current(session)
    source, request = [json.loads(p.read_text()) for p in (source_path, request_path)]
    if (native['splat_context']['evidence_sha256'] != digest(source_path)
            or source['protected_task_state'] != protected_state(session)):
        raise ValueError('Use the current bound field and unchanged frozen state')
    job_path = Path(json.loads((session/'qa/environment_physics/twin/current.json').read_text())['path']).parent/'job.json'
    job = json.loads(job_path.read_text()); geometry = np.load(job['geometry'])
    if job['geometry_sha256'] != digest(job['geometry']): raise ValueError('Audited native geometry changed')
    routing_path = Path(request['routing_manifest']).resolve(); routing = json.loads(routing_path.read_text())
    if routing['asset_sha256'] != source['asset_sha256'] or routing['protected_task_state'] != protected_state(session):
        raise ValueError('Native routing must belong to this same field and frozen state')
    targets, margin = validate_targets(request, routing['appearance_ownership'], job['parts'])
    controls = [Path(request[k]).resolve() for k in ('native_with_gs', 'native_without_gs')]
    full, empty = [json.loads(p.read_text()) for p in controls]
    if (not full.get('diagnostic') or empty.get('diagnostic') != 'context_disabled'
            or any(full.get(k) != empty.get(k) for k in ('usd', 'camera_manifest', 'environment_context', 'effective_photometry', 'input_sha256'))
            or any(p.get('static_scene_preserved') is not True for p in (full, empty))
            or any(p.get(k) for p in (full, empty) for k in ('background_field_override', 'background_transform_override', 'render_settings', 'source_context_override'))
            or full['environment_context']['asset_sha256'] != source['asset_sha256']
            or not np.allclose(json.loads(full['environment_context']['matrix_json']), source['matrix'], rtol=0, atol=1e-8)):
        raise ValueError('Same-field native material with only GS enabled/disabled required')
    stage = Usd.Stage.Open(full['usd']); cache = UsdGeom.XformCache()
    for i, part in enumerate(job['parts']):
        if part['entity'] not in targets or part['part'] not in targets[part['entity']]['parts']: continue
        vertices, faces = geometry['v'+str(i)], geometry['f'+str(i)]
        if not closed_oriented_surface(vertices, np.full(len(faces), 3), faces.ravel()):
            raise ValueError('Replacement needs actual closed oriented native parts')
        meshes = [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)
                  and str(p.GetPath()).startswith('/World/Environment/'+part['entity']+'/')
                  and part['part'] in str(p.GetPath()).split('/')]
        if len(meshes) != 1: raise ValueError('Selected control part identity is ambiguous')
        prim = meshes[0]; mesh = UsdGeom.Mesh(prim); local = np.asarray(mesh.GetPointsAttr().Get())
        world = (np.c_[local, np.ones(len(local))]@np.asarray(cache.GetLocalToWorldTransform(prim)))[:, :3]
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), int); offset = 0; control_faces = []
        for count in mesh.GetFaceVertexCountsAttr().Get():
            f = indices[offset:offset+count]; control_faces.extend([[f[0], f[j], f[j+1]] for j in range(1, count-1)]); offset += count
        if triangles(world, control_faces) != triangles(vertices, faces):
            raise ValueError('Selected native geometry differs from the GS on/off control')
    inputs = {**source['input_sha256'], **full['input_sha256']}
    control_without = {view['source']: view for view in empty['views']}
    causal = {}
    for owner, target in targets.items():
        masks = {}
        for path in target['observations']:
            for row in json.loads(Path(path).read_text())['records']:
                if row['entity'] == owner and row.get('mask'): masks[row['source']] = row['mask']
        visible = []
        for view in full['views']:
            if view['source'] not in masks: continue
            a = cv2.imread(view['renders'][0]); b = cv2.imread(control_without[view['source']]['renders'][0])
            mask = cv2.resize(cv2.imread(masks[view['source']], 0), (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST)>128
            count = int(np.count_nonzero(mask&(abs(a.astype(float)-b).max(2)>16)))
            if count >= 64: visible.append({'source':view['source'], 'affected_pixels':count})
        if len(visible) < 2: raise ValueError('Native replacement needs visible causal GS overlap in two owner-source controls')
        causal[owner] = visible
    worker = Path(__file__).with_name('field_replace_worker.py').resolve()
    for p in (source_path, request_path, routing_path, job_path, Path(job['geometry']), *controls, Path(__file__).resolve(), worker): inputs[str(p)] = digest(p)
    for control in (full, empty):
        for view in control['views']:
            for p in [view['source'], *view['renders']]: inputs[str(Path(p).resolve())] = digest(p)
    cloud = source_path.parent/'context.ply'; vertices = PlyData.read(str(cloud))['vertex'].data
    matrix = np.asarray(source['matrix']); scale = np.cbrt(np.linalg.det(matrix[:3, :3]))
    points = (np.c_[vertices['x'], vertices['y'], vertices['z'], np.ones(len(vertices))]@matrix.T)[:, :3]
    sigma = scale*np.exp(np.c_[vertices['scale_0'], vertices['scale_1'], vertices['scale_2']]).max(1)
    semantic, extra, report = source_replace.prepare(session, points,
        {owner: row['observations'] for owner, row in targets.items()}, targets)
    inputs.update(extra); inputs[str(cloud)] = digest(cloud)
    if any(digest(p) != sha for p, sha in inputs.items()): raise ValueError('Native replacement evidence changed')
    out.mkdir(parents=True); np.savez_compressed(out/'cloud_world.npz', points=points, sigma=sigma)
    selection = {'geometry': job['geometry'], 'parts': job['parts'], 'cloud': str(out/'cloud_world.npz'),
                 'entities': list(targets), 'replacement_parts': {k: v['parts'] for k, v in targets.items()},
                 'margin_m': margin, 'output': str(out/'near_surface.npy')}
    (out/'selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    with (out/'selection.log').open('w') as log:
        subprocess.run([str(ROOT/'tools/run_blender.sh'), '--python', str(worker), '--', str(out/'selection.json')], check=True, stdout=log, stderr=subprocess.STDOUT)
    near = np.load(out/'near_surface.npy')
    if near.dtype != bool or near.shape != semantic.shape: raise ValueError('Actual triangle-distance selection missing')
    mode = request.get('selection_mode', 'surface_and_source')
    removed = semantic if mode == 'source_component' else near&semantic
    if not removed.any() or removed.all(): raise ValueError('Replacement needs a nonempty strict source-owned subset')
    np.savez_compressed(out/'source_rows.npz', retained=np.flatnonzero(~removed), removed=np.flatnonzero(removed))
    PlyData([PlyElement.describe(vertices[~removed], 'vertex')], text=False).write(str(out/'context.ply'))
    with (out/'transcode.log').open('w') as log:
        subprocess.run([str(ROOT/'envs/splat/3dgrut/.venv/bin/python'), '-m', 'threedgrut.export.scripts.transcode', str(out/'context.ply'), '-o', str(out/'context.usdz'), '-f', 'nurec'], check=True, stdout=log, stderr=subprocess.STDOUT)
    config = session.config('environment.json'); entities = sorted({p['entity'] for p in job['parts'] if not p['external']}); owners = backing(entities, config['completion'])
    result = copy.deepcopy(source)
    for key in ('selection_frozen', 'coverage_reuse', 'native_surface_completion', 'source_ray_trial'): result.pop(key, None)
    for p in (out/'selection.json', out/'source_rows.npz', out/'near_surface.npy', out/'context.ply'): inputs[str(p)] = digest(p)
    result.update(asset_sha256=digest(out/'context.usdz'), parameters_sha256=parameter_digest(config['parameters']),
        source_sha256={str(p.resolve()): digest(p) for p in session.config_dir.joinpath('agent/environment').rglob('*.py')},
        appearance_ownership=owners, gap_masks={}, native_entities=entities, background_entities=entities,
        replaced_entities=[e for e, r in owners.items() if r['role'] == 'native_replacement'],
        gap_fill_entities=[e for e, r in owners.items() if r['role'] == 'inferred_enclosure'],
        retained=int((~removed).sum()), removed=source.get('removed', 0)+int(removed.sum()),
        geometry_rebuild_required=True, photorealism_approved=False, input_sha256=inputs,
        native_replacement_trial={'owners': targets, 'source_ownership': report, 'causal_overlap': causal, 'removed_rows': int(removed.sum()),
            'selection_mode': mode, 'bad_depth_review': request.get('bad_depth_review'), 'near_surface_rows': int((near&semantic).sum()),
            'retained_rows_sha256': hashlib.sha256(vertices[~removed].tobytes()).hexdigest(), 'appearance_pass': False, 'scope': __doc__.strip()})
    result['appearance_plan']={'version':1, 'routes':{e:{'route':'mesh','quality':'unresolved','reason':'Provisional backing after source-owned native replacement; fresh actual coverage and source/free-view review required.', 'scores':[], 'free_view_review':{'pass':False,'note':'Not reviewed.'}} for e,r in owners.items() if r['role']!='inferred_enclosure'}, 'input_sha256':{}, 'protected_task_state':protected_state(session)}
    if any(digest(p) != sha for p, sha in inputs.items()): raise ValueError('Native replacement evidence changed during preparation')
    (out/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('session', 'context', 'request'): parser.add_argument(name)
    parser.add_argument('--output', required=True); args = parser.parse_args()
    result = prepare(Session(args.session), args.context, args.request, args.output)
    print('Unapproved source-owned field replacement:', result['native_replacement_trial']['removed_rows'], 'rows')


if __name__ == '__main__': main()
