"""Propose a local GS subset for an explicitly rejected native-replacement owner.

Only kernels supported by repeated reviewed same-owner source masks AND near
named audited mesh parts may be removed. Other owners and frozen foreground
remain untouched. Native PBR is allowed; photographed texture is not assumed.
The proposal requires fresh field, coverage and composed-scene review.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess

import numpy as np
from plyfile import PlyData, PlyElement

from calibrated_scene_reconstruction.core.paths import ROOT
from calibrated_scene_reconstruction.scene.background.blender_env import verify
from calibrated_scene_reconstruction.scene.background.context_refresh import backing
from calibrated_scene_reconstruction.scene.background.environment_physics import current
from calibrated_scene_reconstruction.scene.background.observed_context import parameter_digest
from calibrated_scene_reconstruction.scene.background.part_source_replace import intersection
from calibrated_scene_reconstruction.scene.background import source_replace
from calibrated_scene_reconstruction.scene.review.environment import digest, protected_state


def scope(config, parts, ownership):
    owner, names = config.get('owner'), config.get('parts')
    available = {r['part'] for r in parts if r['entity'] == owner and not r['external']}
    if (not isinstance(owner, str) or not available
            or any(r['entity'] == owner and r['external'] for r in parts)
            or ownership.get(owner, {}).get('role') != 'native_replacement'):
        raise ValueError('Use an explicitly declared native-replacement background owner')
    if not isinstance(names, list) or not names or len(set(names)) != len(names) or not set(names) <= available:
        raise ValueError('Explicit distinct actual background parts required')
    if (not isinstance(config.get('reason'), str) or not config['reason'].strip()
            or not isinstance(config.get('observations'), list) or not config['observations']
            or not isinstance(config.get('evidence_sha256'), dict) or not config['evidence_sha256']):
        raise ValueError('Reviewed replacement reason, source observations and visual evidence required')
    margin = config.get('surface_margin_m', .035)
    if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not .01 <= margin <= .1:
        raise ValueError('Bounded native surface margin required')
    return owner, names, margin


def prepare(session, context_manifest, geometry_job, config, output):
    context_manifest, geometry_job, config, out = map(lambda p: Path(p).resolve(), (context_manifest, geometry_job, config, output))
    if out.exists() or not out.is_relative_to(session.dir.resolve()) or (session/'qa/independent_freeze.json').exists():
        raise ValueError('New unfrozen session-local field proposal required')
    native = verify(session); audit, _ = current(session)
    source, job, request = [json.loads(p.read_text()) for p in (context_manifest, geometry_job, config)]
    cfg = session.config('environment.json')
    if (job['native_manifest_sha256'] != digest(session/'scene/environment/twin/current.json')
            or job['geometry_sha256'] != digest(job['geometry'])
            or json.loads(Path(job['output']).read_text())['usd'] != audit['usd']):
        raise ValueError('Use current physically audited geometry')
    if (native['splat_context']['evidence_sha256'] != digest(context_manifest)
            or source['protected_task_state'] != protected_state(session)
            or source['parameters_sha256'] != parameter_digest(cfg['parameters'])
            or source['camera_sha256'] != digest(session/'parsing/posed_frames/cameras.npz')):
        raise ValueError('Use the exact current field, capture and protected state')
    owner, names, margin = scope(request, job['parts'], source['appearance_ownership'])
    inputs = {**source['input_sha256'], **request['evidence_sha256']}
    cloud = context_manifest.parent/'context.ply'; field = PlyData.read(str(cloud))['vertex'].data
    matrix = np.asarray(source['matrix']); scale = np.cbrt(np.linalg.det(matrix[:3, :3]))
    points = (np.c_[field['x'], field['y'], field['z'], np.ones(len(field))]@matrix.T)[:, :3]
    sigma = scale*np.exp(np.c_[field['scale_0'], field['scale_1'], field['scale_2']]).max(1)
    owned, more, report = source_replace.prepare(session, points, {owner: request['observations']}, {owner})
    inputs.update(more)
    worker = Path(__file__).with_name('field_replace_worker.py').resolve()
    for p in (context_manifest, geometry_job, Path(job['geometry']), Path(job['output']), config, cloud, worker, Path(__file__).resolve()):
        inputs[str(p)] = digest(p)
    if any(digest(p) != sha for p, sha in inputs.items()):
        raise ValueError('Replacement source changed')
    out.mkdir(parents=True)
    np.savez_compressed(out/'cloud_world.npz', points=points, sigma=sigma)
    selection = {'geometry': job['geometry'], 'parts': job['parts'], 'cloud': str(out/'cloud_world.npz'),
                 'entities': [owner], 'replacement_parts': {owner: names}, 'margin_m': margin, 'output': str(out/'near.npy')}
    (out/'selection.json').write_text(json.dumps(selection, indent=2)+'\n')
    with (out/'selection.log').open('w') as log:
        subprocess.run([str(ROOT/'tools/run_blender.sh'), '--python', str(worker), '--', str(out/'selection.json')],
                       check=True, stdout=log, stderr=subprocess.STDOUT)
    near = np.load(out/'near.npy'); removed = intersection(near, owned)
    np.save(out/'removed.npy', removed)
    np.savez_compressed(out/'source_rows.npz', retained=np.flatnonzero(~removed), removed=np.flatnonzero(removed))
    PlyData([PlyElement.describe(field[~removed], 'vertex')], text=False).write(str(out/'context.ply'))
    with (out/'transcode.log').open('w') as log:
        subprocess.run([str(ROOT/'envs/splat/3dgrut/.venv/bin/python'), '-m', 'threedgrut.export.scripts.transcode',
                        str(out/'context.ply'), '-o', str(out/'context.usdz'), '-f', 'nurec'],
                       check=True, stdout=log, stderr=subprocess.STDOUT)
    # Verify the surviving attributes, rather than only the row count.
    saved = PlyData.read(str(out/'context.ply'))['vertex'].data
    if saved.dtype != field.dtype or saved.tobytes() != field[~removed].tobytes():
        raise ValueError('Surviving original kernels changed')
    for p in (out/'selection.json', out/'source_rows.npz', out/'removed.npy', out/'context.ply'):
        inputs[str(p)] = digest(p)
    entities = sorted({r['entity'] for r in job['parts'] if not r['external']})
    owners = backing(entities, cfg['completion'])
    result = copy.deepcopy(source)
    for key in ('selection_frozen', 'coverage_reuse', 'quality_filter'):
        result.pop(key, None)
    result.update(asset_sha256=digest(out/'context.usdz'), appearance_ownership=owners, gap_masks={},
                  native_entities=entities, background_entities=entities,
                  replaced_entities=[e for e, r in owners.items() if r['role'] == 'native_replacement'],
                  gap_fill_entities=[e for e, r in owners.items() if r['role'] == 'inferred_enclosure'],
                  retained=len(saved), removed=source.get('removed', 0)+int(removed.sum()),
                  input_sha256=inputs, geometry_rebuild_required=True, photorealism_approved=False,
                  owned_part_replacement={'owner': owner, 'parts': names, 'removed_rows': int(removed.sum()),
                      'source_manifest': str(context_manifest), 'source_manifest_sha256': digest(context_manifest),
                      'source_rows': int(owned.sum()), 'near_rows': int(near.sum()), 'source_observations': report,
                      'selection_operation': 'reviewed_source_intersection_named_part_proximity', 'quality_pass': False})
    result['appearance_plan'] = {'version': 1, 'routes': {e: {'route': 'mesh', 'quality': 'unresolved',
        'reason': 'Provisional native backing after local field replacement; fresh composed review required.',
        'scores': [], 'free_view_review': {'pass': False, 'note': 'Unapproved candidate'}}
        for e, r in owners.items() if r['role'] != 'inferred_enclosure'},
        'input_sha256': {}, 'protected_task_state': protected_state(session)}
    if any(digest(p) != sha for p, sha in inputs.items()):
        raise ValueError('Replacement evidence changed during export')
    (out/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('session', 'context_manifest', 'geometry_job', 'config'):
        p.add_argument(name)
    p.add_argument('--output', required=True); a = p.parse_args()
    result = prepare(Session(a.session), a.context_manifest, a.geometry_job, a.config, a.output)
    print('Unapproved local replacement:', result['owned_part_replacement']['removed_rows'], 'removed;', result['retained'], 'unchanged rows retained')


if __name__ == '__main__':
    main()
