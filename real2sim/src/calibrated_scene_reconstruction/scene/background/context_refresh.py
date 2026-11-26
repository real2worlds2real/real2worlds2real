"""Reuse exact captured GS bytes while rebuilding unapproved native background.

Background edits invalidate native cutouts, not the captured field itself.
This bootstrap removes every cutout and keeps full native backing. New export,
physics, field/backdrop diagnostics and coverage are required before rebinding
appearance. It never transfers an appearance approval or changes task geometry.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

from calibrated_scene_reconstruction.scene.background.observed_context import parameter_digest
from calibrated_scene_reconstruction.scene.background.appearance_ownership import specification
from calibrated_scene_reconstruction.scene.review.environment import digest, protected_state


def backing(entities, enclosure):
    if not entities or len(set(entities)) != len(entities):
        raise ValueError('Explicit distinct background entities required')
    inferred = {r['entity'] for r in enclosure['surfaces'] if r['provenance'] == 'inferred'}
    if not inferred <= set(entities):
        raise ValueError('Keep every inferred enclosure surface')
    owners = {e: {'role': 'inferred_enclosure' if e in inferred else 'native_replacement',
                  'basis': 'Provisional complete native backing after background edit; fresh actual coverage and full-scene review required.'}
              for e in entities}
    specification(owners, set(entities), set(entities)-inferred, inferred, enclosure)
    return owners


def prepare(session, context_manifest, entities, output):
    out, source = Path(output).resolve(), Path(context_manifest).resolve()
    if out.exists() or not out.is_relative_to(session.dir.resolve()) or (session/'qa/independent_freeze.json').exists():
        raise ValueError('New unfrozen background bootstrap required')
    original = json.loads(source.read_text());camera = session/'parsing/posed_frames/cameras.npz'
    if original['camera_sha256'] != digest(camera) or original['protected_task_state'] != protected_state(session):
        raise ValueError('Capture and frozen task/robot must be unchanged')
    inputs = dict(original['input_sha256'])
    for p, sha in inputs.items():
        if digest(p) != sha:
            raise ValueError('GS selection evidence changed: '+p)
    cloud = source.parent/'context.ply';cfg = session.config('environment.json')
    owners = backing(entities, cfg['completion'])
    out.mkdir(parents=True);shutil.copyfile(cloud, out/'context.ply')
    for p in [source, cloud, camera, Path(__file__).resolve(), out/'context.ply']:
        inputs[str(p.resolve())] = digest(p)
    observed = set(entities)-{e for e, r in owners.items() if r['role'] == 'inferred_enclosure'}
    plan = {'version': 1, 'routes': {e: {'route': 'mesh', 'quality': 'unresolved',
             'reason': 'Provisional native backing; fresh retained-field coverage required.', 'scores': [],
             'free_view_review': {'pass': False, 'note': 'New composite not reviewed.'}} for e in sorted(observed)},
            'input_sha256': {str(source): digest(source)}, 'protected_task_state': protected_state(session)}
    result = copy.deepcopy(original)
    result.update(parameters_sha256=parameter_digest(cfg['parameters']),
                  source_sha256={str(p.resolve()): digest(p) for p in session.config_dir.joinpath('agent/environment').rglob('*.py')},
                  appearance_ownership=owners, gap_masks={}, appearance_plan=plan,
                  native_entities=entities, replaced_entities=sorted(observed),
                  gap_fill_entities=sorted(set(entities)-observed), background_entities=sorted(entities),
                  input_sha256=inputs, photorealism_approved=False, geometry_rebuild_required=True,
                  selection_frozen={'manifest': str(source), 'manifest_sha256': digest(source),
                                    'ply_sha256': hashlib.sha256(cloud.read_bytes()).hexdigest(),
                                    'scope': 'Exact original GS field reused. Native geometry changed: all previous cutouts and acceptance discarded.'})
    if any(digest(p) != sha for p, sha in inputs.items()):
        raise ValueError('GS bootstrap evidence changed')
    (out/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('session');p.add_argument('context_manifest')
    p.add_argument('--entities', nargs='+', required=True);p.add_argument('--output', required=True);a=p.parse_args()
    prepare(Session(a.session), a.context_manifest, a.entities, a.output)


if __name__ == '__main__':
    main()
