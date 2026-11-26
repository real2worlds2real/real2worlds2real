"""Reject observed background instances silently omitted from agent assemblies.

This matches explicit identities, never class names. It does not detect every
object, measure shape/material accuracy, or certify the appearance of a match.
"""
import argparse
import json
from pathlib import Path

from calibrated_scene_reconstruction.scene.review.environment import digest, protected_state


def audit(entities, aliases, records):
    identities = {name: name for name in entities}
    for entity, names in aliases.items():
        if entity not in identities:
            raise ValueError('Observation aliases need an existing explicit entity')
        for name in names:
            if name in identities and identities[name] != entity:
                raise ValueError('Ambiguous source instance alias')
            identities[name] = entity
    frames = {}; unmatched = []; rejected = []
    for row in records:
        if not row.get('mask'):
            continue
        frame, raw = row['frame'], row['entity']
        rejection = row.get('rejected_instance_review')
        if rejection is not None:
            if (not isinstance(rejection, str) or not rejection.strip()
                    or row.get('geometry_fit_scope') in ('whole_object', 'occluded_object')):
                raise ValueError('Explicit rejected-instance explanation cannot coexist with accepted geometry scope')
            rejected.append({'entity': raw, 'frame': frame, 'reason': rejection})
            continue
        canonical = identities.get(raw)
        item = {'entity': raw, 'matched_entity': canonical}
        frames.setdefault(frame, []).append(item)
        if canonical is None:
            unmatched.append({'entity': raw, 'frame': frame,
                              'issue': 'Observed instance has no explicit native/GS owner; class-name matching is forbidden'})
    counts = {frame: {'observed_identities': sorted({r['entity'] for r in rows}),
                      'matched_entities': sorted({r['matched_entity'] for r in rows if r['matched_entity'] is not None}),
                      'unmatched_identities': sorted({r['entity'] for r in rows if r['matched_entity'] is None})}
              for frame, rows in frames.items()}
    return {'pass': bool(frames) and not unmatched, 'frames': counts, 'unmatched': unmatched,
            'rejected': rejected, 'scope': __doc__.strip()}


def run(session, observations, output):
    output = Path(output).resolve()
    if output.exists() or not output.is_relative_to(session.dir.resolve()):
        raise ValueError('New session-local source presence audit required')
    cfgpath = session.config_dir/'environment.json'; cfg = session.config('environment.json')
    camera = session/'parsing/posed_frames/cameras.npz'
    import numpy as np
    names = set(map(str, np.load(camera)['names']))
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = output.with_name(output.stem+'_environment.json')
    solver = output.with_name(output.stem+'_solver.py')
    if snapshot.exists() or solver.exists():
        raise ValueError('New immutable presence audit snapshots required')
    snapshot.write_bytes(cfgpath.read_bytes()); solver.write_bytes(Path(__file__).read_bytes())
    inputs = {str(p.resolve()): digest(p) for p in [snapshot, solver, camera]}; records = []
    for path in observations:
        path = Path(path).resolve(); inputs[str(path)] = digest(path)
        for row in json.loads(path.read_text())['records']:
            if not row.get('mask'):
                continue
            if row['frame'] not in names or Path(row['frame']).name != row['frame']:
                raise ValueError('Use registered source observations')
            source = (session/'splat/data/images'/row['frame']).resolve()
            if Path(row['source']).resolve() != source:
                raise ValueError('Use the exact original registered image')
            for key in ['source', 'mask']:
                p = Path(row[key]).resolve(); expected = row[key+'_sha256']
                if digest(p) != expected:
                    raise ValueError('Source instance observation changed')
                inputs[str(p)] = expected
            records.append(row)
    result = audit({e['id'] for e in cfg['parameters']['entities']}, cfg.get('observation_aliases', {}), records)
    result.update(input_sha256=inputs, protected_task_state=protected_state(session),
                  environment_sha256=digest(snapshot), observed_entities=len({r['entity'] for r in records}))
    if any(digest(p) != sha for p, sha in inputs.items()):
        raise ValueError('Source presence input changed')
    output.write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session'); parser.add_argument('--observations', nargs='+', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--require-pass', action='store_true')
    a = parser.parse_args(); result = run(Session(a.session), a.observations, a.output)
    print('Explicit source instance coverage', 'PASS' if result['pass'] else 'FAIL',
          len({r['entity'] for r in result['unmatched']}), 'unmatched identities')
    if a.require_pass and not result['pass']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
