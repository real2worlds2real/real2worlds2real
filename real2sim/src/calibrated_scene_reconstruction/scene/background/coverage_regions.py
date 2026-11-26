"""Prepare sharp native-gap coverage for newly observed background owners.

An agent-reviewed image identity need not have an old GS inventory claim ID.
These masks measure appearance only: they cannot assign, select or delete
kernel rows, and do not establish whole-object geometry or free-view quality.
"""
import argparse
import json
from pathlib import Path

import cv2

from calibrated_scene_reconstruction.scene.background import gs_regions
from calibrated_scene_reconstruction.scene.review.environment import digest, protected_state


def prepare(session, review, observations, entities, output):
    out, review = Path(output).resolve(), Path(review).resolve()
    if out.exists() or not out.is_relative_to(session.dir.resolve()):
        raise ValueError('New session-local coverage regions required')
    if (session/'qa/independent_freeze.json').exists():
        raise ValueError('Frozen session: create an environment derivative')
    source = json.loads(review.read_text())
    if source.get('diagnostic') != 'context_source':
        raise ValueError('Use original GS source/free-view comparisons')
    available = {e['id'] for e in session.config('environment.json')['parameters']['entities']}
    if not entities or len(set(entities)) != len(entities) or not set(entities) <= available:
        raise ValueError('Explicit distinct current background owners required')
    views = {Path(v['source']).name: v for v in source['views']}
    camera = session/'parsing/posed_frames/cameras.npz'
    inputs = {str(p.resolve()): digest(p) for p in (review, camera, Path(__file__), Path(gs_regions.__file__))}
    rows, seen = [], set()
    out.mkdir(parents=True)
    for file in observations:
        file = Path(file).resolve()
        inputs[str(file)] = digest(file)
        for row in json.loads(file.read_text())['records']:
            key = row['entity'], row['frame']
            if row['entity'] not in entities or row['frame'] not in views or not row.get('mask') or key in seen:
                continue
            if not isinstance(row.get('appearance_owner_review'), str) or not row['appearance_owner_review'].strip():
                raise ValueError('Agent must review the exact appearance owner')
            if Path(row['source']).resolve() != (session/'splat/data/images'/row['frame']).resolve():
                raise ValueError('Use original same-capture source pixels')
            for name in ('source', 'mask'):
                if digest(row[name]) != row[name+'_sha256']:
                    raise ValueError('Source owner evidence changed')
                inputs[str(Path(row[name]).resolve())] = row[name+'_sha256']
            view = views[row['frame']]
            image = cv2.imread(view['renders'][0])
            raw = cv2.imread(row['source'])
            h, w = image.shape[:2]
            good = gs_regions.sharp_mask(cv2.resize(raw, (w, h), interpolation=cv2.INTER_AREA), image,
                              cv2.resize(cv2.imread(row['mask'], 0), (w, h), interpolation=cv2.INTER_NEAREST) > 128)
            mask = out/(row['entity']+'_'+Path(row['frame']).stem+'.png')
            cv2.imwrite(str(mask), cv2.resize(good.astype('uint8')*255, (raw.shape[1], raw.shape[0]), interpolation=cv2.INTER_NEAREST))
            for p in [mask, *map(Path, view['renders'])]:
                inputs[str(p.resolve())] = digest(p)
            rows.append({'entity': row['entity'], 'frame': row['frame'], 'source': row['source'],
                         'mask': str(mask), 'accepted_pixels': int(good.sum())})
            seen.add(key)
    if any(sum(r['entity'] == e and r['accepted_pixels'] >= 32 for r in rows) < 2 for e in entities):
        raise ValueError('Two usable source regions required per owner')
    if any(digest(p) != sha for p, sha in inputs.items()):
        raise ValueError('Coverage source changed during preparation')
    result = {'version': 1, 'coverage_only': True, 'regions': rows, 'input_sha256': inputs,
              'protected_task_state': protected_state(session), 'scope': __doc__.strip()}
    (out/'regions.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('session'); p.add_argument('review')
    p.add_argument('--observations', nargs='+', required=True)
    p.add_argument('--entities', nargs='+', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    prepare(Session(a.session), a.review, a.observations, a.entities, a.output)


if __name__ == '__main__':
    main()
