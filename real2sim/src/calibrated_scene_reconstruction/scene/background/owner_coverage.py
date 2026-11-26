"""Recover explicitly reviewed background coverage behind coarse task masks.

Old semantic/zone masks can also exclude a distant desk of the same class.
Only an agent-reviewed background identity may use this supplementary path.
Its measured opaque GS must remain outside actual frozen geometry projections,
including their existing metric protection margins. Original field bytes,
original masks and other owners are unchanged. This never approves appearance.
"""
import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np

from calibrated_scene_reconstruction.scene.background.gs_regions import sharp_mask
from calibrated_scene_reconstruction.scene.background.gs_opacity import validate_field, MEASUREMENT_KEYS
from calibrated_scene_reconstruction.scene.background.observed_context import parameter_digest
from calibrated_scene_reconstruction.scene.review.environment import digest, protected_state
from calibrated_scene_reconstruction.scene.review.frozen_scope import validate as validate_scope, meshes
from calibrated_scene_reconstruction.scene.review.frozen_scope_cache import margins, same_geometry


def reviewed_mask(observation, frozen, opaque, raw, field, empty):
    """Neither source similarity nor semantic identity can remove frozen pixels."""
    shape = field.shape[:2]
    resize = lambda a: cv2.resize(a.astype('uint8'), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)>128
    keep = resize(observation)&resize(frozen)
    keep = cv2.erode(keep.astype('uint8'), np.ones((3, 3), 'uint8'))>0
    raw = cv2.resize(raw, (shape[1], shape[0]), interpolation=cv2.INTER_AREA)
    return sharp_mask(raw, field, keep)&resize(opaque)&(abs(field.astype(float)-empty).max(2)>16)


def prepare(session, context_path, coverage_path, scope_path, request_path, output):
    context_path, coverage_path, scope_path, request_path, out = [Path(p).resolve() for p in
        (context_path, coverage_path, scope_path, request_path, output)]
    if out.exists() or not out.is_relative_to(session.dir.resolve()) or (session/'qa/independent_freeze.json').exists():
        raise ValueError('New unfrozen session-local supplementary coverage required')
    context, coverage, scope, request = [json.loads(p.read_text()) for p in
        (context_path, coverage_path, scope_path, request_path)]
    if (context['protected_task_state'] != protected_state(session)
            or context['parameters_sha256'] != parameter_digest(session.config('environment.json')['parameters'])
            or scope['protected_task_state'] != context['protected_task_state']
            or scope.get('projection_uncertainty_m') != margins(session)
            or coverage.get('version') != 2 or not coverage.get('opacity_evidence')):
        raise ValueError('Unchanged protected state, parameters and measured opacity required')
    owners = request.get('owners')
    if not isinstance(owners, dict) or not owners:
        raise ValueError('Explicit reviewed background owners required')
    for owner, declaration in owners.items():
        if (owner not in context['appearance_ownership']
                or context['appearance_ownership'][owner]['role'] == 'inferred_enclosure'
                or not declaration.get('background_identity_review', '').strip()):
            raise ValueError('Only explicitly identified observed background owners may supplement coverage')
    field_path = Path(request['field_review']).resolve(); empty_path = Path(request['empty_review']).resolve()
    field, empty = [json.loads(p.read_text()) for p in (field_path, empty_path)]
    validate_field(field, context, context_path)
    validate_field(empty, context, context_path)
    if (field.get('diagnostic') != 'context_only' or empty.get('diagnostic') != 'context_empty'
            or field.get('context_background_override') is not None
            or any(field.get(k) != empty.get(k) for k in (*MEASUREMENT_KEYS, 'context_background_override'))
            or coverage['render_usd'] != field['usd']):
        raise ValueError('Same actual measured field, empty render and frozen geometry required')
    validate_scope(scope_path, {'usd': scope['usd'], 'camera_manifest': scope['camera_manifest'],
                                'input_sha256': scope['input_sha256']})
    if scope['usd'] != field['usd']:
        from pxr import Usd
        graph = session.scene_graph
        args = (graph['task']['support'], graph.get('dynamic_objects', {}))
        if not same_geometry(meshes(Usd.Stage.Open(field['usd']), *args),
                             meshes(Usd.Stage.Open(scope['usd']), *args)):
            raise ValueError('Frozen world triangles differ from the measured field scene')
    # A projection subset is safe only for the exact same registered captured
    # rays. It cannot donate masks to a shifted or novel camera.
    cameras = json.loads(Path(field['camera_manifest']).read_text())
    scope_cameras = json.loads(Path(scope['camera_manifest']).read_text())
    anchors = {str(Path(a['source']).resolve()): a for a in cameras['anchors']}
    for anchor in scope_cameras['anchors']:
        original = anchors.get(str(Path(anchor['source']).resolve()))
        if (original is None or anchor['K'] != original['K']
                or anchor['views'][0] != original['views'][0]):
            raise ValueError('Frozen coverage needs identical captured camera rays')
    opacity_path = Path(coverage['opacity_evidence']).resolve(); opacity = json.loads(opacity_path.read_text())
    if (opacity['field_rows_sha256'] != coverage['field_rows_sha256']
            or opacity['measurement'] != {k: field.get(k) for k in MEASUREMENT_KEYS}):
        raise ValueError('Same-field actual backdrop opacity required')
    inputs = {}
    for proof in (context, coverage, scope, field, empty, opacity):
        for p, sha in proof['input_sha256'].items():
            if p in inputs and inputs[p] != sha:
                raise ValueError('Conflicting supplementary source evidence: '+p)
            inputs[p] = sha
    for p in (context_path, coverage_path, scope_path, request_path, field_path, empty_path, opacity_path,
              Path(__file__).resolve(), Path(margins.__code__.co_filename).resolve(),
              Path(sharp_mask.__code__.co_filename).resolve(), Path(meshes.__code__.co_filename).resolve()):
        inputs[str(p)] = digest(p)
    frozen = {r['source']: r for r in scope['rows']}
    views = {r['source']: r for r in field['views']}; empties = {r['source']: r for r in empty['views']}
    opaque = {r['frame']: r for r in opacity['rows']}; records = []
    for owner, declaration in owners.items():
        path = Path(declaration['observations']).resolve(); inputs[str(path)] = digest(path)
        for row in json.loads(path.read_text())['records']:
            if row['entity'] != owner or not row.get('mask'): continue
            if not row.get('appearance_owner_review', '').strip() or row.get('rejected_instance_review'):
                raise ValueError('Each supplementary mask needs an explicit background identity review')
            source = str(Path(row['source']).resolve())
            if source not in frozen or source not in views or row['frame'] not in opaque:
                raise ValueError('Every supplementary source needs matching frozen and actual-field measurements')
            if Path(source) != (session/'splat/data/images'/row['frame']).resolve():
                raise ValueError('Original registered capture required')
            for key in ('source', 'mask'):
                p = Path(row[key]).resolve(); inputs[str(p)] = row[key+'_sha256']
            records.append((row, source))
    if not records or any(digest(p) != sha for p, sha in inputs.items()):
        raise ValueError('Supplementary source evidence missing or changed')
    out.mkdir(parents=True); rows = []
    for row, source in records:
        frame = row['frame']; ref = views[source]; zero = empties[source]
        image = cv2.imread(ref['renders'][0]); h, w = image.shape[:2]
        mask = reviewed_mask(cv2.imread(row['mask'], 0), cv2.imread(frozen[source]['mask'], 0),
                             cv2.imread(opaque[frame]['mask'], 0), cv2.imread(source), image,
                             cv2.imread(zero['renders'][0]))
        width = min(640, w); height = round(h*width/w)
        target = out/(row['entity']+'_'+Path(frame).stem+'.png')
        cv2.imwrite(str(target), cv2.resize(mask.astype('uint8')*255, (width, height), interpolation=cv2.INTER_NEAREST))
        inputs[str(target)] = digest(target)
        rows.append({'entity': row['entity'], 'frame': frame, 'mask': str(target),
                     'mask_sha256': digest(target), 'size': [width, height], 'pixels': int(mask.sum())})
    result = copy.deepcopy(coverage)
    replaced = {(r['entity'], r['frame']) for r in rows}
    result.update(rows=[r for r in coverage['rows'] if (r['entity'], r['frame']) not in replaced]+rows,
                  input_sha256=inputs, photorealism_approved=False,
                  supplementary_scope={'owners': owners, 'frozen_scope': str(scope_path), 'scope': __doc__.strip()})
    if any(digest(p) != sha for p, sha in inputs.items()):
        raise ValueError('Supplementary evidence changed during measurement')
    (out/'coverage.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('session', 'context', 'coverage', 'frozen_scope', 'request'): parser.add_argument(name)
    parser.add_argument('--output', required=True); args = parser.parse_args()
    result = prepare(Session(args.session), args.context, args.coverage, args.frozen_scope, args.request, args.output)
    print('Supplementary background coverage; appearance unapproved:', len(result['rows']), 'rows')


if __name__ == '__main__': main()
