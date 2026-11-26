"""Verify dark GS using field and empty-scene backdrop controls.

An opaque photographic surface can match the default background colour. Its
RGB difference from one empty image is not a visibility test. Known black/white
empty controls must demonstrate that the background actually reaches the ray;
the same black/white field controls must remain invariant. Frozen projections,
reviewed source ownership, detail and subsequent free-view QA still apply.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from plyfile import PlyData
import hashlib

from calibrated_scene_reconstruction.scene.background.gs_opacity import MEASUREMENT_KEYS, stable_mask, validate_field
from calibrated_scene_reconstruction.scene.background.gs_regions import sharp_mask
from calibrated_scene_reconstruction.scene.review.environment import digest, protected_state
from calibrated_scene_reconstruction.scene.review.frozen_scope import validate as validate_scope


def causal_mask(black, white, empty_black, empty_white):
    arrays = [np.asarray(a) for a in (black, white, empty_black, empty_white)]
    if any(a.shape != arrays[0].shape for a in arrays): raise ValueError('Aligned field/empty backdrop controls required')
    black, white, empty_black, empty_white = arrays
    invariant = stable_mask(black, white)
    reachable = np.max(abs(empty_black.astype(float)-empty_white), axis=2)>32
    reachable = cv2.erode(reachable.astype('uint8'), np.ones((3, 3), 'uint8'))>0
    # Clipping in every channel can hide background transmission. One usable
    # channel is sufficient; a saturated white glow cannot certify opacity.
    measurable = np.any((black < 254)&(white < 254), axis=2)
    return invariant&reachable&measurable


def measure(session, usd, cameras, output):
    from calibrated_scene_reconstruction.scene.background.blender_env import verify
    from calibrated_scene_reconstruction.scene.background.environment_physics import current
    usd, cameras, out = map(lambda p: Path(p).resolve(), (usd, cameras, output))
    if out.exists() or not out.is_relative_to(session.dir.resolve()) or (session/'qa/independent_freeze.json').exists():
        raise ValueError('New unfrozen empty-backdrop measurement required')
    verify(session); audit, _ = current(session)
    if audit['usd'] != str(usd): raise ValueError('Use the current physically audited export')
    pointer = session/'scene/environment/twin/current.json'; before = digest(pointer)
    out.mkdir(parents=True); snapshot = out/'native_pointer_snapshot.json'; snapshot.write_bytes(pointer.read_bytes())
    inputs = {str(p): digest(p) for p in (usd, cameras, snapshot, Path(__file__).resolve())}
    for name, colour in (('black', '0'), ('white', '1')):
        with (out/(name+'.log')).open('w') as log:
            subprocess.run(['tools/run_isaac.sh', 'apps/scene/review_usd_cameras.py', str(usd), str(cameras),
                '--renderer', 'rt', '--context-only', '--context-empty', '--context-background', colour, colour, colour,
                '--output', str(out/name)], check=True, stdout=log, stderr=subprocess.STDOUT)
        if not (out/name/'review.json').is_file(): raise ValueError('Empty-backdrop native render incomplete')
        print('Empty-backdrop control complete', name, flush=True)
    current(session)
    if digest(pointer) != before or any(digest(p) != sha for p, sha in inputs.items()):
        raise ValueError('Backdrop measurement scene changed during rendering')
    (out/'controls.json').write_text(json.dumps({'input_sha256': inputs,
        'checked_native_pointer': {'path':str(pointer), 'sha256':before, 'snapshot':str(snapshot)},
        'photorealism_approved':False}, indent=2)+'\n')


def prepare(session, context_path, measurement, controls, coverage_path, scope_path, request_path, output):
    paths = [Path(p).resolve() for p in (context_path, measurement, controls, coverage_path, scope_path, request_path, output)]
    context_path, measurement, controls, coverage_path, scope_path, request_path, out = paths
    if out.exists() or not out.is_relative_to(session.dir.resolve()) or (session/'qa/independent_freeze.json').exists():
        raise ValueError('New unfrozen causal source-coverage proposal required')
    context, base, scope, request = [json.loads(p.read_text()) for p in (context_path, coverage_path, scope_path, request_path)]
    review_paths = [measurement/'black/review.json', measurement/'white/review.json',
                    controls/'black/review.json', controls/'white/review.json', measurement/'field/review.json']
    black, white, empty_black, empty_white, field = reviews = [json.loads(p.read_text()) for p in review_paths]
    for review in reviews: validate_field(review, context, context_path)
    for review, kind, colour in [(black,'context_only',[0.,0.,0.]), (white,'context_only',[1.,1.,1.]),
                                (empty_black,'context_empty',[0.,0.,0.]), (empty_white,'context_empty',[1.,1.,1.])]:
        if review.get('diagnostic') != kind or review.get('context_background_override') != colour:
            raise ValueError('Known actual field/empty black and white controls required')
    if (any(any(r.get(k) != field.get(k) for k in MEASUREMENT_KEYS) for r in reviews)
            or base['render_usd'] != field['usd'] or scope['usd'] != field['usd']
            or scope['camera_manifest'] != field['camera_manifest']
            or scope['protected_task_state'] != protected_state(session)
            or request['owners'] != base.get('supplementary_scope', {}).get('owners')
            or Path(request['field_review']).resolve() != review_paths[-1]):
        raise ValueError('Same field, frozen scope and already reviewed background identities required')
    validate_scope(scope_path, field)
    vertices = PlyData.read(str(context_path.parent/'context.ply'))['vertex'].data
    if base['field_rows_sha256'] != hashlib.sha256(vertices.tobytes()).hexdigest():
        raise ValueError('Actual retained field rows changed')
    inputs = {}
    controls_path = controls/'controls.json'; controls_proof = json.loads(controls_path.read_text())
    for proof in [base, scope, controls_proof, *reviews]:
        for p, sha in proof['input_sha256'].items():
            if p in inputs and inputs[p] != sha: raise ValueError('Conflicting causal coverage inputs')
            inputs[p] = sha
    for p in [context_path, coverage_path, scope_path, request_path, controls_path, *review_paths, Path(__file__).resolve(),
              Path(stable_mask.__code__.co_filename).resolve(), Path(sharp_mask.__code__.co_filename).resolve()]: inputs[str(p)] = digest(p)
    legacy_path = Path(base['opacity_evidence']); legacy = json.loads(legacy_path.read_text())
    legacy_rows = {r['frame']: r for r in legacy['rows']}; inputs[str(legacy_path)] = digest(legacy_path)
    if legacy['field_rows_sha256'] != base['field_rows_sha256']: raise ValueError('Matching existing opacity evidence required')
    indexes = [{v['source']:v for v in r['views']} for r in reviews]
    frozen = {v['source']:v for v in scope['rows']}
    observations = []
    for owner, declaration in request['owners'].items():
        path = Path(declaration['observations']).resolve()
        if inputs.get(str(path)) != digest(path): raise ValueError('Background identity observations changed')
        observations.extend(r for r in json.loads(path.read_text())['records'] if r['entity'] == owner and r.get('mask'))
    if any(digest(p) != sha for p, sha in inputs.items()): raise ValueError('Causal source evidence changed')
    out.mkdir(parents=True); result = copy.deepcopy(base); rows = []; reports = []; evidence = {}
    for row in observations:
        source = str(Path(row['source']).resolve()); frame = row['frame']
        if not row.get('appearance_owner_review', '').strip() or source not in frozen:
            raise ValueError('Explicit source owner and frozen projection required')
        if source not in evidence:
            views = [index[source] for index in indexes]
            files = [view['renders'][0] for view in views]
            images = [cv2.imread(p) for p in files]
            causal = causal_mask(*images[:4]); h, w = causal.shape
            opaque = cv2.resize(cv2.imread(legacy_rows[frame]['mask'],0), (w,h), interpolation=cv2.INTER_NEAREST)>128
            causal &= opaque
            evidence[source] = (causal, images[-1])
            for view in views:
                for p in [source, *view['renders']]: inputs[str(Path(p).resolve())] = digest(p)
        causal, image = evidence[source]; h, w = causal.shape
        raw = cv2.resize(cv2.imread(source), (w,h), interpolation=cv2.INTER_AREA)
        owner = cv2.resize(cv2.imread(row['mask'],0), (w,h), interpolation=cv2.INTER_NEAREST)>128
        frozen_keep = cv2.resize(cv2.imread(frozen[source]['mask'],0), (w,h), interpolation=cv2.INTER_NEAREST)>128
        if inputs.get(str(Path(row['mask']).resolve())) != row['mask_sha256']:
            raise ValueError('Source owner mask is not bound to the reviewed coverage')
        trusted = sharp_mask(raw,image,owner&frozen_keep)&causal
        width = min(640,w); height = round(h*width/w); path = out/(row['entity']+'_'+Path(frame).stem+'.png')
        cv2.imwrite(str(path),cv2.resize(trusted.astype('uint8')*255,(width,height),interpolation=cv2.INTER_NEAREST))
        inputs[str(path)] = digest(path)
        rows.append({'entity':row['entity'],'frame':frame,'mask':str(path),'mask_sha256':digest(path),
                     'size':[width,height],'pixels':int(trusted.sum())})
        reports.append({'entity':row['entity'],'frame':frame,'causal_opaque_source_pixels':int(trusted.sum())})
    replaced = {(r['entity'],r['frame']) for r in rows}
    result.update(rows=[r for r in base['rows'] if (r['entity'],r['frame']) not in replaced]+rows,
                  input_sha256=inputs, photorealism_approved=False,
                  causal_contribution={'controls':[str(p) for p in review_paths[:4]],'rows':reports,'scope':__doc__.strip()})
    if any(digest(p) != sha for p, sha in inputs.items()): raise ValueError('Causal coverage changed during measurement')
    (out/'coverage.json').write_text(json.dumps(result,indent=2)+'\n'); return result


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='action',required=True)
    measure_parser = sub.add_parser('measure')
    for name in ('session','usd','cameras'): measure_parser.add_argument(name)
    measure_parser.add_argument('--output',required=True)
    prepare_parser = sub.add_parser('prepare')
    for name in ('session','context','measurement','controls','coverage','frozen_scope','request'): prepare_parser.add_argument(name)
    prepare_parser.add_argument('--output',required=True); args = parser.parse_args()
    if args.action == 'measure': measure(Session(args.session),args.usd,args.cameras,args.output)
    else: prepare(Session(args.session),args.context,args.measurement,args.controls,args.coverage,args.frozen_scope,args.request,args.output)


if __name__ == '__main__': main()
