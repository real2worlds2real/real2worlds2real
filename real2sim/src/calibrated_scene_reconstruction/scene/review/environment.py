"""Whole-scene visual acceptance, shared by GS and agent-Blender environments.

Numbers diagnose an image; an explicit evidence-backed review approves a scene.
Changed cousins are reviewed for coherence, never pixel-matched to an unchanged
source photograph. Existing frozen deliveries keep their original gate policy.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from calibrated_scene_reconstruction.core.paths import resolve
from calibrated_scene_reconstruction.scene.session import Session

CHECKS = ('shape_and_proportions', 'materials_and_texture_scale', 'lighting_and_exposure',
          'contact_shadows', 'occlusion_and_intersections', 'layout_relationships',
          'cross_view_consistency', 'blur_and_detail', 'holes_and_coverage',
          'ghosts_and_floaters', 'task_preservation', 'export_renderability')


def validate_hybrid_review(request, render, context, entity_specs):
    """Require object-specific semantic review, independent of physics/closure.

    Silhouettes and front/back semantics need an image-backed reviewer; a
    gravity pass or an opaque enclosure cannot answer either question. Missing
    evidence stays failed, including legacy reviews with only broad checkboxes.
    """
    ownership=context.get('appearance_ownership')
    if not ownership:raise ValueError('Hybrid appearance ownership has not been audited')
    visible={k for k,v in ownership.items() if v['role']!='gs'}
    if visible!=set(context['native_entities']):raise ValueError('Visible mesh ownership disagrees with the composed context')
    gap_entities={k for k,v in ownership.items() if v['role']=='gap_fill'}
    if gap_entities!=set(context.get('gap_masks',{})):raise ValueError('Every gap filler needs a current GS coverage mask')
    from calibrated_scene_reconstruction.scene.background.coverage_support import validate as coverage_support
    coverage_support(ownership, context.get('gap_masks', {}))
    review=request.get('hybrid_review')
    if not isinstance(review,dict):raise ValueError('Hybrid needs per-object completeness/orientation and free-view GS floater review')
    views=render['views'];triplets={(r['source'],r['renders'][0],r['renders'][1]) for r in views}
    required={e['id'] for e in entity_specs if e.get('kind') in ('monitor','laptop','chair','stool','sofa','cabinet','cabinet_open','glass_enclosure','office_printer','folding_screen')}
    rows=review.get('objects')
    if not isinstance(rows,list) or len(rows)!=len(required) or {r.get('entity') for r in rows}!=required:
        raise ValueError('Review completeness and front/back for every directional display, seat, cabinet and device')
    passed=True
    for row in rows:
        evidence=row.get('views')
        if (not isinstance(evidence,list) or len(evidence)<2
                or len({r.get('source') for r in evidence})<2
                or any((r.get('source'),r.get('matched'),r.get('novel')) not in triplets for r in evidence)):
            raise ValueError('Each display/chair needs two current source/matched/free-view comparisons')
        spec=next(e for e in entity_specs if e['id']==row['entity'])
        observed=set(spec.get('source_frames',[]))
        if observed and any(Path(r['source']).name not in observed for r in evidence):
            raise ValueError('Object review must use source frames in which that object was observed')
        for criterion in ('complete','front_back_correct'):
            value=row.get(criterion)
            if not isinstance(value,dict) or type(value.get('pass')) is not bool or not value.get('note','').strip():
                raise ValueError('Explicit object completeness/orientation result and visual explanation required')
            passed &= value['pass']
    scans=review.get('gs_free_views')
    novel={r['renders'][1] for r in views}
    if (not isinstance(scans,list) or len(scans)!=len(novel)
            or {r.get('image') for r in scans}!=novel):
        raise ValueError('Inspect GS floaters, holes and double images in every current free view')
    for row in scans:
        if type(row.get('pass')) is not bool or not row.get('note','').strip():
            raise ValueError('Each GS free view needs a visual result and explanation')
        passed &= row['pass']
    ownership_check=review.get('appearance_boundary')
    if (not isinstance(ownership_check,dict) or type(ownership_check.get('pass')) is not bool
            or not ownership_check.get('note','').strip()):
        raise ValueError('Review GS/mesh boundary coverage and overpainting explicitly')
    return bool(passed and ownership_check['pass'])


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def protected_state(session):
    graph = session.config('scene_graph.json', {})
    support = graph.get('task', {}).get('support')
    return {'task': graph.get('task', {}), 'objects': graph.get('dynamic_objects', {}),
            'support': graph.get('furniture', {}).get(support),
            'robot': graph.get('photo_robot'), 'robot_fit': graph.get('_robot_fit')}


def validate_checks(request, allowed_images):
    if request.get('kind') not in ('twin', 'cousin') or request.get('backend') not in ('gs', 'blender', 'hybrid'):
        raise ValueError('Review kind twin/cousin and backend gs/blender/hybrid required')
    checks = request.get('checks')
    if not isinstance(checks, dict) or set(checks) != set(CHECKS):
        raise ValueError('Review every whole-scene quality criterion: '+', '.join(CHECKS))
    for name, value in checks.items():
        if (not isinstance(value, dict) or type(value.get('pass')) is not bool
                or not isinstance(value.get('note'), str) or not value['note'].strip()
                or not isinstance(value.get('evidence'), list) or not value['evidence']):
            raise ValueError('Each criterion needs boolean pass, explanation and rendered evidence: '+name)
        if any(str(resolve(p).resolve()) not in allowed_images for p in value['evidence']):
            raise ValueError('Evidence must belong to this candidate render review: '+name)
    findings = request.get('findings')
    if not isinstance(findings, list) or any(not isinstance(f, dict) or not f.get('action') for f in findings):
        raise ValueError('Quality findings must be explicit actions')
    return all(value['pass'] for value in checks.values()) and all(f.get('status') == 'resolved' for f in findings)


def record(session, request):
    if (session/'qa/independent_freeze.json').exists():
        raise ValueError('Frozen delivery: create a derivative for new environment reviews')
    variant = request.get('variant', 'twin')
    if not isinstance(variant, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,63}', variant):
        raise ValueError('Use a simple environment variant identifier')
    if request.get('kind') != ('twin' if variant == 'twin' else 'cousin'):
        raise ValueError('Twin and each named cousin need separate reviews')
    source = resolve(request['render_review']).resolve()
    if not source.is_relative_to(session.dir.resolve()):
        raise ValueError('Use this session candidate render review')
    render = json.loads(source.read_text())
    views = render.get('views', [])
    if len(views) < 2 or any(len(row.get('renders', [])) < 2 for row in views):
        raise ValueError('Whole-scene review needs multiple captured and novel camera renders')
    images = {str(resolve(p).resolve()) for row in views for p in [row['source'], *row['renders']]}
    inputs = {str(resolve(p).resolve()): h for p, h in render['input_sha256'].items()}
    inputs.update({p: digest(p) for p in images})
    inputs[str(source)] = digest(source)
    candidates = request.get('candidate_inputs')
    if not isinstance(candidates, list) or not candidates:
        raise ValueError('Bind the current candidate assets/configuration explicitly')
    for path in candidates:
        path = resolve(path).resolve()
        inputs[str(path)] = digest(path)
    environment_manifest = None
    physical_digest, physical_failure = None, None
    completion_failure = None
    hybrid_failure = None
    composition_failure = None
    source_failure = None
    visibility_failure = None
    source_scope_failure = None
    if session.config('twin.json', {}).get('background') == 'blender':
        from calibrated_scene_reconstruction.scene.background.blender_env import verify
        built = verify(session, variant)
        if built['asset'] not in inputs:
            raise ValueError('Bind this Blender variant asset in candidate_inputs')
        rendered = render.get('environment') or {}
        if (request.get('backend') not in ('blender', 'hybrid')
                or rendered.get('variant') != variant
                or rendered.get('asset_sha256') != built['output_sha256'][built['asset']]):
            raise ValueError('Render evidence does not show the current Blender environment variant')
        environment_manifest = digest(session/f'scene/environment/{variant}/current.json')
        context=built.get('splat_context')
        if context:
            from calibrated_scene_reconstruction.scene.background import coverage_support
            inputs[str(Path(coverage_support.__file__).resolve())] = digest(coverage_support.__file__)
            from calibrated_scene_reconstruction.scene.background import capture_scope
            inputs[str(Path(capture_scope.__file__).resolve())]=digest(capture_scope.__file__)
            try:
                capture_scope.validate(context.get('appearance_ownership',{}),context.get('gap_masks',{}))
            except (ValueError,KeyError,TypeError) as exc:
                source_scope_failure=str(exc)
            shown=render.get('environment_context') or {}
            if request.get('backend')!='hybrid' or shown.get('asset_sha256')!=context['asset_sha256'] or shown.get('evidence_sha256')!=context['evidence_sha256']:
                raise ValueError('Bind the actual observed context and identify the review backend as hybrid')
            if context.get('source_asset_sha256') and shown.get('source_asset_sha256')!=context['source_asset_sha256']:
                raise ValueError('Render must bind this context original captured GS reference')
            if context.get('source_background_scope') and json.loads(shown.get('source_background_scope_json','{}'))!=context['source_background_scope']:
                raise ValueError('Render must bind the fixed original background scope')
            if not any(value==context['asset_sha256'] for value in render['input_sha256'].values()):
                raise ValueError('Rendered context payload is not bound by its input hash')
            try:
                if not validate_hybrid_review(request,render,context,session.config('environment.json',{}).get('parameters',{}).get('entities',[])):
                    hybrid_failure='Object completeness/orientation, GS free views or appearance boundary has not passed'
            except (ValueError,KeyError,TypeError) as exc:
                hybrid_failure=str(exc)
            if not context.get('appearance_plan'):
                hybrid_failure=(hybrid_failure+'; ' if hybrid_failure else '')+'Image-backed GS/mesh routing plan required; coverage cannot certify sharpness or unseen completion'
            if context.get('coverage_rebuild_required'):
                hybrid_failure=(hybrid_failure+'; ' if hybrid_failure else '')+'Provisional native backing requires actual current GS coverage before acceptance'
            try:
                from calibrated_scene_reconstruction.scene.review.hybrid_composition import validate as composition_check
                inputs.update(composition_check(request['composition_audit'],source,context['appearance_ownership']))
            except (OSError,ValueError,KeyError,TypeError) as exc:
                composition_failure='Current captured GS preservation audit required: '+str(exc)
            try:
                from calibrated_scene_reconstruction.scene.review.source_preservation import validate as source_check
                inputs.update(source_check(request['source_preservation_audit'],source))
            except (OSError,ValueError,KeyError,TypeError) as exc:
                source_failure='Original captured GS preservation required: '+str(exc)
            if built.get('gs_visibility_required'):
                try:
                    from calibrated_scene_reconstruction.scene.review.gs_visibility import validate as visibility_check
                    inputs.update(visibility_check(request['gs_visibility_audit'],source))
                except (OSError,ValueError,KeyError,TypeError) as exc:
                    visibility_failure='Actual visible GS contribution required: '+str(exc)
        from calibrated_scene_reconstruction.scene.background.environment_physics import current as physical_current
        try:
            _, physical_digest = physical_current(session, variant, render)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            physical_failure = str(exc)
        if context or built.get('completion'):
            from calibrated_scene_reconstruction.scene.background.completion import current as completed
            try:
                proof=completed(session,variant)
                if render['input_sha256'].get(proof['usd'])!=proof['input_sha256'][proof['usd']]:
                    raise ValueError('Completion and render must bind the same exported scene')
                pointer=session/f'qa/environment_completion/{variant}/current.json'
                inputs[str(pointer.resolve())]=digest(pointer)
                record_path=Path(json.loads(pointer.read_text())['path'])
                inputs[str(record_path.resolve())]=digest(record_path)
                inputs.update(proof['input_sha256'])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                completion_failure=str(exc)
    graph = session.config('scene_graph.json', {})
    support = graph.get('task', {}).get('support')
    specs = [*graph.get('dynamic_objects', {}).values(), graph.get('furniture', {}).get(support, {})]
    for spec in specs:
        fix = spec.get('fix') or {}
        for key in ('mesh', 'visual_glb'):
            if fix.get(key):
                path = resolve(fix[key]).resolve()
                inputs[str(path)] = digest(path)
        if fix.get('model'):
            path = (session.config_dir/fix['model']).resolve()
            inputs[str(path)] = digest(path)
    for path, expected in inputs.items():
        if digest(path) != expected:
            raise ValueError('Render/candidate input changed: '+path)
    override_failure = None
    if any(render.get(key) for key in ('render_settings','background_field_override','background_transform_override','diagnostic')):
        override_failure = 'Controlled review overrides are diagnostic only; rebuild/export those settings into the delivered candidate before approval'
    passed = validate_checks(request, images) and all(f is None for f in (physical_failure,completion_failure,override_failure,hybrid_failure,composition_failure,source_failure,visibility_failure,source_scope_failure))
    root = session/'qa/environment_quality'
    if variant != 'twin':
        root = root/'cousins'/variant
    root.mkdir(parents=True, exist_ok=True)
    rounds = [int(p.name[6:]) for p in root.glob('round_*') if p.name[6:].isdigit()]
    output = root/f'round_{max(rounds, default=0)+1}'
    output.mkdir(exist_ok=False)
    result = {'version': 1, 'pass': passed, **request, 'input_sha256': inputs,
              'variant': variant, 'environment_manifest_sha256': environment_manifest,
              'physical_audit_sha256': physical_digest, 'physical_audit_failure': physical_failure,
              'completion_audit_failure':completion_failure,
              'hybrid_audit_failure':hybrid_failure,
              'composition_audit_failure':composition_failure,
              'source_preservation_failure':source_failure,
              'gs_visibility_failure':visibility_failure,
              'source_scope_failure':source_scope_failure,
              'render_override_failure': override_failure,
              'protected_task_state': protected_state(session),
              'scope': 'Whole-scene visual review. No automatic photorealism certification, task CAD, true-size gate or robot operation.'}
    # The result is derived from all criteria; a caller cannot override pass.
    result.update(version=1, **{'pass': passed})
    path = output/'review.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    pointer = {'path': str(path.resolve()), 'sha256': digest(path)}
    (root/'current.json').write_text(json.dumps(pointer, indent=2)+'\n')
    return result, pointer['sha256']


def current(session, variant='twin'):
    root = session/'qa/environment_quality'
    if not isinstance(variant, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,63}', variant):
        raise ValueError('Use a simple environment variant identifier')
    if variant != 'twin':
        root = root/'cousins'/variant
    pointer = json.loads((root/'current.json').read_text())
    path = Path(pointer['path']).resolve()
    if not path.is_relative_to(root.resolve()) or digest(path) != pointer['sha256']:
        raise ValueError('Environment review pointer/source changed')
    result = json.loads(path.read_text())
    if result.get('version') != 1 or result.get('pass') is not True:
        raise ValueError('Whole-scene environment quality has not passed')
    if result.get('variant', 'twin') != variant:
        raise ValueError('Environment review belongs to a different variant')
    if session.config('twin.json', {}).get('background') == 'blender' and not result.get('environment_manifest_sha256'):
        raise ValueError('Current Blender build and mandatory physical audit required')
    if result.get('environment_manifest_sha256'):
        from calibrated_scene_reconstruction.scene.background.blender_env import verify
        verify(session, variant)
        if digest(session/f'scene/environment/{variant}/current.json') != result['environment_manifest_sha256']:
            raise ValueError('Blender environment round changed since review')
        from calibrated_scene_reconstruction.scene.background.environment_physics import current as physical_current
        _, physical_digest = physical_current(session, variant, json.loads(resolve(result['render_review']).read_text()))
        if physical_digest != result.get('physical_audit_sha256'):
            raise ValueError('Current physical support/stability audit required')
    if result['protected_task_state'] != protected_state(session):
        raise ValueError('Protected task/support/robot state changed since environment review')
    for source, expected in result['input_sha256'].items():
        if digest(source) != expected:
            raise ValueError('Environment asset/render changed: '+source)
    render = json.loads(resolve(result['render_review']).read_text())
    if session.config('twin.json',{}).get('background')=='blender':
        from calibrated_scene_reconstruction.scene.background.blender_env import verify
        built=verify(session,variant);context=built.get('splat_context')
        if context and context.get('coverage_rebuild_required'):raise ValueError('Provisional native backing still requires actual current GS coverage')
        if context and not context.get('appearance_plan'):raise ValueError('Image-backed GS/mesh quality routing plan required')
        if context and not validate_hybrid_review(result,render,context,session.config('environment.json',{}).get('parameters',{}).get('entities',[])):
            raise ValueError('Hybrid object/GS/boundary audit has not passed')
        if context:
            from calibrated_scene_reconstruction.scene.background.capture_scope import validate as scope_check
            scope_check(context.get('appearance_ownership',{}),context.get('gap_masks',{}))
            from calibrated_scene_reconstruction.scene.review.hybrid_composition import validate as composition_check
            composition_check(result['composition_audit'],resolve(result['render_review']),context['appearance_ownership'])
            from calibrated_scene_reconstruction.scene.review.source_preservation import validate as source_check
            source_check(result['source_preservation_audit'],resolve(result['render_review']))
            if built.get('gs_visibility_required'):
                from calibrated_scene_reconstruction.scene.review.gs_visibility import validate as visibility_check
                visibility_check(result['gs_visibility_audit'],resolve(result['render_review']))
    if any(render.get(key) for key in ('render_settings','background_field_override','background_transform_override','diagnostic')):
        raise ValueError('Controlled review overrides are not the delivered environment')
    images = {str(resolve(p).resolve()) for row in render['views'] for p in [row['source'], *row['renders']]}
    if not validate_checks(result, images):
        raise ValueError('Environment quality contains failed/open criteria')
    return result, pointer['sha256']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session')
    parser.add_argument('request')
    args = parser.parse_args()
    result, sha = record(Session(args.session), json.loads(Path(args.request).read_text()))
    print('Environment quality', 'PASS' if result['pass'] else 'FAIL', sha)


if __name__ == '__main__':
    main()
