"""Optional post-twin cousin planning and independently validated child runs."""
from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .artifacts import RunStore, atomic_json, sha256_file


def request_cousins(store, *, request):
    run_path = store.root/'run.json'
    run = json.loads(run_path.read_text())
    if run.get('generation_mode', 'twin') not in {'twin','scene'} or not request.strip():
        raise ValueError('Only a twin task may request a nonempty optional cousin stage')
    run['cousin_request'] = request
    atomic_json(run_path, run)
    return {'ok': True, 'request': request, 'status': 'deferred_until_twin_accepted'}


def create_planning_run(parent, executor, *, prompt, count=None, model=None, max_turns=12, families_root=None,
                        scene_runner=None, gaussian_runner=None):
    if (Path(parent)/'final_scene.json').is_file():
        from .scene_cousins import create_planning_run as create_scene_planning
        return create_scene_planning(parent,executor,prompt=prompt,count=count,model=model,
                                     max_turns=max_turns,families_root=families_root,
                                     scene_runner=scene_runner,gaussian_runner=gaussian_runner)
    if scene_runner is not None or gaussian_runner is not None:
        raise ValueError('Scene runner overrides apply only to scene cousins')
    from .capabilities import TASK_TOOLS as CODEX_TOOLS
    parent = RunStore(Path(parent).resolve())
    if hasattr(executor, "for_task"):
        executor = executor.for_task(parent)
    final_path = parent.root/'final.json'
    if not final_path.is_file():
        raise ValueError('Cousins require an accepted twin')
    final = json.loads(final_path.read_text())
    if final.get('status') != 'accepted':
        raise ValueError('Cousins require an accepted twin')
    if count is not None and (type(count) is not int or count < 1):
        raise ValueError('An explicit cousin count must be a positive integer')
    if not prompt.strip() and count is None:
        raise ValueError('Supply a natural-language cousin request or an explicit count')
    parent_run = json.loads((parent.root/'run.json').read_text())
    attempt = parent.attempts_dir/final['attempt_id']
    source, script = attempt/'scene.blend', attempt/'agent_scene.py'
    if not source.is_file() or not script.is_file():
        raise ValueError('The accepted editable source is unavailable')
    photos = sorted(attempt.glob('hero_*.png'))
    if not photos:
        photos = [parent.root/row['path'] for row in parent_run['references']]
    if not photos:
        raise ValueError('The accepted twin needs visual context')
    family_id = datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    family = Path(families_root or parent.root/'cousins')/f'family-{family_id}'
    family.mkdir(parents=True)
    source_digest = sha256_file(source)
    for key in ('simulation', 'hybrid', 'fluid', 'final_render'):
        prior = final.get('report', {}).get(key, {}).get('source_scene_sha256')
        if prior is not None and prior != source_digest:
            raise ValueError('The accepted parent source changed after its retained review')
    structure_path = family/'parent_structure.json'
    extractor = Path(__file__).resolve().parents[2]/'scripts/extract_asset_structure.py'
    with (family/'structure.log').open('w') as stream:
        result = subprocess.run([str(executor.blender), '--background', str(source), '--python', str(extractor), '--', str(structure_path)],
                                stdout=stream, stderr=subprocess.STDOUT, timeout=executor.timeout_seconds, check=False)
    if result.returncode or not structure_path.is_file():
        raise RuntimeError('Parent structure extraction failed: '+(family/'structure.log').read_text(errors='replace')[-3000:])
    if sha256_file(source) != source_digest:
        raise ValueError('Parent source changed during cousin preparation')
    structure = json.loads(structure_path.read_text())
    contract = parent_run.get('asset_contract') or structure.get('asset_contract')
    if not contract or not contract.get('kind'):
        raise ValueError('The accepted parent lacks an inferable asset contract')
    planner = RunStore.create(family/'planning', references=photos, objective=prompt or f'Generate {count} diverse functional cousins.',
                              model=model or parent_run['model'], max_turns=max_turns,
                              tool_schemas=CODEX_TOOLS, asset_kind=contract['kind'], front_axis=contract['front_axis'])
    context = planner.root/'context'
    context.mkdir()
    shutil.copy2(source, context/'parent_scene.blend')
    shutil.copy2(script, context/'parent_source.py')
    shutil.copy2(structure_path, context/'parent_structure.json')
    parent_info = {'run': str(parent.root), 'attempt_id': final['attempt_id'], 'source_scene_sha256': source_digest,
                   'source_script_sha256': sha256_file(script), 'asset_contract': contract,
                   'functional_structure': structure, 'acceptance_reason': final['reason']}
    atomic_json(context/'parent.json', parent_info)
    batch = {'schema_version': 1, 'status': 'awaiting_plan', 'request': prompt, 'explicit_count': count,
             'parent': parent_info, 'planner': str(planner.root.relative_to(family)), 'variants': []}
    atomic_json(family/'family.json', batch)
    run_path = planner.root/'run.json'
    run = json.loads(run_path.read_text())
    run.update(generation_mode='cousin_planning', cousin_family=str(family), cousin_request=prompt,
               cousin_explicit_count=count, parent_context='context/parent.json')
    atomic_json(run_path, run)
    return family, planner


def submit_plan(store, *, count, variants, shared_notes, question, execute=True):
    run = json.loads((store.root/'run.json').read_text())
    if run.get('generation_mode') not in {'cousin_planning','scene_cousin_planning'}:
        raise ValueError('submit_cousin_plan is available only during cousin planning')
    family = Path(run['cousin_family'])
    batch = json.loads((family/'family.json').read_text())
    if any(row.get('run') for row in batch.get('variants',[])):
        raise ValueError('This cousin plan already started; create a new family for changed design intent')
    if type(execute) is not bool:
        raise ValueError('Execution intent must be inferred as a boolean')
    batch['execute'] = execute
    if count is None:
        if variants or not question or not question.strip():
            raise ValueError('An unresolved plan needs a concise question and no scheduled variants')
        batch.update(status='needs_input', question=question)
    else:
        if type(count) is not int or count < 1 or len(variants) != count:
            raise ValueError('The positive count must equal the number of variant plans')
        if batch['explicit_count'] is not None and count != batch['explicit_count']:
            raise ValueError('The plan count differs from the user-specified count')
        for row in variants:
            if not row.get('prompt', '').strip() or type(row.get('geometry_change_expected')) is not bool:
                raise ValueError('Each variant needs a prompt and a geometry-change expectation inferred from the user request')
            layout_request = row.get('layout_change_request')
            if layout_request is not None:
                if run['generation_mode'] != 'scene_cousin_planning':
                    raise ValueError('Layout-change intent applies only to scene cousins')
                if not isinstance(layout_request, str) or not layout_request.strip():
                    raise ValueError('Layout-change intent must describe the user-requested scope or be null')
        batch.update(status='planned', count=count, shared_notes=shared_notes,
                     variants=[{'index': index+1, 'prompt': row['prompt'],
                                'geometry_change_expected': row['geometry_change_expected'],
                                'layout_change_request': row.get('layout_change_request'),
                                'status': 'pending', 'run': None} for index, row in enumerate(variants)])
    atomic_json(family/'family.json', batch)
    store.append_event({'type': 'cousin_plan_submitted', 'count': count, 'status': batch['status']})
    return {'ok': True, 'status': batch['status'], 'family': str(family), 'plan': batch}


def create_child(family, index, *, max_turns):
    from .capabilities import TASK_TOOLS as CODEX_TOOLS
    family = Path(family)
    batch = json.loads((family/'family.json').read_text())
    variant = batch['variants'][index]
    planner = RunStore(family/batch['planner'])
    planner_run = json.loads((planner.root/'run.json').read_text())
    photos = [planner.root/row['path'] for row in planner_run['references']]
    child = RunStore.create(family/f'variant-{index+1:03d}', references=photos,
                            objective=variant['prompt'], model=planner_run['model'], max_turns=max_turns,
                            tool_schemas=CODEX_TOOLS, asset_kind='auto',
                            front_axis=batch['parent']['asset_contract']['front_axis'])
    shutil.copytree(planner.root/'context', child.root/'context')
    siblings = []
    for other in batch['variants']:
        if other['status'] != 'accepted':
            continue
        final = json.loads((family/other['run']/'final.json').read_text())
        structure = final['report']['functional_structure']
        siblings.append({'index': other['index'], 'prompt': other['prompt'],
                         'geometry_fingerprint': structure['geometry_fingerprint'], 'acceptance_reason': final['reason']})
    cousin = {'family': str(family), 'variant_index': index+1, 'variant_count': batch['count'],
              'raw_user_request': batch['request'], 'shared_notes': batch['shared_notes'],
              'variant_prompt': variant['prompt'], 'parent': batch['parent'],
              'allow_surface_only': not variant['geometry_change_expected'], 'accepted_siblings': siblings}
    cousin['execution_authorized'] = batch.get('execute', True)
    cousin['execution_request'] = batch.get('execution_request')
    run_path = child.root/'run.json'
    run = json.loads(run_path.read_text())
    run.update(generation_mode='cousin', cousin=cousin, parent_context='context/parent.json')
    atomic_json(run_path, run)
    atomic_json(child.root/'context/cousin_task.json', cousin)
    variant.update(status='running', run=str(child.root.relative_to(family)))
    atomic_json(family/'family.json', batch)
    return child


def run_family(family, executor, *, agent_options, execute_plan=False):
    from .codex_backend import run_codex_agent
    family = Path(family).resolve()
    batch = json.loads((family/'family.json').read_text())
    if batch.get('family_kind')=='scene':
        from .scene_cousins import run_family as run_scene_family
        return run_scene_family(family,executor,agent_options=agent_options,execute_plan=execute_plan)
    if batch['status'] == 'needs_input':
        return batch
    if batch.get('execute') is False and not execute_plan:
        return batch
    if execute_plan:
        batch['execute'] = True
        batch['execution_request'] = 'A later explicit request starts/resumes execution of this retained plan; earlier planning-only wording is superseded for execution, while design requirements stay in force.'
        atomic_json(family/'family.json', batch)
    if not batch.get('count'):
        raise ValueError('Resume cousin planning before executing an unplanned family')
    for index in range(batch['count']):
        batch = json.loads((family/'family.json').read_text())
        row = batch['variants'][index]
        if row['status'] == 'accepted':
            final_path = family/row['run']/'final.json'
            if not final_path.is_file() or json.loads(final_path.read_text()).get('status') != 'accepted':
                raise ValueError('An accepted cousin is missing its final receipt')
            continue
        existing = bool(row.get('run'))
        child = RunStore(family/row['run']) if existing else create_child(family, index, max_turns=agent_options['max_turns'])
        child_run = json.loads((child.root/'run.json').read_text())
        if (child.root/'final.json').is_file():
            result = {'status': json.loads((child.root/'final.json').read_text())['status']}
        else:
            options = {key: value for key, value in agent_options.items() if key != 'max_turns'}
            try:
                result = run_codex_agent(executor, child, model=child_run['model'], objective=child_run['objective'],
                                         max_turns=child_run['max_turns'], seed_script=None if existing else child.root/'context/parent_source.py',
                                         resume_existing=existing and (child.root/'codex/thread.json').is_file(), **options)
            except Exception as error:
                result = {'status': 'error', 'error_type': type(error).__name__}
        batch = json.loads((family/'family.json').read_text())
        batch['variants'][index]['status'] = result['status']
        if result.get('error_type'):
            batch['variants'][index]['error_type'] = result['error_type']
        batch['status'] = 'running'
        atomic_json(family/'family.json', batch)
        # Preserve unfinished children for resume rather than silently counting them as delivered.
        if result['status'] != 'accepted':
            batch.update(status='incomplete', unfinished_index=index+1)
            atomic_json(family/'family.json', batch)
            return batch
    batch = json.loads((family/'family.json').read_text())
    batch['status'] = 'complete'
    atomic_json(family/'family.json', batch)
    return batch
