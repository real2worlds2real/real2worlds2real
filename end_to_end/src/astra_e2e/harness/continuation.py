"""Explicit additional model grants over immutable completed campaign evidence."""
import copy
from pathlib import Path

from .backends import read
from ..oversight.evidence import digest, file_hash


SCHEMA = 'astra_unified_continuation_v1'


def _history(directory):
    rows, previous, files = [], None, {}
    for path in sorted((directory/'history').glob('*.json')):
        row = read(path)
        body = {key:value for key,value in row.items() if key!='sha256'}
        if row.get('previous') != previous or digest(body) != row.get('sha256'):
            raise ValueError('Continuation parent history integrity failure')
        rows.append(row)
        previous = row['sha256']
        files[str(path.resolve())] = file_hash(path)
    if not rows:
        raise ValueError('Continuation requires a completed immutable parent history')
    return rows, files


def _parent(path):
    path = Path(path).resolve(strict=True)
    report, directory = read(path), path.parent
    config_path, implementation_path = directory/'config.json', directory/'implementation.json'
    config, implementation = read(config_path), read(implementation_path)
    rows, files = _history(directory)
    state = report.get('state', {})
    if (report.get('schema') != 'astra_unified_agent_report_v1' or not state.get('done')
            or report.get('status') != state['done'] or state.get('pending') is not None
            or rows[-1]['state'] != state or (directory/'runner.lock').exists()):
        raise ValueError('Continuation parent must be completed, unlocked and have no pending decision')
    expected = digest({'configuration':config,'implementation':implementation})
    if (report.get('config_sha256') != expected or any(row['config_sha256'] != expected for row in rows)
            or report.get('decision_steps') != state['step']):
        raise ValueError('Continuation parent configuration or cost accounting changed')
    if (report.get('execution_mode') != config['execution_mode']
            or report.get('binding_sha256') != read(config['binding'])['binding_sha256']):
        raise ValueError('Continuation parent summary differs from its bound execution configuration')
    for row in rows:
        pending = row['state'].get('pending') or {}
        if (pending.get('choice', {}).get('operation') == 'confirm'
                or row['state'].get('done') in {'simulation_policy_accepted','sealed_evaluation_failed',
                                              'sealed_evaluation_incomplete','sealed_stage_already_started'}):
            raise ValueError('Continuation cannot reopen a parent that started sealed evaluation')
    for item in (path, config_path, implementation_path):
        files[str(item.resolve())] = file_hash(item)
    seal = directory/'report.integrity.json'
    if seal.exists():
        if read(seal).get('report_sha256') != file_hash(path):
            raise ValueError('Parent completed report integrity seal changed')
        files[str(seal.resolve())] = file_hash(seal)
    # A transport failure may have reserved a model call without producing an
    # audited decision. Count that reservation conservatively, never as free.
    calls = report['decision_steps'] + int(report['status']=='model_decision_failed')
    if calls > config['max_model_calls']:
        raise ValueError('Parent model-call accounting exceeds its declared grant')
    recovery_state = copy.deepcopy(state)
    if report['status']=='model_decision_failed' and (state.get('last') or {}).get('error'):
        # Legacy runners overwrote the current receipt on a transport failure.
        # Recover only hash-linked prior evidence; restore() still verifies its
        # exact current module attempt, request and accepted dependencies.
        for row in reversed(rows):
            previous_state=row['state']
            if (previous_state.get('last') or {}).get('receipt'):
                recovery_state['last']=copy.deepcopy(previous_state['last'])
                recovery_state['review_result']=copy.deepcopy(previous_state.get('review_result'))
                recovery_state['tool_error']={'tool':'decision_transport','error':state['last']['error'],
                    'receipt_recovery':'Last development receipt recovered from the same immutable parent history.'}
                break
    return {'path':str(path), 'sha256':file_hash(path), 'files':files,
            'model_call_reservations':calls, 'config':config, 'report':report,'recovery_state':recovery_state}


def prepare(parent_report, *, grant_id, additional_model_calls, reason, earlier_reports=()):
    """Build a reviewable descriptor; do not start a model, controller or job."""
    if (not isinstance(grant_id,str) or not grant_id.strip() or not isinstance(reason,str) or not reason.strip()
            or type(additional_model_calls) is not int or not 1 <= additional_model_calls <= 100):
        raise ValueError('Declare a named, justified finite additional model-call grant')
    primary = str(Path(parent_report).resolve(strict=True))
    queue, parents, supporting = [primary, *map(str, earlier_reports)], {}, {}
    while queue:
        path = str(Path(queue.pop(0)).resolve(strict=True))
        if path in parents:
            continue
        item = _parent(path)
        parents[path] = item
        previous = item['config'].get('continuation')
        if previous:
            for entry in previous['parents']:
                if file_hash(entry['path']) != entry['sha256']:
                    raise ValueError('Declared earlier campaign report changed')
                queue.append(entry['path'])
        # Import the explicitly recorded operator continuations used before this
        # helper existed. Their prior costs remain part of the shared history.
        handoff = Path(path).parent.parent/'continuation.json'
        if handoff.is_file():
            record = read(handoff)
            if record.get('schema') == 'astra_operator_continuation_v1':
                if record.get('binding_sha256') != item['report']['binding_sha256']:
                    raise ValueError('Operator continuation belongs to a different world binding')
                supporting[str(handoff)] = file_hash(handoff)
                for old, sha in record.get('prior_reports_sha256', {}).items():
                    if file_hash(old) != sha:
                        raise ValueError('Operator continuation prior report changed')
                    queue.append(old)
    return {'schema':SCHEMA, 'primary_parent':primary,
            'parents':[{key:item[key] for key in ('path','sha256','files','model_call_reservations')}
                       for _,item in sorted(parents.items())], 'supporting_evidence':supporting,
            'grant':{'id':grant_id,'additional_model_calls':additional_model_calls,'reason':reason,
                     'authorization':'Operator-configured under the user ongoing execution request; the numeric limit is an operator budget.',
                     'scope':'Explicit additional grant; earlier grants and outer costs are retained, not reset.'}}


def inspect(config, *, completed_report_exists=False):
    descriptor = config.get('continuation')
    if not descriptor:
        return None
    if descriptor.get('schema') != SCHEMA:
        raise ValueError('Expected explicit unified continuation descriptor')
    grant = descriptor['grant']
    if grant.get('additional_model_calls') != config['max_model_calls'] or not grant.get('reason'):
        raise ValueError('New model budget must match the explicit additional grant')
    binding = read(config['binding'])
    parents, evidence = {}, dict(descriptor.get('supporting_evidence', {}))
    for declared in descriptor['parents']:
        item = _parent(declared['path'])
        if any(item[key] != declared[key] for key in ('sha256','files','model_call_reservations')):
            raise ValueError('Continuation parent artifact or history changed')
        if item['report']['binding_sha256'] != binding['binding_sha256']:
            raise ValueError('Continuation requires the same world binding')
        if item['report']['execution_mode'] != config['execution_mode']:
            raise ValueError('Fixture and production continuation evidence cannot be mixed')
        if item['config']['campaign'] == config['campaign']:
            raise ValueError('Continuation needs a new campaign identity for idempotent tool calls')
        old_grant = item['config'].get('continuation', {}).get('grant', {}).get('id', item['config']['campaign'])
        if grant['id'] == old_grant:
            raise ValueError('An additional grant must have a new explicit identity')
        if item['path'] in parents:
            raise ValueError('Do not double-count a parent campaign')
        parents[item['path']] = item
        evidence.update(item['files'])
    primary = parents.get(descriptor['primary_parent'])
    if primary is None:
        raise ValueError('Primary parent is missing from continuation evidence')
    rebuilt = prepare(primary['path'], grant_id=grant['id'], additional_model_calls=config['max_model_calls'],
                      reason=grant['reason'], earlier_reports=list(parents))
    if rebuilt != descriptor:
        raise ValueError('Continuation omitted or changed declared ancestor cost evidence')
    if any(file_hash(path) != sha for path,sha in evidence.items()):
        raise ValueError('Continuation evidence changed')
    outer_rows, _ = _history(Path(binding['oversight_run']))
    outer = outer_rows[-1]['state']
    if (outer.get('holdout_consumed') or outer['modules']['holdout']['attempts']) and not completed_report_exists:
        raise ValueError('Continuation is forbidden after sealed evaluation started')
    if not completed_report_exists and any(outer['modules'][name]['status']=='running'
                                          for name in ('data','train','validation')):
        raise ValueError('Continuation cannot start while another development attempt is running')
    return {'primary_state':copy.deepcopy(primary['recovery_state']),
            'parent_development_snapshot':primary['report'].get('development_snapshot'), 'evidence_inputs':evidence,
            'current_outer_state':outer,
            'accounting':{'schema':'astra_model_grant_accounting_v1', 'grant':grant,
                'prior_campaigns':[{'report':item['path'],'sha256':item['sha256'],
                                    'model_call_reservations':item['model_call_reservations']}
                                   for item in parents.values()],
                'prior_model_call_reservations':sum(item['model_call_reservations'] for item in parents.values()),
                'outer_plan_sha256':binding['plan_sha256'],'outer_jobs_at_start':outer['jobs'],
                'outer_cycle_at_start':outer['cycle'],'outer_budgets_reset':False}}


def restore(info, state, status):
    """Carry development state forward while respecting actual outer invalidation."""
    restored = {**state, **copy.deepcopy(info['primary_state'])}
    restored.update(step=0,pending=None,done=None,decision_error=None,
                    consecutive_retrievals=0)
    if restored.get('tool_error'):
        restored['tool_error']['retained_from_completed_parent'] = True
    modules = status['modules']
    if any(modules[name]['state']=='running' for name in ('data','train','validation')):
        raise ValueError('Continuation cannot start while another development attempt is running')
    if modules['holdout']['attempts'] or modules['holdout']['state'] not in {'pending','invalidated'}:
        raise ValueError('Continuation is forbidden after sealed evaluation started')
    parent_snapshot = info.get('parent_development_snapshot')
    if parent_snapshot:
        for name in ('data','train','validation'):
            before, now = parent_snapshot[name], modules[name]
            if now['state'] == 'invalidated':
                continue  # The independent outer revision is retained in current causal feedback.
            if before['state'] != now['state'] or before['attempts'] != now['attempts']:
                raise ValueError('Current development advanced beyond the completed parent snapshot; an explicit evidence handoff is required')
    last = restored.get('last') or {}
    if not parent_snapshot and not last.get('receipt') and any(
            modules[name]['state']=='accepted' for name in ('data','train','validation')):
        raise ValueError('Legacy parent lacks evidence for current accepted development; an explicit evidence handoff is required')
    awaiting = [name for name in ('data','train','validation') if modules[name]['state']=='awaiting_review']
    if awaiting and not last.get('receipt'):
        raise ValueError('Current awaiting-review attempt has no matching parent receipt')
    if last.get('receipt'):
        if file_hash(last['receipt']) != last.get('runner_receipt_sha256'):
            raise ValueError('Parent development receipt changed')
        operation = read(last['receipt'])['operation']
        if operation == 'confirm':
            raise ValueError('Sealed receipts cannot become continuation development state')
        module = {'collect':'data','import_data':'data','train':'train','develop':'validation'}[operation]
        if awaiting and module not in awaiting:
            raise ValueError('Parent receipt does not match the current awaiting-review module')
        order = {'data':0,'train':1,'validation':2}
        if any(order[name] > order[module] and modules[name]['attempts']
               and modules[name]['state'] not in {'pending','invalidated'} for name in order):
            raise ValueError('Current development advanced beyond the parent receipt; an explicit evidence handoff is required')
        attempt = last.get('execution', {}).get('attempt')
        if modules[module]['state'] != 'invalidated':
            if type(attempt) is not int or attempt != modules[module]['attempts']:
                raise ValueError('Parent receipt is from a different or unbound current module attempt')
            request_path = last.get('request')
            current_attempt = info['current_outer_state']['modules'][module]['attempts'][-1]
            if (not request_path or current_attempt['inputs'].get(request_path) != file_hash(request_path)
                    or read(last['receipt'])['facts']['oversight']['request_sha256'] != digest(read(request_path))):
                raise ValueError('Parent receipt does not bind the current attempt request')
            request = read(request_path)
            if any(current_attempt['dependencies'].get(name) != dependency['id']
                   for name,dependency in request['dependencies'].items()):
                raise ValueError('Parent receipt belongs to different development dependencies')
        if modules[module]['state'] == 'invalidated' or (modules[module]['state']=='accepted' and not modules[module]['current']):
            restored.setdefault('development_history',[]).append({'parent_operation':operation,
                'parent_review':restored.get('review_result'), 'reason':'Current outer revision invalidated this evidence.'})
            restored.update(last=None,review_result=None,diagnosed=False)
    restored['failure_pending'] = any(modules[name]['state']=='failed' for name in ('data','train','validation'))
    if not modules['validation']['current']:
        restored['memory_for'] = None
    restored['continuation_context'] = {
        'prior_model_call_reservations':info['accounting']['prior_model_call_reservations'],
        'additional_grant':info['accounting']['grant'],
        'state_restored':True,
        'invalidated_meaning':'An outer revision cleared dependent acceptance. A ready invalidated module may execute; invalidation is not a new policy-performance result.'}
    restored['grant_accounting'] = copy.deepcopy(info['accounting'])
    return restored
