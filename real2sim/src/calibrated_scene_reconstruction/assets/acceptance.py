"""Input-specified acceptance, with explicit conditional default relaxations.

Raw measurements and raw verdicts are immutable inputs. A relaxation is a
separate effective verdict, and never evidence that the default check passed.
No policy means exactly the previous default behavior.
"""
from copy import deepcopy


PROTECTED = {'native_errors', 'material_validity', 'collision_validity',
             'input_integrity', 'frozen_geometry', 'backend'}


def declaration(task):
    value = deepcopy(task.get('acceptance', {}))
    if not value:
        return {'version': 1, 'criteria': [], 'relaxations': []}
    if value.get('version') != 1:
        raise ValueError('Unsupported input acceptance version')
    criteria = value.get('criteria')
    relaxations = value.get('relaxations', [])
    if not isinstance(criteria, list) or not criteria or not isinstance(relaxations, list):
        raise ValueError('Input acceptance needs criteria and an explicit relaxation list')
    ids = []
    for criterion in criteria:
        if (not isinstance(criterion, dict) or not isinstance(criterion.get('id'), str)
                or not criterion['id'] or criterion['id'] in ids
                or not criterion.get('description') or not criterion.get('source')
                or criterion.get('required') is not True):
            raise ValueError('Input criteria need unique ids, user source, description and required:true')
        checks = criterion.get('checks', [])
        if (not isinstance(checks, list) or any(not isinstance(c, str) or not c for c in checks)
                or len(checks) != len(set(checks))):
            raise ValueError('Requested checks must be unique named measurements')
        if criterion.get('validation', 'evidence') not in {'evidence','native_physics'}:
            raise ValueError('Acceptance validation must be saved evidence or native physics')
        ids.append(criterion['id'])
    gates = []
    for relaxation in relaxations:
        gate = relaxation.get('gate')
        if (not isinstance(gate, str) or not gate or gate in gates or '*' in gate
                or gate.split(':', 1)[0] in PROTECTED
                or not relaxation.get('reason') or not relaxation.get('scope')
                or not isinstance(relaxation.get('requires'), list) or not relaxation['requires']
                or not set(relaxation['requires']) <= set(ids)):
            raise ValueError('Each relaxation needs an exact default gate, scope, reason and required criteria')
        gates.append(gate)
    return value


def evaluate(task, defaults, criteria):
    """Resolve named defaults only after all requested conditions have passed.

    The caller validates simulation provenance and hashes as part of its normal
    result contract. Merely declaring a condition, missing evidence, or a true
    flag without saved evidence cannot activate a relaxation.
    """
    plan = declaration(task)
    criteria = criteria if isinstance(criteria, dict) else {}
    verdicts = {}
    for request in plan['criteria']:
        proof = criteria.get(request['id'], {})
        native = request.get('validation', 'evidence') == 'native_physics'
        passed = (proof.get('pass') is True and bool(proof.get('evidence_files'))
                  and (not native or (proof.get('native_states_read') is True
                       and proof.get('backend', {}).get('name') == 'isaacsim'
                       and str(proof.get('backend', {}).get('version', '')).split('.')[:2] == ['5', '1']))
                  and all(proof.get('checks', {}).get(key, {}).get('pass') is True
                          for key in request.get('checks', [])))
        verdicts[request['id']] = {'pass': passed, 'request': request, 'result': deepcopy(proof)}
    all_criteria = all(row['pass'] for row in verdicts.values())
    relaxations = {row['gate']: row for row in plan['relaxations']}
    gates = {}
    for name, result in defaults.items():
        if not isinstance(result, dict) or type(result.get('pass')) is not bool:
            raise ValueError('Default gate results need explicit boolean verdicts')
        relaxation = relaxations.get(name)
        relaxed = (result['pass'] is False and relaxation is not None and all_criteria
                   and all(verdicts[key]['pass'] for key in relaxation['requires']))
        gates[name] = {'default_pass': result['pass'], 'raw': deepcopy(result),
                       'effective_pass': bool(result['pass'] or relaxed),
                       'relaxed': relaxed,
                       **({'relaxation': deepcopy(relaxation)} if relaxation is not None else {})}
    return {'pass': all_criteria and all(row['effective_pass'] for row in gates.values()),
            'criteria': verdicts, 'defaults': gates,
            'scope': 'Input-specified acceptance; raw default failures are retained separately.'}


def may_attempt(task, failed_gates, *, physical_valid):
    """Permit testing a requested condition without prematurely granting a waiver.

    This breaks the dependency cycle when a waivable appearance condition would
    otherwise block the physical experiment needed to resolve the condition.
    Invalid material/contact states still cannot proceed.
    """
    plan = declaration(task)
    requested = {row['gate'] for row in plan['relaxations']}
    return bool(physical_valid and plan['criteria'] and set(failed_gates) <= requested)
