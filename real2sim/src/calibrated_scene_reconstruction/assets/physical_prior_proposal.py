"""Bounded, unaccepted physical-prior samples for isolated candidate assays."""
import copy
import numbers

import numpy as np

from .physical_priors import expand_families, parameter, prior_groups


def apply(plan, proposal):
    """Change explicit prior-only nominals; never identify or adopt a material.

    The caller must bind the source declaration and proposal, snapshot the
    resulting complete configuration and run the ordinary declaration/native
    checks. Bounds, units, physical parts and validity limits are immutable.
    """
    if (not isinstance(proposal, dict) or set(proposal) != {'version', 'basis', 'overrides'}
            or type(proposal['version']) is not int or proposal['version'] != 1
            or not isinstance(proposal['basis'], str) or not proposal['basis'].strip()
            or not isinstance(proposal['overrides'], dict) or not proposal['overrides']):
        raise ValueError('Versioned bounded diagnostic proposal and explicit basis required')
    result = expand_families(plan)
    changes = []
    for group, objects in proposal['overrides'].items():
        if group not in {'bodies', 'mechanisms', 'supports', *prior_groups()}:
            raise ValueError('Proposal cannot change a physical declaration or validity policy')
        if not isinstance(objects, dict) or not objects:
            raise ValueError('Explicit named physical parameters required')
        for name, parameters in objects.items():
            if name not in result.get(group, {}):
                raise ValueError('Proposal cannot add or omit a physical body')
            if not isinstance(parameters, dict) or not parameters:
                raise ValueError('Explicit existing parameter samples required')
            for key, sample in parameters.items():
                original = result[group][name].get('parameters', {}).get(key)
                if original is None:
                    raise ValueError('Proposal cannot introduce a new material parameter')
                parameter(key, original)
                if original['identification'] != 'prior_only':
                    raise ValueError('An identified estimate cannot be changed by a prior-only diagnostic')
                if (not isinstance(sample, dict) or set(sample) != {'value', 'unit'}
                        or sample['unit'] != original['unit']
                        or isinstance(sample['value'], bool)
                        or not isinstance(sample['value'], numbers.Real)
                        or not np.isfinite(sample['value'])
                        or not original['bounds'][0] <= sample['value'] <= original['bounds'][1]):
                    raise ValueError('Diagnostic sample must preserve the existing unit and prior bounds')
                changes.append({'group': group, 'body': name, 'parameter': key,
                                'old_nominal': original['nominal'], 'sample': float(sample['value']),
                                'unit': original['unit'], 'unchanged_bounds': list(original['bounds']),
                                'identification': original['identification'],
                                'original_prior': copy.deepcopy(original)})
                original['nominal'] = float(sample['value'])
    return result, {'version': 1, 'diagnostic_only': True, 'accepted': False,
                    'material_identified': False, 'task_approved': False,
                    'basis': proposal['basis'], 'changes': changes,
                    'scope': 'Bounded prior samples only. No runtime state change, altered validity '
                             'limits, source approval or adoption. Native response and ordinary '
                             'source/material/semantic/task gates remain required.'}
