"""Unaccepted same-geometry cloth-backend hypotheses for native diagnostics.

Switching a constitutive law is not a new physical observation. The caller must
bind both descriptors, the complete proposal and its source configuration, then
run ordinary source, material-state, native and semantic checks.
"""
import copy

import numpy as np

from calibrated_scene_reconstruction.assets.mechanisms import validate as mechanisms
from calibrated_scene_reconstruction.assets.physical_priors import validate as priors


_CONSTITUTIVE = {'stretch_stiffness', 'shear_stiffness', 'bend_stiffness',
                 'spring_damping', 'youngs_modulus', 'poissons_ratio',
                 'flexural_rigidity', 'elasticity_damping', 'linear_damping',
                 'bend_damping'}


def apply(configuration, proposal, selected):
    """Keep all physical identities, checks, evidence and common material data.

    The initial implementation deliberately changes only particle/surface-FEM
    cloth backends. Ordinary rigid, volume and articulation paths are untouched.
    Material constants need their own explicit priors in the new law's units.
    """
    fields = {'version', 'basis', 'mechanisms', 'physics_priors',
              'object_understanding', 'reference_candidates'}
    if (not isinstance(proposal, dict) or set(proposal) != fields
            or type(proposal['version']) is not int or proposal['version'] != 1
            or not isinstance(proposal['basis'], str) or not proposal['basis'].strip()):
        raise ValueError('Complete, versioned representation hypothesis and basis required')
    old = mechanisms(copy.deepcopy(configuration['agent/mechanisms.json']))
    new = mechanisms(copy.deepcopy(proposal['mechanisms']))
    if proposal['object_understanding'] != configuration['agent/object_understanding.json']:
        raise ValueError('A backend control must preserve full semantic inventory and source evidence')
    restored = copy.deepcopy(new); changed = []
    for obj, body in old['objects'].items():
        for link, request in body['links'].items():
            proposed = new.get('objects', {}).get(obj, {}).get('links', {}).get(link, {})
            before, after = request.get('backend', 'particle'), proposed.get('backend', 'particle')
            if before != after:
                if (obj not in selected or request['kind'] != 'cloth'
                        or proposed.get('kind') != 'cloth'
                        or {before, after} != {'particle', 'surface_fem'}):
                    raise ValueError('Only selected actual cloth links may change native backend')
                changed.append({'object': obj, 'link': link, 'before': before, 'after': after})
                if 'backend' in request:restored['objects'][obj]['links'][link]['backend'] = before
                else:restored['objects'][obj]['links'][link].pop('backend', None)
    if not changed or restored != old:
        raise ValueError('Backend hypothesis must preserve every physical part, mechanism and source condition')
    references = proposal['reference_candidates']
    if (not isinstance(references, dict) or set(references) != {v['object'] for v in changed}
            or any(not isinstance(p, str) or not p.strip() for p in references.values())):
        raise ValueError('Each changed object requires its immutable same-geometry reference descriptor')
    old_physics = priors(copy.deepcopy(configuration['agent/physics_priors.json']), old)
    new_physics = priors(copy.deepcopy(proposal['physics_priors']), new)
    restored_physics = copy.deepcopy(new_physics)
    for item in changed:
        key = item['object']+'/'+item['link']
        before, after = old_physics['bodies'][key], new_physics['bodies'][key]
        b, a = before['parameters'], after['parameters']
        for name in set(b) | set(a):
            if name not in _CONSTITUTIVE and b.get(name) != a.get(name):
                raise ValueError('A same-material backend control cannot change mass, thickness or contact priors')
            if name in b and b[name]['identification'] != 'prior_only' and b[name] != a.get(name):
                raise ValueError('A backend control cannot discard or relabel an identified material estimate')
            if name in a and a[name] != b.get(name) and a[name]['identification'] != 'prior_only':
                raise ValueError('New constitutive assumptions are priors, not identified observations')
        clean = copy.deepcopy(after)
        for name in ('parameters', 'family', 'material_hypothesis'):
            if name in before:clean[name] = copy.deepcopy(before[name])
            else:clean.pop(name, None)
        if clean != before:
            raise ValueError('Representation hypothesis cannot weaken validity or other physical obligations')
        restored_physics['bodies'][key] = copy.deepcopy(before)
    if restored_physics != old_physics:
        raise ValueError('Unselected bodies, supports, material families and mechanisms must remain unchanged')
    result = copy.deepcopy(configuration)
    result['agent/mechanisms.json'] = new
    result['agent/physics_priors.json'] = new_physics
    result['agent/object_understanding.json'] = copy.deepcopy(proposal['object_understanding'])
    return result, {'version': 1, 'basis': proposal['basis'], 'changes': changed,
        'reference_candidates': copy.deepcopy(references), 'diagnostic_only': True,
        'accepted': False, 'material_identified': False, 'task_approved': False,
        'geometry_comparison_required': True}


def same_geometry(reference, candidate, reference_pose, candidate_pose):
    """Verify actual coordinates, topology, natural state, bonds and materials.

    A backend label is the only permissible asset difference. This is stricter
    than matching a silhouette or a vertex count and also catches hidden parts.
    """
    if reference_pose != candidate_pose:
        raise ValueError('Same-geometry control requires the same source pose')
    def equal(a, b):
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return np.array_equal(a, b)
        if isinstance(a, dict) and isinstance(b, dict):
            return set(a) == set(b) and all(equal(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
        return a == b
    a, b = copy.deepcopy(reference), copy.deepcopy(candidate)
    for value in (a, b):
        for part in value['parts']:part.pop('backend', None)
        links = value.get('links') or []
        for part in (links.values() if isinstance(links, dict) else links):part.pop('backend', None)
    if not equal(a, b):
        raise ValueError('Backend hypothesis changed actual geometry, material chart, bonds or visual material')
    return {'identical_actual_geometry': True, 'same_source_pose': True,
            'parts': len(a['parts']), 'object_approved': False}
