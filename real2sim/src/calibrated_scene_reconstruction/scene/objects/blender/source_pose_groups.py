"""Explicit user-authorized pose changes of independently labelled rigid groups.

Several indistinguishable loose parts may share one source mask. Never pretend
that a group mask independently labels each member. Dimensions and native
function remain mandatory, even when the captured arrangement is optional.
"""
import re


def declaration(session, obj):
    fit = getattr(session, 'scene_graph', {}).get('dynamic_objects', {}).get(obj, {}).get('fit') or {}
    groups = fit.get('source_pose_groups', [])
    if not groups:
        return []
    if not isinstance(groups, list):
        raise ValueError('Source pose groups must be explicitly listed')
    from calibrated_scene_reconstruction.assets.acceptance import declaration as acceptance
    from calibrated_scene_reconstruction.scene.objects.mechanisms import declaration as mechanisms
    policy = acceptance(session.config('task.json', {}))
    relaxations = {r['gate']: r for r in policy['relaxations']}
    criteria = {r['id']: r for r in policy['criteria']}
    links = mechanisms(session)['objects'][obj]['links']
    result = []; seen = set(); used = set()
    for group in groups:
        name = group.get('id'); members = group.get('links'); check = group.get('dimension_check')
        if (not isinstance(name, str) or not re.fullmatch(r'[a-zA-Z0-9_]+', name)
                or name in seen or not isinstance(members, list) or not members
                or any(not isinstance(n, str) or n not in links or links[n]['kind'] != 'rigid' for n in members)
                or len(set(members)) != len(members) or used.intersection(members)
                or not isinstance(group.get('part_prompt'), str) or not group['part_prompt'].strip()
                or not group.get('basis') or not isinstance(check, str) or not check):
            raise ValueError('Rigid source groups need unique rigid members, source prompt and dimension check')
        gate = 'source_pose:'+obj+'/'+name
        relaxation = relaxations.get(gate)
        if not relaxation:
            raise ValueError('Rigid pose group needs an exact input-authorized relaxation: '+gate)
        required = [criteria[key] for key in relaxation['requires']]
        if not any(r.get('validation') == 'native_physics' and check in r.get('checks', []) for r in required):
            raise ValueError('Rigid pose relaxation requires native function and explicit dimension evidence')
        seen.add(name); used.update(members)
        result.append({'id': name, 'gate': gate, 'kind': 'rigid_group', 'links': members,
                       'prompt': group['part_prompt'], 'dimension_check': check})
    return result
