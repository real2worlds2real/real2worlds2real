"""Current-session agent physics estimates; no true-size references are read."""
import json
import hashlib

from calibrated_scene_reconstruction.assets.physical_priors import validate, prior_groups
from calibrated_scene_reconstruction.core.session_code import source_path
from calibrated_scene_reconstruction.scene.objects.mechanisms import declaration as mechanisms


def declaration(s, *, required=False):
    path = source_path(s.config_dir, 'physics_priors.json')
    task_path = s.config_dir/'task.json'
    task = json.loads(task_path.read_text()) if task_path.exists() else {}
    if not path.exists():
        if required or task.get('physical_priors_contract'):
            raise ValueError('Agent must estimate physics_priors.json from material/mechanism priors before modeling')
        return None
    mechanism = mechanisms(s)
    if mechanism is None:
        raise ValueError('Physical priors require the task physical-body declaration')
    result = validate(json.loads(path.read_text()), mechanism)
    for group in ('bodies', 'mechanisms', 'supports', *prior_groups()):
        for body in result.get(group, {}).values():
            for value in body.get('parameters', {}).values():
                for relative, expected in value.get('observation_evidence', {}).items():
                    file = (s.dir/relative).resolve()
                    if not file.is_relative_to(s.dir.resolve()) or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
                        raise ValueError('Physical identification evidence changed or is not from this session')
    support = s.scene_graph.get('task', {}).get('support')
    if support and support not in result.get('supports', {}):
        raise ValueError('Agent must also estimate task-support contact properties')
    return result


def mass_estimate(s, obj, build):
    """Mass/density estimates use generated solid volume, never an object size target."""
    prior = declaration(s)
    if prior is None:
        return None
    links = build.get('parts') if build.get('links') else [{'name': 'model', 'geometry': build['geometry']}]
    value, details = 0., []
    for link in links:
        body = prior['bodies'][obj+'/'+link['name']]
        params = body['parameters']
        if 'mass' in params:
            p = params['mass']
            mass, bounds = p['nominal'], p['bounds']
        elif 'areal_density' in params:
            area = link['geometry'].get('material_area_m2') or link['geometry']['area_m2']
            p = params['areal_density']
            mass, bounds = p['nominal']*area, [x*area for x in p['bounds']]
        else:
            volume = link['geometry']['volume_m3']
            p = params['density']
            mass, bounds = p['nominal']*volume, [x*volume for x in p['bounds']]
        value += mass
        details.append({'link': link['name'], 'mass_kg': mass, 'mass_prior_bounds_kg': bounds,
                        'material_hypothesis': body['material_hypothesis'],
                        'basis': p['basis'], 'source': p['source'], 'identification': p['identification']})
    return {'nominal_kg': value, 'links': details, 'identified_from_task_pass': False}
