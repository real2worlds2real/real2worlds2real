"""Validated metric meshes as dynamic objects; the same surface drives the visual and the SDF collider."""
import json
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.core.paths import rel, resolve
from calibrated_scene_reconstruction.assets.metric_mesh import asset as asset, sdf_settings, asset_geometry  # noqa: F401 -- legacy export API
from calibrated_scene_reconstruction.scene.objects.object_fit import placement
from calibrated_scene_reconstruction.scene.objects.blender.contract import contract, geometry, write_json
from calibrated_scene_reconstruction.scene.objects.blender.store import digest, event
from calibrated_scene_reconstruction.assets.material_mass import mass




def verify_round(s, obj, out, exported=False, *, physical_review=True):
    """Reject stale camera/mask evidence, changed scripts and changed exported geometry."""
    from calibrated_scene_reconstruction.scene.objects.blender.fit import model_path
    result = json.loads((out / 'fit.json').read_text())
    if result.get('fixed_candidate_revalidation'):
        from .revalidate import verify as verify_revalidation
        verify_revalidation(s, obj, out, result)
    if 'input_sha256' in result and digest(out / 'input.json') != result['input_sha256']:
        raise ValueError('Round input manifest changed since fitting')
    inputs = json.loads((out / 'input.json').read_text())
    from .raster_contract import verify as verify_raster_contract
    verify_raster_contract(inputs.get('source_raster_contract'))
    from .representation_review import check as representation_check
    if representation_check(s, obj, model_path(s, obj)) != result.get('representation_review'):
        raise ValueError('Representation decision changed; rebuild and review a new round')
    if inputs.get('source_interface_contract',[])!=s.config('agent/interface_checks.json',{}).get('objects',{}).get(obj,[]):
        raise ValueError('Mating-surface observations changed; fit and review a new round')
    geometry_plan=s.config('agent/geometry_checks.json',{}).get('objects',{}).get(obj,[])
    if inputs.get('source_geometry_contract',[])!=geometry_plan:
        raise ValueError('Source solid/void observations changed; fit and review a new round')
    if geometry_plan:
        from .geometry_checks import measure
        from calibrated_scene_reconstruction.core.paths import ROOT
        if not measure(out,geometry_plan,ROOT)['pass']:
            raise ValueError('Exported mesh contradicts source solid/void geometry')
    if inputs.get('effective_fit_request',{}).get('align_to_support',False)!=s.scene_graph['dynamic_objects'][obj]['fit'].get('align_to_support',False):
        raise ValueError('Object support orientation policy changed; fit a new round')
    from calibrated_scene_reconstruction.scene.objects.object_understanding import declaration as understanding, current as understanding_review
    if inputs.get('object_understanding') != understanding(s, obj):
        raise ValueError('Object structure/material understanding changed; fit and review a new round')
    from calibrated_scene_reconstruction.scene.objects.object_understanding import validate_build
    validate_build(s, obj, result['build'])
    if physical_review:
        result['object_understanding_sha256'] = understanding_review(s, obj, out, result['build'])
    from .source_acceptance import policy as source_pose_policy
    if inputs.get('source_pose_contract',[]) != source_pose_policy(s,obj):
        raise ValueError('Input-specific source-pose scope changed; fit and review a new round')
    from .initial_state import policy as initial_state_policy
    if inputs.get('initial_state_contract')!=initial_state_policy(s,obj):
        raise ValueError('Authored material-state validity changed; fit and review a new round')
    if inputs.get('initial_state_contract') and not result.get('build',{}).get('initial_state_admissibility',{}).get('pass'):
        raise ValueError('Authored geometry did not pass initial material-state admissibility')
    from .part_evidence import requirements
    if inputs.get('source_part_contract',[])!=requirements(s,obj):
        raise ValueError('Functional source-part constraints changed; fit and review a new round')
    from .source_pose_groups import declaration as pose_groups
    if requirements(s,obj) or pose_groups(s,obj):
        from .part_evidence import attach
        from .evidence import load
        z,_,_,_=load(s,obj)
        attach(s,obj,z)  # Revalidate source-only label identity, not just a review path.
    from .material_reference import declaration as material_references,attach as material_attach
    if inputs.get('source_material_contract',[])!=material_references(s,obj):
        raise ValueError('Source material reference changed; fit and review a new round')
    if material_references(s,obj):
        from .evidence import load
        z,frames,_,_=load(s,obj)
        if material_attach(s,obj,z,frames)!=inputs.get('source_material_evidence'):
            raise ValueError('Source material area evidence changed; fit and review again')
    from calibrated_scene_reconstruction.scene.capture.supplementary import evidence as supplemental_evidence
    if inputs.get('supplemental_evidence') != supplemental_evidence(s):
        raise ValueError('Supplemental media or its usage plan changed; fit and review again')
    from .data_inputs import files as model_data_files
    current_data = model_data_files(s, obj, model_path(s, obj)) if hasattr(s, 'scene_graph') else {}
    if set(inputs.get('source_assets', {})) != set(map(str, current_data)):
        raise ValueError('Declared model data changed since fitting')
    from calibrated_scene_reconstruction.scene.objects.mask_components import policy as component_policy
    if inputs.get('mask_component_policy') != component_policy(s, obj):
        raise ValueError('Semantic component association changed since fitting')
    if 'semantic_query_policy' in inputs:
        from calibrated_scene_reconstruction.scene.objects.observation_history import semantic_policy, object_request
        if inputs['semantic_query_policy'] != semantic_policy(object_request(s, obj)):
            raise ValueError('Semantic query policy changed since fitting')
        if inputs.get('observation_prompt') != object_request(s, obj).get('prompt'):
            raise ValueError('Object observation prompt changed since fitting')
    from .observation_domain import policy
    if inputs.get('observation_domain') != policy(s, obj):
        raise ValueError('Object observation visibility/negative policy changed since fitting')
    from calibrated_scene_reconstruction.scene.objects.observation_exclusions import declaration as exclusions
    if inputs.get('observation_exclusions',[])!=exclusions(s,obj):
        raise ValueError('Observation-only source identity changed; prepare and fit a new round')
    if exclusions(s,obj):
        from .evidence import load
        load(s,obj)  # Revalidate source/support/mask review, never just a pointer.
    from calibrated_scene_reconstruction.scene.objects.mechanisms import declaration,object_scope
    if object_scope(inputs.get('mechanism_contract'),obj) != object_scope(declaration(s),obj):
        raise ValueError('Mechanism declaration changed since this object round; fit and review again')
    for path, expected in inputs.get('dependencies', {}).items():
        if digest(path) != expected:
            raise ValueError('Geometry source/data dependency changed since fitting')
    if inputs.get('support_contact') is not None:
        from calibrated_scene_reconstruction.scene.objects.blender.fit import support_contact
        from calibrated_scene_reconstruction.scene.support.contact_equivalence import matches
        if not matches(inputs['support_contact'],support_contact(s)):
            raise ValueError('Modeled support changed since this object contact fit')
    if (out / 'model.py').exists() and digest(out / 'model.py') != result['model_sha256']:
        raise ValueError('Round source snapshot changed since fitting')
    if contract(model_path(s, obj))['sha256'] != result['model_sha256']:
        raise ValueError('model.py changed since this round')
    for path, expected in inputs['inputs'].items():
        if digest(s / path) != expected:
            raise ValueError(f'Observation changed since this round: {path}')
    if digest(out / 'mesh.npz') != result['mesh_sha256']:
        raise ValueError('Round mesh changed since fitting')
    from calibrated_scene_reconstruction.scene.objects.structural_review import current as structural_review
    result['structural_review_sha256'] = structural_review(s, obj, out)
    if exported:
        fix = s.scene_graph['dynamic_objects'][obj]['fix']
        expected_sdf = sdf_settings(s.scene_graph['dynamic_objects'][obj]['fit'].get('sdf'))
        if fix.get('sdf') != expected_sdf:
            raise ValueError('Exported SDF settings differ from the current fit request; export and review again')
        if fix.get('link_physics', {}) != s.scene_graph['dynamic_objects'][obj]['fit'].get('link_physics', {}):
            raise ValueError('Exported link physics differs from current request')
        if (fix['round'] != result['round'] or fix['model_sha256'] != result['model_sha256']
                or digest(resolve(fix['mesh'])) != result['mesh_sha256']
                or fix.get('opening') != result['build'].get('opening')
                or fix.get('params') != result['build']['params']
                or fix.get('support_rotation') != result['pose'].get('support_rotation')
                or any(fix.get(k) != result['pose'][k] for k in ('xy', 'yaw_deg', 'support_z', 'local_top', 'down_offset'))):
            raise ValueError('Exported geometry or pose differs from the reviewed fit; fit and export again')
    if not result.get('ok') or not result.get('metrics',{}).get('ok'):
        from .source_acceptance import decision
        result['source_acceptance'] = decision(s, obj, out / 'mesh.npz', result)
        result['effective_ok'] = result['source_acceptance']['pass']
    return result


def pending_source_candidate(result):
    """A valid unrelaxed body may enter native tests, never final approval."""
    acceptance=result.get('source_acceptance',{})
    if acceptance.get('policy')=='task_and_semantics':
        checks=acceptance.get('checks',{})
        return bool(acceptance.get('native_candidate_only') is True
                    and acceptance.get('pending_native_eligible') is True
                    and checks.get('unrelaxed_source_geometry') is True
                    and checks.get('material_and_initial_geometry') is True)
    verdict=acceptance.get('input_acceptance',{})
    defaults=verdict.get('defaults',{})
    required={'source_geometry','material_validity','source_material'}
    return bool(result.get('metrics',{}).get('source_pose_controls',{}).get('policy')
                and required<=defaults.keys()
                and all(defaults[k].get('default_pass') is True for k in required))


def run(s, obj, round_dir, *, allow_pending_source=False):
    out = Path(round_dir)
    # Author the candidate collider needed by native response tests. This does
    # not approve OBJ or delivery; those use the default complete verification.
    result = verify_round(s, obj, out, physical_review=False)
    if (not result['ok'] or not result['metrics']['ok']) and not result.get('source_acceptance',{}).get('pass'):
        if not allow_pending_source or not pending_source_candidate(result):
            raise ValueError('Fit has not met IoU/local-difference/frame thresholds; export refused')
    source = s.config_dir / 'scene_graph.json'
    g = json.loads(source.read_text())
    spec = g['dynamic_objects'][obj]
    from calibrated_scene_reconstruction.scene.objects.physical_priors import mass_estimate
    estimate = mass_estimate(s, obj, result['build'])
    if estimate is not None:
        spec['mass'] = estimate['nominal_kg']
        spec['_mass_why'] = 'Agent physical prior and current generated geometry; see agent/physics_priors.json'
        spec['mass_estimate'] = estimate
    from calibrated_scene_reconstruction.assets.mechanisms import parameter_constraints
    from calibrated_scene_reconstruction.scene.objects.mechanisms import declaration
    mechanism = declaration(s)
    if mechanism:
        fitted = {name: o['fix']['params'] for name, o in g['dynamic_objects'].items() if o.get('fix')}
        fitted[obj] = result['build']['params']
        parameter_constraints(mechanism, fitted, require_all=False)
    collision_settings = sdf_settings(spec['fit'].get('sdf'))
    with np.load(out / 'mesh.npz') as d:
        checks = asset_geometry(d['V'], d['F'], json.loads(str(d['links_json'])) if 'links_json' in d else None)
    if checks['volume_m3'] > 0:
        mass_kg, mass_why = mass(checks['volume_m3'], spec)
    else:
        mass_kg, mass_why = None, 'Native surface mass supplied in link_physics; no artificial solid volume'
    pose = result['pose']
    if not -0.012 <= pose['down_offset'] <= 0 or abs(pose['support_z'] - pose['local_top'] - pose['down_offset']) > 1e-8:
        raise ValueError('Invalid local support placement')
    fix = {'model': spec['fit']['model'], 'mesh': rel(out / 'mesh.npz'), 'params': result['build']['params'],
           'xy': pose['xy'], 'yaw_deg': pose['yaw_deg'], 'support_z': pose['support_z'],
           'local_top': pose['local_top'], 'down_offset': pose['down_offset'], 'placement': placement(s),
           'opening': result['build'].get('opening'), 'color': result['build']['params'].get('color', [0.75]*3),
           'round': result['round'], 'model_sha256': result['model_sha256'], 'sdf': collision_settings}
    if spec['fit'].get('link_physics'):
        fix['link_physics'] = spec['fit']['link_physics']
    if pose.get('support_plane'):fix['support_plane']=pose['support_plane']
    if pose.get('support_rotation') is not None:fix['support_rotation']=pose['support_rotation']
    if 'support_contact_offset_m' in pose:
        fix.update(support_contact_offset_m=pose['support_contact_offset_m'],support_surface_z=pose['support_surface_z'])
    previous = spec.get('fix') or {}
    same_surface = all(previous.get(key) == fix.get(key) for key in
                       ('mesh','model_sha256','params','xy','yaw_deg','support_z','support_rotation','placement'))
    if same_surface and previous.get('look'):
        fix['look'] = previous['look']
    elif previous.get('look') and result.get('fixed_candidate_revalidation'):
        from .revalidate import inherited_look
        appearance = inherited_look(s, obj, out, result, previous, fix)
        if appearance is not None:
            fix['look'] = appearance
    spec['fix'] = fix
    write_json(source, g)
    path = s / 'splat/object_fit.json'
    fits = json.loads(path.read_text()) if path.exists() else {}
    fits[obj] = {'route': 'blender', 'model': fix['model'], 'params': fix['params'], **pose,
                 'iou_median': result['metrics']['iou_median'], 'frames': result['metrics']['frames'],
                 'local_difference': result['metrics']['local_difference'], 'converged': result['converged'],
                 'round': result['round'], 'export_checks': {**checks, 'sdf': collision_settings, 'mass_kg': mass_kg, 'mass_why': mass_why}}
    write_json(path, fits)
    exported = {'ok': True, 'geometry': checks, 'sdf': collision_settings, 'mesh': fix['mesh'],
                'mass_kg': mass_kg, 'mass_why': mass_why,
                'note': 'SDF settings are authored by sim.scene; PhysX cooking and task checks require Isaac'}
    exported['native_candidate_only']=bool(allow_pending_source and not result.get('ok')
        and not result.get('source_acceptance',{}).get('pass'))
    if result.get('source_acceptance'):
        exported['source_acceptance'] = result['source_acceptance']
    write_json(out / 'export.json', exported)
    event(s, obj, 'exported', round=result['round'], model_sha256=result['model_sha256'])
    return exported
