"""Agent-neutral JSON tools: python -m calibrated_scene_reconstruction.objects.tool NAME --request FILE --output DIR."""
import argparse
import hashlib
import itertools
import json
import subprocess
from pathlib import Path

import jsonschema

from calibrated_scene_reconstruction.core.capabilities import entries, invoke

from calibrated_scene_reconstruction.core.paths import ROOT, rel
from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.objects.blender import evidence, export, fit, store, texture
from calibrated_scene_reconstruction.scene.objects.blender.blender_run import Blender
from calibrated_scene_reconstruction.scene.objects.blender.contract import write_json

DESCRIPTIONS = {
    'inputs': 'Read reviewed scene, closeup and motion evidence with object identity, state and scale-source provenance.',
    'blender_run': 'Build and validate a Blender model; output includes mesh, GLB and appearance preview paths.',
    'fit': 'Fit model parameters and pose to cached metric silhouettes; write round evidence and numeric gates.',
    'revalidate': 'Re-evaluate an exact historically frozen rigid candidate in a new round against current source observations. Requires archived freeze lineage; preserves geometry and pose, never grants native approval or activates the candidate.',
    'cloth_proposal': 'Propose source-constrained flexible geometry with actual solid distances and unchanged material/self-contact checks; requires agent rebuild and native/task checks, never exports.',
    'sheet_reference': 'Recover one complete flat material surface and fabric photograph from a reviewed unfolded source reference; no folded pose, paper crease, stiffness identification or task approval.',
    'orientation_review': 'Compare flexible-link direction hypotheses in all reviewed source views while preserving rigid geometry and the material metric; output is a rebuild proposal, never a contact or task pass.',
    'native_cloth_proposal': 'Propose geometry preserving full cloth metric and actual rigid/joint state. Default requires a valid released trace; geometry_seed can use an individually valid state from a failed completed diagnostic, retaining its failures and granting no native evidence. Always requires agent rebuilding and fresh source/native/task gates.',
    'evidence': 'Render exact multi-view silhouette evidence for an existing round; stamp the OBJ review.',
    'texture': 'Bake visible frame colours into the exported model UV atlas; write coverage and evidence images.',
    'export': 'Export a passing model round into the scene graph with surface SDF collision settings.',
    'refine_run': 'Run fit, evidence and gated export for one Blender object; the agent reviews OBJ separately.',
    'tune': 'Evaluate a bounded parameter grid within a round using exact silhouettes, retaining evidence.',
    'task_check': 'Check object-model task feasibility and explicit input acceptance in Isaac; default object fixtures or the specifically requested hand/contact interaction, without autonomous robot execution unless requested.'}

DESCRIPTIONS.update(entries('object_descriptions'))


def schemas():
    return {p.stem: json.loads(p.read_text()) for p in (Path(__file__).with_name('schemas') / 'objects').glob('*.json')}


def session_evidence(session, relative):
    """Resolve an existing session-owned input; reject traversal and symlink escape."""
    from pathlib import PurePosixPath
    value = PurePosixPath(relative)
    if (value.is_absolute() or '..' in value.parts or '\\' in relative or ':' in relative):
        raise ValueError('Revalidation evidence must use session-relative paths')
    path = (session.dir / str(value)).resolve()
    if not path.is_relative_to(session.dir.resolve()) or not path.is_file():
        raise ValueError('Revalidation evidence must be an existing file inside the session')
    return path


def modeling_facts(session,target=None):
    """Give the modeling agent declared reference identity and material family.

    Only modeling input files are read; evaluation dimensions are excluded.
    These declarations retain their evidence/uncertainty, not task approval.
    """
    mechanisms=session.config('agent/mechanisms.json',{})
    priors=session.config('agent/physics_priors.json',{})
    from calibrated_scene_reconstruction.assets.physical_priors import expand_families
    priors=expand_families(priors)
    understanding=session.config('agent/object_understanding.json',{}).get('objects',{})
    understanding={name: value for name,value in understanding.items()
                   if target is None or name==target.split('/')[0]}
    task=session.config('task.json',{})
    from calibrated_scene_reconstruction.assets.acceptance import declaration as acceptance
    input_acceptance=acceptance(task)
    from calibrated_scene_reconstruction.scene.objects.blender.material_reference import declaration
    objects=[target.split('/')[0]] if target else list(mechanisms.get('objects',{}))
    from calibrated_scene_reconstruction.scene.objects.deformable_method import plan as deformable_method_plan
    deformable_methods=deformable_method_plan(mechanisms,objects)
    from calibrated_scene_reconstruction.scene.objects.observation_exclusions import declaration as source_regions
    references=[]
    for obj in objects:
        regions={r['id']:r for r in source_regions(session,obj)}
        references.extend({**r,'source_region':regions.get(r['region'])} for r in declaration(session,obj))
    materials={name:{'material_hypothesis':item.get('material_hypothesis'),
                     'material_class_evidence':item.get('material_class_evidence'),
                     'parameters':item.get('parameters',{})}
               for name,item in priors.get('bodies',{}).items()
               if target is None or name==target or name.startswith(target+'/')}
    response=session.config('agent/material_response.json',{}).get('controls',{})
    response={name:cases for name,cases in response.items() if target is None or name==target or name.startswith(target+'/')}
    contacts=session.config('agent/contact_response.json',{}).get('controls',{})
    contacts={name:cases for name,cases in contacts.items() if target is None or name==target or name.startswith(target+'/')}
    from calibrated_scene_reconstruction.scene.objects.contact_response import recorded_controls
    contact_readiness=recorded_controls(session,objects)
    files=[session.config_dir/name for name in ['task.json','agent/mechanisms.json','agent/physics_priors.json','agent/material_references.json','agent/observation_exclusions.json','agent/material_response.json']]
    files.append(session.config_dir/'agent/contact_response.json')
    files.append(session.config_dir/'agent/object_understanding.json')
    hand=None
    if task.get('hand_interaction_plan'):
        path=(session.config_dir/task['hand_interaction_plan']).resolve()
        if not path.is_relative_to(session.config_dir.resolve()):raise ValueError('Session-owned hand interaction plan required')
        hand=json.loads(path.read_text());files.append(path)
    return {'references':references,'materials':materials,
            'object_understanding':understanding,
            'object_understanding_required_before_fit':True,
            'physical_modeling_requirements':{
                'semantic_review_covers_furniture_and_every_reconstructed_object':True,
                'check_part_count_layout_symmetry_connections_and_support':True,
                'structural_priors_need_source_review_not_blind_symmetry_enforcement':True,
                'flexible_response_accuracy_default':'task_and_semantics',
                'flexible_exact_material_and_local_shape_errors_are_diagnostics_unless_requested':True,
                'flexible_semantics_physical_plausibility_and_native_task_remain_required':True,
                'rigid_precision_is_not_relaxed_by_flexible_policy':True,
                'observed_loaded_shape_is_not_automatically_stress_free':True,
                'static_unactuated_pose_requires_support_and_mass_layout_review':True,
                'task_holders_cannot_certify_unforced_assembly_stability':True,
                'measure_rigid_surface_motion_including_rotation_before_installing_task_holders':True,
                'inspect_visible_end_shapes_and_cross_section_changes':True,
                'silhouette_fit_alone_certifies_support_contact_geometry':False,
                'distinct_materials_may_need_distinct_spatial_mass_moments':True,
                'uniform_compound_density_requires_justification':True,
                'source_support_consistency_identifies_material_parameters':False,
                'loaded_pose_diagnostics':{
                    'inspect_actual_support_contacts_before_material_fitting':True,
                    'check_hidden_load_paths_against_all_source_views':True,
                    'inferred_backings_require_real_geometry_and_independent_review':True,
                    'static_stability_alone_approves_a_hidden_construction':False,
                    'geometry_repair_requires_fresh_native_passive_contact_and_task_checks':True},
                'numerical_material_qualification':{
                    'passive_shape_alone_proves_two_way_coupling':False,
                    'require_independent_force_or_momentum_controls':True,
                    'excitation_must_transfer_a_resolved_signal_between_coupled_domains':True,
                    'include_rigid_coasting_without_reaction_as_a_negative_control':True,
                    'weak_coupling_signal_does_not_qualify_an_attachment':True,
                    'bonded_sheets_need_tangential_and_normal_load_controls':True,
                    'check_fixture_load_direction_boundary_conditions_and_reference_regime':True,
                    'compression_after_buckling_is_not_an_axial_tensile_modulus_measurement':True,
                    'check_discretization_and_timestep_convergence':True,
                    'controls_must_share_runtime_solver_and_external_force_settings':True,
                    'known_SI_spring_control_before_adjusting_elastic_constants':True,
                    'a_numerical_refinement_may_change_material_or_mass':False,
                    'a_failed_candidate_may_redefine_its_acceptance_bounds':False},
                'native_particle_cross_part_contacts':{
                    'separate_systems_provide_cross_cloth_contacts':False,
                    'shared_system_requires_compatible_system_wide_contact_settings':True,
                    'independent_native_positive_and_isolated_negative_controls_required':True,
                    'sharing_does_not_prove_particle_coverage_or_material_accuracy':True},
                'scope':'Review physical construction and geometric/numerical method before fitting material constants. These requirements do not approve a solver, waive source or task gates, or identify unmeasured material properties.'},
            'intrinsic_validation_requirement':'Inventory all semantic parts, materials, alternative hypotheses, connections and expected responses before choosing representation. Include secondary parts. Validate actual geometry, physical link kinds and independent passive/contact/material response; successful task execution cannot approve object identity or physical fidelity. See docs/agent/workflow.md.',
            'link_physics_requirements':{f'{obj}/{name}':{k:body[k] for k in ['required_contact_effects','initial_support'] if k in body}
                 for obj,spec in mechanisms.get('objects',{}).items() for name,body in spec.get('links',{}).items()
                 if obj in objects and any(k in body for k in ['required_contact_effects','initial_support'])},
            'input_acceptance':input_acceptance,'hand_interaction':hand,
            'native_material_response_controls':response,
            'native_contact_response_controls':contacts,
            'native_contact_control_readiness':contact_readiness,
            'deformable_method_plan':deformable_methods,
            'cloth_initialization_requirements':{
                'finite_sheet_solid_clearance':True,
                'particle_coverage_and_nonlocal_initial_envelopes':any(
                    row['particle_coverage_required'] for row in deformable_methods.values()),
                'numerical_offsets_are_material_thickness':False,
                'pose_repair_may_change_intrinsic_material_coordinates':False,
                'rigid_closed_pose_changes_require_source_review':True,
                'geometry_pass_requires_fresh_native_and_task_checks':True},
            'construction_questions':mechanisms.get('open_questions',[]),
            'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file()},
            'scope':'Declared source identity and material evidence/priors for modeling; no evaluation dimensions or source/task approval. A complete unfolded reference constrains its full material shape. User-confirmed cloth family is valid input; numerical constitutive properties remain uncertain unless separately observed.'}


def execute(name, request, output):
    registry = schemas()
    if name not in registry:
        raise ValueError('Unsupported tool: ' + str(name))
    schema = registry[name]
    jsonschema.validate(request, schema['input'])
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    s = Session(request['session'])
    obj = request.get('object')
    if name == 'inputs':
        from calibrated_scene_reconstruction.scene.capture.supplementary import modeling_inputs
        result = modeling_inputs(s, obj)
        result['modeling_facts']=modeling_facts(s,obj)
        result['evidence_images']=list(dict.fromkeys(result['evidence_images']+
            [r['source_region']['source_image'] for r in result['modeling_facts']['references'] if r.get('source_region')]))
    elif name == 'blender_run':
        from calibrated_scene_reconstruction.scene.objects.object_understanding import declaration, validate_build
        declaration(s, obj)
        from calibrated_scene_reconstruction.scene.objects.blender.data_inputs import files as model_data_files
        model = fit.model_path(s, obj)
        from calibrated_scene_reconstruction.scene.objects.blender.contract import contract
        spec = contract(model)
        from calibrated_scene_reconstruction.scene.objects.mechanisms import validate_model
        validate_model(s,obj,spec)
        validate_build(s, obj, {'parts': [dict(name=n,kind=v['kind']) for n,v in spec.get('links',{'model':{'kind':'rigid'}}).items()],
                                'joints': spec.get('joints',[])}, preflight=True)
        assets = model_data_files(s, obj, model)
        with Blender(model, out / 'cache', request.get('timeout', 180), source_files=assets) as builder:
            path, result = builder.evaluate(request.get('params'), artifacts=True)
        validate_build(s, obj, result)
        validate_model(s,obj,{'links':result.get('links',{}),'joints':result.get('joints',[]),
                              'surface_bindings':result.get('surface_bindings',[])})
        result = {**result, 'artifact_dir': rel(path), 'evidence_images': [rel(path / p) for p in result['previews']]}
        result['source_data_sha256'] = {str(p): store.digest(p) for p in assets}
    elif name in ('fit', 'refine_run'):
        result, round_dir = fit.run(s, obj, request.get('max_evaluations', 300))
        result['round_dir'] = str(round_dir.relative_to(s.dir))
        if name == 'refine_run' and result.get('effective_ok',result['ok']):
            result['export'] = export.run(s, obj, round_dir)
        result['evidence_images'] = [result['metrics']['sheet']]
    elif name == 'revalidate':
        from calibrated_scene_reconstruction.scene.objects.blender.revalidate import run
        result, round_dir = run(s, obj,
            inherited_freeze=session_evidence(s, request['inherited_freeze']),
            inherited_scene_graph=session_evidence(s, request['inherited_scene_graph']))
        result.update(round_dir=str(round_dir.relative_to(s.dir)), source_pass=bool(result['ok']),
                      native_approval=False, active_candidate_changed=False)
        result['evidence_images'] = [result['metrics']['sheet']]
        if result['metrics'].get('opening_sheet'):
            result['evidence_images'].append(result['metrics']['opening_sheet'])
    elif name == 'export':
        result = export.run(s, obj, s / 'objects' / obj / f'round_{request["round"]}')
    elif name == 'cloth_proposal':
        from calibrated_scene_reconstruction.scene.objects.blender.cloth_proposal import run
        result=run(s,obj,request['candidate'],request['observation'],out/'proposal',
                   steps=request.get('steps',400),learning_rate=request.get('learning_rate',2e-6),
                   membrane_weight=request.get('membrane_weight',10.),mode=request.get('mode','field'))
        result.update(ok=True,source_ok=bool(result['metrics']['ok']),proposal_only=True,
                      candidate=str((out/'proposal/candidate.json').resolve()),
                      evidence_images=[p for item in result['visual'].values() for p in item['sheets']])
    elif name == 'sheet_reference':
        from calibrated_scene_reconstruction.scene.objects.blender.sheet_reference import run
        result=run(s,obj,request['link'],out/'sheet_reference',spacing_m=request.get('spacing_m',.001))
    elif name == 'orientation_review':
        from calibrated_scene_reconstruction.scene.objects.blender.source_orientation import run
        result=run(s,obj,request['link'],request['candidate'],out/'orientation_review',
                   pivot=request['pivot'],axis=request['axis'],angles=request['angles_deg'])
    elif name == 'native_cloth_proposal':
        from calibrated_scene_reconstruction.scene.objects.blender.native_proposal import run
        result=run(s,obj,request['link'],request['anchor'],request['candidate'],request['probe'],out/'native_proposal',
                   mode=request.get('mode','released_candidate'),time_s=request.get('time_s'))
    elif name in entries('object_tools'):
        result = invoke('object_tools', name, s, obj, request, out)
    elif name == 'texture':
        result = texture.run(s, obj, request.get('resolution', 512))
        result['evidence_images'] = [result['sheet']]
    elif name in ('evidence', 'tune'):
        rd = s / 'objects' / obj / f'round_{request["round"]}'
        r = export.verify_round(s, obj, rd)
        if name == 'tune':
            grid = request['grid']
            count = 1
            for values in grid.values():
                count *= len(values)
            if count > request.get('max_evaluations', 300):
                raise ValueError('Parameter grid exceeds evaluation budget')
            best = (r['metrics']['loss'], r['params'], r['metrics'])
            obs = evidence.load(s, obj)
            import numpy as np
            inputs = json.loads((rd/'input.json').read_text())
            assets = inputs.get('source_assets', {})
            with Blender(rd / 'model.py', s / 'objects' / obj / 'cache',
                         source_files=[*assets, *[rd/path for path in assets.values()]]) as builder:
                for values in itertools.product(*grid.values()):
                    params = {**r['params'], **dict(zip(grid, values))}
                    path, _ = builder.evaluate(params)
                    with np.load(path / 'mesh.npz') as mesh:
                        metrics, _ = evidence.score(mesh['V'], mesh['F'], r['pose'], obs)
                    if metrics['loss'] < best[0]:
                        best = (metrics['loss'], params, metrics)
            result = {'ok': best[2]['ok'], 'params': best[1], 'loss': best[0], 'metrics': best[2], 'evaluations': count,
                      'note': 'Set these PARAMS init values in model.py, then refine_run a new round for acceptance.'}
            store.event(s, obj, 'tuned', round=request['round'], evaluations=count, params=best[1])
        else:
            result = evidence.run(s, obj, rd / 'mesh.npz', r['pose'], rd, request['round'])
            result['evidence_images'] = [result['sheet']]
    elif name == 'task_check':
        (s / 'qa/task_check.json').unlink(missing_ok=True)
        with (out / 'task_check.log').open('w') as log:
            subprocess.run([str(ROOT / 'tools/run_isaac.sh'), 'apps/scene/task_check.py', str(s.dir),
                            '--background', request.get('background', 'zone')], cwd=ROOT,
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=request.get('timeout', 600))
        result = json.loads((s / 'qa/task_check.json').read_text())
        result['ok'] = bool(result['pass'])
        result['evidence_images'] = ['qa/task_check.jpg']
    result.setdefault('evidence_images', [])
    if name != 'blender_run':
        result['evidence_images'] = [rel(s / path) for path in result['evidence_images']]
    jsonschema.validate(result, schema['output'])
    write_json(out / 'result.json', result)
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('name', choices=list(DESCRIPTIONS))
    p.add_argument('--request', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    try:
        result = execute(a.name, json.loads(Path(a.request).read_text()), a.output)
    except Exception as exc:
        write_json(Path(a.output) / 'result.json', {'ok': False, 'error': str(exc), 'evidence_images': []})
        raise
    print(json.dumps({'ok': result.get('effective_ok',result['ok']), 'default_ok':result['ok'], 'result': str(Path(a.output) / 'result.json')}))
    if not result.get('effective_ok',result['ok']):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
