"""Mechanism inputs and task-evidence provenance for reconstruction sessions."""
import hashlib
import json
from pathlib import Path
import numpy as np

from calibrated_scene_reconstruction.assets.mechanisms import digest, validate
from calibrated_scene_reconstruction.core.paths import resolve
from calibrated_scene_reconstruction.core.session_code import source_path


def declaration(s):
    if not hasattr(s, 'config_dir'):
        return None  # Lightweight legacy geometry clients have no session configuration.
    path = source_path(s.config_dir, 'mechanisms.json')
    task_path = Path(s.config_dir) / 'task.json'
    task = json.loads(task_path.read_text()) if task_path.exists() else {}
    if not path.exists():
        if task.get('mechanism_contract'):
            raise ValueError('Agent mechanism declaration required before modeling or task validation')
        return None
    value = validate(json.loads(path.read_text()))
    if task.get('physical_observation_contract') and not value.get('observation_contract'):
        raise ValueError('Task requires physical observation consistency before operation checks')
    for observation in value.get('observations', []):
        for relative, expected in observation['evidence_files'].items():
            source = (s.dir/relative).resolve()
            if (not source.is_relative_to(s.dir.resolve()) or not source.is_file()
                    or hashlib.sha256(source.read_bytes()).hexdigest() != expected):
                raise ValueError('Physical observation evidence changed: '+relative)
    from calibrated_scene_reconstruction.core.capabilities import entries, invoke
    covered = set(value['objects'])
    for name in entries('task_objects'):
        covered.update(invoke('task_objects', name, value))
    if covered != set(task.get('objects', {})):
        raise ValueError('Mechanism declaration must cover exactly the task objects')
    return value


def task_check_inputs(s):
    """Resolve declared external assay inputs, rejecting stale or escaped files."""
    manifest = source_path(s.config_dir, 'task_check_inputs.json')
    if not manifest.exists():
        return {}
    record = json.loads(manifest.read_text())
    if record.get('version') != 1 or not isinstance(record.get('files'), dict):
        raise ValueError('Invalid task-check input declaration')
    result = {}
    for name, expected in record['files'].items():
        path = (s.dir/name).resolve()
        if (not path.is_relative_to(s.dir.resolve()) or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
            raise ValueError('Task-check input missing or changed: '+name)
        result[name] = path
    return result


def object_scope(contract,obj):
    """Source-model dependencies owned by an object, including shared mechanisms.

    Full-scene task and delivery fingerprints remain global. A change to an
    unrelated object's internal flexible discretization must not invalidate
    already fitted geometry of every other object.
    """
    if contract is None:return None
    names=set(contract.get('objects',{}))
    def owners(value):
        if isinstance(value,str):return {name for name in names if value==name or value.startswith(name+'/')}
        if isinstance(value,dict):return set().union(*(owners(v) for pair in value.items() for v in pair))
        if isinstance(value,list):return set().union(*(owners(v) for v in value))
        return set()
    def relevant(row):
        owned=owners(row)
        # Unscoped requirements/questions are global, never silently dropped.
        return not owned or obj in owned
    return {**{k:contract[k] for k in ['version','observation_contract','material_state_contract'] if k in contract},
        'objects':{obj:contract.get('objects',{}).get(obj)},
        **{key:[row for row in contract.get(key,[]) if relevant(row)]
           for key in ['mechanisms','observations','open_questions']}}


def sampling_tolerances(s):
    """Scene-derived secondary cloth bounds; primary contacts stay strict."""
    from calibrated_scene_reconstruction.assets.metric_mesh import asset
    from calibrated_scene_reconstruction.assets.response_policy import secondary_sampling_tolerances
    from .object_understanding import declaration as understanding
    result={}
    primary={link for mechanism in (declaration(s) or {}).get('mechanisms',[])
             if mechanism.get('role')=='primary' for link in mechanism.get('links',[])}
    for obj,spec in s.scene_graph.get('dynamic_objects',{}).items():
        fix=spec.get('fix') or {}
        if not fix.get('model'):continue
        parts=asset(fix)['parts']
        if not any(p['kind']=='cloth' for p in parts):continue
        result.update({obj+'/'+name:value for name,value in secondary_sampling_tolerances(
            parts,understanding(s,obj),s.config('task.json',{}).get('flexible_accuracy')).items()
                       if obj+'/'+name not in primary})
    return result


def fingerprint(s):
    """Bind QA to current geometry, physics, task, agent check and mechanism plan.

    Historical sessions without this contract keep their existing review rules.
    """
    mechanism = declaration(s)
    if mechanism is None:
        return None
    from calibrated_scene_reconstruction.scene.objects.physical_priors import declaration as physics_priors
    physical = physics_priors(s)
    files = {}
    files['mechanism_fingerprint_engine']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    pose_engine=Path(__file__).resolve().parents[2]/'assets/object_pose.py'
    files['object_pose_engine']=hashlib.sha256(pose_engine.read_bytes()).hexdigest()
    intrinsic_engine = Path(__file__).with_name('object_understanding.py')
    files['object_understanding_engine'] = hashlib.sha256(intrinsic_engine.read_bytes()).hexdigest()
    understanding = s.config_dir/'agent/object_understanding.json'
    files['object_understanding'] = hashlib.sha256(understanding.read_bytes()).hexdigest() if understanding.is_file() else 'pending'
    for name in ['geometry_checks.json','interface_checks.json']:
        path=s.config_dir/'agent'/name
        if path.is_file():files['source_contract/'+name]=hashlib.sha256(path.read_bytes()).hexdigest()
    for name, spec in s.scene_graph.get('dynamic_objects', {}).items():
        # This review consumes native test recordings. Including its verdict
        # here creates a cycle: reviewing a run would invalidate that same run.
        # OBJ/delivery separately verify its candidate and raw-evidence hashes.
        from .structural_review import review_path
        structural = review_path(s, name)
        if structural is not None:
            files[f'{name}/structural_review'] = (hashlib.sha256(structural.read_bytes()).hexdigest()
                                                   if structural.is_file() else 'pending')
        fix = spec.get('fix') or {}
        for key in ('mesh',):
            if fix.get(key):
                path = resolve(fix[key])
                files[f'{name}/{key}'] = hashlib.sha256(path.read_bytes()).hexdigest()
        for key in ('mesh', 'albedo', 'photometric', 'material_finishes'):
            value = (fix.get('look') or {}).get(key)
            if value:
                path = s.dir/value
                files[f'{name}/look/{key}'] = hashlib.sha256(path.read_bytes()).hexdigest()
        from calibrated_scene_reconstruction.assets.optical_print import declaration as optical_print
        printing=optical_print(fix.get('look'),s.dir) if fix.get('look') else None
        for key,path in (printing or {}).items():
            files[f'{name}/look/optical_print/{key}']=hashlib.sha256(path.read_bytes()).hexdigest()
        if spec.get('fit', {}).get('model'):
            path = source_path(s.config_dir, spec['fit']['model'])
            files[f'{name}/model'] = hashlib.sha256(path.read_bytes()).hexdigest()
    script = source_path(s.config_dir, 'task_check.py')
    files['task_check'] = hashlib.sha256(script.read_bytes()).hexdigest()
    inputs_manifest = source_path(s.config_dir, 'task_check_inputs.json')
    if inputs_manifest.exists():
        files['task_check_inputs'] = hashlib.sha256(inputs_manifest.read_bytes()).hexdigest()
    for name, path in task_check_inputs(s).items():
        files['task_check_input/'+name] = hashlib.sha256(path.read_bytes()).hexdigest()
    # Hash source of the authoring and validation tools, not just the settings.
    root = Path(__file__).resolve().parents[2]
    hand_name=s.config('task.json',{}).get('hand_interaction_plan')
    if hand_name:
        hand_path=(s.config_dir/hand_name).resolve()
        if not hand_path.is_relative_to(s.config_dir.resolve()):
            raise ValueError('Hand interaction plan must belong to the session configuration')
        hand=json.loads(hand_path.read_text());files['hand_interaction_plan']=hashlib.sha256(hand_path.read_bytes()).hexdigest()
        for name in hand.get('asset_dependencies',[]):
            path=(root.parents[1]/name).resolve()
            if not path.is_relative_to(root.parents[1]/'assets') or not path.is_file():
                raise ValueError('Hand interaction requires current local robot asset dependencies')
            files['hand_asset/'+name]=hashlib.sha256(path.read_bytes()).hexdigest()
        if not hand.get('asset_dependencies'):
            raise ValueError('Record the actual composed hand asset layers before validation')
        for name in ['sim/hand_fixture.py','sim/hand_collision.py','sim/hand_materials.py','assets/hand_cloth.py','apps/reconstruction/hand_acceptance.py']:
            files[name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
        from calibrated_scene_reconstruction.core.capabilities import entries
        for key, relative in entries('hand_fingerprint').items():
            files[key]=hashlib.sha256((root.parents[1]/relative).read_bytes()).hexdigest()
    if any((item.get('fit') or {}).get('source_identity_review') is not None
           for item in s.scene_graph.get('dynamic_objects', {}).values()):
        relative = 'scene/objects/object_label_union.py'
        files[relative] = hashlib.sha256((root/relative).read_bytes()).hexdigest()
    from calibrated_scene_reconstruction.core.capabilities import entries
    from calibrated_scene_reconstruction.core.capabilities import source_files
    for path in source_files():
        files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for key, relative in entries('fingerprint').items():
        files[key] = hashlib.sha256((root/relative).read_bytes()).hexdigest()
    for relative in ('sim/scene_stage.py', 'sim/mechanism_stage.py', 'sim/usd.py', 'sim/object_materials.py',
                     'sim/source_moments.py','sim/contact_carrier.py','sim/collision_inventory.py','assets/articulation_initial_state.py',
                     'scene/objects/blender/articulation_path.py',
                     'assets/particle_contact_profile.py','assets/particle_registration.py','assets/particle_batch_order.py',
                     'sim/particle_batch.py','tasks/rigid/batched_flexible_trace.py',
                     'sim/physics.py', 'sim/scene_physics.py', 'sim/mass_properties.py', 'assets/metric_mesh.py',
                     'core/capabilities.py', 'assets/mechanisms.py', 'assets/shell_geometry.py', 'tasks/mechanisms/contact.py', 'tasks/mechanisms/bounded_control.py', 'assets/cloth_sampling.py', 'assets/cloth_mass.py', 'tasks/rigid/task_checks.py',
                     'tasks/rigid/check_script.py', 'assets/acceptance.py','scene/acceptance_evidence.py',
                     'tasks/rigid/native_nesting.py', 'tasks/rigid/nesting.py', 'tasks/rigid/contact_probe.py',
                     'tasks/mechanisms/cloth.py','tasks/mechanisms/seam_contact.py',
                     'tasks/mechanisms/interface_contact.py','assets/interface_geometry.py','assets/ray_triangles.py','assets/joint_state.py',
                     'assets/passage_geometry.py','scene/objects/blender/geometry_checks.py',
                     'assets/contact_control_profile.py',
                     'apps/reconstruction/observation_match.py', 'core/geometry/visibility.py',
                     'scene/objects/physical_observations.py','scene/objects/part_label_review.py',
                     'scene/objects/mask_cache.py','scene/objects/blender/evidence.py','scene/objects/blender/source_acceptance.py',
                     'scene/objects/blender/qualitative_source.py',
                     'scene/objects/optical_visibility.py','assets/optical_surfaces.py','assets/optical_print.py',
                     'tasks/mechanisms/fixed.py',
                     'tasks/mechanisms/force_fixture.py', 'tasks/mechanisms/tabletop.py',
                     'tasks/mechanisms/preparation.py',
                     'assets/surface_distance.py', 'sim/rigid_wrench.py', 'assets/cloth_contact.py','assets/cloth_shell_clearance.py', 'sim/surface_cloth.py', 'sim/physics_errors.py',
                     'assets/physical_priors.py', 'assets/cloth_rest.py', 'assets/cloth_metric.py','assets/cloth_topology.py','assets/cloth_state.py','assets/cloth_curvature.py','assets/support_route.py',
                     'assets/surface_bindings.py','assets/bond_footprint.py','assets/rigid_partition.py','assets/collision_slices.py','assets/static_support.py',
                     'sim/rigid_partition.py','sim/particle_attachment.py','sim/particle_surface.py','sim/particle_contacts.py','assets/particle_contacts.py','sim/application.py',
                     'assets/surface_conditioning.py','assets/surface_refinement.py','assets/rooted_strip.py','assets/surface_mass.py','assets/support_placement.py','sim/support_placement.py','sim/surface_attachment.py','assets/cloth_clearance.py',
                     'scene/objects/blender/root_geometry.py','scene/objects/blender/surface_remesh.py',
                     'assets/constitutive_features.py','sim/physical_priors.py', 'scene/objects/physical_priors.py',
                     'scene/objects/physics_trials.py', 'sim/support_stage.py', 'sim/setpieces.py',
                     'sim/clearance.py', 'sim/zone.py', 'scene/twin/runtime_inputs.py', 'scene/twin/variants.py',
                     'sim/simulation.py', 'scene/simulation.py',
                     'tasks/mechanisms/rigid_probe.py', 'tasks/rigid/insertion_corridor.py',
                     'tasks/rigid/contact_pairs.py', 'assets/rigid_trajectory.py',
                     'tasks/rigid/native_assay.py','assets/rigid_frame_error.py','tasks/rigid/flexible_trace.py','tasks/rigid/flexible_trajectory.py','tasks/rigid/flexible_assessment.py', 'tasks/rigid/assembly_measurements.py',
                     'tasks/rigid/contact_regions.py', 'assets/observation_state.py',
                     'assets/collision_precision.py','assets/cloth_contact_tolerance.py',
                     'assets/rigid_initial_state.py', 'assets/convex_source_solids.py',
                     'assets/revolved_source_cells.py', 'assets/collision_groups.py',
                     'sim/convex_cooking.py', 'scene/support/runtime_inputs.py'):
        path = root / relative
        if path.exists():
            files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    for relative in ['scene/objects/material_response.py','scene/objects/contact_response.py','assets/response_policy.py','assets/structural_layout.py','assets/particle_lattice.py',
                     'sim/surface_cloth.py','assets/flexure_reference.py','assets/surface_contact_profile.py']:
        path=root/relative;files[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
    response=s.config('agent/material_response.json',{})
    contacts=s.config('agent/contact_response.json',{})
    for link,cases in contacts.get('controls',{}).items():
        for case,effects in cases.items():
            for effect,pair in effects.items():
                for kind,entry in pair.items():
                    path=(s.dir/entry['path']).resolve()
                    if not path.is_relative_to(s.dir.resolve()/'qa') or not path.is_file():
                        raise ValueError('Current session native contact-response evidence required')
                    files[f'{link}/contact_response/{case}/{effect}/{kind}']=hashlib.sha256(path.read_bytes()).hexdigest()
    input_acceptance=s.config('agent/acceptance_evidence.json',{})
    for link,cases in response.get('controls',{}).items():
        for case,controls in cases.items():
            for kind,entry in controls.items():
                path=(s.dir/entry['path']).resolve()
                if not path.is_relative_to(s.dir.resolve()/'qa') or not path.is_file():
                    raise ValueError('Current session native material-response evidence required')
                files[f'{link}/material_response/{case}/{kind}']=hashlib.sha256(path.read_bytes()).hexdigest()
    task_app = root.parents[1] / 'apps/scene/task_check.py'
    if task_app.exists():
        files['apps/scene/task_check.py'] = hashlib.sha256(task_app.read_bytes()).hexdigest()
    response_app=root.parents[1]/'apps/scene/check_surface_response.py'
    files['apps/scene/check_surface_response.py']=hashlib.sha256(response_app.read_bytes()).hexdigest()
    response_app=root.parents[1]/'apps/scene/check_surface_flexure.py'
    files['apps/scene/check_surface_flexure.py']=hashlib.sha256(response_app.read_bytes()).hexdigest()
    graph = s.scene_graph
    from .blender.material_reference import declaration as material_references
    source_material={name:material_references(s,name) for name in graph.get('dynamic_objects',{})}
    source_material={name:value for name,value in source_material.items() if value}
    if source_material:
        from .observation_exclusions import declaration as exclusions
        path=root/'scene/objects/blender/material_reference.py'
        files['source_material_engine']=hashlib.sha256(path.read_bytes()).hexdigest()
        path=root/'assets/cloth_outline.py'
        files['source_material_outline_engine']=hashlib.sha256(path.read_bytes()).hexdigest()
        for name in source_material:
            regions={item['id']:item for item in exclusions(s,name)}
            for item in source_material[name]:
                region=regions[item['region']]
                for label,path in [('source',s.dir/region['source_image']),
                                   ('mask',source_path(s.config_dir,region['mask']))]:
                    files[f'{name}/material_reference/{item["region"]}/{label}']=hashlib.sha256(path.read_bytes()).hexdigest()
    from calibrated_scene_reconstruction.scene.simulation import settings as simulation_settings
    from calibrated_scene_reconstruction.scene.capture.supplementary import evidence as supplemental_evidence
    extra = supplemental_evidence(s)
    return digest({'mechanisms': mechanism, 'physics_priors': physical, 'task': s.config('task.json', {}),
                   **({'input_acceptance_evidence':input_acceptance} if input_acceptance else {}),
                   **({'native_material_response':response} if response else {}),
                   **({'native_contact_response':contacts} if contacts else {}),
                   **({'source_material_reference':source_material} if source_material else {}),
                   **({'supplemental_evidence': extra} if extra is not None else {}),
                   'simulation': simulation_settings(s),
                   'objects': graph.get('dynamic_objects'), 'support': graph.get('furniture'),
                   'set_pieces': graph.get('set_pieces'), 'static_objects': graph.get('static_objects'),
                   'movable_furniture': graph.get('movable_furniture'),
                   'scene_variant': s.read_json('scene/twin.json') if (s/'scene/twin.json').exists() else None,
                   'files': files})


def validate_model(s, obj, model_contract):
    value = declaration(s)
    if value is None:
        task_path = Path(s.config_dir)/'task.json' if hasattr(s, 'config_dir') else None
        task = json.loads(task_path.read_text()) if task_path is not None and task_path.exists() else {}
        if task.get('task') and obj in task.get('objects', {}):
            raise ValueError('Before a new task-object fit, agent must declare mechanisms.json (also for rigid-only objects)')
        return
    expected = value['objects'][obj]['links']
    actual = model_contract.get('links', {'model': {'kind': 'rigid'}})
    if {n: p['kind'] for n, p in expected.items()} != {n: p['kind'] for n, p in actual.items()}:
        raise ValueError(f'{obj}: model LINKS differ from the agent mechanism declaration')
    if any(p.get('backend','particle')!=actual[n].get('backend','particle') for n,p in expected.items()):
        raise ValueError(f'{obj}: native material backend differs from the declared mechanism')
    declared_joints = model_contract.get('joints', [])
    expected_bonds={(b['surface'],b['rigid']) for b in value['objects'][obj].get('surface_connections',[])}
    if 'surface_bindings' in model_contract:
        actual_bonds={(b['surface'],b['rigid']) for b in model_contract['surface_bindings']}
        if actual_bonds!=expected_bonds:raise ValueError('Actual rooted physical connections differ from the declaration')
    elif expected_bonds and not model_contract.get('surface_bindings_dynamic'):
        raise ValueError('Required rooted physical connections missing from the model')
    if model_contract.get('joints_dynamic'):
        return  # Actual parameter-derived joint frames are checked after Blender evaluates them.
    for mechanism in value['mechanisms']:
        refs = mechanism['links']
        if len(refs) != 2 or any(not r.startswith(obj+'/') for r in refs):
            continue
        a, b = (r.split('/', 1)[1] for r in refs)
        matching = [j for j in declared_joints if {j['parent'], j['child']} == {a, b}]
        if mechanism['type'] == 'contact' and matching:
            raise ValueError('Contact mechanism incorrectly replaced by a joint')
        if mechanism['type'] in {'fixed', 'revolute', 'prismatic'} and mechanism['role'] != 'secondary':
            if len(matching) != 1 or matching[0]['type'] != mechanism['type']:
                raise ValueError('Required real joint missing or wrong type')
            if mechanism['type'] != 'fixed' and any(not np.allclose(matching[0].get(k), mechanism[k])
                                                    for k in ('axis', 'limits')):
                raise ValueError('Joint axis or limits differ from declared mechanism')


def evidence_files(s, results):
    """Hash numeric/image/video evidence; forbid paths outside this session."""
    files = {}
    if not isinstance(results, dict):
        raise ValueError('Mechanism results must be an object')
    for result in results.values():
        if not isinstance(result, dict):
            raise ValueError('Each mechanism result must be an object')
        for name in result.get('evidence_files', []):
            if not isinstance(name, str):
                raise ValueError('Evidence file names must be strings')
            path = (s.dir/name).resolve()
            if not path.is_relative_to(s.dir.resolve()) or not path.is_file():
                raise ValueError('Mechanism evidence must be an existing file inside this session')
            files[str(path.relative_to(s.dir.resolve()))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files
