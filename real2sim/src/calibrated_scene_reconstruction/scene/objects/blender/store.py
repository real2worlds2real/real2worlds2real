"""Append-only object attempts with immutable script snapshots and input fingerprints."""
import hashlib
import json
import shutil
import time
from pathlib import Path

from calibrated_scene_reconstruction.scene.objects.blender.contract import contract, write_json


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def event(s, obj, kind, **data):
    path = s / 'objects' / obj / 'events.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps({'time': time.time(), 'event': kind, **data}, ensure_ascii=False) + '\n')


def start(s, obj, model, budget=300):
    from calibrated_scene_reconstruction.scene.objects.object_understanding import declaration as understanding
    object_contract = understanding(s, obj)
    spec = contract(model)
    from calibrated_scene_reconstruction.scene.objects.mechanisms import declaration
    mechanism = declaration(s)
    root = s / 'objects' / obj
    root.mkdir(parents=True, exist_ok=True)
    events = root / 'events.jsonl'
    for line in events.read_text().splitlines() if events.exists() else []:
        r = json.loads(line)
        if r['event'] == 'rejected' and r.get('model_sha256') == spec['sha256']:
            raise ValueError('This model.py was rejected and has not changed; revise the script before resubmission')
    rounds = [int(p.name[6:]) for p in root.glob('round_*') if p.name[6:].isdigit()]
    k = max(rounds, default=0) + 1
    out = root / f'round_{k}'
    out.mkdir()  # exclusive creation: concurrent attempts never overwrite one another
    shutil.copyfile(model, out / 'model.py')
    engine = out/'engine'
    engine.mkdir()
    from .raster_contract import current as raster_contract, sources as raster_sources
    source_raster_contract = raster_contract()
    for source in raster_sources():
        shutil.copyfile(source, engine/source.name)
    shutil.copyfile(Path(__file__).resolve().parent.parent/'object_understanding.py', engine/'object_understanding.py')
    shutil.copyfile(Path(__file__).resolve().parents[3]/'assets/object_understanding.py', engine/'asset_object_understanding.py')
    shutil.copyfile(Path(__file__).resolve().parents[3]/'assets/object_pose.py',engine/'object_pose.py')
    shutil.copyfile(Path(__file__).resolve().parents[2]/'capture/valid_pixels.py',engine/'valid_pixels.py')
    for name in ('evidence.py', 'boundary_search.py', 'fit.py', 'store.py', 'projection.py', 'pose_polish.py', 'surface_observations.py', 'observation_domain.py', 'data_inputs.py','part_evidence.py','proposal_stage.py','material_reference.py','initial_state.py','source_acceptance.py','qualitative_source.py','source_pose_groups.py'):
        shutil.copyfile(Path(__file__).with_name(name), engine/name)
    shutil.copyfile(Path(__file__).resolve().parent.parent/'mask_components.py', engine/'mask_components.py')
    shutil.copyfile(Path(__file__).resolve().parent.parent/'mask_cache.py', engine/'mask_cache.py')
    shutil.copyfile(Path(__file__).resolve().parent.parent/'observation_history.py', engine/'observation_history.py')
    shutil.copyfile(Path(__file__).resolve().parent.parent/'observation_exclusions.py', engine/'observation_exclusions.py')
    request = getattr(s,'scene_graph',{}).get('dynamic_objects',{}).get(obj,{}).get('fit') or {}
    if request.get('collection_context'):
        shutil.copyfile(Path(__file__).resolve().parent.parent/'collection_observations.py', engine/'collection_observations.py')
    if request.get('source_identity_review') is not None:
        for name in ['object_label_union.py', 'part_label_review.py']:
            shutil.copyfile(Path(__file__).resolve().parent.parent/name, engine/name)
    from .part_evidence import requirements
    part_contract=requirements(s,obj)
    from .source_pose_groups import declaration as pose_groups
    group_contract=pose_groups(s,obj)
    if part_contract or group_contract:
        shutil.copyfile(Path(__file__).resolve().parent.parent/'physical_observations.py',engine/'physical_observations.py')
        shutil.copyfile(Path(__file__).resolve().parent.parent/'part_label_review.py',engine/'part_label_review.py')
        shutil.copyfile(Path(__file__).resolve().parents[3]/'core/geometry/visibility.py',engine/'state_visibility.py')
        for relative in ['assets/optical_surfaces.py','assets/surface_regions.py','scene/objects/optical_visibility.py']:
            path=Path(__file__).resolve().parents[3]/relative;shutil.copyfile(path,engine/path.name)
    if digest(out / 'model.py') != spec['sha256']:
        raise ValueError('Model changed while snapshotting this round')
    inputs = {str(p.relative_to(s.dir)): digest(p) for p in (
        s / 'parsing/posed_frames/cameras.npz', s / 'splat/object_fit' / f'{obj}.npz',
        s / 'splat/object_fit' / f'{obj}.start.json', s / 'parsing/world_frame.json',
        s / 'splat/align.json', s / 'splat/data/sparse/full/points3D.bin') if p.exists()}
    inputs.update({str(p.relative_to(s.dir)): digest(p) for p in engine.glob('*.py')})
    if request.get('support_seed'):
        from .support_seed import inputs as support_inputs
        inputs.update(support_inputs(s,obj))
        path=engine/'support_seed.py'
        shutil.copyfile(Path(__file__).with_name('support_seed.py'),path)
        inputs[str(path.relative_to(s.dir))]=digest(path)
    if request.get('support_placement') is not None:
        from .triangle_support import inputs as triangle_support_inputs
        inputs.update(triangle_support_inputs(s,obj))
        for relative in ['scene/objects/blender/triangle_support.py','assets/support_placement.py',
                         'assets/rigid_initial_state.py','assets/surface_distance.py']:
            source=Path(__file__).resolve().parents[3]/relative
            path=engine/source.name;shutil.copyfile(source,path)
            inputs[str(path.relative_to(s.dir))]=digest(path)
    if request.get('source_support_occlusion'):
        path=engine/'context_visibility.py'
        shutil.copyfile(Path(__file__).with_name('context_visibility.py'),path)
        inputs[str(path.relative_to(s.dir))]=digest(path)
    if part_contract or group_contract:
        component_archive=(s/'splat/object_fit'/obj).with_suffix('.components.acquired.npz')
        suffixes=['.acquired.npz']+(['.components.acquired.npz'] if component_archive.exists()
                                  else ['.collections.acquired.npz','.collections.json'])
        for suffix in suffixes:
            path=(s/'splat/object_fit'/obj).with_suffix(suffix)
            inputs[str(path.relative_to(s.dir))]=digest(path)
    if group_contract:
        for relative in ['assets/rigid_initial_state.py','assets/surface_distance.py']:
            path=Path(__file__).resolve().parents[3]/relative
            destination=engine/path.name;shutil.copyfile(path,destination)
            inputs[str(destination.relative_to(s.dir))]=digest(destination)
    if (getattr(s, 'scene_graph', {}).get('dynamic_objects', {}).get(obj, {}).get('fit') or {}).get('part_instances'):
        for suffix in ('.collections.acquired.npz', '.collections.json'):
            path=(s/'splat/object_fit'/obj).with_suffix(suffix)
            inputs[str(path.relative_to(s.dir))]=digest(path)
    if request.get('collection_context'):
        for suffix in ['.acquired.npz','.components.acquired.npz','.instances.acquired.npz','.instances.json']:
            path=(s/'splat/object_fit'/obj).with_suffix(suffix)
            inputs[str(path.relative_to(s.dir))]=digest(path)
    from .data_inputs import snapshot
    assets = snapshot(s, obj, model, out)
    for relative in assets.values():
        inputs[str((out/relative).relative_to(s.dir))] = digest(out/relative)
    from .observation_domain import policy
    from calibrated_scene_reconstruction.scene.objects.observation_exclusions import declaration as exclusions
    from calibrated_scene_reconstruction.scene.objects.mask_components import policy as component_policy
    from .evidence import load
    from calibrated_scene_reconstruction.scene.capture.supplementary import evidence as supplemental_evidence
    from calibrated_scene_reconstruction.scene.objects.observation_history import semantic_policy, object_request
    z, frames, masks, windows = load(s, obj)
    inputs.update(z.get('_object_domain_inputs',{}))
    pose_seed = None
    if request.get('source_pose_seed') is not None:
        from .source_pose_seed import snapshot as snapshot_pose_seed
        pose_seed = snapshot_pose_seed(s, obj, out, observations=load(s, obj, parts=True))
        path = engine/'source_pose_seed.py'
        shutil.copyfile(Path(__file__).with_name('source_pose_seed.py'), path)
        inputs[str(path.relative_to(s.dir))] = digest(path)
        for path in (out/'source_pose_seed').iterdir():
            inputs[str(path.relative_to(s.dir))] = digest(path)
    from .material_reference import declaration as material_references,attach as material_attach
    material_contract=material_references(s,obj)
    from .initial_state import policy as initial_state_policy
    from .source_acceptance import policy as source_pose_policy
    state_contract=initial_state_policy(s,obj)
    if state_contract:
        for relative in ['assets/cloth_metric.py','assets/cloth_rest.py','assets/cloth_state.py','assets/response_policy.py',
                         'assets/surface_bindings.py','assets/bond_footprint.py','assets/rigid_partition.py','assets/collision_slices.py','assets/static_support.py','assets/surface_conditioning.py','assets/support_placement.py','assets/surface_mass.py',
                         'scene/objects/blender/root_geometry.py','scene/objects/blender/surface_remesh.py',
                         'sim/surface_attachment.py','sim/rigid_partition.py','sim/particle_attachment.py','sim/particle_surface.py','sim/particle_contacts.py','assets/particle_contacts.py',
                         'sim/application.py','sim/physics_errors.py','assets/cloth_contact.py','assets/cloth_clearance.py','assets/surface_distance.py']:
            path=Path(__file__).resolve().parents[3]/relative
            destination=engine/Path(relative).name;shutil.copyfile(path,destination)
            inputs[str(destination.relative_to(s.dir))]=digest(destination)
    material_evidence=[]
    if material_contract:
        z,frames,_,_=load(s,obj)
        material_evidence=material_attach(s,obj,z,frames)
        inputs.update(z['_object_domain_inputs'])
    write_json(out / 'input.json', {'model_sha256': spec['sha256'], 'inputs': inputs, 'budget': budget, 'round': k,
                                    'source_raster_contract': source_raster_contract,
                                    'effective_fit_request':getattr(s,'scene_graph',{}).get('dynamic_objects',{}).get(obj,{}).get('fit',{}),
                                    'object_understanding': object_contract,
                                    'source_geometry_contract':s.config('agent/geometry_checks.json',{}).get('objects',{}).get(obj,[]),
                                    'source_interface_contract':s.config('agent/interface_checks.json',{}).get('objects',{}).get(obj,[]),
                                    'mechanism_contract': mechanism,'source_part_contract':part_contract,
                                    'dependencies': {**spec.get('dependencies', {}),
                                                     **{str(path): digest(path) for path in assets}},
                                    'source_assets': {str(path): str(relative) for path, relative in assets.items()},
                                    'supplemental_evidence': supplemental_evidence(s),
                                    'semantic_query_policy': semantic_policy(object_request(s, obj)),
                                    'observation_prompt': object_request(s, obj).get('prompt'),
                                    'observation_domain': policy(s, obj),
                                    'observation_exclusions':exclusions(s,obj),
                                    'source_material_contract':material_contract,
                                    'initial_state_contract':state_contract,
                                    'source_pose_contract':source_pose_policy(s,obj),
                                    'source_pose_seed':pose_seed,
                                    'source_material_evidence':material_evidence,
                                    'mask_component_policy': component_policy(s, obj)})
    event(s, obj, 'started', round=k, model_sha256=spec['sha256'], inputs=inputs, budget=budget)
    review = s / 'qa/review'
    review.mkdir(parents=True, exist_ok=True)
    (review / f'OBJ.{obj}.ran').touch()
    return out, k
