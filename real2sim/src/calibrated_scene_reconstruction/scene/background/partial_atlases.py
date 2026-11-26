"""Validate calibrated partial background photos for unchanged native meshes.

Unknown pixels retain original PBR. Geometry, semantic part names and the GS
field stay intact; source/free-view appearance review is still mandatory.
"""
import json
from pathlib import Path

from calibrated_scene_reconstruction.scene.review.environment import digest


def prepare(session, entity, bindings, spec, assets, protected, context):
    from calibrated_scene_reconstruction.scene.background.display_response import validate_profile
    from calibrated_scene_reconstruction.scene.background.observed_context import parameter_digest
    keys = {'mesh', 'radiance', 'evidence', 'observed_mask', 'attachment'}
    if (set(bindings) != keys or bindings['attachment'] != 'usd_partial'
            or any(bindings[k] not in assets for k in keys-{'attachment'})):
        raise ValueError('Partial atlas requires explicit mesh/radiance/evidence/observed_mask asset IDs')
    definitions = {e['id']:e for e in spec['parameters']['entities']}
    reserved = {protected['task']['support'], *protected['objects']}
    if entity not in definitions or entity in reserved:
        raise ValueError('Partial source appearance is only for an existing background owner')
    paths = {k:Path(assets[bindings[k]]).resolve() for k in keys-{'attachment'}}
    proof = json.loads(paths['evidence'].read_text())
    camera = session/'parsing/posed_frames/cameras.npz'
    if (proof.get('entity') != entity
            or proof.get('entity_parameters_sha256') != parameter_digest(definitions[entity])
            or proof.get('protected_task_state') != protected
            or proof.get('input_sha256',{}).get(str(camera.resolve())) != digest(camera)):
        raise ValueError('Partial source atlas must bind this exact owner, geometry and capture')
    parts = proof.get('source_parts')
    if not parts or any(p.get('entity') != entity or not p.get('path','').startswith('/World/Environment/') for p in parts):
        raise ValueError('Partial source atlas contains foreign mesh parts')
    fraction = proof.get('observed_fraction')
    if type(fraction) not in (int,float) or not 0 < fraction <= 1:
        raise ValueError('Partial source atlas must contain observed surface pixels')
    if proof.get('radiance_calibration',{}).get('mode') != 'measured_display_inversion':
        raise ValueError('Partial source appearance needs measured display calibration')
    validate_profile(proof['display_response'], spec['render'], context)
    inputs = {**proof['input_sha256'], str(Path(__file__).resolve()):digest(__file__)}
    for key in ['mesh', 'radiance', 'observed_mask']:
        if digest(paths[key]) != proof.get(key+'_sha256'):
            raise ValueError('Partial source atlas payload changed: '+key)
    for path in paths.values():inputs[str(path)] = digest(path)
    if any(digest(p) != sha for p,sha in inputs.items()):
        raise ValueError('Partial source appearance evidence changed')
    # Use declared immutable build assets, including the observed-pixel mask.
    proof = {**proof, 'observed_mask':str(paths['observed_mask'])}
    return {'proof':proof, 'mesh':str(paths['mesh']), 'radiance':str(paths['radiance'])}, inputs


def apply(asset, atlases, output):
    from pxr import Usd
    from calibrated_scene_reconstruction.assets import source_atlas
    if not atlases:return {}, {}
    stage = Usd.Stage.Open(str(asset)); audits = {}
    for entity,row in atlases.items():
        audits[entity] = source_atlas.attach(stage,row['proof'],row['mesh'],row['radiance'],
            path_prefixes=('/World/Environment','/Environment'),pixel_blend=True,
            output=Path(output)/entity)
    stage.GetRootLayer().Save()
    return audits, {str(Path(source_atlas.__file__).resolve()):digest(source_atlas.__file__)}
