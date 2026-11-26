"""Probe layered native cutouts against the same measured, retained GS field.

Probe an existing gap owner's layer handling or optional Gaussian-centre depth
hypothesis. Source identity, actual opaque support, independent rays and bounded
spatial support remain required. No field rows, geometry or active configuration
change; the original appearance acceptance criteria remain unchanged.
"""
import inspect
import json
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.scene.review.environment import digest


def declarations(context, owners, layered=True, without_centre_depth=False):
    if not isinstance(owners, list) or not owners or len(set(owners)) != len(owners):
        raise ValueError('Distinct existing gap owners required')
    if not layered and not without_centre_depth:
        raise ValueError('An explicit changed layer or centre-depth hypothesis required')
    result = {}
    for owner in owners:
        row = context['appearance_ownership'].get(owner, {})
        original = row.get('mesh_coverage', {})
        if (row.get('role') != 'gap_fill' or original.get('compositing') != 'captured_projection'
                or owner not in context.get('gap_masks', {})
                or original.get('ray_mode', 'self_visible') != 'self_visible'):
            raise ValueError('Probe only an existing captured-projection gap owner in self-visible mode')
        if without_centre_depth and original.get('depth_band_m') is None:
            raise ValueError('Centre-depth probe requires an existing explicit depth hypothesis')
        result[owner] = dict(original)
        if layered:
            result[owner]['ray_mode'] = 'opaque_layers'
        if without_centre_depth:
            result[owner].pop('depth_band_m')
    return result


def prepare(session, owners, coverage, output, apply_mask, layered=True, without_centre_depth=False):
    from plyfile import PlyData
    from pxr import Usd
    from calibrated_scene_reconstruction.core.paths import resolve
    from calibrated_scene_reconstruction.scene.background import capture_gaps
    from calibrated_scene_reconstruction.scene.background.blender_env import verify
    from calibrated_scene_reconstruction.scene.background.environment_physics import current
    from calibrated_scene_reconstruction.scene.background.context_rebind import geometry_signature
    from calibrated_scene_reconstruction.scene.review import ablation, frozen_scope
    from calibrated_scene_reconstruction.scene.review.frozen_scope_cache import same_geometry

    out = Path(output).resolve(); coverage = Path(coverage).resolve()
    if out.exists() or not out.is_relative_to(session.dir.resolve()) or (session/'qa/independent_freeze.json').exists():
        raise ValueError('Use a new unfrozen session-local layer preview')
    built = verify(session); audit, _ = current(session)
    audit_pointer = session/'qa/environment_physics/twin/current.json'
    audit_path = Path(json.loads(audit_pointer.read_text())['path'])
    job_path = audit_path.parent/'job.json'
    job = json.loads(job_path.read_text())
    context_path = resolve(session.config('environment.json')['assets']['observed_context_evidence']).resolve()
    if digest(context_path) != built['splat_context']['evidence_sha256']:
        raise ValueError('Preview must use the actual current context evidence')
    context = json.loads(context_path.read_text()); requests = declarations(context, owners, layered, without_centre_depth)
    if context['input_sha256'].get(str(coverage)) != digest(coverage):
        raise ValueError('Use measured coverage already bound by the current context')
    protected = {session.scene_graph['task']['support'], *session.scene_graph.get('dynamic_objects', {})}
    if protected & set(owners):
        raise ValueError('Cannot change frozen task/support materials')
    if context.get('selection_frozen', {}).get('geometry_signature') != geometry_signature(job):
        raise ValueError('Preview needs the exact geometry bound to the retained field')
    field = PlyData.read(str(context_path.parent/'context.ply'))['vertex'].data
    points = (np.c_[field['x'], field['y'], field['z'], np.ones(len(field))]@np.asarray(context['matrix']).T)[:, :3]
    pointer = session/'scene/environment/twin/current.json'
    inputs = {str(p.resolve()): digest(p) for p in [context_path, coverage, Path(audit['usd']), pointer, audit_pointer, audit_path, job_path,
              Path(__file__), Path(ablation.__file__), Path(inspect.getfile(apply_mask))]}
    out.mkdir(parents=True)
    stage = ablation.copy_stage(audit['usd'], out/'scene.usda', 'Unapproved layered native cutout probe; field and physical geometry unchanged')
    changes = {}
    for owner, declaration in requests.items():
        directory = out/owner; directory.mkdir()
        masks, evidence = capture_gaps.bake(session, job, owner, declaration, field, points, coverage, directory)
        for path, payload in masks['meshes'].items():
            prim = stage.GetPrimAtPath(path)
            if not prim or not apply_mask(stage, prim, payload):
                raise ValueError('Layer probe must apply every actual same-owner mesh mask')
        inputs.update(evidence)
        changes[owner] = {'declaration': declaration, 'masks': masks}
    original = Usd.Stage.Open(audit['usd'])
    if not same_geometry(frozen_scope.meshes(original, session.scene_graph['task']['support'], session.scene_graph.get('dynamic_objects', {})),
                         frozen_scope.meshes(stage, session.scene_graph['task']['support'], session.scene_graph.get('dynamic_objects', {}))):
        raise ValueError('Layer preview changed frozen foreground')
    stage.GetRootLayer().Save()
    if any(digest(p) != sha for p, sha in inputs.items()):
        raise ValueError('Layer preview evidence changed')
    result = {'diagnostic_only': True, 'approved': False, 'scene': str(out/'scene.usda'),
              'source_usd': audit['usd'], 'changes': changes, 'input_sha256': inputs,
              'changed_hypotheses': {'layered': layered, 'without_centre_depth': without_centre_depth},
              'field_rows_unchanged': True, 'frozen_geometry_unchanged': True,
              'scope': __doc__.strip()}
    (out/'preview.json').write_text(json.dumps(result, indent=2)+'\n')
    return result
