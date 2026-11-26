"""Preview source appearance on fully observed background faces; retain GS/PBR."""
import argparse
import json
from pathlib import Path

def prepare(session, request, output):
    from pxr import Usd
    from calibrated_scene_reconstruction.assets import source_atlas
    from calibrated_scene_reconstruction.scene.background.blender_env import verify
    from calibrated_scene_reconstruction.scene.background.display_response import validate_profile
    from calibrated_scene_reconstruction.scene.background.environment_physics import current
    from calibrated_scene_reconstruction.scene.background.observed_context import parameter_digest
    from calibrated_scene_reconstruction.scene.review import ablation, frozen_scope
    from calibrated_scene_reconstruction.scene.review.environment import digest
    from calibrated_scene_reconstruction.scene.review.frozen_scope_cache import same_geometry
    from calibrated_scene_reconstruction.sim import context_stage

    out = Path(output).resolve(); request = Path(request).resolve()
    if out.exists() or not out.is_relative_to(session.dir.resolve()) or (session/'qa/independent_freeze.json').exists():
        raise ValueError('New unfrozen session-local appearance preview required')
    built = verify(session); audit, _ = current(session)
    config = session.config('environment.json'); entities = {r['id']: r for r in config['parameters']['entities']}
    protected = {session.scene_graph['task']['support'], *session.scene_graph.get('dynamic_objects', {})}
    owners = json.loads(request.read_text())['owners']
    if not owners or not set(owners) <= set(entities)-protected:
        raise ValueError('Explicit existing background owners required')
    inputs = {str(p): digest(p) for p in [request, Path(audit['usd']), Path(__file__).resolve(), Path(source_atlas.__file__).resolve(), Path(context_stage.__file__).resolve()]}
    out.mkdir(parents=True)
    (out/'environment_input.json').write_bytes((session.config_dir/'environment.json').read_bytes())
    inputs[str(out/'environment_input.json')] = digest(out/'environment_input.json')
    stage = ablation.copy_stage(audit['usd'], out/'scene.usda', 'Unapproved partial source-face appearance; GS rows and every original mesh polygon unchanged')
    changes = {}
    for owner, bindings in owners.items():
        proof_path = Path(bindings['evidence']).resolve(); proof = json.loads(proof_path.read_text())
        if (proof['entity'] != owner or proof['entity_parameters_sha256'] != parameter_digest(entities[owner])
                or any(p['entity'] != owner or not p['path'].startswith('/World/Environment/') for p in proof['source_parts'])):
            raise ValueError('Source appearance must retain the exact current owner and geometry hypothesis')
        for p, sha in proof['input_sha256'].items():
            if digest(p) != sha:raise ValueError('Source atlas evidence changed: '+p)
        validate_profile(proof['display_response'], built['render'], built.get('splat_context'))
        changes[owner] = source_atlas.attach(stage, proof, bindings['mesh'], bindings['radiance'], partial=True,
                                             pixel_blend=True, output=out/owner)
        masks = (built.get('splat_context') or {}).get('gap_masks', {}).get(owner)
        if masks:
            if masks.get('kind') != 'surface_atlas':raise ValueError('Prepared per-part GS masks required')
            for path, payload in masks['meshes'].items():
                context_stage.apply_gap_mask(stage, stage.GetPrimAtPath(path), payload)
        inputs.update(proof['input_sha256'])
        for path in [proof_path, Path(bindings['mesh']), Path(bindings['radiance'])]:inputs[str(path.resolve())] = digest(path)
    original = Usd.Stage.Open(audit['usd']); support = session.scene_graph['task']['support']; objects = session.scene_graph.get('dynamic_objects', {})
    if not same_geometry(frozen_scope.meshes(original, support, objects), frozen_scope.meshes(stage, support, objects)):
        raise ValueError('Appearance preview changed frozen foreground')
    stage.GetRootLayer().Save()
    if any(digest(p) != sha for p, sha in inputs.items()):raise ValueError('Source appearance evidence changed')
    result = {'diagnostic_only': True, 'appearance_pass': False, 'source_usd': audit['usd'], 'scene': str(out/'scene.usda'),
              'changes': changes, 'frozen_geometry_unchanged': True, 'gs_masks_reapplied': True, 'input_sha256': inputs}
    (out/'preview.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('session'); p.add_argument('request'); p.add_argument('--output', required=True)
    args = p.parse_args()
    from calibrated_scene_reconstruction.scene.session import Session
    print(prepare(Session(args.session), args.request, args.output)['scene'])


if __name__ == '__main__':main()
