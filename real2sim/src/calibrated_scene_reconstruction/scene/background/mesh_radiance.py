"""Calibrate source-baked native background appearance to measured display RGB.

Captured lighting remains captured lighting; these are not recovered albedos.
The geometry and semantic part identities remain those of the agent mesh.
"""
import argparse
import copy
import json
from pathlib import Path

from calibrated_scene_reconstruction.scene.review.environment import digest


def calibrate(manifest, response_path, output):
    from calibrated_scene_reconstruction.scene.background.photo_assets import convert
    from calibrated_scene_reconstruction.scene.background import photo_assets
    source, response_path, out = map(lambda p: Path(p).resolve(), [manifest, response_path, output])
    if out.exists():
        raise ValueError('New calibrated atlas output required')
    original = json.loads(source.read_text()); response = json.loads(response_path.read_text())
    inputs = dict(original['input_sha256'])
    for key in ['mesh', 'texture', 'radiance']:
        if digest(original[key]) != original[key+'_sha256']:
            raise ValueError('Raw atlas payload changed')
        inputs[str(Path(original[key]).resolve())] = original[key+'_sha256']
    if any(digest(p) != sha for p, sha in inputs.items()):
        raise ValueError('Raw source atlas evidence changed')
    out.mkdir(parents=True)
    # A later native build legitimately replaces its current pointer. Archive
    # the exact old bytes before adoption, preserving historical source evidence.
    snapshots = {}
    for path, sha in list(inputs.items()):
        if Path(path).name == 'current.json' and '/scene/environment/' in path:
            snapshot = out/'source_native_pointer.json'; snapshot.write_bytes(Path(path).read_bytes())
            snapshots[path] = {'sha256': sha, 'snapshot': str(snapshot)}
            del inputs[path]; inputs[str(snapshot)] = digest(snapshot)
    converted = convert(['appearance'], {'appearance': original['texture']}, response, response_path, out/'colour')['appearance']
    result = copy.deepcopy(original)
    observed = source.parent/'observed.png'
    if observed.is_file():
        result.update(observed_mask=str(observed), observed_mask_sha256=digest(observed))
        inputs[str(observed)] = digest(observed)
    result.update(radiance=converted['radiance'], radiance_sha256=converted['radiance_sha256'],
                  display_response=response, source_native_snapshots=snapshots,
                  radiance_calibration={'raw_manifest': str(source), 'response': str(response_path),
                       'legacy_preview_gain': original.get('emission_gain'), 'mode': 'measured_display_inversion'},
                  photorealism_approved=False, scope=__doc__.strip())
    for path in [source, response_path, Path(__file__).resolve(), Path(photo_assets.__file__).resolve(), Path(converted['radiance'])]:
        inputs[str(path)] = digest(path)
    result['input_sha256'] = inputs
    (out/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def run(session, config, geometry_job, output):
    from calibrated_scene_reconstruction.scene.background.blender_env import verify
    from calibrated_scene_reconstruction.scene.background.display_response import validate_profile
    from calibrated_scene_reconstruction.scene.background.mesh_textures import run as bake
    out = Path(output).resolve()
    if out.exists() or not out.is_relative_to(session.dir.resolve()) or (session/'qa/independent_freeze.json').exists():
        raise ValueError('New unfrozen session-local calibrated source atlas required')
    built = verify(session)
    response_path = Path(built['display_response']['asset'])
    if digest(response_path) != built['display_response']['sha256']:
        raise ValueError('Measured native display response changed')
    validate_profile(json.loads(response_path.read_text()), built['render'], built.get('splat_context'))
    out.mkdir(parents=True)
    bake(session, config, geometry_job, out/'raw')
    return calibrate(out/'raw/manifest.json', response_path, out/'calibrated')


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('session'); p.add_argument('config'); p.add_argument('geometry_job'); p.add_argument('--output', required=True)
    args = p.parse_args(); result = run(Session(args.session), args.config, args.geometry_job, args.output)
    print('Calibrated native source atlas:', result['entity'], 'observed fraction', result['observed_fraction'])


if __name__ == '__main__':
    main()
