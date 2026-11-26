"""Bind an exact same-capture calibration copy to its destination evidence.

This never transfers task models, reviews or a scale estimate between captures.
Every source map, image, mask, calibration setting and derived camera must match
byte for byte. New geometry, mechanisms and delivery still need fresh checks.
"""
from copy import deepcopy
import json

from calibrated_scene_reconstruction.core.paths import resolve
from .robot_visual import digest, verify_anchor


def bind(source, destination):
    name = 'splat/robot_visual_fit.json'
    if source.dir.resolve() == destination.dir.resolve():
        raise ValueError('Expected distinct same-capture experiment directories')
    if (destination/'qa/independent_freeze.json').exists():
        raise ValueError('Frozen destination calibration')
    original = verify_anchor(source, source.scene_graph)
    if digest(source/name) != digest(destination/name):
        raise ValueError('Destination must hold the unchanged source anchor before rebinding')
    output_names = ['parsing/posed_frames/cameras.npz', 'splat/align.json',
                    'parsing/robot_visual_state.json']
    outputs = {}
    for path in output_names:
        outputs[path] = digest(source/path)
        if digest(destination/path) != outputs[path]:
            raise ValueError('Destination is a different calibrated capture: '+path)

    changes = []
    def rebind(paths):
        mapped = {}
        for name, expected in paths.items():
            old = resolve(name).resolve()
            new = old
            for before, after in [(source.dir, destination.dir),
                                  (source.config_dir, destination.config_dir)]:
                if old.is_relative_to(before.resolve()):
                    new = after.resolve()/old.relative_to(before.resolve())
                    break
            if digest(new) != expected:
                raise ValueError('Same-capture calibration copy differs: '+str(new))
            mapped[str(new)] = expected
            if new != old:
                changes.append({'source': str(old), 'destination': str(new), 'sha256': expected})
        return mapped

    result = deepcopy(original)
    result['input_sha256'] = rebind(original['input_sha256'])
    for section in ['observation_evidence', 'fixed_base_evidence']:
        if 'mask_inputs' in original.get(section, {}):
            result[section]['mask_inputs'] = rebind(original[section]['mask_inputs'])
    result['derived_outputs_sha256'] = outputs
    result['copied_anchor'] = {'source_session': str(source.dir.resolve()),
                              'source_anchor_sha256': digest(source/name),
                              'scope': 'Exact map, image, mask and derived-camera copy; no task or delivery reviews transferred'}
    # Validate the destination graph/state before publishing any changed anchor.
    alignment = json.loads((destination/'splat/align.json').read_text())
    photo = destination.scene_graph.get('photo_robot', {})
    if (alignment['sim3'] != result['applied_sim3']
            or digest(destination/'parsing/robot_visual_state.json') != result['visual_state_sha256']
            or photo.get('base') != result['applied_base'] or photo.get('spec') != result['robot']):
        raise ValueError('Destination alignment, joint state or robot configuration differs')
    root = destination/'qa/calibration_copy'
    number = 1
    while (root/f'round_{number}').exists():
        number += 1
    out = root/f'round_{number}'
    out.mkdir(parents=True)
    (out/'source_anchor.json').write_bytes((source/name).read_bytes())
    (out/'binding.json').write_text(json.dumps({'copies': changes, 'derived_outputs_sha256': outputs,
                                               'copied_anchor': result['copied_anchor']}, indent=2)+'\n')
    (destination/name).write_text(json.dumps(result, indent=2)+'\n')
    verify_anchor(destination, destination.scene_graph)
    return out
