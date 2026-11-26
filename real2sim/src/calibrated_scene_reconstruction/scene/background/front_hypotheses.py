"""Validate source-backed direction hypotheses before a new background fit."""
from pathlib import Path
import numpy as np

from calibrated_scene_reconstruction.scene.review.environment import digest


def validate(request, entities, parts, names, directory):
    checks = request.get('checks')
    if not isinstance(request.get('reason'), str) or not request['reason'].strip() or not isinstance(checks, list) or not checks:
        raise ValueError('Source direction hypothesis needs explicit checks and reasoning')
    evidence = request.get('source_evidence', {})
    inputs = {}
    for check in checks:
        owner = check.get('entity')
        if owner not in entities:
            raise ValueError('Direction hypothesis must stay within selected background owners')
        available = {p['part'] for p in parts if p['entity'] == owner and not p['external']}
        for key in ('front_parts', 'reference_parts'):
            value = check.get(key)
            if not isinstance(value, list) or not value or len(set(value)) != len(value) or not set(value) <= available:
                raise ValueError('Direction hypothesis needs exact actual front/reference parts')
        axis = np.asarray(check.get('local_front', []), float)
        if axis.shape != (3,) or not np.isfinite(axis).all() or not np.isclose(np.linalg.norm(axis), 1.):
            raise ValueError('Source-facing local axis must be normalized')
        frames = check.get('frames')
        if not isinstance(frames, list) or len(set(frames)) < 2:
            raise ValueError('Two distinct source-facing observations required')
        for frame in frames:
            if frame not in names or Path(frame).name != frame:
                raise ValueError('Use registered source frame basenames')
            source = (Path(directory)/frame).resolve()
            if evidence.get(str(source)) != digest(source):
                raise ValueError('Direction hypothesis source pixels changed or unbound')
            inputs[str(source)] = digest(source)
    return checks, inputs
