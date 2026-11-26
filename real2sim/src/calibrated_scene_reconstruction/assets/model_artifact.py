"""Reviewed metric mesh artifacts consumed by scene and physics authoring.

This contract carries source and observation fingerprints. It contains no measured
reference sizes and never decides acceptance from an object's physical dimensions.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.assets.metric_mesh import geometry


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def contained(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Artifact path must stay inside its recording')
    return path


def read_mesh(path, expected=None):
    if expected is not None and digest(path) != expected:
        raise ValueError('Model mesh changed after acceptance')
    with np.load(path) as data:
        V, F = data['V'], data['F']
    geometry(V, F)
    return V, F


def require_reviewed(root, source):
    """Read an approved artifact; missing/stale assets stop before fallback execution."""
    root, source = Path(root), Path(source)
    manifest = root / 'objects/object/approved.json'
    if not manifest.exists():
        qa = root / 'qa'
        qa.mkdir(parents=True, exist_ok=True)
        (qa / 'model_requests.json').write_text(json.dumps({
            'route': 'agent_blender', 'object': 'object', 'model': str(source),
            'next': 'Write model.py, fit calibrated silhouettes, inspect evidence, revise and record a passing review.',
        }, indent=2) + '\n')
        raise ValueError('Agent Blender iteration required; see qa/model_requests.json. No fallback was run.')
    proof = json.loads(manifest.read_text())
    if proof.get('route') != 'agent_blender' or digest(source) != proof.get('model_sha256'):
        raise ValueError('Approved model source changed; run a new Blender round')
    for relative, expected in proof['inputs'].items():
        if digest(contained(root, relative)) != expected:
            raise ValueError('Observation changed after acceptance: ' + relative)
    fit_path = contained(root, proof['fit'])
    if digest(fit_path) != proof['fit_sha256']:
        raise ValueError('Fit evidence changed after acceptance')
    fit = json.loads(fit_path.read_text())
    if (fit.get('model_sha256') != proof['model_sha256']
            or fit.get('mesh_sha256') != proof['mesh_sha256'] or fit.get('round') != proof['round']):
        raise ValueError('Approved artifact does not match its fitted round')
    metrics = fit['metrics']
    if not (fit.get('ok') is True and metrics['iou_median'] >= .85
            and metrics['local_difference'] <= .02 and len(metrics['frames']) >= 8):
        raise ValueError('Model round failed the unchanged image-evidence thresholds')
    review_path = contained(root, proof['review'])
    review = json.loads(review_path.read_text())
    if (review.get('pass') is not True or review.get('model_sha256') != proof['model_sha256']
            or review.get('round') != fit['round'] or not isinstance(review.get('findings'), list)
            or any(f.get('status') != 'resolved' for f in review['findings'])
            or review_path.stat().st_mtime_ns <= fit_path.stat().st_mtime_ns):
        raise ValueError('Current passing object review required')
    V, F = read_mesh(contained(root, proof['mesh']), proof['mesh_sha256'])
    return proof, V, F
