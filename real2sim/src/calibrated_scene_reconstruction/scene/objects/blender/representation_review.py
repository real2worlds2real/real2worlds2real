"""Honor source-grounded model-family rejection before more parameter fitting.

This verifies provenance, not semantics. Eligible models still need every
normal source, semantic, material, contact and task check.
"""
import hashlib
import json
from pathlib import Path

from calibrated_scene_reconstruction.core.session_code import source_path


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(session, obj, model):
    declared = session.scene_graph['dynamic_objects'][obj].get('fit', {}).get('representation_review')
    if declared is None:
        return None
    if not isinstance(declared, str) or not declared:
        raise ValueError('Representation review must name a config-relative file')
    path = source_path(session.config_dir, declared)
    if not path.resolve().is_relative_to(session.config_dir.resolve()):
        raise ValueError('Representation review must remain in the session config')
    review = json.loads(path.read_text())
    if (type(review.get('version')) is not int or review['version'] != 1 or review.get('object') != obj
            or review.get('decision') not in {'refine_parameters', 'replace_representation'}
            or not isinstance(review.get('basis'), str) or not review['basis'].strip()):
        raise ValueError('Explicit source-based representation decision required')
    if review.get('model_sha256') != _digest(model):
        raise ValueError('Representation review is stale; inspect the changed model before fitting')
    for kind in ['source_evidence', 'candidate_evidence']:
        items = review.get(kind)
        if not isinstance(items, dict) or not items:
            raise ValueError('Representation review needs source and actual candidate evidence')
        for name, expected in items.items():
            if not isinstance(name, str) or not isinstance(expected, str):
                raise ValueError('Evidence paths and SHA256 digests must be strings')
            evidence = (session.dir / name).resolve()
            if (not evidence.is_relative_to(session.dir.resolve()) or not evidence.is_file()
                    or _digest(evidence) != expected):
                raise ValueError('Representation review evidence changed: ' + str(name))
    if review['decision'] == 'replace_representation':
        raise ValueError('Replace the rejected representation before parameter fitting: ' + review['basis'])
    return {'path': str(path), 'sha256': _digest(path), 'engine_sha256': _digest(__file__),
            'accepted': False, 'scope': 'Model-family eligibility only; all other gates remain required'}
