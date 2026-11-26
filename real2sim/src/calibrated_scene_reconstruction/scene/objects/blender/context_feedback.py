"""Source checks for a support candidate and its separately modeled contents.

A container can pass its own masked silhouette while its inner lip hides real
contents. Evaluate both with the actual candidate triangles and unchanged source
observations. This returns proposal evidence, never semantic/native approval.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from . import evidence
from .source_pose_seed import evidence_inputs
from .store import digest


def score(session, parent, candidate, children):
    """Score a parent and nonempty ``name: {mesh, pose}`` child candidates.

    Mesh paths must be actual current-session/config files. Every declared
    child must already use this parent as its opaque source context; arbitrary
    image exclusions or new relationships cannot be introduced here.
    """
    if not children or parent in children:
        raise ValueError('A parent needs distinct contained-object candidates')
    inputs = {str(path.resolve()): digest(path) for path in
              [Path(__file__), Path(evidence.__file__),
               Path(__file__).with_name('context_visibility.py'),
               Path(__file__).with_name('source_pose_seed.py')]}
    poses = {}; observations = {}; meshes = {}; metrics = {}
    for name, spec in {parent: candidate, **children}.items():
        path = Path(spec['mesh']).resolve()
        if not path.is_file() or not any(path.is_relative_to(Path(root).resolve())
                                        for root in (session.dir, session.config_dir)):
            raise ValueError('Context feedback requires actual current-session candidate files')
        inputs[str(path)] = digest(path)
        with np.load(path, allow_pickle=False) as raw:
            mesh = {key: raw[key] for key in raw.files}
        if name == parent:
            materials = json.loads(str(mesh.get('materials_json', '{}')))
            if not materials or any(row.get('transmission', 0) > .01 or row.get('opacity', 1) < .99
                                    for row in materials.values()):
                raise ValueError('Opaque source feedback requires declared opaque parent materials')
        obs = evidence.load(session, name, parts=True)
        inputs.update(evidence_inputs(session, name, observations=obs))
        # Proposal search may have an optional smooth boundary objective. Keep
        # the final original source metrics/objective here, with every view.
        obs = (dict(obs[0]), *obs[1:]); obs[0].pop('_boundary_search', None)
        observations[name] = obs; meshes[name] = mesh
        poses[name] = json.loads(json.dumps(spec['pose'], allow_nan=False))
    parent_mesh = evidence.placed(meshes[parent]['V'], meshes[parent]['F'], poses[parent])
    for name in [parent, *children]:
        mesh = meshes[name]; obs = observations[name]
        if name != parent:
            old = obs[0].get('_source_context')
            if old is None or old.get('object') != parent:
                raise ValueError('Child source observations do not declare this parent context')
            obs[0]['_source_context'] = {**old, 'mesh': parent_mesh,
                'inputs': {str(Path(candidate['mesh']).resolve()): inputs[str(Path(candidate['mesh']).resolve())]},
                'round': 'explicit_joint_source_proposal', 'candidate_only': True,
                'parent_source_geometry_pass': metrics[parent]['ok']}
        parts = json.loads(str(mesh['links_json'])) if 'links_json' in mesh else None
        metrics[name] = evidence.score(mesh['V'], mesh['F'], poses[name], obs, model_parts=parts)[0]
    unchanged = all(Path(path).is_file() and digest(path) == value for path, value in inputs.items())
    if not unchanged:
        raise ValueError('Joint source inputs changed during scoring')
    return {'accepted': False, 'native_tested': False,
            'numeric_source_pass': all(row['ok'] for row in metrics.values()),
            'source_objective': sum(evidence.search_objective(row) for row in metrics.values()),
            'objects': metrics, 'candidate_poses': poses,
            'candidate_pose_sha256': hashlib.sha256(json.dumps(poses, sort_keys=True).encode()).hexdigest(),
            'inputs': inputs, 'input_unchanged': unchanged,
            'scope': 'Actual parent depth occlusion and all original per-object source gates. '
                     'No labels/frames changed; valid solids, semantics, materials and fresh native checks remain required.'}
