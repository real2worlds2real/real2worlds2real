"""Offline source-shape proposals that preserve authored contact regions.

This constrains modeling degrees of freedom, not simulation. It never creates
a joint, a drive, pinned particles or an acceptance verdict. The exact support
solid is used, including its cavities. Fresh source, solid-contact, semantic
and native checks remain necessary after every proposal.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import trimesh


def _array_hash(array):
    array = np.ascontiguousarray(array)
    header = json.dumps([array.dtype.str, array.shape]).encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def _file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def _mesh(vertices, faces):
    vertices, faces = np.asarray(vertices, dtype=float), np.asarray(faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4
            or not np.isfinite(vertices).all() or faces.ndim != 2
            or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer)
            or len(faces) < 4 or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError('Finite triangle geometry required')
    faces = faces.astype(np.int64)
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    if not mesh.is_volume:
        raise ValueError('Contact-preserving proposals require a closed outward solid')
    return vertices.copy(), faces.copy()


@dataclass(frozen=True)
class ContactRegion:
    source: np.ndarray
    faces: np.ndarray
    weights: np.ndarray
    support_vertices: np.ndarray
    support_faces: np.ndarray
    freeze_m: float
    blend_m: float
    maximum_displacement_m: float
    minimum_normal_dot: float

    def apply(self, target):
        """Blend a modeling target while keeping selected source vertices exact."""
        target = np.asarray(target, float)
        if target.shape != self.source.shape or not np.isfinite(target).all():
            raise ValueError('Proposal must retain the source vertex correspondence')
        result = self.source + self.weights[:, None] * (target - self.source)
        self.audit(result)
        return result

    def audit(self, candidate):
        """Check construction bounds; this is not a contact or stability pass."""
        candidate, faces = _mesh(candidate, self.faces)
        fixed = self.weights == 0
        if not np.array_equal(candidate[fixed], self.source[fixed]):
            raise ValueError('Authored contact-region vertices changed')
        displacement = float(np.linalg.norm(candidate - self.source, axis=1).max())
        if displacement > self.maximum_displacement_m + 1e-12:
            raise ValueError('Source-shape proposal exceeds its modeling budget')
        old = trimesh.Trimesh(self.source, faces, process=False).face_normals
        new = trimesh.Trimesh(candidate, faces, process=False).face_normals
        normal_dot = float(np.einsum('ij,ij->i', old, new).min())
        if normal_dot < self.minimum_normal_dot:
            raise ValueError('Source-shape proposal folds or over-rotates a face')
        return {'construction_constraints_pass': True, 'accepted': False,
                'frozen_vertices': int(fixed.sum()),
                'maximum_displacement_m': displacement,
                'minimum_normal_dot': normal_dot,
                'native_constraints_created': False}

    def save(self, output, *, inputs):
        """Create an exclusive snapshot bound to exact source/support bytes.

        ``inputs`` must include the actual candidate, support and applicable
        source/material/semantic manifests chosen by the caller. Their hashes
        are checked both here and at reload; this module does not interpret or
        weaken those contracts.
        """
        inputs = {str(Path(name).resolve()): expected for name, expected in inputs.items()}
        if not inputs or any(_file_hash(name) != value for name, value in inputs.items()):
            raise ValueError('Unchanged candidate/support/source contract files required')
        output = Path(output)
        output.mkdir()  # Exclusive, never rewrite completed proposals.
        arrays = {name: getattr(self, name) for name in
                  ('source', 'faces', 'weights', 'support_vertices', 'support_faces')}
        np.savez_compressed(output / 'geometry.npz', **arrays)
        manifest = {'version': 1, 'construction_only': True, 'accepted': False,
                    'inputs': inputs,
                    'array_hashes': {name: _array_hash(value) for name, value in arrays.items()},
                    'geometry_sha256': _file_hash(output / 'geometry.npz'),
                    'limits': {name: getattr(self, name) for name in
                               ('freeze_m', 'blend_m', 'maximum_displacement_m', 'minimum_normal_dot')}}
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        return manifest


def prepare(source, faces, support_vertices, support_faces, *, freeze_m,
            blend_m, maximum_displacement_m, minimum_normal_dot=.4, device='cpu'):
    """Use distance to the real closed support material, never a filled hull."""
    from calibrated_scene_reconstruction.assets.surface_distance import SignedSurface
    values = np.asarray([freeze_m, blend_m, maximum_displacement_m, minimum_normal_dot], float)
    if (not np.isfinite(values).all() or freeze_m < 0 or blend_m <= 0
            or maximum_displacement_m <= 0 or not 0 < minimum_normal_dot <= 1):
        raise ValueError('Finite positive construction limits required')
    source, faces = _mesh(source, faces)
    support_vertices, support_faces = _mesh(support_vertices, support_faces)
    distance = SignedSurface(support_vertices, support_faces, device=device)(source)
    if not np.isfinite(distance).all():
        raise ValueError('Finite actual-solid contact distances required')
    t = np.clip((distance - freeze_m) / blend_m, 0, 1)
    weights = t * t * (3 - 2 * t)
    arrays = (source, faces, weights, support_vertices, support_faces)
    for array in arrays:
        array.setflags(write=False)
    return ContactRegion(*arrays, *map(float, values))


def load(output):
    """Reject stale provenance or altered constraint geometry before reuse."""
    output = Path(output)
    manifest = json.loads((output / 'manifest.json').read_text())
    if (manifest.get('version') != 1 or manifest.get('construction_only') is not True
            or manifest.get('accepted') is not False or not manifest.get('inputs')
            or _file_hash(output / 'geometry.npz') != manifest['geometry_sha256']
            or any(_file_hash(name) != expected for name, expected in manifest['inputs'].items())):
        raise ValueError('Construction snapshot or candidate/support/source contract changed')
    arrays = dict(np.load(output / 'geometry.npz', allow_pickle=False))
    if {name: _array_hash(value) for name, value in arrays.items()} != manifest['array_hashes']:
        raise ValueError('Construction constraint geometry changed')
    for array in arrays.values():
        array.setflags(write=False)
    return ContactRegion(**arrays, **manifest['limits'])
