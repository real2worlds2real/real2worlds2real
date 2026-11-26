"""Preserve an explicitly supplied, immutable Blender atlas in a native look.

This is an alternative to resampling source photographs, not an appearance
approval. A caller must still review the actual native material and lighting.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def face_uv(mesh):
    """Return UVs in the exact physical-part order consumed by metric_mesh."""
    faces = np.asarray(mesh['F'])
    uv = np.asarray(mesh['UV'])
    if (faces.ndim != 2 or faces.shape[1] != 3 or uv.shape != (*faces.shape, 2)
            or not np.isfinite(uv).all()):
        raise ValueError('Authored native texture needs finite per-face corner UVs')
    parts = json.loads(str(mesh.get('links_json', '[]')))
    if not parts:
        return uv
    indices = []
    for part in parts:
        start, count = part['face_start'], part['triangle_count']
        if type(start) is not int or type(count) is not int or start < 0 or count <= 0:
            raise ValueError('Physical texture parts need valid face ranges')
        indices.extend(range(start, start + count))
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) != len(faces) or not np.array_equal(np.sort(indices), np.arange(len(faces))):
        raise ValueError('Physical texture parts must cover every source face exactly once')
    return uv[indices]


def create(round_dir, output, *, atlas='assets/albedo.png'):
    """Write a fresh look for one explicitly declared shared atlas.

    Returns absolute file paths; callers may make them session-relative when
    inserting the look into a scene graph. Geometry and atlas bytes are never
    changed, and existing output folders are never overwritten.
    """
    root = Path(round_dir).resolve()
    output = Path(output).resolve()
    inputs = json.loads((root/'input.json').read_text())
    fit = json.loads((root/'fit.json').read_text())
    if (digest(root/'input.json') != fit.get('input_sha256')
            or digest(root/'mesh.npz') != fit.get('mesh_sha256')
            or digest(root/'model.py') != fit.get('model_sha256')):
        raise ValueError('Authored look requires an unchanged formal round')
    source = (root/atlas).resolve()
    sources = [name for name, relative in inputs.get('source_assets', {}).items()
               if relative == atlas]
    if not source.is_relative_to(root/'assets') or len(sources) != 1:
        raise ValueError('Atlas must be one explicitly declared immutable model asset')
    expected = inputs.get('dependencies', {}).get(sources[0])
    if not source.is_file() or digest(source) != expected:
        raise ValueError('Authored atlas changed since the formal model snapshot')
    with Image.open(source) as image:
        image.verify()
    with np.load(root/'mesh.npz') as raw:
        mesh = dict(raw)
    uv = face_uv(mesh)
    output.mkdir(parents=True, exist_ok=False)
    destination = output/'mesh.npz'
    np.savez_compressed(destination, face_uv=uv)
    look = {'mesh': str(destination), 'albedo': str(source), 'gain': 1.,
            'preserves_links': bool(json.loads(str(mesh.get('links_json', '[]')))),
            'preserves_surface': True, 'source_mesh_sha256': fit['mesh_sha256'],
            'appearance_source': 'Explicit immutable producer atlas; native appearance review required'}
    report = {'accepted': False, 'look': look,
              'source_sha256': {str(p): digest(p) for p in
                               [root/'input.json', root/'fit.json', root/'model.py',
                                root/'mesh.npz', source]},
              'look_mesh_sha256': digest(destination),
              'producer_sha256': digest(__file__),
              'uv_order': 'Physical links in declared order, then source faces within each link',
              'scope': 'Exact atlas transfer only; no geometric, physical or appearance approval'}
    (output/'authored_look.json').write_text(json.dumps(report, indent=2)+'\n')
    return look
