"""Resolve reviewed Blender support assets without importing the simulator."""
import numpy as np
import trimesh
from calibrated_scene_reconstruction.core.paths import resolve


def models(session):
    from .blender import verify
    return _models(session, lambda name: verify(session, name))


def _models(session, verify_source):
    """Read surfaces after the caller's explicit source validation boundary.

    Ordinary runtime calls complete review above. The separate unaccepted assay
    composition root may supply current candidate integrity checks instead.
    """
    result = {}
    for name, spec in session.scene_graph.get('furniture', {}).items():
        fix = spec.get('fix') or {}
        if not fix.get('model'):
            continue
        verify_source(name)
        data = np.load(resolve(fix['mesh']))
        visual = trimesh.load(resolve(fix['visual_glb']), force='scene', process=False)
        parts = []
        textures = {entry['part']: entry for entry in fix.get('textures', [])}
        for index, node in enumerate(visual.graph.nodes_geometry):
            matrix, key = visual.graph[node]
            mesh = visual.geometry[key].copy()
            mesh.apply_transform(matrix)
            material = getattr(mesh.visual, 'material', None)
            from .materials import validate_texture_coordinates
            validate_texture_coordinates(mesh)
            rgba = getattr(material, 'baseColorFactor', None)
            # glTF's omitted factor is white. Applying a grey fallback to a
            # source image silently darkens every transferred texture.
            color = np.asarray(rgba if rgba is not None else [255, 255, 255, 255], float)[:3]
            if color.max() > 1:
                color /= 255.
            texture = textures.get(index)
            uv = getattr(mesh.visual, 'uv', None)
            if getattr(material, 'baseColorTexture', None) is not None and (texture is None or uv is None):
                raise ValueError('Support GLB image material has no reviewed texture asset/UV')
            from .materials import sampler
            wrap=sampler(resolve(fix['visual_glb']),material.name) if texture else None
            if texture and 'wrap' in texture and texture['wrap']!=wrap:
                raise ValueError('Support texture sampler differs from its reviewed GLB')
            parts.append({'vertices': np.asarray(mesh.vertices), 'faces': np.asarray(mesh.faces),
                          'uv': uv, 'texture': str(resolve(texture['path'])) if texture else None,
                          'wrap':wrap,
                          'color': color.tolist(), 'roughness': float(getattr(material, 'roughnessFactor', .6) or .6),
                          'metallic': float(getattr(material, 'metallicFactor', 0.) or 0.)})
        if not parts:
            raise ValueError('Support GLB has no visual surfaces')
        collision={}
        if fix.get('collision_representation','surface')=='source_convex':
            from .blender import sha
            from calibrated_scene_reconstruction.assets.convex_source_solids import load as load_convex_solids
            solids=fix['source_solids'];path=resolve(solids['path'])
            if sha(path)!=solids['sha256']:raise ValueError('Source support collider solids changed')
            collision['collision_solids']=load_convex_solids(path)
        result[name] = {'vertices': data['V'], 'faces': data['F'], 'parts': parts,**collision,
                        'xy': fix['xy'], 'yaw_deg': fix['yaw_deg'], 'model_sha256': fix['model_sha256']}
    return result
