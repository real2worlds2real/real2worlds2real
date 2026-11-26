"""Validated metric object surfaces shared by scene export and simulation authoring."""
import json
from numbers import Real

import numpy as np
import trimesh

from calibrated_scene_reconstruction.core.paths import resolve

DEFAULT_SDF = {'approximation': 'sdf', 'resolution': 256, 'contact_offset': .002, 'rest_offset': 0.}


def sdf_settings(overrides=None):
    """Validate declared collider precision without changing the object surface."""
    if overrides is not None and (not isinstance(overrides, dict) or set(overrides)-set(DEFAULT_SDF)):
        raise ValueError('SDF settings must use approximation, resolution, contact_offset, rest_offset only')
    settings = {**DEFAULT_SDF, **(overrides or {})}
    resolution = settings['resolution']
    if settings['approximation'] != 'sdf' or type(resolution) is not int or not 64 <= resolution <= 2048:
        raise ValueError('Surface SDF collision requires integer resolution64..2048')
    contact, rest = settings['contact_offset'], settings['rest_offset']
    if (any(isinstance(v, bool) or not isinstance(v, Real) for v in (contact, rest))
            or not np.isfinite([contact, rest]).all()
            or not 0 <= rest < contact <= .1):
        raise ValueError('SDF offsets require finite0 <= rest_offset < contact_offset <= .1m')
    return settings


def geometry(V, F, *, max_extent_m=2., require_bottom_origin=True, min_extent_m=.005):
    V, F = np.asarray(V, float), np.asarray(F)
    if V.ndim != 2 or V.shape[1:] != (3,) or len(V) < 4 or not np.isfinite(V).all():
        raise ValueError('nonempty finite (N,3) vertices required')
    if F.ndim != 2 or F.shape[1:] != (3,) or not len(F) or F.dtype.kind not in 'iu' or F.min() < 0 or F.max() >= len(V):
        raise ValueError('valid nonempty triangle indices required')
    m = trimesh.Trimesh(V, F, process=True)
    extent = np.ptp(V, axis=0)
    if not np.isfinite(max_extent_m) or max_extent_m < .005:
        raise ValueError('A finite positive metric extent bound is required')
    if not np.isfinite(min_extent_m) or min_extent_m <= 0:
        raise ValueError('A positive minimum extent is required')
    if extent.min() < min_extent_m or extent.max() > max_extent_m:
        raise ValueError(f'object extent outside 0.005..{max_extent_m:g} m: {extent.tolist()}')
    if not m.is_watertight or not m.is_winding_consistent or m.volume <= 0:
        raise ValueError('mesh must be watertight with consistent outward winding and positive solid volume')
    centre = (V.min(0) + V.max(0)) / 2
    if require_bottom_origin and (abs(V[:, 2].min()) > 1e-5 or np.max(np.abs(centre[:2])) > 1e-5):
        raise ValueError('origin must be the bottom bounding-box centre, z up, in metres')
    return {'watertight': True, 'winding_consistent': True, 'volume_m3': float(m.volume),
            'extent_m': extent.tolist(), 'vertices': len(V), 'triangles': len(F)}


def asset(fix):
    with np.load(resolve(fix['mesh'])) as d:
        V, F = d['V'], d['F']
        links = json.loads(str(d['links_json'])) if 'links_json' in d else []
        joints = json.loads(str(d['joints_json'])) if 'joints_json' in d else []
        materials = json.loads(str(d['materials_json'])) if 'materials_json' in d else {}
        material_names = d['material_names'].tolist() if 'material_names' in d else []
        face_material = d['face_material'] if 'face_material' in d else None
        normals = d['normals'] if 'normals' in d else None
        cloth_rest = json.loads(str(d['cloth_rest_json'])) if 'cloth_rest_json' in d else {}
        volume_initial = json.loads(str(d['volume_initial_json'])) if 'volume_initial_json' in d else {}
        bindings = json.loads(str(d['surface_bindings_json'])) if 'surface_bindings_json' in d else []
    if normals is not None and (normals.shape!=(len(F),3,3) or not np.isfinite(normals).all()
            or not np.allclose(np.linalg.norm(normals,axis=2),1.,atol=1e-4)):
        raise ValueError('Authored corner normals must preserve finite unit vectors for every triangle')
    asset_geometry(V, F, links)
    material = {'color': fix.get('color', [0.75, 0.75, 0.75]), 'roughness': 0.5}
    parts = []
    for link in links:
        a, n, b, count = (link[k] for k in ('vertex_start', 'vertex_count', 'face_start', 'triangle_count'))
        vertices, faces = V[a:a+n], F[b:b+count]-a
        link_geometry(vertices, faces, link['kind'])
        rest = cloth_rest.get(link['name'])
        volume = volume_initial.get(link['name'])
        if link.get('backend')=='volume_fem':
            if volume is None:raise ValueError('Native volume FEM requires explicit original reference and initial state')
            from .volume_initial_state import skin_mapping
            skin_mapping(volume,vertices,faces)
        elif volume is not None:raise ValueError('Explicit volume state requires a native volume_fem link')
        if rest is not None:
            from calibrated_scene_reconstruction.assets.cloth_rest import validate as validate_rest
            if link['kind'] != 'cloth':
                raise ValueError('Only a native cloth link may have cloth rest data')
            validate_rest(rest, vertices, faces)
        parts.append({'name': link['name'], 'kind': link['kind'], 'V': vertices, 'F': faces,
                      **({'backend':link.get('backend','particle')} if link['kind']=='cloth' else {}),
                      **({'backend':link['backend']} if link['kind'] in {'deformable','rope'} and 'backend' in link else {}),
                      'UV': None, 'material': material,
                      'normals': normals[b:b+count] if normals is not None and link['kind']=='rigid' else None,
                      'face_material': face_material[b:b+count] if face_material is not None else None,
                      'material_names': material_names, 'materials': materials,
                      **({'cloth_rest': rest} if rest is not None else {}),
                      **({'volume_initial':volume} if volume is not None else {})})
    if set(cloth_rest)-{p['name'] for p in parts}:
        raise ValueError('Cloth rest data refers to an absent physical link')
    if set(volume_initial)-{p['name'] for p in parts}:
        raise ValueError('Volume initial data refers to an absent physical link')
    if not parts:
        parts = [{'name': 'model', 'kind': 'rigid', 'V': V, 'F': F, 'UV': None, 'material': material,
                  'normals': normals,
                  'face_material': face_material, 'material_names': material_names, 'materials': materials}]
    from .surface_bindings import validate as validate_bindings
    validate_bindings(bindings,links,V,F)
    from .surface_bindings import validate_islands
    validate_islands(cloth_rest,bindings,links)
    return {'parts': parts, 'links': links, 'joints': joints,'surface_bindings':bindings,
            'height': float(np.ptp(V[:, 2])), 'half': np.max(np.abs(V[:, :2]), axis=0).tolist()}


def link_geometry(V, F, kind):
    if kind != 'cloth':
        return geometry(V, F, require_bottom_origin=False, min_extent_m=1e-5)
    V, F = np.asarray(V, float), np.asarray(F)
    if (V.ndim != 2 or V.shape[1:] != (3,) or len(V) < 3 or not np.isfinite(V).all()
            or F.ndim != 2 or F.shape[1:] != (3,) or not len(F) or F.dtype.kind not in 'iu'
            or F.min() < 0 or F.max() >= len(V)):
        raise ValueError('Cloth requires a finite triangular simulation surface')
    areas = np.linalg.norm(np.cross(V[F[:, 1]]-V[F[:, 0]], V[F[:, 2]]-V[F[:, 0]]), axis=1)/2
    if areas.min() <= 1e-14:
        raise ValueError('Degenerate cloth triangles')
    return {'area_m2': float(areas.sum()), 'vertices': len(V), 'triangles': len(F), 'physical_kind': kind}


def asset_geometry(V, F, links=None, *, max_extent_m=2.):
    if not links or all(p['kind'] == 'rigid' for p in links):
        return geometry(V, F, max_extent_m=max_extent_m)
    V, F = np.asarray(V), np.asarray(F)
    extent = np.ptp(V, axis=0)
    if extent.max() > max_extent_m or np.sort(extent)[1] < .005:
        raise ValueError('Flexible asset outside metric extent bounds')
    centre = (V.min(0)+V.max(0))/2
    if abs(V[:, 2].min()) > 1e-5 or np.max(abs(centre[:2])) > 1e-5:
        raise ValueError('Flexible asset origin must be bottom bounding-box centre')
    checks = []
    for part in links:
        a, n, b, count = (part[k] for k in ('vertex_start', 'vertex_count', 'face_start', 'triangle_count'))
        checks.append(link_geometry(V[a:a+n], F[b:b+count]-a, part['kind']))
    return {'extent_m': extent.tolist(), 'links': checks, 'vertices': len(V), 'triangles': len(F),
            'volume_m3': sum(p.get('volume_m3', 0.) for p in checks)}
