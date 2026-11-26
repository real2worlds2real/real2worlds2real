"""Choose GS or mesh appearance explicitly; collision proxies are independent.

A coarse collision mesh is not evidence against a captured Gaussian. A gap
filler needs a measured coverage mask; an inferred enclosure must be identified
as such. These rules do not certify object completeness or photorealism.
"""
from pathlib import Path

import numpy as np


ROLES = {'gs', 'native_replacement', 'gap_fill', 'inferred_enclosure'}


def bind_planar_radiance(session, entity, plane, evidence, surface_id):
    """Bind capture-baked radiance only to its exact declared world-space quad."""
    import json
    from calibrated_scene_reconstruction.scene.review.environment import digest,protected_state
    evidence=Path(evidence).resolve();proof=json.loads(evidence.read_text())
    inputs={str(evidence):digest(evidence),**proof['input_sha256']}
    camera=session/'parsing/posed_frames/cameras.npz'
    if proof.get('protected_task_state')!=protected_state(session) or inputs.get(str(camera.resolve()))!=digest(camera):
        raise ValueError('Planar radiance requires the current capture and protected geometry')
    row=next((r for r in proof['surfaces'] if r['id']==surface_id),None)
    origin,axes,extent,*_=plane_specification(plane)
    quad=np.array([origin,origin+axes[0]*extent[0],origin+axes[0]*extent[0]+axes[1]*extent[1],origin+axes[1]*extent[1]])
    if (not row or row.get('entity')!=entity or np.asarray(row.get('quad_world')).shape!=(4,3)
            or not np.allclose(row['quad_world'],quad,atol=1e-6,rtol=0)):
        raise ValueError('Planar radiance must match the exact gap surface and entity')
    if not 0<float(row.get('observed_fraction',0))<=1:
        raise ValueError('Planar radiance needs actual observed pixels; a plain fallback is not capture evidence')
    path=Path(row['emission_texture']).resolve();inputs[str(path)]=row['emission_sha256']
    for p,expected in inputs.items():
        if digest(p)!=expected:raise ValueError('Planar radiance evidence changed: '+p)
    return {'path':str(path),'sha256':row['emission_sha256'],'appearance_mode':row['appearance_mode']},inputs


def specification(value, entities, replacements, fills, enclosure):
    if not isinstance(value, dict) or set(value) != set(entities):
        raise ValueError('Declare appearance ownership for every background entity')
    inferred = {r['entity'] for r in enclosure['surfaces'] if r['provenance'] == 'inferred'}
    identities={k:k for k in value}
    for entity, row in value.items():
        if (not isinstance(row, dict) or row.get('role') not in ROLES
                or not isinstance(row.get('basis'), str) or not row['basis'].strip()):
            raise ValueError('Each appearance owner needs a role and capture/inference basis: '+entity)
        role = row['role']
        aliases=row.get('observation_entities',[])
        if not isinstance(aliases,list) or any(not isinstance(a,str) or not a.strip() for a in aliases) or len(set(aliases))!=len(aliases):
            raise ValueError('Explicit distinct source instance aliases required')
        for alias in aliases:
            if alias in identities and identities[alias]!=entity:raise ValueError('Ambiguous source instance association: '+alias)
            identities[alias]=entity
        if role == 'inferred_enclosure' and entity not in inferred:
            raise ValueError('Unmasked inferred mesh is restricted to declared inferred enclosure surfaces')
        if role == 'gap_fill' and (sum(isinstance(row.get(k),dict) for k in ('plane','source_crop','mesh_coverage'))!=1):
            raise ValueError('Gap-only mesh needs one explicit coverage plane, verified source crop or mesh coverage: '+entity)
    if {k for k, v in value.items() if v['role'] == 'native_replacement'} != set(replacements):
        raise ValueError('GS removal must match explicitly declared native replacements')
    if {k for k, v in value.items() if v['role'] in ('gap_fill', 'inferred_enclosure')} != set(fills):
        raise ValueError('Visible mesh fillers must match their declared appearance owners')
    return value


def plane_specification(value):
    origin = np.asarray(value.get('origin'), float)
    axes = np.asarray(value.get('axes'), float)
    extent = np.asarray(value.get('extent_m'), float)
    cell = value.get('pixel_m', .02)
    band = value.get('band_m', .035)
    radius = value.get('max_radius_m', .06)
    if value.get('compositing','cutout') not in ('cutout','depth_backing','solid_backing'):
        raise ValueError('Use cutout coverage or a capture-textured depth backing')
    parts=value.get('mesh_parts')
    if parts is not None and (not isinstance(parts,list) or not parts or any(not isinstance(p,str) or not p.strip() for p in parts)
                              or len(set(parts))!=len(parts)):
        raise ValueError('Distinct mesh part names required for a scoped plane binding')
    if (origin.shape != (3,) or axes.shape != (2, 3) or extent.shape != (2,)
            or not all(np.isfinite(a).all() for a in (origin, axes, extent))
            or not np.allclose(axes@axes.T, np.eye(2), atol=1e-6) or np.any(extent <= 0)
            or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v)
                   for v in (cell, band, radius))
            or not .005 <= cell <= .1 or not 0 < band <= .08 or not cell <= radius <= .15
            or np.prod(np.ceil(extent/cell)) > 16_000_000):
        raise ValueError('Bounded metric plane, orthonormal axes and coverage resolution required')
    return origin, axes, extent, float(cell), float(band), float(radius)


def plane_gap_mask(points, sigma, declaration, output, opacity=None):
    """Opaque only where the retained GS has no plane support.

    Retained rows must already have independent capture support. This projects
    only rows near the declared surface, so a desk above the floor cannot erase
    the floor mesh underneath it. No task dimensions, object CAD or proxy fit.
    """
    from PIL import Image
    from scipy.ndimage import gaussian_filter
    from calibrated_scene_reconstruction.scene.review.environment import digest
    origin, axes, extent, cell, band, maximum = plane_specification(declaration)
    points = np.asarray(points, float); sigma = np.asarray(sigma, float)
    if points.ndim != 2 or points.shape[1] != 3 or sigma.shape != (len(points),):
        raise ValueError('Coverage rows must match the retained world-space cloud')
    opacity=np.ones(len(points)) if opacity is None else np.asarray(opacity,float)
    if opacity.shape!=(len(points),) or not np.isfinite(opacity).all() or np.any((opacity<0)|(opacity>1)):
        raise ValueError('Coverage needs finite per-row opacity in0..1')
    local = (points-origin)@axes.T
    distance = abs((points-origin)@np.cross(*axes))
    valid = (np.isfinite(points).all(1) & np.isfinite(sigma) & (sigma > 0)
             & (distance <= band) & np.all((local >= 0) & (local < extent), axis=1))
    width, height = np.ceil(extent/cell).astype(int)
    # Use the true extent rather than the rounded nominal cell for USD UVs.
    pixel = extent/[width, height]
    density = np.zeros((height, width), np.float64)
    xy = np.floor(local[valid]/pixel).astype(int)
    # A lone faint row is not opaque coverage. Accumulate a conservative alpha
    # estimate instead of dilating every point into a fully transparent hole.
    radii=np.minimum(sigma[valid],maximum/2)/min(pixel)
    bins=np.maximum(1,np.ceil(radii*4).astype(int))
    weights=-np.log1p(-np.minimum(opacity[valid],1-1e-6))
    for size in np.unique(bins):
        seed=np.zeros_like(density);group=xy[bins==size]
        np.add.at(seed,(group[:,1],group[:,0]),weights[bins==size])
        spread=float(size)/4
        # gaussian_filter normalizes its kernel; restore a unit-height kernel.
        impulse=np.zeros((2*int(np.ceil(3*spread))+1,)*2);mid=len(impulse)//2;impulse[mid,mid]=1.
        peak=gaussian_filter(impulse,spread,mode='constant',truncate=3)[mid,mid]
        density+=gaussian_filter(seed,spread,mode='constant',truncate=3)/peak
    coverage=-np.expm1(-density)
    gap=np.round(255*np.clip((.98-coverage)/.98,0,1)).astype(np.uint8)
    output = Path(output)
    Image.fromarray(np.flipud(gap)).save(output)
    return {'path': str(output.resolve()), 'sha256': digest(output),
            'origin': origin.tolist(), 'axes': axes.tolist(), 'extent_m': extent.tolist(),
            'supported_rows': int(valid.sum()), 'gs_covered_fraction': float((coverage >= .98).mean()),
            'partial_gs_coverage_fraction':float((coverage>=.1).mean()),
            'compositing':declaration.get('compositing','cutout'),
            'mesh_parts':declaration.get('mesh_parts'),
            'scope': ('Capture-textured solid plane backs the field with ordinary depth; foreground GS remains visible, below-plane GS is occluded. Requires composed source/free-view preservation review.'
                      if declaration.get('compositing')=='depth_backing' else
                      'Opaque PBR plane backs source-qualified GS; no coverage-based holes. Requires composed preservation review.' if declaration.get('compositing')=='solid_backing' else
                      'GS-supported plane texels are transparent in the native mesh; collision geometry stays solid. Requires composed novel-view coverage review.')}
