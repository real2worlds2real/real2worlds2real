"""Explicit collision partitions taken from the generated source triangles.

A partition changes no rigid ownership or mass and adds no authored geometry.
Convex cooking still requires native shape and task validation, especially for
cavities. Material names identify source faces; they do not select a physics
strategy automatically.
"""
import re
import numpy as np


def normalize(vertices, *, principal_axes=False, scale_mode='maximum_extent'):
    """Rigid coordinate conditioning and uniform scale, preserving every point."""
    vertices=np.asarray(vertices,float)
    if vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all() or len(vertices)<4:
        raise ValueError('Finite source collision vertices required')
    centre=(vertices.min(0)+vertices.max(0))/2
    basis=np.eye(3)
    if principal_axes:
        _,_,vh=np.linalg.svd(vertices-vertices.mean(0),full_matrices=False)
        basis=vh.T
        if np.linalg.det(basis)<0:basis[:,-1]*=-1
    local=(vertices-centre)@basis
    if scale_mode not in {'maximum_extent','minimum_extent','axis_extents'} or scale_mode!='maximum_extent' and not principal_axes:
        raise ValueError('Minimum-feature conditioning requires the principal-axis frame')
    extents=np.ptp(local,axis=0)
    extent=extents if scale_mode=='axis_extents' else float(extents.max() if scale_mode=='maximum_extent' else extents.min())
    if not np.isfinite(extent).all() or np.any(np.asarray(extent)<=0):raise ValueError('Cannot normalize empty collision geometry')
    return local/extent,centre,basis,extent


def partition(part, groups):
    if not isinstance(groups,list) or not groups:raise ValueError('Nonempty collision group list required')
    labels=part.get('face_material');names=part.get('material_names',[])
    if labels is None or len(labels)!=len(part['F']):raise ValueError('Collision partitions need source face labels')
    covered=np.zeros(len(labels),int);seen=set();result=[]
    for spec in groups:
        if not isinstance(spec,dict) or set(spec)-{'name','materials','approximation','cooking','normalize_coordinates','align_principal_axes','normalization_scale','max_cell_aspect_ratio'}:
            raise ValueError('Invalid collision partition declaration')
        if type(spec.get('normalize_coordinates',False)) is not bool:
            raise ValueError('Collision coordinate normalization must be boolean')
        if (type(spec.get('align_principal_axes',False)) is not bool
                or spec.get('align_principal_axes',False) and not spec.get('normalize_coordinates',False)):
            raise ValueError('Principal-axis conditioning requires explicit coordinate normalization')
        mode=spec.get('normalization_scale','maximum_extent')
        if mode not in {'maximum_extent','minimum_extent','axis_extents'} or mode!='maximum_extent' and not spec.get('align_principal_axes',False):
            raise ValueError('Minimum-feature conditioning requires principal axes and normalized coordinates')
        name=spec.get('name')
        if not isinstance(name,str) or not re.fullmatch('[A-Za-z][A-Za-z0-9_]*',name) or name in seen:
            raise ValueError('Unique collision group names required')
        seen.add(name);materials=spec.get('materials')
        if not isinstance(materials,list) or not materials or len(set(materials))!=len(materials) or set(materials)-set(names):
            raise ValueError('Collision group refers to absent or repeated source material')
        approximation=spec.get('approximation')
        ratio=spec.get('max_cell_aspect_ratio')
        if ratio is not None and (type(ratio) not in (int,float) or not np.isfinite(ratio) or not 2<=ratio<=100 or approximation!='convexHull'):
            raise ValueError('Exact convex source cells require an aspect ratio in2..100 and convexHull')
        if approximation not in {'convexHull','convexDecomposition','sdf'}:
            raise ValueError('Explicit supported source partition approximation required')
        if spec.get('normalize_coordinates',False) and approximation!='convexHull':
            raise ValueError('Coordinate normalization is qualified only for single hulls; decomposition controls use metric local geometry')
        if approximation=='convexDecomposition':
            from calibrated_scene_reconstruction.assets.convex_cooking import validate
            validate(spec.get('cooking',{}))
        elif approximation=='sdf' and 'cooking' in spec:
            raise ValueError('SDF partitions cannot use convex cooking settings')
        elif 'cooking' in spec:
            from calibrated_scene_reconstruction.assets.convex_cooking import validate_hull
            validate_hull(spec['cooking'])
        mask=np.isin(labels,[names.index(m) for m in materials]);covered+=mask
        if not mask.any():raise ValueError('Empty collision group')
        faces=np.asarray(part['F'])[mask];ids,local=np.unique(faces,return_inverse=True)
        if approximation=='sdf':
            from calibrated_scene_reconstruction.assets.metric_mesh import geometry
            geometry(np.asarray(part['V'])[ids],local.reshape(-1,3),
                     min_extent_m=1e-5,require_bottom_origin=False)
        result.append({**spec,'V':np.asarray(part['V'])[ids], 'F':local.reshape(-1,3),
                       'source_faces':np.flatnonzero(mask)})
    if not np.all(covered==1):raise ValueError('Each source triangle must belong to exactly one collision group')
    return result
