"""Evaluate the full declared initial support relation in one place.

Routing above a support and an end actually near that support are distinct.
Neither geometrical statement proves loaded or retained native contact.
"""
from calibrated_scene_reconstruction.assets.surface_distance import triangle_samples
from .support_route import above_unthreaded,end_contact


def validate_fold_support(request):
    import numpy as np
    if (not isinstance(request,dict)or set(request)!={'maximum_material_path_m','maximum_end_distance_m','basis'}
            or not isinstance(request['basis'],str)or not request['basis'].strip()
            or not np.isfinite([request['maximum_material_path_m'],request['maximum_end_distance_m']]).all()
            or not 0<request['maximum_end_distance_m']<request['maximum_material_path_m']):
        raise ValueError('Explicit finite local material-support path, end distance and physical/source basis required')


def connected_fold_support(points,rest,faces,vertices,support_faces,request,*,tolerance_m):
    """Screen a retained end carried by a nearby connected part of its sheet.

    This does not infer contact forces or stiffness from connectivity. Preserve
    the direct skin-contact distance and its raw area measurement; point/line
    contacts of a lower fold need not cover the upper end's area. Require native retention
    without material fixtures and the requested task separately.
    """
    import numpy as np
    import trimesh
    from scipy import sparse
    from scipy.sparse.csgraph import dijkstra
    from .cloth_rest import triangle_edges
    from .support_route import upward_patch,_closest_actual
    options=request['connected_folds'];validate_fold_support(options)
    p,r,f,v,sf=map(np.asarray,[points,rest,faces,vertices,support_faces]);direct=request['material_contact']
    if options['maximum_material_path_m']>.5*float(np.ptp(r,axis=0).max()):
        raise ValueError('Connected-fold support must stay local to the retained material end')
    up=np.asarray(request['up'],float)
    region=upward_patch(v,sf,up,tolerance_m,request['surface_region'])
    top=trimesh.Trimesh(v,sf[region],process=False);centres=p[f].mean(1)
    closest,ids,query=_closest_actual(v,sf[region],centres)
    delta=centres-closest;distance=np.linalg.norm(delta,axis=1)
    height=np.einsum('ij,ij->i',delta,top.face_normals[ids])
    rc=r[f].mean(1);axis=direct['axis'];span=float(np.ptp(r[:,axis]));low=float(r[:,axis].min())
    fraction=(rc[:,axis]-low)/span
    end=(fraction<=direct['fraction']if direct['end']=='minimum'else fraction>=1-direct['fraction'])
    areas=np.linalg.norm(np.cross(r[f[:,1]]-r[f[:,0]],r[f[:,2]]-r[f[:,0]]),axis=1)/2
    if not end.any()or(areas<=0).any():raise ValueError('Resolved positive-area material end required')
    nodes=np.unique(f[end]);edges=triangle_edges(f);length=np.linalg.norm(r[edges[:,1]]-r[edges[:,0]],axis=1)
    graph=sparse.coo_matrix((np.r_[length,length],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(len(r),len(r))).tocsr()
    path=dijkstra(graph,directed=False,indices=nodes,min_only=True,limit=options['maximum_material_path_m'])
    local=np.min(path[f],axis=1)<=options['maximum_material_path_m']
    touching=(distance<=direct['maximum_gap_m'])&(height>=-tolerance_m)&local
    contact_area=float(areas[touching].sum());end_area=float(areas[end].sum())
    required_area=end_area*direct['minimum_area_fraction']
    near=(distance[end]<=options['maximum_end_distance_m'])&(height[end]>=-tolerance_m)
    share=float(areas[end][near].sum()/end_area)
    return {'pass':bool(contact_area>0 and share>=direct['minimum_area_fraction']),
            'near_support_end_area_fraction':share,'actual_local_direct_contact_area_m2':contact_area,
            'actual_local_direct_contact_triangles':int(touching.sum()),
            'reference_direct_end_area_m2':required_area,
            'direct_contact_distance_m':direct['maximum_gap_m'],'maximum_material_path_m':options['maximum_material_path_m'],
            'maximum_end_distance_m':options['maximum_end_distance_m'],'nearest_query':query,
            'loaded_support_contact_proven':False,'native_retention_required':True,
            'scope':'Local indexed material connection to real upward support contact. No air contact, attachments or loaded-force claim; strict routing, self/solid geometry and native released/open-state retention remain mandatory.'}


def contact_state(points,rest,faces,vertices,support_faces,request,*,tolerance_m):
    contact=end_contact(points,rest,faces,vertices,support_faces,request['up'],request['material_contact'],
                        surface_region=request['surface_region'],tolerance=tolerance_m)
    if request.get('connected_folds'):
        folded=connected_fold_support(points,rest,faces,vertices,support_faces,request,tolerance_m=tolerance_m)
        contact={**contact,'pass':bool(contact['pass']or folded['pass']),
                 'direct_end_contact':dict(contact),'connected_folds':folded,
                 'native_retention_required':True,'support_mode':'direct_or_local_connected_folds'}
    return contact


def check(points,rest,faces,vertices,support_faces,request,*,thickness_m,tolerance_m):
    route=above_unthreaded(triangle_samples(points,faces),vertices,support_faces,request['up'],
        tolerance=tolerance_m,surface_region=request['surface_region'])
    contact=None
    if request.get('connected_folds')and not request.get('material_contact'):
        raise ValueError('Connected-fold support requires the original direct-contact contract')
    if request.get('material_contact'):
        if thickness_m is None or request['material_contact']['maximum_gap_m']>2*thickness_m:
            raise ValueError('End-support gap must be within two declared sheet thicknesses')
        contact=contact_state(points,rest,faces,vertices,support_faces,request,tolerance_m=tolerance_m)
    return {'pass':bool(route['pass'] and (contact is None or contact['pass'])),
            'routing':route,'material_contact':contact,'loaded_support_contact_proven':False,
            'scope':'Complete declared geometric support relation only; native retention, material and task checks remain separate.'}
