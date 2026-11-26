"""Validate a volume's original material reference and separate initial pose.

No reference is inferred from the posed surface. This does not approve its
physical priors, source fit, native authoring or export/reload behavior.
"""
import numpy as np

from .tetra_boundary import determinants


def indexed_nodes(buffer, reference, indices):
    """Exclude native allocation padding using the explicit reference count.

    The complete reference and every tetrahedral index remain mandatory. An
    out-of-range physical node cannot be discarded as padding.
    """
    points,rest=np.asarray(buffer,float),np.asarray(reference,float)
    determinants(rest,np.asarray(indices))
    if points.ndim!=2 or points.shape[1:]!=(3,) or len(points)<len(rest):
        raise ValueError('Native buffer must contain every referenced volume node')
    active=points[:len(rest)]
    if not np.isfinite(active).all():raise ValueError('Finite active native volume nodes required')
    return active.copy()


def embedding(points, reference, tetrahedra, *, tolerance_m=1e-8):
    """Find containing reference cells without projecting across empty space."""
    from scipy.spatial import cKDTree
    p,r,t=np.asarray(points,float),np.asarray(reference,float),np.asarray(tetrahedra)
    det=determinants(r,t)
    if (p.ndim!=2 or p.shape[1:]!=(3,) or not len(p) or not np.isfinite(p).all()
            or np.any(abs(det)<1e-20) or not np.isfinite(tolerance_m) or not 0<tolerance_m<=1e-5):
        raise ValueError('Finite embedded points and resolved reference volume required')
    cells=r[t];centres=cells.mean(1);radius=np.linalg.norm(cells-centres[:,None],axis=2).max()
    tree=cKDTree(centres);lo,hi=cells.min(1),cells.max(1)
    basis=np.stack([cells[:,1]-cells[:,0],cells[:,2]-cells[:,0],cells[:,3]-cells[:,0]],axis=2)
    inverse=np.linalg.inv(basis);tolerance=tolerance_m*np.linalg.norm(inverse,axis=2).max(1)
    result=[]
    for point in p:
        candidates=np.array(sorted(tree.query_ball_point(point,float(radius+tolerance_m))),dtype=int)
        candidates=candidates[((point>=lo[candidates]-tolerance_m)&(point<=hi[candidates]+tolerance_m)).all(1)]
        if not len(candidates):raise ValueError('Collision point lies outside original simulation material')
        weights=np.einsum('nij,nj->ni',inverse[candidates],point-cells[candidates,0])
        weights=np.c_[1-weights.sum(1),weights]
        inside=(weights>=-tolerance[candidates,None]).all(1)&(weights<=1+tolerance[candidates,None]).all(1)
        if not inside.any():raise ValueError('Collision point lies outside original simulation material')
        accepted=candidates[inside];result.append(int(accepted[np.argmax(weights[inside].min(1))]))
    return np.array(result,dtype=int)


def validate(state):
    required={'version','basis','skin_rest_points','simulation_rest_points','simulation_points',
              'simulation_indices','collision_rest_points','collision_points','collision_indices'}
    if (not isinstance(state,dict) or set(state)!=required or state['version']!=1
            or not isinstance(state['basis'],str) or not state['basis'].strip()):
        raise ValueError('Explicit version1 material reference and initial volume state required')
    result={'version':1,'basis':state['basis']};report={}
    skin=np.asarray(state['skin_rest_points'],float)
    if skin.ndim!=2 or skin.shape[1:]!=(3,) or len(skin)<4 or not np.isfinite(skin).all():
        raise ValueError('Finite original skin coordinates required')
    result['skin_rest_points']=skin.copy()
    for kind in ['simulation','collision']:
        rest=np.asarray(state[kind+'_rest_points'],float)
        points=np.asarray(state[kind+'_points'],float)
        tets=np.asarray(state[kind+'_indices'])
        if points.shape!=rest.shape or not np.isfinite(points).all():
            raise ValueError('Current volume nodes must correspond to original material nodes')
        reference=determinants(rest,tets)
        if np.any(abs(reference)<1e-20):raise ValueError('Degenerate volume reference cells')
        ratio=determinants(points,tets)/reference
        if (ratio<=0).any() or not np.isfinite(ratio).all():
            raise ValueError('Initial volume material cells are inverted or nonfinite')
        for key,value in [('rest_points',rest),('points',points),('indices',tets)]:
            result[kind+'_'+key]=value.copy()
        report[kind]={'minimum_volume_jacobian':float(ratio.min()),
                      'maximum_volume_jacobian':float(ratio.max())}
    return result,{'task_approved':False,'native_reload_qualified':False,'cells':report}


def skin_mapping(state,vertices,faces,*,tolerance_m=1e-6):
    """Bind an unchanged surface topology to a conforming posed volume."""
    from itertools import product
    from .tetra_boundary import surface
    value,_=validate(state)
    v,f=np.asarray(vertices,float),np.asarray(faces)
    p,t=value['simulation_points'],value['simulation_indices']
    if (v.ndim!=2 or v.shape[1:]!=(3,) or not len(v) or not np.isfinite(v).all()
            or f.ndim!=2 or f.shape[1:]!=(3,) or not len(f) or f.dtype.kind not in 'iu'
            or f.min()<0 or f.max()>=len(v) or value['skin_rest_points'].shape!=v.shape):
        raise ValueError('Corresponding complete skin, rest skin and indexed volume boundary required')
    # This validator also runs in Blender's embedded Python, which has NumPy
    # but does not require SciPy. Search every neighbouring tolerance cell;
    # rounding alone could lose a valid vertex across a cell boundary.
    boundary=surface(p,t);boundary_ids=np.unique(boundary)
    buckets={}
    for index in boundary_ids:
        key=tuple(np.floor(p[index]/tolerance_m).astype(np.int64))
        buckets.setdefault(key,[]).append(int(index))
    neighbourhood=np.array(list(product([-1,0,1],repeat=3)),dtype=np.int64)
    mapping=[];distance=[]
    for point in v:
        key=np.floor(point/tolerance_m).astype(np.int64)
        candidates=[index for nearby in key+neighbourhood for index in buckets.get(tuple(nearby),[])]
        if not candidates:raise ValueError('Skin is not the corresponding original/posed material boundary')
        gaps=np.linalg.norm(p[candidates]-point,axis=1);best=int(np.argmin(gaps))
        mapping.append(candidates[best]);distance.append(gaps[best])
    mapping=np.array(mapping,dtype=int);distance=np.array(distance)
    if (distance.max()>tolerance_m or len(np.unique(mapping))!=len(v)
            or np.linalg.norm(value['simulation_rest_points'][mapping]-value['skin_rest_points'],axis=1).max()>tolerance_m):
        raise ValueError('Skin is not the corresponding original/posed material boundary')
    mapped=mapping[f]
    if (mapped.shape!=boundary.shape or
            set(map(tuple,np.sort(mapped,axis=1)))!=set(map(tuple,np.sort(boundary,axis=1)))):
        raise ValueError('Skin must preserve the complete conforming volume boundary topology')
    return mapping,mapped


def proposal(skin, reference, simulation_points, collision_points, position, rotation, *, basis, tolerance_m=1e-6):
    """Propose a posed skin and retain the actual reference in one rigid frame.

    Only exact source-skin/collision-node correspondence is supported here.
    Do not project an unmatched visual surface onto nearby material. Velocities
    are not restored: a static asset proposal needs a fresh unforced check.
    """
    from scipy.spatial import cKDTree
    skin=np.asarray(skin,float);p=np.asarray(position,float);r=np.asarray(rotation,float)
    matrix=np.asarray(reference['local_to_world_row_matrix'],float)
    if (p.shape!=(3,) or r.shape!=(3,3) or matrix.shape!=(4,4)
            or not all(np.isfinite(x).all() for x in [p,r,matrix])
            or not np.allclose(r.T@r,np.eye(3),atol=1e-8,rtol=0) or not np.isclose(np.linalg.det(r),1.,atol=1e-8)
            or not np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-8,rtol=0)
            or not np.isclose(np.linalg.det(matrix[:3,:3]),1.,atol=1e-8)
            or not np.allclose(matrix[:,3],[0,0,0,1],atol=1e-8,rtol=0)
            or not np.isfinite(tolerance_m) or not 0<tolerance_m<=1e-5):
        raise ValueError('Proper metric source/anchor frames and small correspondence tolerance required')
    rest_collision=np.asarray(reference['collision_rest_points_local'],float)
    distance,indices=cKDTree(rest_collision).query(skin)
    if not len(skin) or not np.isfinite(skin).all() or distance.max()>tolerance_m:
        raise ValueError('Exact original skin/collision material correspondence required; no nearest-surface projection')
    def local_to_anchor(value):return (np.asarray(value)@matrix[:3,:3]+matrix[3,:3]-p)@r
    value={'version':1,'basis':basis,'skin_rest_points':local_to_anchor(skin)}
    for kind,buffer in [('simulation',simulation_points),('collision',collision_points)]:
        rest=np.asarray(reference[kind+'_rest_points_local'],float);tets=np.asarray(reference[kind+'_tetrahedra'])
        current=indexed_nodes(buffer,rest,tets)
        value[kind+'_rest_points']=local_to_anchor(rest)
        value[kind+'_points']=(current-p)@r
        value[kind+'_indices']=tets.copy()
    value,report=validate(value)
    posed=value['collision_points'][indices]
    report.update(maximum_skin_correspondence_error_m=float(distance.max()),velocities_restored=False,
                  requires_fresh_unforced_reload=True)
    return posed,value,report
