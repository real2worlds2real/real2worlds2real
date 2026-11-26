"""Independent signed rest dihedrals; the intrinsic membrane metric stays fixed."""
import numpy as np


def reference_geometry(chart, faces, thickness, *, device='cpu'):
    """Diagnose an explicit natural shape separately from the loaded state.

    Exact subdivision can expose a sub-thickness crease in an otherwise
    isometric reference. A valid loaded surface does not resolve that issue.
    This is a history/preload review, not a measured material identification
    or a blanket prohibition of intentionally prestressed objects.
    """
    reference = chart.get('bend_reference')
    if reference is None:
        return None
    from calibrated_scene_reconstruction.assets.cloth_contact import self_intersections
    from calibrated_scene_reconstruction.assets.cloth_clearance import measure
    points = np.asarray(reference['points'], float)
    intersections = self_intersections(points, faces, device=device)
    clearance = measure(points, faces, chart['points'], thickness, device=device)
    passed = bool(intersections['pass'] and clearance['pass'])
    return {'pass': passed, 'history_review_required': not passed,
            'self_intersection': intersections, 'finite_thickness': clearance,
            'source': reference['source'], 'basis': reference['basis'],
            'scope': 'Natural-reference geometry only; actual-state checks remain independent. '
                     'An unresolved natural crease requires a history/preload explanation and native release tests; '
                     'do not infer memory from the captured drape or change material thickness to hide a failure.'}


def require_supported(rest, *, diagnostic=False):
    if 'bend_angles' in (rest or {}) and not diagnostic:
        raise ValueError('Independent rest bend angles are unvalidated in the installed Isaac5.1 authoring path: matched positive/reverse native controls had identical motion. Use validated isometric bend_reference or an explicit diagnostic.')


def adjacency(faces):
    faces=np.asarray(faces,int)
    if faces.ndim!=2 or faces.shape[1:]!=(3,):raise ValueError('Triangle faces required for curvature')
    if not len(faces):return np.empty((0,2),int),np.empty((0,2),int)
    directed=np.stack([faces,np.roll(faces,-1,axis=1)],axis=2).reshape(-1,2)
    keys=np.sort(directed,axis=1);order=np.lexsort((keys[:,1],keys[:,0]));sorted_keys=keys[order]
    starts=np.r_[0,np.flatnonzero(np.any(sorted_keys[1:]!=sorted_keys[:-1],axis=1))+1]
    counts=np.diff(np.r_[starts,len(order)])
    if np.any(counts>2):raise ValueError('Curvature needs manifold triangle edges')
    shared=starts[counts==2]
    first=np.minimum(order[shared],order[shared+1]);second=np.maximum(order[shared],order[shared+1])
    if np.any(directed[first]!=directed[second,::-1]):
        raise ValueError('Curvature needs consistent triangle winding')
    # Preserve the original traversal order: a pair is emitted when its
    # second directed edge is encountered, and carries the first edge's sign.
    sequence=np.argsort(second)
    return np.c_[first[sequence]//3,second[sequence]//3],directed[first[sequence]]


def measured(points,faces):
    """Geometry-derived proposal angles, never an identified material history."""
    p=np.asarray(points,float);f=np.asarray(faces,int)
    pairs,edges=adjacency(f)
    normals=np.cross(p[f[:,1]]-p[f[:,0]],p[f[:,2]]-p[f[:,0]])
    lengths=np.linalg.norm(normals,axis=1)
    if not np.isfinite(normals).all() or (lengths<=1e-14).any():raise ValueError('Nondegenerate curvature triangles required')
    normals/=lengths[:,None]
    tangent=p[edges[:,1]]-p[edges[:,0]];tangent/=np.linalg.norm(tangent,axis=1)[:,None]
    left,right=normals[pairs[:,0]],normals[pairs[:,1]]
    angles=np.degrees(np.arctan2(np.einsum('ij,ij->i',np.cross(left,right),tangent),np.einsum('ij,ij->i',left,right)))
    return pairs,(angles+180)%360-180


def orientation_diagnostic(points,faces,reference):
    """Review signed natural folds without inferring or changing their history.

    A surface sweep can inadvertently reflect its material frame. Rigid
    rotations preserve signed dihedrals; a reflection reverses them. Loaded
    folds can legitimately oppose their natural state, so this is a review
    trigger, never permission to copy the observed pose into rest coordinates.
    Near-flat and near-180-degree folds have ambiguous sign and are excluded.
    """
    _,actual=measured(points,faces);_,natural=measured(reference,faces)
    selected=((abs(actual)>=10)&(abs(actual)<=170)&
              (abs(natural)>=10)&(abs(natural)<=170))
    a,n=actual[selected],natural[selected]
    count=len(a);opposed=int(np.sum(a*n<0))
    distance=lambda x:abs((x+180)%360-180)
    original=float(np.median(distance(a-n))) if count else None
    reflected=float(np.median(distance(a+n))) if count else None
    suspected=bool(count and opposed/count>=.9 and original-reflected>10)
    return {'resolved_natural_fold_edges':count,'opposed_fold_edges':opposed,
            'opposed_fraction':opposed/count if count else None,
            'median_angle_difference_deg':original,
            'median_difference_if_reference_reflected_deg':reflected,
            'possible_reflected_material_frame':suspected,
            'review_required':suspected,'reference_changed':False,
            'scope':'Check material-frame handedness and independently justified preload/history. Opposite folds alone do not identify a rest shape or approve a correction.'}


def validate(value,faces):
    if (not isinstance(value,dict) or set(value)!={'pairs','degrees','source','basis'}
            or any(not isinstance(value.get(k),str) or not value[k].strip() for k in ('source','basis'))):
        raise ValueError('Independent bend angles need adjacent triangle pairs, degrees, source and construction basis')
    pairs=np.asarray(value['pairs']);angles=np.asarray(value['degrees'],float)
    if (pairs.ndim!=2 or pairs.shape[1:]!=(2,) or not len(pairs) or pairs.dtype.kind not in 'iu'
            or angles.shape!=(len(pairs),) or not np.isfinite(angles).all() or (angles < -180).any() or (angles>=180).any()):
        raise ValueError('Finite signed rest angles in [-180,180) degrees required')
    available={tuple(sorted(row)) for row in adjacency(faces)[0]}
    selected=[tuple(sorted(map(int,row))) for row in pairs]
    if len(set(selected))!=len(selected) or not set(selected)<=available:
        raise ValueError('Bend angles must reference distinct actual adjacent triangles')
    return pairs,angles
