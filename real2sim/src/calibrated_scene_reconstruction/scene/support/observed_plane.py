"""Estimate the task top from semantically observed tracks, not every nearby object."""
import numpy as np


def use_labelled_tracks(config, alignment):
    """Keep capture's semantic plane evidence unless explicitly overridden."""
    return config.get('align', {}).get('photo_planes', bool(alignment.get('photo_planes')))


def top_height(points, minimum=30, band=.01):
    points=np.asarray(points,float)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
        raise ValueError('Finite observed support tracks required')
    if len(points)<minimum:
        raise ValueError('Insufficient independent observed support tracks')
    z=np.sort(points[:,2])
    # A sliding metric window is independent of the previous draft box height.
    right=np.searchsorted(z,z+2*band,side='right')
    first=int(np.argmax(right-np.arange(len(z))))
    inliers=z[first:right[first]]
    if len(inliers)<minimum or len(inliers)/len(z)<.6:
        raise ValueError('Observed support tracks do not determine one dominant horizontal plane')
    height=float(np.median(inliers))
    return height,{'source':'actual support-labelled image tracks',
                   'tracks':len(z),'inliers':len(inliers),
                   'median_absolute_residual_m':float(np.median(np.abs(inliers-height))),
                   'band_m':band}


def surface_plane(points,minimum=30,band=.01):
    """Fit observed near-horizontal support geometry without forcing it level.

    These are reconstructed source coordinates, not a gravity estimate or true
    furniture dimensions. Correlated SfM errors are not identified by this fit.
    """
    from scipy.optimize import least_squares
    p=np.asarray(points,float)
    if (p.ndim!=2 or p.shape[1:]!=(3,) or not np.isfinite(p).all()
            or len(p)<minimum or not np.isfinite(band) or band<=0):
        raise ValueError('Enough finite independent source support tracks required')
    xy=p[:,:2];centre=xy.mean(0);scale=np.linalg.svd(xy-centre,compute_uv=False)
    if scale[-1]<=max(1e-6,scale[0]*1e-3):
        raise ValueError('Collinear source tracks cannot identify a support normal')
    A=np.c_[xy-centre,np.ones(len(p))]
    initial=[0.,0.,float(np.median(p[:,2]))]
    fit=least_squares(lambda b:A@b-p[:,2],initial,loss='soft_l1',f_scale=band/3)
    selected=abs(A@fit.x-p[:,2])<=band
    if selected.sum()<minimum or selected.mean()<.6:
        raise ValueError('Source tracks do not determine one dominant planar support')
    fit=least_squares(lambda b:A[selected]@b-p[selected,2],fit.x,loss='soft_l1',f_scale=band/3)
    b=fit.x;normal=np.r_[-b[:2],1.];normal/=np.linalg.norm(normal)
    offset=float((b[2]-centre@b[:2])*normal[2]);residual=p[selected]@normal-offset
    return {'normal':normal.tolist(),'offset':offset,'tracks':len(p),'inliers':int(selected.sum()),
            'median_absolute_residual_m':float(np.median(abs(residual))),
            'horizontal_median_absolute_residual_m':float(np.median(abs(p[selected,2]-np.median(p[selected,2])))),
            'tilt_deg':float(np.degrees(np.arccos(normal[2]))),'band_m':float(band),
            'source':'actual independently support-labelled image tracks',
            'gravity_direction_inferred':False,'true_dimensions_read':False}


def height(plane,xy):
    n=np.asarray(plane['normal'],float);xy=np.asarray(xy,float);d=float(plane['offset'])
    if n.shape!=(3,) or xy.shape!=(2,) or not np.isfinite(np.r_[n,xy,d]).all() or not np.isclose(n@n,1.) or n[2]<=.2:
        raise ValueError('Upward observed support plane and finite XY required')
    return float((d-xy@n[:2])/n[2])


def model_plane(vertices,faces,face_material,material_names,top_material):
    """Inspect the actual source-labelled tabletop faces, never the box top."""
    p=np.asarray(vertices,float);f=np.asarray(faces,int)
    if top_material not in list(material_names):raise ValueError('Named actual tabletop material required for plane review')
    mask=np.asarray(face_material)==list(material_names).index(top_material)
    tri=p[f];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);area=np.linalg.norm(cross,axis=1)
    selected=mask&(area>1e-14)&(cross[:,2]>.9*area)
    if not selected.any():raise ValueError('No actual upward tabletop faces in the declared material')
    n=cross[selected].sum(0);n/=np.linalg.norm(n);points=tri[selected].reshape(-1,3);offset=float(np.mean(points@n))
    error=float(np.max(abs(points@n-offset)))
    return {'normal':n.tolist(),'offset':offset,'maximum_planarity_error_m':error,
            'top_material':top_material,'selected_triangles':int(selected.sum())}


def model_error(vertices,faces,face_material,material_names,observed):
    model=model_plane(vertices,faces,face_material,material_names,observed['top_material'])
    p=np.asarray(vertices,float);f=np.asarray(faces,int)
    labels=np.asarray(face_material)==list(material_names).index(observed['top_material'])
    tri=p[f];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);size=np.linalg.norm(cross,axis=1)
    selected=labels&(size>1e-14)&(cross[:,2]>.9*size)
    n=np.asarray(observed['normal'],float);d=float(observed['offset'])
    if n.shape!=(3,) or not np.isfinite(np.r_[n,d]).all() or not np.isclose(n@n,1.) or n[2]<=.2:
        raise ValueError('Finite upward observed tabletop plane required')
    maximum=float(np.max(abs(tri[selected]@n-d)))
    return maximum,{**model,'observed_normal':n.tolist(),'observed_offset':d,
                    'maximum_source_plane_distance_m':maximum,
                    'normal_difference_deg':float(np.degrees(np.arccos(np.clip(np.dot(n,model['normal']),-1,1))))}
