"""Material-area distribution around an observed container opening."""
import numpy as np


def measure(points, rest, faces, opening):
    points,rest,faces=np.asarray(points,float),np.asarray(rest,float),np.asarray(faces,int)
    if points.shape!=rest.shape or points.ndim!=2 or points.shape[1]!=3 or not np.isfinite([points,rest]).all():
        raise ValueError('Matching finite actual and intrinsic cloth coordinates required')
    centre=np.asarray(opening['xy'],float)
    if (centre.shape!=(2,) or not np.isfinite(centre).all()
            or not np.isfinite([opening['radius'],opening['rim_z'],opening['depth']]).all()
            or min(opening['radius'],opening['depth'])<=0):
        raise ValueError('Finite physical opening with positive radius and depth required')
    tri=rest[faces];third_area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/6
    weights=np.zeros(len(rest))
    for column in range(3):np.add.at(weights,faces[:,column],third_area)
    if weights.sum()<=0:raise ValueError('Positive complete material area required')
    radial=np.linalg.norm(points[:,:2]-opening['xy'],axis=1)
    inside=(radial<float(opening['radius']))&(points[:,2]<opening['rim_z'])&(points[:,2]>opening['rim_z']-opening['depth'])
    above=(radial<float(opening['radius']))&(points[:,2]>=opening['rim_z'])
    inside_weight=float(weights[inside].sum());total=float(weights.sum())
    depth=(float(opening['rim_z'])-points[inside,2])/float(opening['depth'])
    return {'inside_fraction':inside_weight/total,
            'above_opening_fraction':float(weights[above].sum()/total),
            'opening_column_fraction':float(weights[inside|above].sum()/total),
            'outside_fraction':float(weights[~(inside|above)].sum()/total),
            'mean_inside_depth_fraction':float(weights[inside]@depth/inside_weight) if inside_weight>0 else None,
            'maximum_inside_depth_m':float(depth.max()*opening['depth']) if len(depth) else None,
            'material_area_m2':float(weights.sum()),
            'scope':'Intrinsic material-area distribution and below-rim insertion depth. Above-mouth material is reported separately; opening-column occupancy does not establish support, retention or stability.'}


def check(points,rest,faces,opening,requirements):
    measured=measure(points,rest,faces,opening);checks={}
    for key,interval in requirements.items():
        bounds=np.asarray(interval,float)
        if key not in {'inside_fraction','above_opening_fraction','opening_column_fraction','outside_fraction','mean_inside_depth_fraction'} or bounds.shape!=(2,) or not np.isfinite(bounds).all() or not 0<=bounds[0]<=bounds[1]<=1:
            raise ValueError('Explicit material fraction bounds required')
        checks[key]=bool(measured[key] is not None and bounds[0]<=measured[key]<=bounds[1])
    return {'pass':all(checks.values()),'measurements':measured,'checks':checks,'requirements':requirements}
