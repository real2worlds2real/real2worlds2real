"""Geometry proposals for loose material fed into a contents/wall gap.

Use actual closed meshes and arbitrary ray directions. Free target points do
not certify a complete folded sheet, normal preload, native retention or task.
"""
import numpy as np


def gap(contents,wall,origin,direction,*,maximum_m,step_m,clearance_m):
    o,d=np.asarray(origin,float),np.asarray(direction,float)
    if (o.shape!=(3,) or d.shape!=(3,) or not np.isfinite([o,d]).all()
            or not np.isfinite([maximum_m,step_m,clearance_m]).all()
            or not 0<step_m<=clearance_m<maximum_m<=10 or np.linalg.norm(d)<1e-12):
        raise ValueError('Finite ray and bounded resolved contact-gap search required')
    d=d/np.linalg.norm(d);n=int(np.ceil(maximum_m/step_m))+1
    if n>100000:raise ValueError('Contact-gap ray exceeds sample budget')
    t=np.linspace(0.,maximum_m,n);p=o+t[:,None]*d
    soft=contents(p);rigid=wall(p)
    if soft[0]>=0 or rigid[0]<=0:
        raise ValueError('Gap ray must start inside contents and outside the container solid')
    exits=np.flatnonzero(soft>0.)
    if not len(exits):raise ValueError('Ray never exits contents')
    first=int(exits[0]);hits=np.flatnonzero((np.arange(n)>first)&(rigid<0.))
    if not len(hits):raise ValueError('Ray has no enclosing wall after contents; outside air is not a pocket')
    last=int(hits[0]);eligible=(np.arange(n)>=first)&(np.arange(n)<last)&(soft>=clearance_m)&(rigid>=clearance_m)
    ids=np.flatnonzero(eligible)
    if not len(ids) or np.any(np.diff(ids)!=1):raise ValueError('No single resolved free channel at the requested clearance')
    distance=float((t[ids[0]]+t[ids[-1]])/2);target=o+distance*d
    return {'point_m':target.tolist(),'ray_distance_m':distance,
            'sampled_free_gap_width_m':float(t[last]-t[first]),'sampling_step_bound_m':float(maximum_m/(n-1)),
            'contents_clearance_m':float(contents(target[None])[0]),'wall_clearance_m':float(wall(target[None])[0]),
            'normal_pressure_proven':False,'whole_sheet_fit_proven':False,'task_approved':False}


def screen_route(points,solids,*,clearance_m,step_m,offsets_m=None):
    """Screen a target path and optional unchanged, translated patch samples.

    A blocked patch proposal may need a different orientation or deformation;
    it does not show that the physical task is impossible. Even a clear sampled
    patch leaves the rest of the connected sheet and native dynamics unchecked.
    """
    p=np.asarray(points,float)
    if (p.ndim!=2 or p.shape[1:]!=(3,) or len(p)<2 or not np.isfinite(p).all()
            or not 0<step_m<=clearance_m or not solids):
        raise ValueError('Finite route, explicit rigid surfaces and positive resolved clearance required')
    samples=[]
    for a,b in zip(p[:-1],p[1:]):
        count=max(2,int(np.ceil(np.linalg.norm(b-a)/step_m))+1)
        if count>100000:raise ValueError('Route segment exceeds sample budget')
        samples.append(np.linspace(a,b,count))
    sampled=np.concatenate(samples)
    offsets=np.zeros((1,3)) if offsets_m is None else np.asarray(offsets_m,float)
    if (offsets.ndim!=2 or offsets.shape[1:]!=(3,) or not len(offsets)
            or not np.isfinite(offsets).all() or len(sampled)*len(offsets)>5000000):
        raise ValueError('Finite actual patch offsets within the sampled-route query budget required')
    distances={name:float('inf') for name in solids}
    # Bound transient memory independently of the complete route/patch product.
    chunk=max(1,50000//len(offsets))
    for start in range(0,len(sampled),chunk):
        query=(sampled[start:start+chunk,None,:]+offsets[None,:,:]).reshape(-1,3)
        for name,solid in solids.items():
            value=np.asarray(solid(query),float)
            if value.shape!=(len(query),) or not np.isfinite(value).all():
                raise ValueError('Finite actual signed surface distances required')
            distances[name]=min(distances[name],float(value.min()))
    return {'pass':all(value>=clearance_m for value in distances.values()),
            'minimum_distances_m':distances,'samples':len(sampled),
            'sampled_footprint_points':len(offsets),
            'fixed_orientation_shape_proposal':offsets_m is not None,
            'whole_material_checked':False,'normal_pressure_proven':False,'task_approved':False,
            'scope':('Sampled translated patch proposal' if offsets_m is not None else
                     'Sampled target centre path only')+
                    '; full connected material and native contact checks remain mandatory'}
