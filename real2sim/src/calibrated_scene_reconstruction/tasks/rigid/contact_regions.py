"""Measure source-surface proximity in explicitly identified support regions.

This complements unforced native stability. Proximity alone does not prove
load-bearing contact; no geometry, pose or physics state is modified.
"""
import numpy as np
from .assembly_measurements import rotations


def measure(samples, body_pose, owner_pose, signed_surface, regions, *, maximum_gap_m, maximum_penetration_m):
    samples=np.asarray(samples,float)
    if (samples.ndim!=2 or samples.shape[1]!=3 or not len(samples)
            or not np.isfinite(samples).all() or not regions
            or not np.isfinite([maximum_gap_m,maximum_penetration_m]).all()
            or min(maximum_gap_m,maximum_penetration_m)<0):
        raise ValueError('Source samples, explicit regions and finite nonnegative tolerances required')
    body_pose=np.asarray(body_pose,float);owner_pose=np.asarray(owner_pose,float)
    Rb,Ro=rotations(body_pose),rotations(owner_pose)
    points=(samples@Rb.T+body_pose[:3]-owner_pose[:3])@Ro
    distances=signed_surface(points)
    rows={}
    for name,region in regions.items():
        lower,upper=np.asarray(region['lower'],float),np.asarray(region['upper'],float)
        if lower.shape!=(3,) or upper.shape!=(3,) or not np.isfinite([lower,upper]).all() or (upper<=lower).any():
            raise ValueError('Each support region requires a finite nonempty owner-local box')
        selected=((points>=lower)&(points<=upper)).all(1)
        d=distances[selected]
        gap=float(d.min()) if len(d) else None
        rows[name]={'pass':gap is not None and -maximum_penetration_m<=gap<=maximum_gap_m,
                    'minimum_signed_gap_m':gap,'source_samples':int(selected.sum()),'region':region}
    return {'pass':all(row['pass'] for row in rows.values()),'regions':rows,
            'maximum_gap_m':maximum_gap_m,'maximum_penetration_m':maximum_penetration_m,
            'scope':'Finite source-surface proximity only; pair with native unforced support/stability and contact audit.'}
