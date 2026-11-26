"""Conservative axial nesting geometry from actual hollow rigid mesh sections.

This is a geometric precursor to native contact/retention/withdrawal tests.
It reads no object dimensions, templates or previous scene results.
"""
import numpy as np
import cv2


def release_containment(positions, heights, moving_height_m, receiver_height_m):
    """Keep release settling observable and contained; not a source-state waiver."""
    positions, heights = np.asarray(positions, float), np.asarray(heights, float)
    dimensions = np.asarray([moving_height_m, receiver_height_m], float)
    if (positions.ndim != 2 or positions.shape[1:] != (3,) or not len(positions)
            or heights.shape != (len(positions),) or not np.isfinite(positions).all()
            or not np.isfinite(heights).all() or not np.isfinite(dimensions).all()
            or dimensions.min() <= 0):
        raise ValueError('Finite observed release trajectory and positive geometry required')
    overlap = receiver_height_m - heights
    return {'pass_check': bool(overlap.min() >= .2 * moving_height_m and heights.min() >= 0.),
            'minimum_overlap_m': float(overlap.min()),
            'minimum_height_m': float(heights.min()),
            'maximum_transient_displacement_m': float(np.linalg.norm(positions-positions[0], axis=1).max()),
            'guide_disabled': True,
            'scope': 'Task-release containment only; final unforced stability and all-phase contact/penetration checks remain required'}


def inscribed_radius(loop):
    xy=np.asarray(loop[:,:2],float)
    if cv2.pointPolygonTest(xy.astype('float32'),(0.,0.),False)<=0:
        raise ValueError('Cup cavity must contain the declared nesting axis')
    delta=np.roll(xy,-1,axis=0)-xy
    length=np.sum(delta*delta,axis=1)
    t=np.clip(-np.sum(xy*delta,axis=1)/np.maximum(length,1e-30),0.,1.)
    return float(np.linalg.norm(xy+t[:,None]*delta,axis=1).min())


def radial_profile(mesh, levels=256):
    if (type(levels) is not int or not 32<=levels<=2048 or not mesh.is_volume
            or not np.isfinite(mesh.vertices).all()):
        raise ValueError('A closed outward cup solid and bounded section resolution required')
    lo,hi=mesh.bounds[:,2]
    z=np.linspace(lo+1e-6,hi-1e-6,levels)
    outer,inner=[],[]
    for height in z:
        section=mesh.section(plane_normal=[0.,0.,1.],plane_origin=[0.,0.,height])
        loops=[] if section is None else section.discrete
        loops=[loop for loop in loops if len(loop)>3]
        loops.sort(key=lambda loop:float(np.median(np.linalg.norm(loop[:,:2],axis=1))))
        radii=[np.linalg.norm(loop[:,:2],axis=1) for loop in loops]
        if not radii:raise ValueError('Missing closed horizontal cup section')
        outer.append(float(radii[-1].max()))
        inner.append(inscribed_radius(loops[0]) if len(radii)==2 else 0.)
        if len(radii)>2:raise ValueError('Ambiguous non-axisymmetric cup section topology')
    if not np.any(np.asarray(inner)>0):raise ValueError('Goal has no observed model cavity')
    return {'z':z,'outer':np.asarray(outer),'inner':np.asarray(inner),'height':float(hi-lo)}


def clearance(target,goal,offset,inner_shrink=0.,floor_min=0.):
    # A circumscribed target section fits inside the inscribed goal section,
    # regardless of polygon azimuth. Include all target sample levels in overlap.
    z=target['z']+offset
    keep=z<goal['z'][-1]
    if not keep.any():return float('inf')
    if z[keep].min()<floor_min:return -float(floor_min-z[keep].min())
    cavity=np.interp(z[keep],goal['z'],goal['inner'],left=0.)-inner_shrink
    return float(np.min(cavity-target['outer'][keep]))


def seating(target,goal,inner_shrink=0.,floor_min=0.):
    low,high=0.,goal['height']+target['height']
    for _ in range(60):
        middle=(low+high)/2
        if clearance(target,goal,middle,inner_shrink,floor_min)>=0:high=middle
        else:low=middle
    overlap=max(0.,goal['height']-high)
    path=np.linspace(high,goal['height']+target['height'],80)
    gaps=[clearance(target,goal,float(value),inner_shrink,floor_min) for value in path]
    return {'offset_m':float(high),'overlap_m':float(overlap),
            'overlap_target_height_fraction':float(overlap/target['height']),
            'minimum_sampled_path_clearance_m':float(min(gaps)),
            'path_samples':len(path),'pass':bool(overlap>=.2*target['height'] and min(gaps)>=-1e-7)}
