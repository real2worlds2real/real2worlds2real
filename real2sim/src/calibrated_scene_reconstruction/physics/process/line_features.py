"""Reviewed straight-part observations without hallucinating occluded tips.

Two image-row probes encode a projected line's position and direction. They
are measurement coordinates, not simulated poses or forces. Source segments
and their uncertainty remain available for independent visual review.
"""
import numpy as np

from .contract import require


def line_probes(endpoints, y):
    points=np.asarray(endpoints,float);rows=np.asarray(y,float)
    require(points.shape==(2,2) and rows.shape==(2,) and np.isfinite(points).all()
            and np.isfinite(rows).all() and abs(rows[1]-rows[0])>1e-6,
            'Two finite endpoints and distinct measurement rows required')
    delta=points[1]-points[0]
    require(abs(delta[1])>1e-6,'Horizontal segments require another observation representation')
    return points[0,0]+delta[0]/delta[1]*(rows-points[0,1])


def reviewed_groups(lines,groups,y,count,*,max_spread_px,max_extrapolation_ratio):
    """Aggregate reviewed edge proposals; never assert object centers or IDs.

    Missing groups become NaN points, so the unordered observation metric can
    match the visible subset. Excessive edge disagreement or extrapolation
    is rejected rather than silently converted to a precise observation.
    """
    require(isinstance(count,int) and 0<count<=256 and len(groups)<=count,
            'Declare a bounded physical part count')
    require(max_spread_px>0 and max_extrapolation_ratio>=0,'Declare observation uncertainty limits')
    bank={row['id']:row for row in lines}
    require(len(bank)==len(lines),'Duplicate source proposal IDs')
    used=[];result=np.full((count,2),np.nan);reports=[]
    for index,ids in enumerate(groups):
        require(ids and all(i in bank for i in ids),'Each visible group needs reviewed source segments')
        used.extend(ids);values=[];weights=[];segments=[]
        for key in ids:
            points=np.asarray(bank[key]['endpoints'],float);values.append(line_probes(points,y))
            lo,hi=sorted(points[:,1]);span=hi-lo
            outside=np.maximum(lo-np.asarray(y),np.maximum(np.asarray(y)-hi,0))
            require(np.max(outside)/span<=max_extrapolation_ratio,
                    'Source fragment too short or far from the measurement rows')
            weights.append(float(np.linalg.norm(points[1]-points[0])**2));segments.append(points.tolist())
        center=np.average(values,axis=0,weights=weights)
        spread=float(np.max(np.abs(np.asarray(values)-center)))
        require(spread<=max_spread_px,'Reviewed source edges disagree beyond the declared uncertainty')
        result[index]=center
        reports.append({'proposal_ids':ids,'source_segments':segments,'probe_x':center.tolist(),
                        'measurement_kind':'edge_group_proposal',
                        'max_edge_deviation_px':spread})
    require(len(used)==len(set(used)),'A source edge cannot count as two independent physical parts')
    return result,reports
