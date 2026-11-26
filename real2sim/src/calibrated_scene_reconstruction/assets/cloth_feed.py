"""Offline sheet-placement proposals at unchanged intrinsic material coordinates."""
import numpy as np


def along_columns(points,rest,distance_m,tail_direction,*,maximum_extension_m,axis=1):
    """Feed a regular material chart along its current drape, extending its tail.

    Each resolved material column is sampled farther along the existing posed
    curve. Beyond its end, the agent supplies a unit physical continuation
    direction and explicit extrapolation budget. This changes initial geometry
    only; it neither changes the source outline/rest metric nor writes a native
    state. Irregular charts without resolved columns require another method.
    New pose strain, self/solid clearance, source, support and native release
    checks remain necessary, including at the proposed continuation junction.
    """
    p,r,d=np.asarray(points,float),np.asarray(rest,float),np.asarray(tail_direction,float)
    if (p.ndim!=2 or p.shape[1:]!=(3,) or r.shape!=p.shape or not len(p)
            or not np.isfinite([p,r]).all() or d.shape!=(3,) or not np.isfinite(d).all()
            or not np.isclose(np.linalg.norm(d),1.,rtol=0.,atol=1e-8)
            or type(axis) is not int or axis not in (0,1)
            or not np.isfinite([distance_m,maximum_extension_m]).all()
            or distance_m==0 or maximum_extension_m<=0
            or abs(distance_m)>maximum_extension_m or np.ptp(r[:,2])>1e-9):
        raise ValueError('Matching planar material chart, signed feed, unit tail direction and explicit extension budget required')
    coordinate=np.sign(distance_m)*r[:,axis];lateral=r[:,1-axis]
    # Chart columns must really exist. Rounding to force an unstructured mesh
    # into columns could merge distinct material and silently corrupt the pose.
    values,inverse=np.unique(lateral,return_inverse=True)
    result=np.empty_like(p);extension=0.
    for column in range(len(values)):
        ids=np.flatnonzero(inverse==column);ids=ids[np.argsort(coordinate[ids])]
        source=coordinate[ids]
        if len(ids)<2 or np.any(np.diff(source)<=1e-10):
            raise ValueError('Feed requires distinct resolved material columns; use another placement method for this chart')
        if abs(source[0]-coordinate.min())>1e-9 or abs(source[-1]-coordinate.max())>1e-9:
            raise ValueError('Clipped column ends cannot stand in for the drape tail; use a complete shared spine')
        query=source+abs(distance_m)
        for j in range(3):result[ids,j]=np.interp(query,source,p[ids,j])
        extra=np.maximum(query-source[-1],0.)
        result[ids]+=extra[:,None]*d
        extension=max(extension,float(extra.max()))
    return result,{'proposal_only':True,'task_approved':False,'native_positions_written':False,
                   'material_coordinates_changed':False,'material_outline_changed':False,
                   'feed_distance_m':float(distance_m),'axis':axis,'material_columns':len(values),
                   'maximum_extension_m':extension,'extension_budget_m':float(maximum_extension_m),
                   'scope':'Initial geometry proposal only. Independently check all material, contact, source, support and released native behavior before adoption.'}


def along_spine(points,coordinates,spine_coordinates,spine_points,distance_m,tail_direction,
                *,width_direction,maximum_extension_m):
    """Feed an unchanged cut sheet along an explicit complete planar drape.

    A proper rotation about the supplied transverse axis transports each actual
    section offset. Individual clipped column ends never become fake drape
    endpoints. The supplied spine covers the whole existing material interval;
    its continuation is an agent hypothesis, subject to full independent checks.
    """
    from scipy.interpolate import PchipInterpolator
    p,t,s,c,d,w=map(lambda x:np.asarray(x,float),
                  [points,coordinates,spine_coordinates,spine_points,tail_direction,width_direction])
    if (p.ndim!=2 or p.shape[1:]!=(3,) or t.shape!=(len(p),) or s.ndim!=1 or len(s)<3
            or c.shape!=(len(s),3) or d.shape!=(3,) or w.shape!=(3,)
            or not all(np.isfinite(x).all() for x in [p,t,s,c,d,w])
            or (np.diff(s)<=0).any() or t.min()<s[0]-1e-10 or t.max()>s[-1]+1e-10
            or not np.allclose([np.linalg.norm(d),np.linalg.norm(w)],[1,1],rtol=0.,atol=1e-8)
            or abs(d@w)>1e-8 or not np.isfinite([distance_m,maximum_extension_m]).all()
            or not 0<distance_m<=maximum_extension_m):
        raise ValueError('Complete resolved spine, matching material samples, unit transverse/tail axes and bounded forward feed required')
    curve=PchipInterpolator(s,c,axis=0,extrapolate=False)
    def sample(query):
        clipped=np.clip(query,s[0],s[-1]);position=curve(clipped);tangent=curve.derivative()(clipped)
        beyond=query>s[-1];position[beyond]=c[-1]+(query[beyond]-s[-1])[:,None]*d
        tangent[beyond]=d
        tangent-=np.outer(tangent@w,w)
        norm=np.linalg.norm(tangent,axis=1)
        if (norm<1e-8).any():raise ValueError('Spine has an unresolved longitudinal tangent')
        return position,tangent/norm[:,None]
    before,a=sample(t);after,b=sample(t+distance_m)
    angle=np.arctan2(np.cross(a,b)@w,(a*b).sum(1))
    offset=p-before;cos=np.cos(angle)[:,None];sin=np.sin(angle)[:,None]
    result=after+cos*offset+sin*np.cross(w,offset)+(1-cos)*np.outer(offset@w,w)
    return result,{'proposal_only':True,'task_approved':False,'native_positions_written':False,
                   'material_coordinates_changed':False,'material_outline_changed':False,
                   'feed_distance_m':float(distance_m),'maximum_extension_m':float(max(0,t.max()+distance_m-s[-1])),
                   'extension_budget_m':float(maximum_extension_m),'spine_samples':len(s),
                   'maximum_section_rotation_rad':float(abs(angle).max()),
                   'scope':'Offline proper section transport along a complete drape; source, material, junction, self/solid, support and native release remain unchecked.'}


def redistribute(points,coordinates,spine_coordinates,spine_points,new_spine_points,
                 *,width_direction,maximum_trim_m):
    """Redistribute a drape at fixed longitudinal arc lengths and inner endpoint.

    For example, sag an interior bridge while shortening the exposed free end.
    The complete material stays present. Extra centreline length is consumed at
    the free end rather than scaling the sheet or extending its supported end.
    This is offline geometry authoring, not a simulated packing history.
    """
    p,t,s,a,b,w=map(lambda x:np.asarray(x,float),
                  [points,coordinates,spine_coordinates,spine_points,new_spine_points,width_direction])
    if (p.ndim!=2 or p.shape[1:]!=(3,) or t.shape!=(len(p),) or s.ndim!=1 or len(s)<3
            or a.shape!=(len(s),3) or b.shape!=a.shape or w.shape!=(3,)
            or not all(np.isfinite(x).all() for x in [p,t,s,a,b,w])
            or (np.diff(s)<=0).any() or t.min()<s[0]-1e-10 or t.max()>s[-1]+1e-10
            or not np.isclose(np.linalg.norm(w),1.,rtol=0.,atol=1e-8)
            or not np.allclose(a[-1],b[-1],rtol=0.,atol=1e-10)
            or not np.isfinite(maximum_trim_m) or maximum_trim_m<=0):
        raise ValueError('Complete matching drapes, fixed inner endpoint, unit width axis and explicit trim budget required')
    def path(c):
        edge=np.diff(c,axis=0);length=np.linalg.norm(edge,axis=1)
        if (length<1e-10).any():raise ValueError('Drape has unresolved duplicate points')
        direction=edge/length[:,None]
        tangent=np.vstack([direction[0],direction[:-1]+direction[1:],direction[-1]])
        tangent-=np.outer(tangent@w,w);norm=np.linalg.norm(tangent,axis=1)
        if (norm<1e-8).any():raise ValueError('Drape has unresolved longitudinal turn')
        return np.r_[0.,np.cumsum(length)],tangent/norm[:,None]
    aa,at=path(a);ba,bt=path(b);trim=float(ba[-1]-aa[-1])
    if not -1e-10<=trim<=maximum_trim_m:raise ValueError('New drape length exceeds the unchanged-material trim budget')
    trim=max(0.,trim);old_arc=np.interp(t,s,aa);new_arc=np.minimum(old_arc+trim,ba[-1])
    interp=lambda values,query,domain:np.column_stack([np.interp(query,domain,values[:,j])for j in range(3)])
    before=interp(a,old_arc,aa);after=interp(b,new_arc,ba)
    u=interp(at,old_arc,aa);v=interp(bt,new_arc,ba)
    u/=np.linalg.norm(u,axis=1)[:,None];v/=np.linalg.norm(v,axis=1)[:,None]
    angle=np.arctan2(np.cross(u,v)@w,(u*v).sum(1));offset=p-before
    cos=np.cos(angle)[:,None];sin=np.sin(angle)[:,None]
    result=after+cos*offset+sin*np.cross(w,offset)+(1-cos)*np.outer(offset@w,w)
    return result,{'proposal_only':True,'task_approved':False,'native_positions_written':False,
                   'material_coordinates_changed':False,'material_outline_changed':False,
                   'old_spine_length_m':float(aa[-1]),'new_spine_length_m':float(ba[-1]),
                   'free_end_trim_m':trim,'maximum_trim_m':float(maximum_trim_m),
                   'supported_spine_endpoint_preserved':True,
                   'scope':'Arc-length based initial-pose proposal. Full surface material, support, self/solid, source and native checks still required.'}
