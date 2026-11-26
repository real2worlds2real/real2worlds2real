"""Conservative sampling screen for a particle representation of sheet contact."""
import numpy as np


def native_probe_coverage(report, *, secondary_approximation=False):
    """Sampling is required; a conservative skin-overlap proxy is diagnostic.

    Particle radii are numerical envelopes, not the physical sheet thickness.
    Their overlap with a reference-distant vertex does not establish either
    the engine's pair filtering or actual material intersection. Keep that
    raw rejection, then require actual finite-thickness, native contact and
    unforced response checks instead of preventing those checks from running.
    This policy is for the diagnostic native probe, not asset approval.
    A separately validated secondary-part declaration may use the existing
    bounded sampling allowance; the raw physical-coverage failure is retained.
    """
    if type(secondary_approximation) is not bool:
        raise ValueError('Explicit semantic secondary-approximation decision required')
    if report.get('particle_sampling_required') is False:
        return {'pass':True,'applicable':False,'task_approved':False}
    physical=report.get('physical_coverage_pass') is True
    gap=report.get('maximum_approximation_tolerance_m',0.)
    approximate=bool(secondary_approximation and report.get('pass') is True
                     and np.isfinite(gap) and 0<gap<=.002)
    return {'pass':physical or approximate,'applicable':True,
            'physical_coverage_pass':physical,'secondary_approximation_applied':approximate and not physical,
            'initial_numerical_envelope_pass':report.get('initial_nonlocal_envelopes',{}).get('pass'),
            'native_contact_and_release_required':True,'task_approved':False}


def initial_overlaps(points,rest,solid_rest_offset,*,tolerance=1e-6,maximum_pairs=1<<24,proposal_candidates=False):
    """Conservative nonlocal particle-envelope screen, not native filtering.

    Reference-near points are excluded explicitly. This does not emulate the
    undocumented native pair-filter cutoff or approve actual collision forces.
    """
    from scipy.spatial import cKDTree
    p,r=np.asarray(points,float),np.asarray(rest,float)
    if (p.ndim!=2 or p.shape[1:]!=(3,) or r.shape!=p.shape or not len(p)
            or not np.isfinite(p).all() or not np.isfinite(r).all()
            or not np.isfinite([solid_rest_offset,tolerance]).all() or solid_rest_offset<=0
            or not 0<=tolerance<2*solid_rest_offset or type(maximum_pairs) is not int or maximum_pairs<1):
        raise ValueError('Finite matching particle/reference points and bounded positive envelope required')
    diameter=2*solid_rest_offset;exclusion=diameter;tree=cKDTree(p)
    radius=1.1*diameter if proposal_candidates else diameter
    count=int((tree.query_ball_point(p,radius,return_length=True).sum()-len(p))//2)
    base={'solid_rest_offset_m':float(solid_rest_offset),'reference_exclusion_distance_m':float(exclusion),
          'tolerance_m':float(tolerance),'candidate_pairs':count,'maximum_pairs':maximum_pairs,
          'scope':'Conservative initial nonlocal envelope compatibility only; no native filter, force, material or task approval'}
    if count>maximum_pairs:return {**base,'pass':False,'reason':'Particle envelope query exceeds the explicit pair budget'}
    pairs=tree.query_pairs(radius,output_type='ndarray')
    distance=np.linalg.norm(p[pairs[:,0]]-p[pairs[:,1]],axis=1)
    reference=np.linalg.norm(r[pairs[:,0]]-r[pairs[:,1]],axis=1)
    nonlocal_=reference>exclusion;bad=nonlocal_&(distance<diameter-tolerance)
    result={**base,'pass':not bool(bad.any()),'violating_nonlocal_pairs':int(bad.sum()),
            'maximum_envelope_overlap_m':float(np.maximum(0.,diameter-distance[nonlocal_]).max()) if nonlocal_.any() else 0.,
            'examples':pairs[bad][:8].tolist()}
    if proposal_candidates:result['nearby_nonlocal_pairs']=pairs[nonlocal_]
    return result


def _obtuse_covering_radius(triangles):
    """Exact vertex covering radius on obtuse triangles, with roundoff padding.

    Inside each nearest-vertex Voronoi cell, squared distance is convex, so
    its maximum is at a cell vertex. An obtuse triangle's circumcenter lies
    outside the triangle: candidates are its boundary/bisector intersections.
    The original triangle vertices have distance zero. Work in a local frame
    and bounded chunks; neither a longest-edge circle nor dense point samples
    are an exact coverage test.
    """
    result=np.empty(len(triangles))
    for start in range(0,len(triangles),16384):
        tri=triangles[start:start+16384]
        tri=tri-tri[:,:1]
        radius=np.zeros(len(tri))
        for a,b in [(0,1),(1,2),(2,0)]:
            origin=tri[:,a];edge=tri[:,b]-origin
            for i,j in [(0,1),(1,2),(2,0)]:
                vi,vj=tri[:,i]-origin,tri[:,j]-origin
                denominator=2*np.einsum('ij,ij->i',edge,vj-vi)
                valid=denominator!=0
                t=np.divide(np.sum(vj*vj-vi*vi,axis=1),denominator,
                            out=np.zeros_like(denominator),where=valid)
                valid&=(t>=0)&(t<=1)
                # Invalid intersections are excluded rather than projected
                # onto an edge, which could invent a Voronoi boundary.
                point=origin+np.where(valid,t,0)[:,None]*edge
                nearest=np.linalg.norm(point[:,None]-tri,axis=2).min(1)
                radius=np.maximum(radius,np.where(valid,nearest,0))
        scale=np.linalg.norm(tri-np.roll(tri,1,axis=1),axis=2).max(1)
        result[start:start+len(tri)]=radius+64*np.finfo(float).eps*scale
    return result


def coverage(points,faces,solid_rest_offset,*,maximum_gap_m=0.):
    """Vertex contact spheres must cover their source triangle interiors.

    A sufficiently dense mesh does not prove native collision correctness or
    continuous contact. It removes one obvious source of unsampled holes. Never
    enlarge physical thickness just to pass this check; refine the geometry or
    validate a representation with actual face contact instead.
    """
    p,f=np.asarray(points,float),np.asarray(faces)
    if (p.ndim!=2 or p.shape[1:]!=(3,) or not np.isfinite(p).all() or f.ndim!=2
            or f.shape[1:]!=(3,) or not len(f) or f.dtype.kind not in 'iu'
            or f.min()<0 or f.max()>=len(p) or not np.isfinite(solid_rest_offset)
            or solid_rest_offset<=0):
        raise ValueError('Valid finite particle sheet geometry and positive rest radius required')
    tri=p[f];a=np.linalg.norm(tri[:,1]-tri[:,0],axis=1)
    b=np.linalg.norm(tri[:,2]-tri[:,1],axis=1);c=np.linalg.norm(tri[:,0]-tri[:,2],axis=1)
    double_area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    if (double_area<=1e-14).any():raise ValueError('Degenerate particle cloth triangle')
    circum=a*b*c/(2*double_area)
    edges=np.sort(np.c_[a,b,c],axis=1)
    # An obtuse triangle lies in the circle on its longest side. Circumradius
    # becomes arbitrarily pessimistic for skinny obtuse faces. Both circles
    # conservatively cover distances to the nearest material vertex: the
    # barycentric average squared distance is at most the enclosing radius^2.
    obtuse=edges[:,2]**2>edges[:,0]**2+edges[:,1]**2
    enclosing=np.where(obtuse,edges[:,2]/2,circum)
    radii=circum.copy()
    # The enclosing circle can overestimate the actual gap on a skinny
    # obtuse face: its third vertex may cover the longest edge's midpoint.
    # Refine the bound analytically, retaining both old diagnostics.
    radii[obtuse]=_obtuse_covering_radius(tri[obtuse])
    gap=np.asarray(maximum_gap_m,float)
    if gap.shape not in [(),(len(f),)] or not np.isfinite(gap).all() or (gap<0).any() or (gap>.002).any():
        raise ValueError('Explicit per-triangle secondary approximation at most2mm required')
    ratio=radii/solid_rest_offset
    physical=bool((ratio<=1+1e-5).all())
    return {'pass':bool((radii<=solid_rest_offset+gap+1e-10).all()),
            'physical_coverage_pass':physical,'maximum_triangle_circumradius_m':float(circum.max()),
            'maximum_triangle_enclosing_radius_m':float(enclosing.max()),
            'maximum_triangle_vertex_covering_radius_m':float(radii.max()),
            'coverage_measurement':'Maximum distance to the nearest triangle vertex over its full surface; obtuse boundary Voronoi intersections are evaluated analytically',
            'maximum_unsampled_distance_bound_m':float(np.maximum(0.,radii-solid_rest_offset).max()),
            'maximum_approximation_tolerance_m':float(gap.max()),
            'solid_rest_offset_m':float(solid_rest_offset),'maximum_coverage_ratio':float(ratio.max()),
            'uncovered_triangles':int((ratio>1+1e-5).sum()),'triangles':len(f),
            'scope':'Conservative particle sampling screen only; native self-contact, thickness, CCD and task tests still required'}


def rigid_coverage(points,faces,contact,*,maximum_gap_m=0.):
    """Cover non-particle contact with its actual native restOffset.

    PhysX5.1 separates restOffset (rigid/deformable partners) from
    solidRestOffset (particle partners). Changing the latter cannot fill a
    hole in the rigid-contact envelope. Keep coverage()'s legacy geometric
    API, but bind native rigid checks to explicitly named contact settings.
    """
    if not isinstance(contact,dict) or not {'rest_offset','contact_offset'}<=contact.keys():
        raise ValueError('Explicit native non-particle rest/contact offsets required')
    rest,offset=contact['rest_offset'],contact['contact_offset']
    if not np.isfinite([rest,offset]).all() or not 0<rest<offset:
        raise ValueError('Finite positive rigid rest offset below its contact offset required')
    result=coverage(points,faces,rest,maximum_gap_m=maximum_gap_m)
    result['rest_offset_m']=result.pop('solid_rest_offset_m')
    result['contact_partner']='non_particle'
    result['scope']='Conservative native rigid/deformable-partner particle sampling only; self-contact, thickness, CCD and task tests remain separate'
    return result
