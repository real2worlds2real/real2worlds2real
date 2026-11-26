"""Unscaled construction-footprint proposals on actual planar triangles.

Holes and disconnected regions remain empty. This is an explicit bounded
translation search, not physical contact, material or native-state approval.
"""
import numpy as np


def surface_placement(footprint,support_triangles,*,clearance_m,step_m,max_candidates=32768):
    from scipy.spatial import ConvexHull
    import shapely
    p,t=np.asarray(footprint,float),np.asarray(support_triangles,float)
    if (p.ndim!=2 or p.shape[1:]!=(2,) or len(p)<3 or not np.isfinite(p).all()
            or t.ndim!=3 or t.shape[1:]!=(3,2) or not len(t) or not np.isfinite(t).all()
            or not np.isfinite([clearance_m,step_m]).all() or clearance_m<0 or step_m<=0
            or type(max_candidates) is not int or not 1<=max_candidates<=131072
            or np.linalg.matrix_rank(p-p.mean(0),tol=1e-10)<2):
        raise ValueError('Resolved finite planar footprint, actual triangles and bounded explicit search required')
    tri=t[:,1:]-t[:,:1]
    if (abs(tri[:,0,0]*tri[:,1,1]-tri[:,0,1]*tri[:,1,0])<=1e-16).any():
        raise ValueError('Actual planar support triangles must have resolved area')
    surface=shapely.union_all(shapely.polygons(t))
    if surface.is_empty or not surface.is_valid:raise ValueError('Actual planar support union is invalid')
    centre=(p.min(0)+p.max(0))/2;p=p-centre
    hull=p[ConvexHull(p).vertices]
    low=t.min((0,1))-p.min(0);high=t.max((0,1))-p.max(0)
    report={'proposal_only':True,'task_approved':False,'contact_proven':False,
            'actual_surface_footprint_fit':False,'geometry_scaled':False,'collision_holes_filled':False,
            'support_triangles':len(t),'required_outline_clearance_m':float(clearance_m),
            'search_step_m':float(step_m),'maximum_candidates':max_candidates,
            'scope':'Translation of the unchanged convex footprint on the union of actual planar support triangles. No hole filling. Native contact, material, deformation and released-state checks remain required.'}
    if (low>high).any():return {**report,'tested_candidates':0,'reason':'Footprint exceeds support bounds'}
    counts=np.ceil((high-low)/step_m).astype(int)+1;count=int(np.prod(counts))
    if count>max_candidates:
        return {**report,'tested_candidates':0,'candidate_count':count,'reason':'Explicit search budget exceeded; no infeasibility claim'}
    x,y=np.meshgrid(*(np.linspace(a,b,n) for a,b,n in zip(low,high,counts)))
    centres=np.c_[x.ravel(),y.ravel()]
    shapes=shapely.polygons(hull[None]+centres[:,None])
    shapely.prepare(surface)
    contained=shapely.covers(surface,shapes)
    selected=np.flatnonzero(contained)
    if not len(selected):return {**report,'tested_candidates':count,'reason':'No complete footprint fits actual triangles in this bounded search'}
    # Exact segment distances avoid approximating clearance by a polygonal
    # buffer around an aperture. Containment alone could touch an outlet edge.
    margin=shapely.distance(shapely.boundary(surface),shapes[selected])
    best=int(np.argmax(margin));distance=float(margin[best])
    return {**report,'tested_candidates':count,'actual_surface_footprint_fit':bool(distance>=clearance_m-1e-12),
            'translation_xy_m':(centres[selected[best]]-centre).tolist(),
            'minimum_actual_boundary_clearance_m':distance}
