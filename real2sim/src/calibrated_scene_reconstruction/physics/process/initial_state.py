"""Source-grounded initial part geometry and pose observations.

Helpers are selected by part geometry, not object/task names. Hidden contacts
are explicit model priors. Fitting observations does not certify native stability.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares,minimize
from scipy.spatial.transform import Rotation

from .contract import require


def project(points,camera):
    K,D,T=camera;R=T[:3,:3].T
    return cv2.projectPoints(np.asarray(points,float),cv2.Rodrigues(R)[0],-R@T[:3,3],K,D)[0].reshape(-1,2)


def segment_distance(a,b,c,d):
    """Closest solid-axis distance, including parallel and endpoint cases."""
    a,b,c,d=map(lambda v:np.asarray(v,float),(a,b,c,d))
    u,v,w=b-a,d-c,a-c;aa,bb,cc,dd,ee=u@u,u@v,v@v,u@w,v@w
    require(aa>0 and cc>0,'Nonzero segment axes required')
    clamp=lambda x:float(np.clip(x,0,1));pairs=[];det=aa*cc-bb*bb
    if det>1e-18:
        s,t=(bb*ee-cc*dd)/det,(aa*ee-bb*dd)/det
        if 0<=s<=1 and 0<=t<=1:pairs.append((s,t))
    pairs.extend([(0,clamp(ee/cc)),(1,clamp((ee+bb)/cc)),(clamp(-dd/aa),0),(clamp((bb-dd)/aa),1)])
    return min(float(np.linalg.norm(w+s*u-t*v)) for s,t in pairs)


def fit_upright_landmarks(points,observations,cameras,initial_pose,*,fit_indices,
                          validation_indices,translation_bounds,yaw_bound,pivot=(0,0,0),
                          point_sigma_px=1.,region_terms=()):
    """Fit static material landmarks with an explicit upright support prior.

    Asset identities must be independently checked in native rendered views.
    This estimates one initial pose, never a prescribed response trajectory.
    """
    points=np.asarray(points,float);observations=np.asarray(observations,float)
    fit=np.asarray(fit_indices,int);held=np.asarray(validation_indices,int)
    bounds=np.asarray(translation_bounds,float);B=np.asarray(initial_pose,float);pivot=np.asarray(pivot,float)
    require(points.ndim==2 and points.shape[1]==3 and observations.shape==(len(cameras),len(points),2),
            'Static observations and model identities must match')
    require(len(fit)>=3 and len(held)>=2 and not set(fit)&set(held),'Independent static view split required')
    require(bounds.shape==(3,2) and (bounds[:,0]<0).all() and (bounds[:,1]>0).all() and 0<yaw_bound<=np.pi,
            'Explicit supported pose bounds required')
    def pose(x):
        out=B.copy();out[:3,:3]=Rotation.from_euler('z',x[3]).as_matrix()@B[:3,:3]
        out[:3,3]=B[:3,3]+x[:3]+B[:3,:3]@pivot-out[:3,:3]@pivot
        return out
    def residual(x,indices):
        P=pose(x);world=points@P[:3,:3].T+P[:3,3]
        errors=np.array([project(world,cameras[i])-observations[i] for i in indices])
        valid=np.isfinite(errors).all(2)
        require(valid.sum()>=6,'Insufficient source-static material observations')
        return errors[valid].ravel()
    require(point_sigma_px>0,'Positive point observation uncertainty required')
    def combined(x):
        rows=[residual(x,fit)/point_sigma_px];P=pose(x)
        for term in region_terms:
            from .material_projection import project_material
            observed=np.asarray(term['observations'],float)
            require(observed.shape==(len(cameras),2) and term['sigma_px']>0,'Explicit region observation uncertainty required')
            ids=[i for i in fit if np.isfinite(observed[i]).all()]
            require(len(ids)>=3,'Insufficient static source region observations')
            predicted=np.array([project_material(term['vertices'],term['faces'],P[:3,3],P[:3,:3],cameras[i],
                opening=term.get('opening',(1,1)))[0] for i in ids])
            rows.append(((predicted-observed[ids])/term['sigma_px']).ravel())
        return np.concatenate(rows)
    lower=np.r_[bounds[:,0],-yaw_bound];upper=np.r_[bounds[:,1],yaw_bound]
    solved=least_squares(lambda x:residual(x,fit),np.zeros(4),bounds=(lower,upper),max_nfev=300)
    if region_terms:
        # Pixel raster regions are quantized; numerical infinitesimal Jacobians
        # would incorrectly report zero sensitivity for those observations.
        solved=minimize(lambda x:float(np.mean(combined(x)**2)),solved.x,method='Powell',
            bounds=list(zip(lower,upper)),options={'maxiter':80,'xtol':1e-6,'ftol':1e-7})
    return {'pose':pose(solved.x),'translation_world':solved.x[:3], 'yaw_rad':float(solved.x[3]),
        'fit_rmse_px':float(np.sqrt(np.mean(residual(solved.x,fit)**2))),
        'validation_rmse_px':float(np.sqrt(np.mean(residual(solved.x,held)**2))),
        'optimizer_converged':bool(solved.success)}


def cylinder_surface_samples(bottom,top,radius,*,spacing,angles=64):
    """Sample a reviewed flat-ended circular shaft, including its middle.

    This supplements a sparse mesh's end-ring vertices for solid preparation.
    It does not change render/collision geometry and is not a native certificate.
    Use only when the reviewed part really is a constant-radius cylinder.
    """
    a,b=np.asarray(bottom,float),np.asarray(top,float);axis=b-a;length=np.linalg.norm(axis)
    require(length>0 and radius>0 and spacing>0 and type(angles) is int and angles>=16,'Finite cylinder sampling dimensions required')
    count=int(np.ceil(length/spacing))+1
    require(count*angles<=200000,'Cylinder surface sample budget exceeded')
    axis/=length;reference=np.eye(3)[np.argmin(abs(axis))];u=np.cross(axis,reference);u/=np.linalg.norm(u);v=np.cross(axis,u)
    phase=np.arange(angles)*2*np.pi/angles;radial=radius*(np.cos(phase)[:,None]*u+np.sin(phase)[:,None]*v)
    centers=a+np.linspace(0,length,count)[:,None]*axis
    return np.vstack([(centers[:,None]+radial).reshape(-1,3),a,b])


def fit_region_translation(vertices,faces,initial_pose,cameras,centers,*,fit_indices,validation_indices,
                           basis=None,bounds=(-.05,.05),opening=(1,1)):
    """Align an observed part region with supported translation, fixed geometry."""
    from scipy.optimize import minimize
    from .material_projection import project_material
    basis=np.eye(3)[:,:2] if basis is None else np.asarray(basis,float)
    require(basis.shape==(3,2) and np.linalg.matrix_rank(basis)==2,'Independent supported translation basis required')
    ids=np.asarray(fit_indices,int);held=np.asarray(validation_indices,int);centers=np.asarray(centers,float)
    require(len(ids)>=3 and len(held)>=2 and not set(ids)&set(held),'Disjoint source pose views required')
    def predict(x,indices):
        return np.array([project_material(vertices,faces,initial_pose[:3,3]+basis@x,initial_pose[:3,:3],cameras[i],opening=opening)[0]
                         for i in indices])
    solved=minimize(lambda x:float(np.mean((predict(x,ids)-centers[ids])**2)),np.zeros(2),method='Powell',
                    bounds=[bounds,bounds],options={'xtol':1e-5,'ftol':1e-5,'maxiter':50})
    pose=np.array(initial_pose,copy=True);pose[:3,3]+=basis@solved.x
    return {'pose':pose,'fit_rmse_px':float(np.sqrt(np.mean((predict(solved.x,ids)-centers[ids])**2))),
            'validation_rmse_px':float(np.sqrt(np.mean((predict(solved.x,held)-centers[held])**2))),
            'geometry_changed':False,'optimizer_converged':bool(solved.success)}


def straight_branches(mask,anchor,*,count,min_length,tip_distance):
    """Observe independently visible straight branches meeting near an anchor.

    The source mask and anchor may come from an agent-reviewed segmenter. No
    simulated shape supplies the image branch directions or endpoints.
    """
    from skimage.morphology import skeletonize
    ys,xs=np.nonzero(mask);require(len(xs)>0,'Missing source part mask')
    lo=np.maximum([xs.min()-2,ys.min()-2],0);hi=np.minimum([xs.max()+3,ys.max()+3],[mask.shape[1],mask.shape[0]])
    small=mask[lo[1]:hi[1],lo[0]:hi[0]]>0
    skel=skeletonize(small).astype(np.uint8)
    neighbors=cv2.filter2D(skel,-1,np.ones((3,3),np.uint8))
    y,x=np.nonzero((skel>0)&(neighbors==2));tips=np.c_[x,y]
    root=np.asarray(anchor)-lo
    tips=[p for p in tips if np.linalg.norm(p-root)>=tip_distance]
    # Independent tip identity is ordered by source height; callers must review
    # temporal association when branches cross or occlude one another.
    tips=sorted(tips,key=lambda p:p[1])
    require(len(tips)==count,'Source straight-branch identity is ambiguous')
    result=[]
    for tip in tips:
        path=[tip];previous=None;current=tip
        for _ in range(int(skel.sum())+1):
            if np.linalg.norm(current-root)<min_length*.25:break
            options=[]
            for dy in [-1,0,1]:
                for dx in [-1,0,1]:
                    p=current+[dx,dy]
                    if (dx or dy) and 0<=p[0]<skel.shape[1] and 0<=p[1]<skel.shape[0] and skel[p[1],p[0]]:
                        if previous is None or not np.array_equal(p,previous):options.append(p)
            if len(options)!=1:break
            previous,current=current,options[0];path.append(current)
        points=np.asarray(path,float)
        require(len(points)>=min_length,'Insufficient independent straight branch support')
        # Exclude the junction/lower shared part from the direction estimate.
        fit=points[:max(10,round(len(points)*.8))]
        direction=cv2.fitLine(fit.astype(np.float32),cv2.DIST_HUBER,0,.01,.01).ravel()[:2]
        if direction[1]<0:direction=-direction
        require(direction[1]>.1,'Unsupported near-horizontal branch; choose another observer')
        origin=points[0];foot=origin+direction*(root[1]-origin[1])/direction[1]
        result.append({'tip':origin+lo,'direction':direction,'anchor_line_point':foot+lo,'support':len(fit)})
    return result


def fit_contact_segment(tips,directions,cameras,*,length,radius,floor_z,opening_center,opening_z,opening_radius,
                        fit_indices,validation_indices,seed=None,length_bounds=None,support_radius=None,other_segments=(),pair_margin=5e-5,
                        radial_contact=False,contact_angle_limit=None):
    """A straight solid rests on a plane and leans on a circular aperture.

    Dimensions come from the reviewed model, never measured real dimensions.
    The three fitted degrees of freedom specify independent contact azimuth and
    line direction. Contacts and stability must subsequently pass native tests.
    """
    tips,directions=np.asarray(tips,float),np.asarray(directions,float)
    require(length>0 and 0<radius<opening_radius,'Invalid reviewed segment geometry')
    ids=np.asarray(fit_indices,int);held=np.asarray(validation_indices,int)
    require(len(ids)>=3 and len(held)>=2 and not set(ids)&set(held),'Disjoint initial-state fit/held views required')
    center=np.asarray(opening_center,float)
    require(contact_angle_limit is None or 0<contact_angle_limit<np.pi/2,
            'Declare a physical circular-contact angular prior below pi/2')
    def contact_angle(x):
        if radial_contact or np.linalg.norm(x[:2])<1e-10:return 0.
        slope=np.asarray(x[:2]);normal=np.array([np.cos(x[2]),np.sin(x[2])])
        return abs(np.arctan2(slope[0]*normal[1]-slope[1]*normal[0],slope@normal))
    def geometry(x):
        current_length=x[-1] if length_bounds is not None else length
        slope=np.array(x[:2]);direction=np.r_[slope,1.];direction/=np.linalg.norm(direction)
        bottom_z=floor_z+radius*np.linalg.norm(direction[:2])+1e-5
        # Circular silhouette envelope of a slanted cylinder at the aperture.
        envelope=radius*np.sqrt(1+np.dot(slope,slope))
        normal=slope/max(np.linalg.norm(slope),1e-12) if radial_contact else np.array([np.cos(x[2]),np.sin(x[2])])
        rim=center+(opening_radius-envelope-2e-5)*normal
        bottom=np.r_[rim-slope*(opening_z-bottom_z),bottom_z]
        return bottom,bottom+direction*current_length,direction
    def predict(x,indices):
        bottom,top,d=geometry(x);uv=[];angles=[]
        for i in indices:
            points=project([top,bottom],cameras[i]);delta=points[1]-points[0]
            uv.append(points[0]);angles.append(np.arctan2(delta[0],delta[1]))
        return np.asarray(uv),np.asarray(angles)
    observed=np.arctan2(directions[:,0],directions[:,1])
    def residual(x):
        uv,angles=predict(x,ids)
        residuals=np.r_[((uv-tips[ids])/2.).ravel(),np.arctan2(np.sin(angles-observed[ids]),np.cos(angles-observed[ids]))/.02]
        if support_radius is not None:
            bottom,_,_=geometry(x)
            residuals=np.r_[residuals,max(0.,np.linalg.norm(bottom[:2]-center)+radius-support_radius)/.0005]
        return residuals
    solutions=[]
    for phi in np.linspace(-np.pi,np.pi,9)[:-1]:
        initial=np.r_[seed[:2] if seed is not None else [0.,0.],phi]
        low=[-1.2,-1.2,-2*np.pi];high=[1.2,1.2,2*np.pi]
        if radial_contact:
            # A freely seated, axisymmetric support has no preferred azimuth.
            # Keep the shaft and contact normal in a common radial plane,
            # avoiding a tangential contact torque in the initial hypothesis.
            initial=.3*np.array([np.cos(phi),np.sin(phi)]);low=[-1.2,-1.2];high=[1.2,1.2]
        if length_bounds is not None:
            require(0<length_bounds[0]<=length<=length_bounds[1],'Geometry length seed outside declared modeling bounds')
            initial=np.r_[initial,length];low.append(length_bounds[0]);high.append(length_bounds[1])
        result=least_squares(residual,initial,bounds=(low,high),loss='soft_l1',max_nfev=180)
        if support_radius is not None or other_segments or contact_angle_limit is not None:
            def clearance(x):
                bottom,top,_=geometry(x);values=[]
                if support_radius is not None:values.append(support_radius-radius-np.linalg.norm(bottom[:2]-center))
                values.extend(segment_distance(bottom,top,a,b)-radius-r-pair_margin for a,b,r in other_segments)
                if contact_angle_limit is not None:values.append((contact_angle_limit-contact_angle(x))*.01)
                return np.asarray(values)
            constrained=minimize(lambda x:float(np.sum(2*(np.sqrt(1+residual(x)**2)-1))),result.x,
                method='SLSQP',bounds=list(zip(low,high)),constraints=[{'type':'ineq','fun':clearance}],
                options={'maxiter':250,'ftol':1e-8})
            if np.min(clearance(constrained.x))<-1e-6:continue
            result=constrained
        solutions.append(result)
    require(bool(solutions),'No initial contact hypothesis satisfies the reviewed support footprint')
    result=min(solutions,key=lambda r:np.mean(residual(r.x)**2));bottom,top,d=geometry(result.x)
    metrics={}
    for name,indices in [('fit',ids),('validation',held)]:
        uv,angles=predict(result.x,indices);delta=np.arctan2(np.sin(angles-observed[indices]),np.cos(angles-observed[indices]))
        metrics[name]={'tip_rmse_px':float(np.sqrt(np.mean((uv-tips[indices])**2))),
            'axis_rmse_deg':float(np.degrees(np.sqrt(np.mean(delta**2))))}
    return {'bottom':bottom,'top':top,'direction':d,'parameters':result.x,'metrics':metrics,
            'length':float(np.linalg.norm(top-bottom)),
            'contact_angle_limit_rad':contact_angle_limit,'contact_misalignment_rad':float(contact_angle(result.x)),
            'geometry_refinement_requested':length_bounds is not None,
            'native_stability_checked':False,'scope':'Source-image initial contact hypothesis with reviewed geometry and explicit support/aperture priors.'}


def rigid_segment_pose(local_bottom,local_top,world_bottom,world_top,*,twist=0.):
    """Transfer a reviewed rigid segment without changing its length or scale."""
    a=np.asarray(local_top)-local_bottom;b=np.asarray(world_top)-world_bottom
    require(np.isclose(np.linalg.norm(a),np.linalg.norm(b),atol=1e-6,rtol=1e-4),'Initial pose cannot change reviewed segment length')
    rotation=Rotation.align_vectors([b/np.linalg.norm(b)],[a/np.linalg.norm(a)])[0].as_matrix()
    rotation=Rotation.from_rotvec(b/np.linalg.norm(b)*twist).as_matrix()@rotation
    M=np.eye(4);M[:3,:3]=rotation;M[:3,3]=world_bottom-rotation@local_bottom
    return M


def aperture_assembly_pose(local_bottom,local_top,head_points,head_centers,head_extents,source_lines,cameras,*,
                           radius,floor_z,opening_center,opening_z,opening_radius,support_radius,
                           fit_indices,validation_indices,other_segments=()):
    """Initial rigid assembly pose from a visible shaft and attached region.

    The model is unchanged: pose, contact azimuth and axial twist are inferred
    together. Actual native settling and projected visible contours remain
    separate checks, including for a flexible region initialized on the assembly.
    """
    from scipy.spatial import ConvexHull
    head_points=np.asarray(head_points,float)
    hull=ConvexHull(head_points);head_points=head_points[hull.vertices]
    ids=np.asarray(fit_indices,int);held=np.asarray(validation_indices,int)
    local_bottom=np.asarray(local_bottom,float);local_top=np.asarray(local_top,float)
    length=np.linalg.norm(local_top-local_bottom);center=np.asarray(opening_center,float)
    local_axis=(local_top-local_bottom)/length
    def geometry(x):
        slope=np.array(x[:2]);d=np.r_[slope,1.];d/=np.linalg.norm(d)
        z=floor_z+radius*np.linalg.norm(d[:2])+1e-5
        rim=center+(opening_radius-radius*np.sqrt(1+slope@slope)-2e-5)*np.array([np.cos(x[2]),np.sin(x[2])])
        bottom=np.r_[rim-slope*(opening_z-z),z]
        M=rigid_segment_pose(local_bottom,local_top,bottom,bottom+d*length,twist=x[3])
        return M,bottom,d
    def predictions(x,indices):
        M,_,_=geometry(x);points=head_points@M[:3,:3].T+M[:3,3]
        axis=np.stack([local_bottom,local_top])@M[:3,:3].T+M[:3,3]
        centers=[];extents=[];line_errors=[]
        for i in indices:
            uv=project(points,cameras[i]);polygon=cv2.convexHull(uv.astype(np.float32));moments=cv2.moments(polygon)
            centers.append([moments['m10']/moments['m00'],moments['m01']/moments['m00']]);extents.append(np.ptp(uv,axis=0))
            projected=project(axis,cameras[i]);n,c=source_lines[i,:2],source_lines[i,2]
            line_errors.append(projected@n+c)
        return np.asarray(centers),np.asarray(extents),np.asarray(line_errors)
    def residual(x):
        c,e,line=predictions(x,ids)
        return np.r_[((c-np.asarray(head_centers)[ids])/2.).ravel(),
                     ((e-np.asarray(head_extents)[ids])/5.).ravel(),(line/2.).ravel()]
    def clearance(x):
        _,bottom,d=geometry(x);top=bottom+d*length
        values=[support_radius-radius-np.linalg.norm(bottom[:2]-center)]
        values.extend(segment_distance(bottom,top,a,b)-radius-r-5e-5 for a,b,r in other_segments)
        return np.asarray(values)
    low=[-.7,-.7,-2*np.pi,-2*np.pi];high=[.7,.7,2*np.pi,2*np.pi]
    results=[]
    for phi in [-np.pi,-np.pi/2,0,np.pi/2]:
        for twist in [-np.pi,-np.pi/2,0,np.pi/2]:
            seed=[.05,-.05,phi,twist]
            fitted=least_squares(residual,seed,bounds=(low,high),loss='soft_l1',max_nfev=100)
            bounded=minimize(lambda x:float(np.sum(2*(np.sqrt(1+residual(x)**2)-1))),fitted.x,
                method='SLSQP',bounds=list(zip(low,high)),constraints=[{'type':'ineq','fun':clearance}],
                options={'ftol':1e-7,'maxiter':180})
            if np.min(clearance(bounded.x))>=-1e-6:results.append(bounded)
    require(results,'No supported assembly pose matches the declared contact model')
    solved=min(results,key=lambda r:np.sum(2*(np.sqrt(1+residual(r.x)**2)-1)))
    M,bottom,d=geometry(solved.x);metrics={}
    for name,indices in [('fit',ids),('validation',held)]:
        c,e,line=predictions(solved.x,indices)
        metrics[name]={'head_center_rmse_px':float(np.sqrt(np.mean((c-np.asarray(head_centers)[indices])**2))),
                       'head_extent_rmse_px':float(np.sqrt(np.mean((e-np.asarray(head_extents)[indices])**2))),
                       'shaft_line_rmse_px':float(np.sqrt(np.mean(line**2)))}
    return {'pose':M,'bottom':bottom,'direction':d,'parameters':solved.x,'metrics':metrics,
            'support_clearance_m':float(np.min(clearance(solved.x))),'geometry_changed':False,'native_stability_checked':False}


def seat_rigid_pose(vertices,pose,fixtures,*,floor_z=None,margin=1e-5,max_translation=.005,iterations=15,
                    capsule=None,other_capsules=(),minimum_z=None):
    """Minimum rigid translation out of reviewed solid fixtures, never mesh warping.

    Vertex clearance is a preparation check, not a replacement for native full
    contact checks. Caller supplies all relevant rigid surfaces and support.
    """
    import trimesh
    from scipy.optimize import minimize
    pose=np.asarray(pose,float).copy();original=pose[:3,3].copy();vertices=np.asarray(vertices,float)
    for _ in range(iterations):
        points=vertices@pose[:3,:3].T+pose[:3,3];normals=[];depths=[];peak=0.;vertex_peak=0.
        for fixture in fixtures:
            bounds=np.asarray(fixture.bounds);inside_box=np.all((points>=bounds[0]-margin)&(points<=bounds[1]+margin),axis=1)
            ids=np.flatnonzero(inside_box)
            if not len(ids):continue
            # A large point batch near a dense mesh can create enormous
            # intermediate triangle candidate arrays. Bound query memory.
            signed=np.concatenate([trimesh.proximity.signed_distance(fixture,points[part])
                                   for part in np.array_split(ids,max(1,int(np.ceil(len(ids)/128))))])
            peak=max(peak,float(signed.max()));vertex_peak=max(vertex_peak,float(signed.max()))
            selected=ids[signed>margin*.5]
            if len(selected):
                # Keep the deepest points across the contact patch, not only
                # one vertex whose correction can worsen another boundary.
                select_order=np.argsort(signed[signed>margin*.5])[-128:];selected=selected[select_order]
                near,dist,_=trimesh.proximity.closest_point(fixture,points[selected])
                n=(near-points[selected])/np.maximum(dist[:,None],1e-10)
                normals.extend(n);depths.extend(dist+margin)
        if capsule is not None:
            a,b,r=capsule;a,b=np.asarray([a,b])@pose[:3,:3].T+pose[:3,3]
            for c,d,s in other_capsules:
                gap=segment_distance(a,b,c,d)-r-s;peak=max(peak,-gap)
                if gap>=margin*.5:continue
                epsilon=1e-6
                gradient=np.array([(segment_distance(a+e*epsilon,b+e*epsilon,c,d)-
                    segment_distance(a-e*epsilon,b-e*epsilon,c,d))/(2*epsilon) for e in np.eye(3)])
                require(np.linalg.norm(gradient)>1e-6,'Coincident capsule axes require a new initial hypothesis')
                normals.append(gradient);depths.append(margin-gap)
        if minimum_z is not None:
            depth=minimum_z+margin-float(points[:,2].min());normals.append([0,0,1]);depths.append(depth)
            peak=max(peak,max(0.,depth))
        if floor_z is not None:
            delta_z=floor_z+margin-float(points[:,2].min())
            normals.extend([[0,0,1],[0,0,-1]]);depths.extend([delta_z,-delta_z])
        if peak<=margin and (floor_z is None or abs(points[:,2].min()-floor_z-margin)<2e-6):
            return {'pose':pose,'translation_m':pose[:3,3]-original,'max_vertex_penetration_m':max(0.,vertex_peak),
                    'max_preparation_penetration_m':max(0.,peak),'native_checked':False}
        if not len(normals):break
        N=np.asarray(normals);b=np.asarray(depths)
        solved=minimize(lambda x:float(x@x),np.zeros(3),method='SLSQP',
                        constraints=[{'type':'ineq','fun':lambda x:N@x-b}],
                        options={'maxiter':100,'ftol':1e-12})
        require(solved.success,'Initial rigid contacts conflict; revise pose/geometry instead of forcing a mesh')
        pose[:3,3]+=solved.x
        require(np.linalg.norm(pose[:3,3]-original)<=max_translation,'Initial contact correction exceeds source pose uncertainty')
    raise ValueError('Initial contact projection did not converge; native run not qualified')
def source_metrics(predicted, observed, splits, *, angular=False):
    """Keep fit and held-out source agreement separate, including angle wrap."""
    predicted, observed = np.asarray(predicted), np.asarray(observed)
    require(predicted.shape == observed.shape, 'Initial source feature shape mismatch')
    error = predicted - observed
    if angular:
        error = np.arctan2(np.sin(error), np.cos(error))
    result = {}
    for label, indices in splits.items():
        values = error[indices].reshape(len(indices), -1)
        valid = np.isfinite(values).all(1)
        require(valid.sum() >= 2, 'Not enough independent source initial views: ' + label)
        result[label + '_rmse'] = float(np.sqrt(np.mean(values[valid] ** 2)))
    return result


def aperture_clearance(bottom, top, radius, *, center, height, opening_radius):
    """Clearance of a finite straight shaft through a horizontal circular rim.

    A tilted cylinder has an elliptical section. Use its enclosing circle so a
    positive clearance is conservative. A segment missing the rim fails.
    """
    bottom, top = np.asarray(bottom, float), np.asarray(top, float)
    delta = top - bottom
    if abs(delta[2]) < 1e-10:
        return -float(opening_radius)
    fraction = (height - bottom[2]) / delta[2]
    if not 0 <= fraction <= 1:
        return -float(opening_radius)
    entry = bottom + fraction * delta
    section_radius = radius * np.linalg.norm(delta) / abs(delta[2])
    return float(opening_radius - np.linalg.norm(entry[:2] - center) - section_radius)


def mesh_aperture_clearance(vertices, faces, *, center, height, opening_radius):
    """Measure the actual solid surface section, avoiding a global radius proxy.

    Mesh coordinates must already be in the container frame. A missing section
    fails rather than certifying a shaft whose end never reaches the opening.
    """
    triangles = np.asarray(vertices)[np.asarray(faces, int)]
    a = triangles[:, [0, 1, 2]].reshape(-1, 3)
    b = triangles[:, [1, 2, 0]].reshape(-1, 3)
    delta = b[:, 2] - a[:, 2]
    valid = (np.minimum(a[:, 2], b[:, 2]) <= height) & (np.maximum(a[:, 2], b[:, 2]) >= height) & (abs(delta) > 1e-12)
    if not valid.any():
        return -float(opening_radius)
    f = (height - a[valid, 2]) / delta[valid]
    section = a[valid] + f[:, None] * (b[valid] - a[valid])
    return float(opening_radius - np.linalg.norm(section[:, :2] - center, axis=1).max())


def seat_cylinder(pose,bottom,top,radius,*,apertures,other_capsules=(),floor_z=None,
                  margin=1e-5,max_translation=.002):
    """Conservative clearance preparation for a reviewed circular container.

    Apertures are source-model inner profile sections in the fixture frame,
    above the lower end-cap region. All intervening linear-profile segments
    are bounded by those sections. Other shapes need their own surface tests.
    This changes rigid translation only and still requires native execution.
    """
    pose=np.asarray(pose,float).copy();a,b=np.asarray([bottom,top])@pose[:3,:3].T+pose[:3,3]
    axis=b-a;length=np.linalg.norm(axis)
    require(length>0 and radius>0 and apertures,'Reviewed shaft and source aperture profile required')
    lower=min(a[2],b[2])-radius*np.linalg.norm(axis[:2])/length
    def constraints(u):
        delta=u*.001;values=[]
        for ring in apertures:
            values.append(aperture_clearance(a+delta,b+delta,radius,center=ring['center'],height=ring['height'],opening_radius=ring['radius'])-margin)
        values.extend(segment_distance(a+delta,b+delta,c,d)-radius-r-margin for c,d,r in other_capsules)
        if floor_z is not None:values.append(lower+delta[2]-floor_z-margin)
        values.append(max_translation-np.linalg.norm(delta))
        return np.asarray(values)/.001
    solved=minimize(lambda u:float(u@u),np.zeros(3),method='SLSQP',
                    bounds=[(-max_translation/.001,max_translation/.001)]*3,
                    constraints=[{'type':'ineq','fun':constraints}],options={'maxiter':150,'ftol':1e-12})
    require(constraints(solved.x).min()>=-1e-4,'No source-profile shaft preparation within the allowed pose uncertainty')
    delta=solved.x*.001;pose[:3,3]+=delta
    return {'pose':pose,'translation_m':delta,'minimum_constraint_margin_m':float(constraints(solved.x).min()*.001),
            'native_checked':False}


def seat_cylinders(poses,segments,*,apertures,floor_z=None,margin=1e-5,max_translation=.002,
                   bottom_profile=None):
    """Prepare coupled free shafts jointly, retaining every rigid orientation.

    Moving one shaft at a time can force it through a neighbor or the rim.
    Solve their bounded translations together; this authors no simulation
    constraints, joints, pins or target trajectories.
    """
    poses=np.asarray(poses,float).copy();n=len(poses);require(0<n<=12 and len(segments)==n and apertures,'Bounded reviewed shaft assembly required')
    axes=[];lower=[];bottom_surfaces=[]
    for M,(a,b,r) in zip(poses,segments):
        a,b=np.asarray([a,b])@M[:3,:3].T+M[:3,3];d=b-a;L=np.linalg.norm(d)
        require(L>0 and r>0,'Positive shaft dimensions required')
        axes.append((a,b,r));lower.append(min(a[2],b[2])-r*np.linalg.norm(d[:2])/L)
        if bottom_profile is not None:
            endpoint=a if a[2]<b[2] else b;direction=d/L if a[2]<b[2] else -d/L
            ref=np.eye(3)[np.argmin(abs(direction))];u=np.cross(direction,ref);u/=np.linalg.norm(u);v=np.cross(direction,u)
            phase=np.arange(64)*2*np.pi/64;radial=r*(np.cos(phase)[:,None]*u+np.sin(phase)[:,None]*v)
            centers=endpoint+np.linspace(0,min(.003,L),13)[:,None]*direction
            bottom_surfaces.append((centers[:,None]+radial).reshape(-1,3))
    def constraints(u):
        delta=np.asarray(u).reshape(n,3)*.001;values=[]
        for i,(a,b,r) in enumerate(axes):
            for ring in apertures:
                values.append(aperture_clearance(a+delta[i],b+delta[i],r,center=ring['center'],height=ring['height'],opening_radius=ring['radius'])-margin)
            if floor_z is not None:values.append(lower[i]+delta[i,2]-floor_z-margin)
            if bottom_profile is not None:
                points=bottom_surfaces[i]+delta[i]
                radii=np.interp(points[:,2],bottom_profile['heights'],bottom_profile['radii'])
                values.extend(radii-np.linalg.norm(points[:,:2]-bottom_profile['center'],axis=1)-margin)
            values.append(max_translation-np.linalg.norm(delta[i]))
            for j in range(i+1,n):
                c,d,s=axes[j];values.append(segment_distance(a+delta[i],b+delta[i],c+delta[j],d+delta[j])-r-s-margin)
        return np.asarray(values)/.001
    solved=minimize(lambda u:float(u@u),np.zeros(n*3),method='SLSQP',
                    bounds=[(-max_translation/.001,max_translation/.001)]*(n*3),
                    constraints=[{'type':'ineq','fun':constraints}],options={'maxiter':200,'ftol':1e-12})
    require(constraints(solved.x).min()>=-1e-4,'No coupled shaft preparation within the allowed source pose uncertainty')
    delta=solved.x.reshape(n,3)*.001;poses[:,:3,3]+=delta
    return {'poses':poses,'translations_m':delta,'minimum_constraint_margin_m':float(constraints(solved.x).min()*.001),'native_checked':False}


def fit_seated_segment(bottom, top, cameras, tips, directions, *, center, radius,
                       fit_indices, validation_indices, length_bounds=(.9, 1.1),
                       max_axis_rmse=None,max_tip_rmse=None):
    """Refine length/azimuth from a native seated contact hypothesis.

    Rotation about the container axis preserves an axisymmetric contact
    hypothesis. This is modeling, not a force or pose controller; every result
    needs fresh native settling and interactions with the other bodies.
    """
    bottom, top, center = map(lambda v: np.asarray(v, float), (bottom, top, center))
    tips, directions = np.asarray(tips), np.asarray(directions)
    fit, held = np.asarray(fit_indices, int), np.asarray(validation_indices, int)
    require(not set(fit) & set(held), 'Disjoint source fitting views required')
    def geometry(x):
        R = Rotation.from_euler('z', x[0]).as_matrix()
        start = center + R @ (bottom-center)
        return start, start + R @ (top-bottom)*x[1]
    def residual(x, indices):
        start, end = geometry(x); values=[]
        for i in indices:
            uv = project(np.array([end, start]), cameras[i]); d = uv[1]-uv[0]; d /= np.linalg.norm(d)
            if np.isfinite(tips[i]).all(): values.extend((uv[0]-tips[i])/3.)
            if np.isfinite(directions[i]).all():
                target=directions[i];values.append(np.arctan2(d[0]*target[1]-d[1]*target[0],d@target)/.1)
        return np.asarray(values)
    solved=least_squares(lambda x: residual(x,fit),[0.,1.],bounds=([-np.pi,length_bounds[0]],[np.pi,length_bounds[1]]))
    if max_axis_rmse is not None or max_tip_rmse is not None:
        def margins(x):
            a,b=geometry(x);tip=[];angles=[]
            for i in fit:
                uv=project(np.array([b,a]),cameras[i]);d=uv[1]-uv[0];d/=np.linalg.norm(d)
                if np.isfinite(tips[i]).all():tip.extend(uv[0]-tips[i])
                if np.isfinite(directions[i]).all():
                    target=directions[i];angles.append(np.arctan2(d[0]*target[1]-d[1]*target[0],d@target))
            values=[]
            if max_axis_rmse is not None:values.append(max_axis_rmse-np.sqrt(np.mean(np.array(angles)**2)))
            if max_tip_rmse is not None:values.append(max_tip_rmse-np.sqrt(np.mean(np.array(tip)**2)))
            return np.array(values)
        solved=minimize(lambda x:float(np.mean(residual(x,fit)**2)),solved.x,method='SLSQP',
            bounds=[(-np.pi,np.pi),length_bounds],constraints=[{'type':'ineq','fun':margins}],
            options={'maxiter':100,'ftol':1e-10})
        require(margins(solved.x).min()>=-1e-6,'No seated source fit meets the declared tip and direction constraints')
    start,end=geometry(solved.x)
    metrics={}
    for label,indices in [('fit',fit),('validation',held)]:
        predicted=np.asarray([project(np.array([end,start]),cameras[i]) for i in indices])
        good=np.isfinite(tips[indices]).all(1)
        metrics[label+'_tip_rmse_px']=float(np.sqrt(np.mean((predicted[good,0]-tips[indices][good])**2)))
    return {'bottom':start,'top':end,'azimuth':float(solved.x[0]),'length_ratio':float(solved.x[1]),'metrics':metrics,
            'native_checked':False}


def fit_seated_segments(segments, cameras, tips, directions, *, center, fit_indices,
                        validation_indices, pair_margin=0., length_bounds=(.8,1.2),azimuth_groups=(),
                        max_fit_tip_rmse=None,max_fit_axis_rmse=None,
                        azimuth_bounds=(-np.pi,np.pi),assignments=None):
    """Joint source fit of an independent, axisymmetrically seated assembly.

    Source branch identities may permute identical shafts. Selection reads only
    fit views and solid clearance; held-out observations only report errors.
    """
    import itertools
    center=np.asarray(center,float);tips=np.asarray(tips);directions=np.asarray(directions)
    n=len(segments);require(1<=n<=4,'Enumerate larger identity hypotheses explicitly')
    require(len(azimuth_bounds)==2 and np.isfinite(azimuth_bounds).all()
            and azimuth_bounds[0]<=0<=azimuth_bounds[1] and azimuth_bounds[0]<azimuth_bounds[1],
            'Finite azimuth bounds containing the native seed required')
    hypotheses=list(itertools.permutations(range(n))) if assignments is None else list(assignments)
    require(bool(hypotheses) and all(len(a)==n and sorted(a)==list(range(n)) for a in hypotheses),
            'Each source assignment must be a complete permutation')
    fit=np.asarray(fit_indices,int);held=np.asarray(validation_indices,int)
    require(not set(fit)&set(held),'Disjoint source fitting views required')
    def geometry(x):
        result=[]
        for i,(a,b,r) in enumerate(segments):
            R=Rotation.from_euler('z',x[i*2]).as_matrix();start=center+R@(a-center)
            result.append((start,start+R@(b-a)*x[i*2+1],r))
        return result
    def clearance(x):
        axes=geometry(x)
        return np.array([segment_distance(a,b,c,d)-r-s-pair_margin
                         for i,(a,b,r) in enumerate(axes) for c,d,s in axes[i+1:]])
    bounds=[b for _ in range(n) for b in [azimuth_bounds,length_bounds]];solutions=[]
    for assignment in hypotheses:
        def residual(x,indices):
            values=[]
            for j,(a,b,r) in enumerate(geometry(x)):
                for i in indices:
                    uv=project(np.array([b,a]),cameras[i]);d=uv[1]-uv[0];d/=np.linalg.norm(d);k=assignment[j]
                    if np.isfinite(tips[i,k]).all():values.extend((uv[0]-tips[i,k])/3.)
                    if np.isfinite(directions[i,k]).all():
                        target=directions[i,k];values.append(np.arctan2(d[0]*target[1]-d[1]*target[0],d@target)/.1)
            return np.asarray(values)
        start=np.tile([0.,1.],n)
        guess=least_squares(lambda x:residual(x,fit),start,bounds=np.array(bounds).T,max_nfev=50).x
        seeds=[guess,start]
        for phase in [-1.2,1.2]:
            extra=start.copy();extra[::2]=np.clip(phase,*azimuth_bounds);seeds.append(extra)
        constraints=[{'type':'ineq','fun':clearance}]
        def source_clearance(x):
            margins=[]
            for j,(a,b,r) in enumerate(geometry(x)):
                tip_errors=[];axis_errors=[];k=assignment[j]
                for i in fit:
                    uv=project(np.array([b,a]),cameras[i]);d=uv[1]-uv[0];d/=np.linalg.norm(d)
                    if np.isfinite(tips[i,k]).all():tip_errors.extend(uv[0]-tips[i,k])
                    if np.isfinite(directions[i,k]).all():
                        target=directions[i,k];axis_errors.append(np.arctan2(d[0]*target[1]-d[1]*target[0],d@target))
                if max_fit_tip_rmse is not None:margins.append(max_fit_tip_rmse-np.sqrt(np.mean(np.array(tip_errors)**2)))
                if max_fit_axis_rmse is not None:margins.append(max_fit_axis_rmse-np.sqrt(np.mean(np.array(axis_errors)**2)))
            return np.array(margins)
        if max_fit_tip_rmse is not None or max_fit_axis_rmse is not None:constraints.append({'type':'ineq','fun':source_clearance})
        pairs=[(group[0],i) for group in azimuth_groups for i in group[1:]]
        if pairs:constraints.append({'type':'eq','fun':lambda x:np.array([x[i*2]-x[j*2] for i,j in pairs])})
        for seed in seeds:
            solved=minimize(lambda x:float(np.mean(residual(x,fit)**2)),seed,method='SLSQP',bounds=bounds,
                            constraints=constraints,options={'maxiter':150,'ftol':1e-9})
            if len(clearance(solved.x)) and clearance(solved.x).min()<-1e-6:continue
            if pairs and max(abs(solved.x[i*2]-solved.x[j*2]) for i,j in pairs)>1e-6:continue
            if (max_fit_tip_rmse is not None or max_fit_axis_rmse is not None) and source_clearance(solved.x).min()<-1e-6:continue
            solutions.append((float(np.mean(residual(solved.x,fit)**2)),assignment,solved.x))
    require(bool(solutions),'No nonintersecting source-fitted seated hypothesis')
    loss,assignment,x=min(solutions,key=lambda row:row[0]);rows=[]
    for j,(a,b,r) in enumerate(geometry(x)):
        k=assignment[j];metrics={}
        for label,indices in [('fit',fit),('validation',held)]:
            predicted=np.array([project(np.array([b,a]),cameras[i])[0] for i in indices]);valid=np.isfinite(tips[indices,k]).all(1)
            metrics[label+'_tip_rmse_px']=float(np.sqrt(np.mean((predicted[valid]-tips[indices,k][valid])**2)))
        rows.append({'bottom':a,'top':b,'radius_m':r,'source_index':k,'length_ratio':float(x[j*2+1]),
                     'azimuth':float(x[j*2]),'metrics':metrics,'native_checked':False})
    return rows
