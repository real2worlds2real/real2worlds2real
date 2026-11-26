"""Source-instance axes for geometric initialization, never acceptance.

An occluding cable can split one physical leg into multiple image components.
Preserve the acquired instance identity instead of assigning each fragment to
a different physical leg. Full support masks and part gates remain unchanged.
"""
import numpy as np


def instance_axes(instances, *, minimum_pixels=100, minimum_elongation=2.5):
    selected = []
    for item in sorted(instances, key=lambda row: -np.count_nonzero(row['mask'])):
        mask = np.asarray(item['mask'], bool)
        if mask.ndim != 2:
            raise ValueError('Source instance masks must be two-dimensional')
        if mask.sum() < minimum_pixels:
            continue
        # Independent duplicate detections of the same visible source region.
        if any((mask&old['mask']).sum() >= .8*min(mask.sum(), old['mask'].sum()) for old in selected):
            continue
        y, x = np.nonzero(mask); points = np.c_[x, y]; centre = points.mean(0)
        _, singular, vh = np.linalg.svd(points-centre, full_matrices=False)
        elongation = float(singular[0]/max(singular[1], 1e-12))
        if elongation < minimum_elongation:
            continue
        axis = vh[0]; interval = np.percentile((points-centre)@axis, [15, 85])
        selected.append({'path': item['path'], 'mask': mask, 'pixels': int(mask.sum()),
                         'elongation': elongation, 'centroid': centre.tolist(),
                         'endpoints': (centre+interval[:, None]*axis).tolist()})
    return [{key: value for key, value in row.items() if key != 'mask'} for row in selected]


def symmetric_leg_proposal(observations, prompt, *, size, top_z, thickness, slopes,
                           xy, yaw_deg, initial, bounds):
    """Initialize four symmetric legs from independent source instance axes.

    The caller supplies its current *candidate* tabletop, never true furniture
    dimensions. Occluded leg fragments constrain line direction and lateral
    position, not invented visible endpoints. This is only an optimizer seed;
    complete source masks, geometry and native contact still require checking.
    """
    from scipy.optimize import least_squares, linear_sum_assignment
    from calibrated_scene_reconstruction.scene.objects.object_hull import project
    size=np.asarray(size,float);slopes=np.asarray(slopes,float);xy=np.asarray(xy,float)
    initial=np.asarray(initial,float);bounds=np.asarray(bounds,float)
    if (size.shape!=(2,) or slopes.shape!=(2,) or xy.shape!=(2,) or initial.shape!=(4,)
            or bounds.shape!=(2,4) or not np.isfinite(np.r_[size,slopes,xy,initial,bounds.ravel(),top_z,thickness,yaw_deg]).all()
            or (size<=0).any() or not 0<thickness<top_z or (bounds[0]>=bounds[1]).any()):
        raise ValueError('Finite candidate tabletop and bounded four-parameter leg initializer required')
    rows=[]
    for frame in observations:
        axes=instance_axes(frame.get('part_instances',{}).get(prompt,[]))
        if axes:rows.append((frame,axes[:4]))
    if len(rows)<3:raise ValueError('At least three independent source views of leg instances required')
    angle=np.radians(yaw_deg);R=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    signs=np.array([[-1,-1],[-1,1],[1,-1],[1,1]])
    def endpoints(q):
        top=signs*(size/2-q[:2]);bottom=top+signs*q[2:]
        ends=np.zeros((4,2,3));ends[:,0,:2]=bottom;ends[:,1,:2]=top
        ends[:,1,2]=top_z-thickness+top@slopes
        ends[:,:,:2]=ends[:,:,:2]@R.T+xy
        return ends
    def residual(q, axis_only=False):
        ends=endpoints(q);res=[]
        for frame,axes in rows:
            u,v,z=project(ends.reshape(-1,3),frame['K'],frame['R'],frame['C'])
            x0,y0,_,_,factor=frame['window'];pred=np.c_[(u-x0)*factor,(v-y0)*factor].reshape(4,2,2)
            direction=pred[:,1]-pred[:,0];length=np.linalg.norm(direction,axis=1)
            normal=np.c_[-direction[:,1],direction[:,0]]/np.maximum(length[:,None],1e-6)
            samples=np.asarray([a['endpoints'] for a in axes])
            distance=np.einsum('aslc,lc->asl',samples[:,:,None,:]-pred[None,None,:,0,:],normal)
            cost=np.mean(distance**2,axis=1)
            # A line extension is not a physical leg: observed samples must
            # lie between the projected floor and underside, with pixel noise.
            along=np.einsum('aslc,lc->asl',samples[:,:,None,:]-pred[None,None,:,0,:],direction)/np.maximum(length[None,None,:]**2,1e-6)
            excess=np.maximum(0,-along)+np.maximum(0,along-1)
            cost+=np.mean((excess*length[None,None,:])**2,axis=1)
            a,b=linear_sum_assignment(cost)
            res.extend(distance[a,:,b].ravel())
            if not axis_only:res.extend((excess[a,:,b]*length[b,None]).ravel())
        return np.asarray(res)
    result=least_squares(residual,initial,bounds=bounds,loss='soft_l1',f_scale=2.,max_nfev=500,
                         diff_step=1e-3,xtol=1e-9,ftol=1e-9,gtol=1e-9)
    return {'proposal_only':True,'acceptance':False,'views':len(rows),
            'attachment_insets_m':result.x[:2].tolist(),'foot_splay_m':result.x[2:].tolist(),
            'median_axis_residual_px':float(np.median(abs(residual(result.x,axis_only=True)))),
            'endpoints_world_m':endpoints(result.x).tolist(),
            'basis':'Independent source part instances; symmetric manufacturing hypothesis; full final checks unchanged'}
