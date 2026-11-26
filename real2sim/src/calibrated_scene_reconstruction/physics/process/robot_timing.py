"""Align cameras of one event to measured robot kinematics, before physics fitting.

Only robot material/joint landmarks enter this problem. Response-object motion
and forces are deliberately absent. A shared rate describes the recorder clock;
each video has its own offset and independent camera calibration.
"""
import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator
from scipy.optimize import differential_evolution, least_squares, minimize


def fit(trajectory_time, world_landmarks, observations, *, rate_bounds,
        offset_bounds, uncertainty_px=3., max_iterations=160, nominal_clock=None):
    """Return an unapproved timing proposal and unchanged absolute residuals.

    observations maps camera names to time, pixels, K, R_world_cam, centre and
    training arrays. Optional landmark_indices selects distinct measured CAD
    identities visible in that camera; cameras need not see the same fingers.
    NaN pixels explicitly denote temporarily unavailable correspondences.
    Training and held-out rows must both contain source motion. Fixed initial
    annotation/CAD surface bias is removed only from the timing objective.
    Boundary poses are held ONLY to compare recorder lead/trail photographs;
    replay must use the returned recorded-overlap intervals.
    """
    t=np.asarray(trajectory_time,float);xyz=np.asarray(world_landmarks,float)
    if (t.ndim!=1 or len(t)<4 or not np.isfinite(t).all() or np.any(np.diff(t)<=0)
            or xyz.ndim!=3 or xyz.shape[0]!=len(t) or xyz.shape[2]!=3
            or not np.isfinite(xyz).all() or not observations
            or set(observations)!=set(offset_bounds)
            or not np.isfinite(uncertainty_px) or uncertainty_px<=0
            or type(max_iterations) is not int or max_iterations<1):
        raise ValueError('Finite measured landmark trajectory and explicit camera clocks required')
    bounds=[rate_bounds,*[offset_bounds[name] for name in observations]]
    if any(len(b)!=2 or not np.isfinite(b).all() or b[0]>=b[1] for b in bounds) or rate_bounds[0]<=0:
        raise ValueError('Ordered finite rate/offset bounds required')
    motion=interp1d(t,xyz,axis=0,bounds_error=False,fill_value=(xyz[0],xyz[-1]))
    clock_size=len(bounds);knots=None
    if nominal_clock is not None:
        if (not isinstance(nominal_clock,dict) or nominal_clock.get('kind')!='nominal_frame_rate'
                or not nominal_clock.get('basis')):
            raise ValueError('Nonlinear clock correction requires explicit nominal-clock provenance')
        count=nominal_clock.get('knots',8);deviation=nominal_clock.get('maximum_deviation_s')
        if (type(count)is not int or not 4<=count<=12 or not isinstance(deviation,(int,float))
                or not np.isfinite(deviation) or not 0<deviation<=.4*(t[-1]-t[0])/(count-1)):
            raise ValueError('Bounded monotone nominal clock correction required')
        knots=np.linspace(t[0],t[-1],count)
    cameras=[]
    for name,source in observations.items():
        q={k:np.asarray(source[k]) for k in ['time','pixels','K','R_world_cam','centre','training']}
        indices=np.asarray(source.get('landmark_indices',np.arange(xyz.shape[1])))
        if (indices.ndim!=1 or len(indices)<1 or indices.dtype.kind not in 'iu'
                or len(np.unique(indices))!=len(indices)
                or np.any(indices<0) or np.any(indices>=xyz.shape[1])):
            raise ValueError('Camera landmark_indices must select distinct measured CAD identities')
        q['landmark_indices']=indices
        n=len(q['time']);landmarks=len(indices)
        shapes={'time':(n,),'pixels':(n,landmarks,2),'K':(n,3,3),
                'R_world_cam':(n,3,3),'centre':(n,3),'training':(n,)}
        if (n<8 or any(q[k].shape!=shape for k,shape in shapes.items())
                or any(not np.isfinite(q[k]).all() for k in ['time','K','R_world_cam','centre'])
                or np.any(np.diff(q['time'])<=0) or q['training'].dtype.kind!='b'
                or q['training'].sum()<4 or (~q['training']).sum()<3
                or not np.isfinite(q['pixels'][0]).all() or not q['training'][0]):
            raise ValueError('Independent calibrated source frames and held-out motion required')
        R=q['R_world_cam']
        if (not np.allclose(R.transpose(0,2,1)@R,np.eye(3),atol=1e-5)
                or not np.allclose(np.linalg.det(R),1,atol=1e-5)):
            raise ValueError('Camera rotations must be right-handed orthonormal matrices')
        if (np.any(q['K'][:,[0,1],[0,1]]<=0)
                or not np.allclose(q['K'][:,2],[0,0,1])):
            raise ValueError('Positive calibrated focal lengths and pinhole K required')
        finite=np.isfinite(q['pixels'])
        if np.isinf(q['pixels']).any() or np.any(finite[:,:,0]!=finite[:,:,1]):
            raise ValueError('Unavailable correspondences must use a complete NaN pixel pair')
        valid=np.isfinite(q['pixels']).all(2)
        if np.any(valid.sum(0)<4) or not valid[q['training']].any() or not valid[~q['training']].any():
            raise ValueError('Source identities need enough actual observations')
        for training in [True,False]:
            displacement=np.linalg.norm(q['pixels'][q['training']==training]-q['pixels'][0],axis=2)
            if not np.isfinite(displacement).any() or np.nanmax(displacement)<3*uncertainty_px:
                raise ValueError('Static landmarks cannot identify a video clock')
        q['valid']=valid;q['name']=name;cameras.append(q)

    def project(x,q,i):
        mapped=x[0]*q['time']+x[i+1]
        if knots is not None and len(x)>clock_size:
            # Retain source sample order and endpoints. Only uncertain nominal
            # sample times are inferred; measured joint positions never change.
            warped=knots+np.r_[0.,x[clock_size:],0.]
            inside=(mapped>=t[0])&(mapped<=t[-1])
            mapped=mapped.copy();mapped[inside]=PchipInterpolator(knots,warped)(mapped[inside])
        pc=np.einsum('nji,nkj->nki',q['R_world_cam'],motion(mapped)[:,q['landmark_indices']]-q['centre'][:,None,:])
        if np.any(pc[:,:,2]<=1e-5):raise ValueError('Robot landmark behind source camera')
        uv=np.einsum('nij,nkj->nki',q['K'],pc)
        return uv[:,:,:2]/uv[:,:,2:],mapped

    def residual(x):
        rows=[]
        for i,q in enumerate(cameras):
            predicted,_=project(x,q,i)
            delta=(predicted-predicted[0])-(q['pixels']-q['pixels'][0])
            selected=q['valid']&q['training'][:,None]
            # Equal camera weight; a high frame rate is not extra evidence of
            # scale or clock consistency relative to another video.
            rows.append(delta[selected].ravel()/np.sqrt(selected.sum()))
        return np.concatenate(rows)

    def objective(x):
        # Robust cost in pixels before per-camera normalization.
        value=0.
        for i,q in enumerate(cameras):
            try:
                predicted,_=project(x,q,i)
            except ValueError:
                return np.inf
            delta=(predicted-predicted[0])-(q['pixels']-q['pixels'][0])
            v=delta[q['valid']&q['training'][:,None]]/uncertainty_px
            value+=float(np.mean(2*uncertainty_px**2*(np.sqrt(1+v*v)-1)))
        if knots is not None and len(x)>clock_size:
            shifts=np.r_[0.,x[clock_size:],0.]/deviation
            value+=.1*float(np.mean(shifts**2))+.5*float(np.mean(np.diff(shifts,2)**2))
        return value

    search=differential_evolution(objective,bounds,seed=0,maxiter=max_iterations,tol=1e-9)
    if not np.isfinite(search.fun):
        raise ValueError('No clock hypothesis places the robot landmarks in front of the cameras')
    # Preserve the robust optimum; a Jacobian diagnoses identifiability without
    # substituting a differently weighted least-squares solution.
    x=search.x
    selected_bounds=list(bounds);warp_fit=None
    if knots is not None:
        selected_bounds += [(-deviation,deviation)]*(len(knots)-2)
        initial=np.r_[x,np.zeros(len(knots)-2)]
        warp_fit=minimize(objective,initial,bounds=selected_bounds,method='L-BFGS-B',
                          options={'maxiter':max_iterations*5,'maxls':50,'ftol':1e-10})
        if np.isfinite(warp_fit.fun) and warp_fit.fun<objective(initial):x=warp_fit.x
        else:x=initial
    diagnostic=least_squares(residual,x,bounds=np.asarray(selected_bounds).T,max_nfev=1)
    singular=np.linalg.svd(diagnostic.jac,compute_uv=False)
    result={'shared_rate':float(x[0]),'cameras':{},'accepted':False,
            'jacobian_singular_values':singular.tolist(),'objective':float(objective(x)),
            'locally_identifiable':bool(singular[-1]>max(1e-8,singular[0]*1e-6)),
            'optimizer_converged':bool(search.success),
            'optimizer_message':str(search.message),
            'rate_bound_hit':bool(min(x[0]-bounds[0][0],bounds[0][1]-x[0])<1e-5),
            'scope':__doc__.strip(),'boundary_hold_scope':'alignment_only; never native response playback'}
    comparisons={}
    if knots is not None:
        result['nominal_clock_warp']={
            'kind':'monotone_nominal_sample_clock','basis':nominal_clock['basis'],
            'input_time_knots_s':knots.tolist(),'trajectory_time_knots_s':(knots+np.r_[0.,x[clock_size:],0.]).tolist(),
            'maximum_allowed_deviation_s':float(deviation),'interpolation':'pchip',
            'optimizer_converged':bool(warp_fit.success),'optimizer_message':str(warp_fit.message),
            'bound_hit':bool(np.any(np.abs(x[clock_size:])>=deviation-1e-5)),
            'scope':'Inferred timing of nominal samples only; same measured positions and order, no object response used'}
        result['optimizer_converged'] &= bool(warp_fit.success)
    for i,q in enumerate(cameras):
        pred,mapped=project(x,q,i);delta=(pred-pred[0])-(q['pixels']-q['pixels'][0])
        error=np.linalg.norm(delta,axis=2);absolute=np.linalg.norm(pred-q['pixels'],axis=2)
        valid=q['valid'];training=q['training'][:,None]&valid;held=~q['training'][:,None]&valid
        result['cameras'][q['name']]={'offset_s':float(x[i+1]),
            'landmark_indices':q['landmark_indices'].tolist(),
            'training_relative_rmse_px':float(np.sqrt(np.mean(error[training]**2))),
            'held_out_relative_rmse_px':float(np.sqrt(np.mean(error[held]**2))),
            'absolute_error_percentiles_px':np.percentile(absolute[valid],[50,90,95,100]).tolist(),
            'initial_projection_bias_px':(pred[0]-q['pixels'][0]).tolist(),
            'recorded_overlap_video_s':[(float(t[0])-x[i+1])/x[0],(float(t[-1])-x[i+1])/x[0]],
            'bound_hit':bool(min(x[i+1]-bounds[i+1][0],bounds[i+1][1]-x[i+1])<1e-5)}
        comparisons[q['name']]={'predicted':pred,'observed':q['pixels'],
            'mapped_trajectory_time':mapped,'recorded_overlap':(mapped>=t[0])&(mapped<=t[-1]),
            'training':q['training'],'valid':valid,'time':q['time']}
    return result,comparisons
