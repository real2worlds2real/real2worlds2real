"""Motion-specific numerical initializers; never native acceptance evidence."""
import numpy as np
from scipy.optimize import least_squares

from calibrated_scene_reconstruction.physics.video.fit import fit_slide,slide_positions
from .contract import require


def free_slide(clips,*,gravity=9.81):
    """Shared contact deceleration, separate release states for independent clips.

    Source contact annotations choose the free windows before fitting. Geometry
    and camera uncertainty remain outside this local fit; reported covariance
    is not a posterior confidence interval for the material. Mass and unknown
    human forces cannot be separated by this motion alone.
    """
    require(bool(clips),'At least one observed free-motion clip is required')
    prepared=[];initial=[]
    for clip in clips:
        t,xy=np.asarray(clip['t'],float),np.asarray(clip['xy'],float)
        indices=np.flatnonzero((t>=clip['free_start'])&np.isfinite(xy).all(1))
        require(len(indices)>=10,'Not enough free-motion observations')
        train=indices[indices%3!=0];held=indices[indices%3==0]
        require(len(train)>=5 and len(held)>=3,'Free motion requires fit and held-out frames')
        noise=float(clip.get('position_noise_m',.001))
        require(np.isfinite(noise) and noise>0,'Positive source position uncertainty required')
        travel=float(np.max(np.linalg.norm(xy[indices]-np.median(xy[indices],axis=0),axis=1)))
        require(travel>6*noise,'No resolved free motion above observation uncertainty')
        # One canonical origin per clip: preserve time for held-out early frames.
        origin=t[indices[0]]
        fitted=fit_slide(t[train],xy[train])
        p=fitted['p'].copy();delta=t[train[0]]-origin
        # Rewind the fitted release state; no observed sample changes split/time.
        p[:2]-=(p[3]*delta+.5*p[4]*delta**2)*np.array([np.cos(p[2]),np.sin(p[2])])
        p[3]+=p[4]*delta
        prepared.append((clip,t-origin,xy,train,held,indices,origin))
        initial.append(p)
    seed=np.r_[np.median([p[4] for p in initial]),np.concatenate([p[:4] for p in initial])]
    low=np.full(len(seed),-np.inf);high=np.full(len(seed),np.inf)
    low[0]=.001;high[0]=2*gravity
    for i in range(len(clips)):low[1+4*i+3]=.001;high[1+4*i+3]=10.
    # The unconstrained legacy seed may have negative deceleration for a
    # constant-speed/noisy clip. It initializes, rather than defines, evidence.
    seed=np.clip(seed,low,high)

    def predictions(x,index,t):
        return slide_positions(np.r_[x[1+4*index:5+4*index],x[0]],t)

    def residual(x):
        return np.concatenate([(predictions(x,i,t[train])-xy[train]).ravel()
            for i,(_,t,xy,train,_,_,_) in enumerate(prepared)])

    solved=least_squares(residual,seed,bounds=(low,high),loss='soft_l1',f_scale=.005,max_nfev=300)
    rows=[]
    for i,(clip,t,xy,train,held,indices,origin) in enumerate(prepared):
        p=np.r_[solved.x[1+4*i:5+4*i],solved.x[0]]
        errors=predictions(solved.x,i,t)-xy
        # A constant-velocity or all-rest window cannot establish deceleration.
        line=np.c_[t[train],np.ones(len(train))]
        linear=np.linalg.lstsq(line,xy[train],rcond=None)[0]
        curvature=float(np.sqrt(np.mean(np.sum((line@linear-xy[train])**2,axis=1))))
        noise=float(clip.get('position_noise_m',.001))
        rows.append({'name':clip['name'],'release_t':float(origin),'fit_indices':train.tolist(),'validation_indices':held.tolist(),
            'fit_rmse_m':float(np.sqrt(np.mean(np.sum(errors[train]**2,axis=1)))),
            'validation_rmse_m':float(np.sqrt(np.mean(np.sum(errors[held]**2,axis=1)))),
            'release_speed_m_s':float(p[3]),'stop_t':float(origin+p[3]/p[4]),'parameters':p.tolist(),
            'fit_frames':len(train),'held_out_frames':len(held),
            'deceleration_resolved':bool(curvature>3*noise),
            'linear_model_rmse_m':curvature,'source_position_noise_m':noise})
    return {'kinetic_friction_initial_guess':float(solved.x[0]/gravity),'clips':rows,
        'optimizer_converged':bool(solved.success),'native_simulation_run':False,'accepted':False,
        'mass':'prior_only','external_force':'not identified by the free-motion fit',
        'deceleration_evidence':'resolved' if any(r['deceleration_resolved'] for r in rows) else 'unresolved',
        'limitations':['Planar feature-height approximation; rocking, lens, scale and shutter uncertainty are not marginalized.',
            'Held-out frames of these clips only; no independent-video material validation.',
            'Initializer for an agent-selected model; native whole-process replay and review are still required.']}


def pulse_motion(t,*,mass,friction,peak_force,skew,heading,onset,release,origin,dt=.001):
    """Planar Coulomb surrogate for initialization, not an alternative final simulator."""
    clock=np.arange(0,float(np.max(t))+dt,dt)
    phase=np.clip((clock-onset)/(release-onset),0,1)
    force=peak_force*np.sin(np.pi*phase)*(1+skew*(1-2*phase))
    force[(clock<onset)|(clock>release)]=0.
    integral=np.cumsum(force/mass-friction*9.81)*dt
    velocity=integral-np.minimum.accumulate(np.minimum(integral,0.))
    distance=np.cumsum(velocity)*dt
    return np.asarray(origin)+np.interp(t,clock,distance)[:,None]*np.array([np.cos(heading),np.sin(heading)])


def contact_pulse(clip,*,mass,friction_prior,onset_bounds,release_bounds):
    """A bounded, low-dimensional human-input hypothesis; mass remains a prior."""
    t,xy=np.asarray(clip['t'],float),np.asarray(clip['xy'],float)
    valid=np.isfinite(xy).all(1);train=valid&(np.arange(len(t))%3!=0);held=valid&~train
    require(train.sum()>=15 and held.sum()>=5,'Insufficient process observations')
    before=valid&(t<onset_bounds[0]);require(before.sum()>=5,'Need a stationary initial observation window')
    origin=np.median(xy[before],axis=0)
    delta=np.median(xy[np.flatnonzero(valid)[-5:]],axis=0)-origin
    heading=np.arctan2(delta[1],delta[0])
    lo=[.01,.05,-.9,heading-.4,onset_bounds[0],release_bounds[0]]
    hi=[1.2,10.,.9,heading+.4,onset_bounds[1],release_bounds[1]]
    seed=[friction_prior,1.5,0.,heading,np.mean(onset_bounds),np.mean(release_bounds)]
    def predict(x):
        mu,force,skew,angle,start,end=x
        require(end>start,'Human-input windows must be ordered')
        return pulse_motion(t,mass=mass,friction=mu,peak_force=force,skew=skew,heading=angle,
                            onset=start,release=end,origin=origin)
    result=least_squares(lambda x:(predict(x)[train]-xy[train]).ravel(),seed,bounds=(lo,hi),
                         loss='soft_l1',f_scale=.005,max_nfev=300,diff_step=1e-3)
    mu,force,skew,angle,start,end=result.x
    error=predict(result.x)-xy
    return {'parameters':{'pair.kinetic_friction':float(mu),'push.peak_force':float(force),'push.skew':float(skew),
        'push.heading':float(angle),'push.onset':float(start),'push.release':float(end),'object.mass':mass},
        'origin':origin.tolist(),'fit_rmse_m':float(np.sqrt(np.mean(np.sum(error[train]**2,axis=1)))),
        'validation_rmse_m':float(np.sqrt(np.mean(np.sum(error[held]**2,axis=1)))),
        'optimizer_converged':bool(result.success),'native_simulation_run':False,'accepted':False,
        'limitations':['Planar, non-rotating Coulomb initializer only; not whole-scene physics proof.',
            'Force amplitude is conditional on the supplied mass prior; multiplying mass and force together leaves this trajectory unchanged.',
            'Contact times are bounded by source-only annotations; no per-frame force or target-pose driver.']}
