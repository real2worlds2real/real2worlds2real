"""Bounded wrist corrections while closing around moving material.

Only fixture commands are produced. Material observations are read-only;
native contacts, finite actuation and the causal friction test stay separate.
The negative control replays every recorded command, not a feedback policy.
"""
import numpy as np


def capture_lead(previous,current,patch,*,dt,closing_seconds,up,maximum_shift_m):
    """Bounded intercept proposal from measured material motion, never actuation.

    Aim near the expected material position halfway through closure. This
    constant-velocity prior needs actual open-path and arrival checks; contacts
    may invalidate the prediction. It cannot diagnose a material law.
    """
    previous,current,patch=np.asarray(previous,float),np.asarray(current,float),np.asarray(patch)
    up=np.asarray(up,float)
    if (previous.shape!=current.shape or current.ndim!=2 or current.shape[1:]!=(3,)
            or patch.ndim!=1 or patch.dtype.kind not in 'iu' or not len(patch)
            or patch.min()<0 or patch.max()>=len(current) or up.shape!=(3,)
            or not all(np.isfinite(x).all()for x in [previous,current,up]) or np.linalg.norm(up)<1e-12
            or not np.isfinite([dt,closing_seconds,maximum_shift_m]).all()
            or min(dt,closing_seconds,maximum_shift_m)<=0):
        raise ValueError('Matching measured material states, time interval and bounded interception prior required')
    up=up/np.linalg.norm(up);velocity=np.median((current[patch]-previous[patch])/dt,axis=0)
    distance=min(maximum_shift_m,max(0.,-float(velocity@up))*closing_seconds/2)
    return {'shift_world_m':(-distance*up).tolist(),'material_velocity_world_m_s':velocity.tolist(),
        'observation_dt_s':float(dt),'prediction_horizon_s':float(closing_seconds/2),
        'maximum_shift_m':float(maximum_shift_m),'proposal_only':True,'cloth_actuated':False}


def native_correction(previous, material, pads, standoff):
    """Feedback uses measured fingertip positions, including native joint lag.

    The tracker consumes an absolute correction to its nominal path. Adding
    the residual to the previous correction avoids losing half the correction
    at equilibrium. Limits and slew rates remain the tracker's responsibility.
    """
    previous=np.asarray(previous,float);material=np.asarray(material,float)
    pads=np.asarray(pads,float);standoff=np.asarray(standoff,float)
    if (previous.shape!=(3,) or material.shape!=(3,) or pads.shape!=(2,3) or standoff.shape!=(3,)
            or not all(np.isfinite(x).all() for x in [previous,material,pads,standoff])):
        raise ValueError('Finite actual finger pads, material point and previous wrist correction required')
    return previous+material+standoff-pads.mean(0)


class ClosureTracking:
    def __init__(self, *, physics_hz, steps, maximum_shift_m=.008,
                 speed_m_s=.01, replay=None):
        if (type(physics_hz) is not int or not 240<=physics_hz<=7680
                or type(steps) is not int or steps<=0
                or not np.isfinite([maximum_shift_m,speed_m_s]).all()
                or not 0<maximum_shift_m<=.012 or not 0<speed_m_s<=.02):
            raise ValueError('Bounded finite fixture tracking profile required')
        self.profile={'physics_hz':physics_hz,'steps':steps,
                      'maximum_shift_m':float(maximum_shift_m),'speed_m_s':float(speed_m_s)}
        self.dt=1/physics_hz;self.limit=maximum_shift_m;self.speed=speed_m_s
        self.correction=np.zeros(3);self.desired=np.zeros(3);self.positions=[]
        self.replay=None
        if replay is not None:
            if (replay.get('complete') is not True or replay.get('profile')!=self.profile
                    or replay.get('cloth_actuated') is not False):
                raise ValueError('Replay needs a complete matching fixture-only closure trace')
            value=np.asarray(replay['positions'],float)
            if value.shape!=(steps,3) or not np.isfinite(value).all():
                raise ValueError('One finite wrist target per closure physics step required')
            self.replay=value.copy()

    def step(self, index, nominal, observed_correction=None):
        nominal=np.asarray(nominal,float)
        if (index!=len(self.positions) or index>=self.profile['steps']
                or nominal.shape!=(3,) or not np.isfinite(nominal).all()):
            raise ValueError('Ordered closure step and finite nominal wrist position required')
        if self.replay is not None:
            target=self.replay[index].copy();correction=target-nominal
            if (np.linalg.norm(correction)>self.limit+1e-10
                    or np.linalg.norm(correction-self.correction)>self.speed*self.dt+1e-10):
                raise ValueError('Replayed closure command exceeds declared tracking bounds')
        else:
            if observed_correction is not None:
                value=np.asarray(observed_correction,float)
                if value.shape!=(3,) or not np.isfinite(value).all():
                    raise ValueError('Finite read-only material observation required')
                self.desired=value*min(1.,self.limit/max(np.linalg.norm(value),1e-30))
            delta=self.desired-self.correction
            correction=self.correction+delta*min(1.,self.speed*self.dt/max(np.linalg.norm(delta),1e-30))
            target=nominal+correction
        self.correction=correction;self.positions.append(target.tolist())
        return target

    def report(self):
        return {'profile':self.profile,'complete':len(self.positions)==self.profile['steps'],
                'positions':self.positions,'cloth_actuated':False,
                'scope':'Finite wrist-drive targets only; no material positions, forces or attachments. '
                        'Replayed exactly for the matched control; no grasp or task approval.'}
