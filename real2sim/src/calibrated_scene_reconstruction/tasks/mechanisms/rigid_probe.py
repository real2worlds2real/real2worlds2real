"""Finite translation-force probe for independent rigid parts.

The target belongs to a disclosed test tool. Actual body motion comes from
native dynamics, with gravity and all existing collisions retained.
"""
import numpy as np

from .force_fixture import implicit_pd


def rigid_transform(rotation, translation):
    """An assay setup may move an asset, never rescale or mirror its geometry."""
    R, t = np.asarray(rotation, float), np.asarray(translation, float)
    if (R.shape != (3, 3) or t.shape != (3,) or not np.isfinite(R).all() or not np.isfinite(t).all()
            or not np.allclose(R.T@R, np.eye(3), atol=1e-8, rtol=0)
            or not np.isclose(np.linalg.det(R), 1., atol=1e-8, rtol=0)):
        raise ValueError('Companion placement must be rigid; scaling or deformation is forbidden')
    return R, t


def validate(plan, names, seconds):
    required = {'link', 'start_s', 'duration_s', 'translation_m', 'force_limit_N',
                'stiffness_N_m', 'gravity_compensation'}
    if (not isinstance(plan, dict) or not required <= set(plan)
            or set(plan)-required-{'orientation_hold','motion_profile','end_hold_s','release_ramp_s'} or plan['link'] not in names):
        raise ValueError('Force probe needs one existing link and the explicit tool fields')
    values = np.asarray([plan[k] for k in ['start_s', 'duration_s', 'force_limit_N', 'stiffness_N_m']], float)
    delta = np.asarray(plan['translation_m'], float)
    if (not np.isfinite(values).all() or values[0] < 0 or (values[1:] <= 0).any()
            or values[0]+values[1] >= seconds or delta.shape != (3,)
            or not np.isfinite(delta).all() or np.linalg.norm(delta) <= 0
            or type(plan['gravity_compensation']) is not bool):
        raise ValueError('Finite bounded force, motion and active interval followed by release required')
    if 'orientation_hold' in plan:
        hold = plan['orientation_hold']
        if (not isinstance(hold, dict) or set(hold) != {'stiffness_Nm_rad', 'torque_limit_Nm'}
                or not np.isfinite(list(hold.values())).all() or min(hold.values()) <= 0):
            raise ValueError('Orientation tool needs finite positive stiffness and torque bound')
    hold,ramp=plan.get('end_hold_s',0.),plan.get('release_ramp_s',0.)
    if (type(hold) not in (int,float) or type(ramp) not in (int,float)
            or not np.isfinite([hold,ramp]).all() or min(hold,ramp)<0
            or hold+ramp>=plan['duration_s']
            or plan.get('motion_profile','linear') not in {'linear','minimum_jerk'}):
        raise ValueError('Finite hold/release intervals must leave a positive motion interval')
    return dict(plan)


def tool_schedule(plan,time):
    """Target progress and wrench envelope; neither prescribes body motion."""
    t=time-plan['start_s'];duration=plan['duration_s'];ramp=plan.get('release_ramp_s',0.)
    moving=duration-plan.get('end_hold_s',0.)-ramp
    progress=float(np.clip(t/moving,0,1))
    if plan.get('motion_profile','linear')=='minimum_jerk':
        progress=progress**3*(10-15*progress+6*progress**2)
    envelope=1.
    if t<0 or t>=duration:envelope=0.
    elif ramp and t>duration-ramp:
        u=(t-duration+ramp)/ramp;envelope=1-u*u*(3-2*u)
    return progress,envelope


class TranslationProbe:
    def __init__(self, plan, mass, dt, inertia_local=None):
        if not np.isfinite([mass, dt]).all() or min(mass, dt) <= 0:
            raise ValueError('Positive finite native mass and timestep required')
        if plan['gravity_compensation'] and plan['force_limit_N'] <= 9.81*mass:
            raise ValueError('Tool force budget cannot hold the native mass against gravity; '
                             'declare adequate capacity before testing controlled insertion')
        self.plan, self.mass, self.dt = plan, float(mass), float(dt)
        self.origin = None
        self.peak_force = 0.
        self.orientation = None
        self.peak_torque = 0.
        self.inertia_axes=None;self.inertia_values=None
        if plan.get('orientation_hold'):
            inertia=np.asarray(inertia_local,float)
            if (inertia.shape!=(3,3) or not np.isfinite(inertia).all()
                    or not np.allclose(inertia,inertia.T,rtol=1e-6,atol=1e-12)):
                raise ValueError('Read the complete symmetric native local inertia tensor before applying tool torque')
            values,axes=np.linalg.eigh((inertia+inertia.T)/2)
            if (values<=0).any():raise ValueError('Positive native inertia required')
            self.inertia_values,self.inertia_axes=values,axes

    def torque(self, time, rotation, angular_velocity):
        """Bounded test-tool torque; never a joint or prescribed orientation."""
        hold = self.plan.get('orientation_hold')
        t = time-self.plan['start_s']
        if not hold or t < 0 or t >= self.plan['duration_s']:
            return np.zeros(3)
        R, _ = rigid_transform(rotation, [0., 0., 0.])
        omega = np.asarray(angular_velocity, float)
        if omega.shape != (3,) or not np.isfinite(omega).all():
            raise ValueError('Finite native angular velocity required')
        if self.orientation is None:
            self.orientation = R.copy()
        from scipy.spatial.transform import Rotation
        error = Rotation.from_matrix(self.orientation@R.T).as_rotvec()
        # Thin objects have very different roll and tilt inertias. Using the
        # smallest moment for every axis severely underdamps the heavy axes.
        basis=R@self.inertia_axes
        error=basis.T@error;omega=basis.T@omega
        k, inertia = hold['stiffness_Nm_rad'], self.inertia_values
        damping = 2*np.sqrt(k*inertia)
        torque = (k*(error-self.dt*omega)-damping*omega)/(1+damping*self.dt/inertia+k*self.dt**2/inertia)
        torque=basis@torque
        torque *= min(1., hold['torque_limit_Nm']/max(np.linalg.norm(torque), 1e-12))*tool_schedule(self.plan,time)[1]
        self.peak_torque = max(self.peak_torque, float(np.linalg.norm(torque)))
        return torque

    def force(self, time, position, velocity):
        p, v = np.asarray(position, float), np.asarray(velocity, float)
        if (p.shape != (3,) or v.shape != (3,) or not np.isfinite(np.r_[p, v, time]).all()):
            raise ValueError('Finite actual native position and velocity required')
        t = time-self.plan['start_s']
        if t < 0 or t >= self.plan['duration_s']:
            return np.zeros(3)
        if self.origin is None:
            self.origin = p.copy()
        progress,envelope=tool_schedule(self.plan,time)
        goal = self.origin+np.asarray(self.plan['translation_m'])*progress
        k = self.plan['stiffness_N_m']
        f = implicit_pd(p[None], v[None], goal, np.array([self.mass]), k,
                        2*np.sqrt(k*self.mass), self.dt)[0]
        if self.plan['gravity_compensation']:
            f += np.array([0., 0., 9.81*self.mass])
        f *= min(1., self.plan['force_limit_N']/max(np.linalg.norm(f), 1e-12))*envelope
        self.peak_force = max(self.peak_force, float(np.linalg.norm(f)))
        return f
