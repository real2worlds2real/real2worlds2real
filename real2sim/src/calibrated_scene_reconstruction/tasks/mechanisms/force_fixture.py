"""Bounded force applicators: move a controller target, never cloth vertices."""
import numpy as np


def implicit_pd(points,velocities,goal,masses,stiffness,damping,dt):
    """Finite per-node backward-Euler spring/damper fixture forces, in SI.

    Caller applies force bounds and any explicit gravity compensation. This
    does not impose a position or change the object's constitutive parameters.
    """
    p,v,g,m,k,c=map(lambda x:np.asarray(x,float),[points,velocities,goal,masses,stiffness,damping])
    if (p.ndim!=2 or p.shape[1:]!=(3,) or not len(p) or v.shape!=p.shape or g.shape not in {(3,),p.shape}
            or m.shape!=(len(p),) or k.shape not in {(),m.shape} or c.shape not in {(),m.shape}
            or not all(np.isfinite(x).all() for x in [p,v,g,m,k,c])
            or (m<=0).any() or (k<0).any() or (c<0).any() or not np.isfinite(dt) or dt<=0):
        raise ValueError('Finite matching native state, positive masses/time and nonnegative fixture gains required')
    k=np.broadcast_to(k,m.shape);c=np.broadcast_to(c,m.shape)
    return (k[:,None]*(g-p-dt*v)-c[:,None]*v)/(1+c*dt/m+k*dt*dt/m)[:,None]


def material_patch(material_points,centre_fraction,size_m):
    """Select one finite area in unfolded coordinates, never nearby plies."""
    points=np.asarray(material_points,float);fraction=np.asarray(centre_fraction,float);size=np.asarray(size_m,float)
    if (points.ndim!=2 or points.shape[1:]!=(3,) or not np.isfinite(points).all()
            or fraction.shape!=(2,) or size.shape!=(2,) or not np.isfinite([fraction,size]).all()
            or (fraction<0).any() or (fraction>1).any() or (size<=0).any()
            or (size>np.ptp(points[:,:2],axis=0)).any()):
        raise ValueError('Finite unfolded coordinates, bounded material fractions and physical patch size required')
    centre=points[:,:2].min(0)+fraction*np.ptp(points[:,:2],axis=0)
    selected=np.flatnonzero((abs(points[:,:2]-centre)<=size/2+1e-10).all(1))
    if len(selected)<3 or np.linalg.matrix_rank(points[selected]-points[selected].mean(0),tol=1e-10)<2:
        raise ValueError('Material handle needs a resolved finite-area patch, not a point or line')
    return selected


def edge_patch(material_points,depth,*,axis=1,width=None,lateral_axis=0,centre_fraction=1.):
    """A finite material patch, optionally bounded along the selected edge.

    A full material-width strip is retained for explicit legacy controls only.
    On folded sheets it can reach several spatially adjacent but materially
    distant plies at once. Supply a physical ``width`` for a local handle;
    proximity in the folded shape is never a reason to select another ply.
    The centre fraction is a fixture hypothesis, not a source grasp detection.
    """
    points=np.asarray(material_points,float)
    if (points.ndim!=2 or points.shape[1:]!=(3,) or not np.isfinite(points).all()
            or type(axis) is not int or axis not in (0,1,2) or not np.isfinite(depth) or depth<=0
            or depth>np.ptp(points[:,axis])/2):
        raise ValueError('Finite material coordinates and a bounded positive grasp depth required')
    selected=np.flatnonzero(points[:,axis]>=points[:,axis].max()-depth-1e-10)
    if width is not None:
        if (type(lateral_axis) is not int or lateral_axis not in (0,1,2) or lateral_axis==axis
                or not np.isfinite([width,centre_fraction]).all() or width<=0
                or width>np.ptp(points[:,lateral_axis]) or not 0<=centre_fraction<=1):
            raise ValueError('Bounded physical width, distinct material axis and edge centre fraction required')
        # A complete cloth may have clipped/frayed corners. Locate the handle
        # on the actual selected edge strip, not an absent bounding-box corner.
        low,high=points[selected,lateral_axis].min(),points[selected,lateral_axis].max()
        if high-low<width-1e-10:raise ValueError('Actual material edge cannot resolve the declared grasp width')
        centre=low+centre_fraction*(high-low)
        start=np.clip(centre-width/2,low,high-width)
        selected=selected[(points[selected,lateral_axis]>=start-1e-10)&
                          (points[selected,lateral_axis]<=start+width+1e-10)]
    if len(selected)<3:raise ValueError('Finite-area grasp needs at least three material nodes')
    return selected


def approach(current, goal, maximum_step):
    current, goal = np.asarray(current, float), np.asarray(goal, float)
    if (current.shape != goal.shape or not np.isfinite(current).all() or not np.isfinite(goal).all()
            or not np.isfinite(maximum_step) or maximum_step <= 0):
        raise ValueError('Finite matching targets and a positive step bound required')
    delta = goal-current
    return current+delta*min(1., maximum_step/max(float(np.linalg.norm(delta)), 1e-12))


class ForceHandle:
    """A compliant distributed edge handle, without a rigid patch or position pin.

    Speed, force and gain are fixture assumptions, not measured material values.
    Per-node damping is bounded using actual native mass and timestep. The force
    budget bounds the sum of nodal force magnitudes, including gravity support.
    """
    def __init__(self, points, masses, *, dt, held_mass, force_limit, speed=.05, stiffness=20., ramp_seconds=.5):
        points, masses = np.asarray(points, float), np.asarray(masses, float)
        values = [dt, held_mass, force_limit, speed, stiffness, ramp_seconds]
        if (points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 3
                or masses.shape != (len(points),) or not np.isfinite(points).all()
                or not np.isfinite(masses).all() or (masses <= 0).any()
                or not np.isfinite(values).all() or min(values) <= 0 or dt > 1/960+1e-12):
            raise ValueError('Finite native patch state, positive physical parameters and fine timestep required')
        self.weights = masses/masses.sum()
        self.target = np.average(points, axis=0, weights=self.weights)
        self.count, self.dt, self.speed = len(points), float(dt), float(speed)
        self.force_limit, self.stiffness = float(force_limit), float(stiffness)
        self.ramp_seconds, self.elapsed = float(ramp_seconds), 0.
        self.requested_damping = float(2*np.sqrt(stiffness*held_mass)/len(points))
        self.damping = np.minimum(2*np.sqrt(stiffness*held_mass)*self.weights, .5*masses/dt)
        self.peak_force, self.peak_target_step = 0., 0.

    def forces(self, points, velocities, goal, *, load=(0., 0., 0.)):
        points, velocities, load = np.asarray(points, float), np.asarray(velocities, float), np.asarray(load, float)
        if (points.shape != (self.count, 3) or velocities.shape != points.shape or load.shape != (3,)
                or not np.isfinite(points).all() or not np.isfinite(velocities).all() or not np.isfinite(load).all()):
            raise ValueError('Read finite matching native particle state for each force application')
        before = self.target.copy()
        self.target = approach(self.target, goal, self.speed*self.dt)
        centre = np.average(points, axis=0, weights=self.weights)
        spring = self.stiffness*(self.target-centre)*self.weights[:, None]
        force = spring-self.damping[:, None]*velocities
        force += load*min(1., self.elapsed/self.ramp_seconds)*self.weights[:, None]
        cap = self.force_limit*self.weights
        magnitude = np.linalg.norm(force, axis=1)
        force *= np.minimum(1., cap/np.maximum(magnitude, 1e-12))[:, None]
        self.peak_force = max(self.peak_force, float(np.linalg.norm(force, axis=1).sum()))
        self.peak_target_step = max(self.peak_target_step, float(np.linalg.norm(self.target-before)))
        self.elapsed += self.dt
        return force

    def report(self):
        return {'positions_prescribed': False, 'rigid_patch_imposed': False,
                'force_distribution': 'Actual selected nodal mass fractions; invariant to unequal material sampling',
                'target_speed_m_s': self.speed, 'force_limit_N': self.force_limit,
                'gravity_ramp_s': self.ramp_seconds,
                'peak_sum_nodal_force_magnitudes_N': self.peak_force,
                'peak_controller_target_step_m': self.peak_target_step,
                'requested_per_node_damping_Ns_m': self.requested_damping,
                'maximum_applied_per_node_damping_Ns_m': float(self.damping.max())}
