"""Conditional gravity loads for source-observed static articulation states."""
import numpy as np


def gravity_effort(masses, centres, anchor, axis, *, gravity=(0.,0.,-9.81)):
    """General revolute-axis load, using current actual link COMs and mass priors.

    This is not friction identification. For an observed resting, unsupported
    articulation, opposing torque can come from friction, elasticity, a detent
    or unobserved contact. The agent must select/test these model hypotheses.
    No motor or positional constraint is inferred or added by this function.
    """
    mass=np.asarray(masses,float);com=np.asarray(centres,float)
    anchor=np.asarray(anchor,float);axis=np.asarray(axis,float);g=np.asarray(gravity,float)
    if (mass.ndim!=1 or not len(mass) or com.shape!=(len(mass),3)
            or anchor.shape!=(3,) or axis.shape!=(3,) or g.shape!=(3,)
            or not all(np.isfinite(x).all() for x in [mass,com,anchor,axis,g])
            or (mass<=0).any() or not np.isclose(np.linalg.norm(axis),1.,atol=1e-6)):
        raise ValueError('Positive masses, actual COMs and a finite unit hinge axis required')
    per_link=np.cross(com-anchor,mass[:,None]*g)@axis
    effort=float(per_link.sum())
    return {'gravity_effort_Nm':effort,'per_link_effort_Nm':per_link.tolist(),
            'opposing_effort_required_Nm':abs(effort),
            'friction_identified':False,'task_approved':False,
            'scope':'Conditional static load only. Source stationarity, mass/COM priors, contacts and the passive joint model require separate evidence.'}
