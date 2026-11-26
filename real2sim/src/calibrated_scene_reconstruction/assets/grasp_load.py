"""Explicit load-based grasp-force proposals, independent of object identity."""
import numpy as np


def normal_force(mass_kg,friction,*,gravity_m_s2=9.81,fingers=2,safety_factor=2.):
    """Estimate opposing normal force per finger for a gravity-supported load.

    Coulomb friction is a proposal only. Contact geometry, other contact
    partners, uncertain material properties and native task tests remain
    necessary; this does not identify the object's real friction or mass.
    Gravity alone does not account for drag against a container, a snag or
    other supported material. The result is not a measured gripping force.
    """
    if (not np.isfinite([mass_kg,friction,gravity_m_s2,safety_factor]).all()
            or min(mass_kg,friction,gravity_m_s2)<=0 or safety_factor<1
            or type(fingers) is not int or fingers<2):
        raise ValueError('Positive load/friction and explicit opposing-finger safety margin required')
    return {'normal_force_per_finger_N':float(safety_factor*mass_kg*gravity_m_s2/(fingers*friction)),
        'mass_kg':float(mass_kg),'effective_friction':float(friction),'gravity_m_s2':float(gravity_m_s2),
        'fingers':fingers,'safety_factor':float(safety_factor),'proposal_only':True,
        'scope':'Gravity-load estimate for force-limited grasp planning; no material identification or task approval.'}
