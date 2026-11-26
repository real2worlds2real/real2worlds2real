"""Generic native contact and bounded-wrench measurements."""
import numpy as np


def axial_wrench(force_N, velocity_m_s, *, damping_Ns_m, compensation_N, limit_N):
    """Finite tool force with viscous resistance, without an axial position target."""
    if (not np.isfinite([force_N, velocity_m_s, damping_Ns_m, compensation_N, limit_N]).all()
            or damping_Ns_m < 0 or limit_N <= 0):
        raise ValueError('Finite axial wrench and nonnegative damping required')
    return float(np.clip(force_N+compensation_N-damping_Ns_m*velocity_m_s, -limit_N, limit_N))


def stable_guide_damping(inertia_kg_m2, dt, stiffness_Nm_rad, requested_Nms_rad):
    """Bound a tool's explicit PD damping by actual inertia and integration dt.

    Semi-implicit angular integration requires k*dt**2/I+2*d*dt/I<4.
    This changes the external test guide, never the inferred object's material.
    """
    I, k, d = inertia_kg_m2, stiffness_Nm_rad, requested_Nms_rad
    if not np.isfinite([I, dt, k, d]).all() or min(I, dt, k) <= 0 or d < 0:
        raise ValueError('Finite positive inertia, time step and guide stiffness required')
    upper = 2*I/dt-.5*k*dt
    if upper <= 0:
        raise ValueError('Guide stiffness needs a smaller physics time step')
    return float(min(d, 2*np.sqrt(k*I), .8*upper))


def speed_controller_gains(inertia_kg_m2, dt):
    """Inertia-scaled critically damped PI speed loop with resolved bandwidth.

    This supplies wrench gains, never an angular pose or velocity reset. The
    force runner separately limits and records actual applied torque.
    """
    if not np.isfinite([inertia_kg_m2, dt]).all() or min(inertia_kg_m2, dt) <= 0:
        raise ValueError('Positive actual inertia and physics time step required')
    bandwidth = min(80., .1/dt)
    return 2*bandwidth*inertia_kg_m2, bandwidth**2*inertia_kg_m2, bandwidth
