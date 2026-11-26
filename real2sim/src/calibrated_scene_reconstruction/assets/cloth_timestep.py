"""Observed one-step travel relative to thin contact scales; diagnostic only."""
import numpy as np


def measure(velocities,physics_hz,thickness_m,rest_offset_m):
    v=np.asarray(velocities,float)
    if (v.ndim!=2 or v.shape[1:]!=(3,) or not len(v) or not np.isfinite(v).all()
            or type(physics_hz)is not int or physics_hz<=0
            or not np.isfinite([thickness_m,rest_offset_m]).all()
            or thickness_m<=0 or rest_offset_m<0):
        raise ValueError('Finite actual nodal velocities, positive native clock and resolved contact scales required')
    speed=np.linalg.norm(v,axis=1);travel=float(speed.max()/physics_hz)
    return {'physics_hz':physics_hz,'observed_peak_speed_m_s':float(speed.max()),
            'observed_peak_one_step_travel_m':travel,'physical_thickness_m':float(thickness_m),
            'travel_to_physical_thickness':travel/thickness_m,
            'native_rest_band_m':float(2*rest_offset_m),
            'travel_to_native_rest_band':travel/(2*rest_offset_m) if rest_offset_m else None,
            'contact_approved':False,'task_approved':False,
            'scope':'Instantaneous velocity-times-step diagnostic, not maximum intervening travel, CCD or solver convergence. Compare identical geometry/material at finer clocks when thin-contact release fails; never infer stability from a coupon or velocity cap alone.'}
