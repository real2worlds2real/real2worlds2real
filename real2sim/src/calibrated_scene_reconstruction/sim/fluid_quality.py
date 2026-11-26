"""Resolution-aware passive liquid diagnostics; never source/task approval.

A maximum velocity alone confuses sub-resolution constraint jitter with bulk
motion. Require bounded bulk/surface drift, kinetic motion, per-step extrema
and an unchanged mass ledger. Compare only an explicitly settled time window.
"""
import numpy as np


def passive(history, points, velocity, *, spacing_m, physics_hz, window_start_s,
            mass_error_kg, initial_mass_kg):
    p=np.asarray(points,float);v=np.asarray(velocity,float)
    if (p.ndim!=2 or p.shape[1]!=3 or len(p)<8 or v.shape!=p.shape
            or not np.isfinite(p).all() or not np.isfinite(v).all()
            or not np.isfinite([spacing_m,physics_hz,window_start_s,mass_error_kg,initial_mass_kg]).all()
            or min(spacing_m,physics_hz,initial_mass_kg)<=0):
        raise ValueError('Finite native fluid state and explicit numerical resolution required')
    rows=[r for r in history if r['time_s']>=window_start_s]
    times=np.asarray([r['time_s']for r in rows])
    if len(rows)<5 or np.any(np.diff(times)<=0) or times[-1]-times[0]<.75:
        raise ValueError('Passive fluid review needs at least five samples over 0.75 seconds')
    centres=np.asarray([r['centre']for r in rows],float)
    surface=np.asarray([r['surface_height_p95_m']for r in rows],float)
    if centres.shape!=(len(rows),3) or not np.isfinite(centres).all() or not np.isfinite(surface).all():
        raise ValueError('Finite sampled native centre and free-surface height required')
    speed=np.linalg.norm(v,axis=1)
    values=dict(centre_drift_m=float(np.linalg.norm(np.ptp(centres,axis=0))),
                surface_variation_m=float(np.ptp(surface)),
                rms_speed_m_s=float(np.sqrt(np.mean(speed**2))),
                maximum_step_motion_m=float(speed.max()/physics_hz),
                mass_error_kg=float(abs(mass_error_kg)))
    limits=dict(centre_drift_m=.25*spacing_m,surface_variation_m=.25*spacing_m,
                rms_speed_m_s=.1*np.sqrt(9.81*spacing_m),
                maximum_step_motion_m=.5*spacing_m,
                mass_error_kg=max(1e-10,initial_mass_kg*1e-6))
    checks={k:dict(pass_check=bool(values[k]<=limits[k]),value=values[k],limit=limits[k])for k in values}
    return dict(pass_numerical=all(r['pass_check']for r in checks.values()),checks=checks,
                accepted=False,scope=__doc__.strip(),window_s=[float(times[0]),float(times[-1])],
                limitations=['Container retention and spill require separate geometry checks',
                             'No source appearance, material identifiability or task acceptance'])
