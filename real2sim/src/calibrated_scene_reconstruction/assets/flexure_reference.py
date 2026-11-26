"""Small-amplitude free-strip bending reference for numerical controls only."""
import numpy as np
from scipy.signal import find_peaks


def mode(coordinate,length):
    coordinate=np.asarray(coordinate,float)
    if not np.isfinite(coordinate).all() or not np.isfinite(length) or length<=0:
        raise ValueError('Finite axial coordinates and positive strip length required')
    k=4.730040744862704;u=k*coordinate/length
    ratio=(np.cosh(k)-np.cos(k))/(np.sinh(k)-np.sin(k))
    return np.cosh(u)+np.cos(u)-ratio*(np.sinh(u)+np.sin(u))


def frequency(length,rigidity,areal_density):
    if not np.isfinite([length,rigidity,areal_density]).all() or min(length,rigidity,areal_density)<=0:
        raise ValueError('Positive finite free-strip properties required')
    return float(4.730040744862704**2/(2*np.pi*length**2)*np.sqrt(rigidity/areal_density))


def oscillation(times,amplitudes):
    t=np.asarray(times,float);q=np.asarray(amplitudes,float)
    if (t.ndim!=1 or q.shape!=t.shape or len(t)<20 or not np.isfinite([t,q]).all()
            or (np.diff(t)<=0).any() or not np.allclose(np.diff(t),np.diff(t).mean(),rtol=.01)):
        raise ValueError('Finite uniformly sampled flexure transient required')
    # A linear baseline cannot manufacture repeated oscillations. Require
    # independent positive and negative extrema, with no expected-frequency
    # search window. Small numerical chatter is below the amplitude floor.
    value=q-np.polyval(np.polyfit(t,q,1),t)
    prominence=max(float(np.ptp(value))*.05,1e-8)
    high,_=find_peaks(value,prominence=prominence)
    low,_=find_peaks(-value,prominence=prominence)
    periods=np.r_[np.diff(t[high]),np.diff(t[low])]
    if len(high)<2 or len(low)<2 or not len(periods):
        return {'pass':False,'why':'No resolved repeated free-flexure oscillation','frequency_hz':None}
    period=float(np.median(periods));scatter=float(np.std(periods)/period)
    return {'pass':bool(scatter<.15),'frequency_hz':1/period,'period_scatter':scatter,
            'positive_extrema_s':t[high].tolist(),'negative_extrema_s':t[low].tolist(),
            'scope':'Unforced oscillation measurement only; not an object material identification'}
