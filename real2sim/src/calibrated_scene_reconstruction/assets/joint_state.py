"""Source-declared initial articulation intervals shared by independent checks.

Intervals constrain task initialization, not unloaded seating or visual accuracy.
Legacy callers provide their existing defaults when no source interval is set.
"""
import numpy as np


def interval(mechanism,default):
    declaration=mechanism.get('initial_state')
    if declaration is None:return np.asarray(default,float)
    if (mechanism.get('type') not in {'revolute','prismatic'} or not isinstance(declaration,dict)
            or set(declaration)!={'position_interval','basis'}
            or not isinstance(declaration['basis'],str) or not declaration['basis'].strip()):
        raise ValueError('Initial joint state requires a source basis and position interval')
    values=np.asarray(declaration['position_interval'],float);limits=np.asarray(mechanism.get('limits'),float)
    if (values.shape!=(2,) or limits.shape!=(2,) or not np.isfinite(values).all()
            or not np.isfinite(limits).all() or values[0]>values[1] or values[0]<limits[0] or values[1]>limits[1]):
        raise ValueError('Initial position interval must lie within finite joint limits')
    return values


def measure(mechanism,position,*,default):
    bounds=interval(mechanism,default)
    if not np.isfinite(position) or bounds.shape!=(2,) or not np.isfinite(bounds).all() or bounds[0]>bounds[1]:
        raise ValueError('Finite joint state and ordered initial interval required')
    return {'pass':bool(bounds[0]<=position<=bounds[1]),'actual':float(position),
            'position_interval':bounds.tolist(),'unit':'rad' if mechanism.get('type')=='revolute' else 'm',
            'basis':mechanism.get('initial_state',{}).get('basis','Legacy checker default'),
            'scope':'Initial joint coordinate only; source shape, trapped-material contacts and unloaded seating remain separate checks.'}


def select(contract,parent,child):
    rows=[m for m in contract.get('mechanisms',[]) if m.get('type') in {'revolute','prismatic'}
          and set(m.get('links',[]))=={parent,child}]
    if len(rows)!=1:raise ValueError('One declared articulation required for the initial-state measurement')
    return rows[0]
