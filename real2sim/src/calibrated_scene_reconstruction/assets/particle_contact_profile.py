"""Sample particle contact envelopes from material thickness, not mesh holes."""
import numpy as np


def sample(reference,thickness):
    h0=reference['thickness'];rest0=reference['rest_offset'];solid0=reference['solid_rest_offset']
    contact0=reference['contact_offset'];particle0=reference['particle_contact_offset']
    if (not np.isfinite([h0,rest0,solid0,contact0,particle0,thickness]).all() or min(h0,thickness)<=0
            or min(rest0,solid0)<h0/2*(1-1e-6) or contact0<=rest0 or particle0<=solid0):
        raise ValueError('Coherent physical sheet thickness and explicit particle contact reference required')
    rest=thickness/2+max(0.,rest0-h0/2);solid=thickness/2+max(0.,solid0-h0/2)
    return {'rest_offset':float(rest),'solid_rest_offset':float(solid),
            'contact_offset':float(rest+contact0-rest0),
            'particle_contact_offset':float(solid+particle0-solid0)}
