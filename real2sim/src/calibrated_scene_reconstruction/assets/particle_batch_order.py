"""Verify a bulk native readback against independent single-body views."""
import numpy as np


def check(single,bulk):
    a,b=np.asarray(single),np.asarray(bulk)
    if (a.ndim!=3 or a.shape[2]!=3 or a.shape!=b.shape or len(a)<2
            or not np.isfinite([a,b]).all()):raise ValueError('Equal finite body/node readbacks required')
    tolerance=8*np.finfo(np.float32).eps*max(1.,float(abs(a).max()))
    error=float(abs(a-b).max())
    if error>tolerance:raise ValueError('Bulk particle order differs from independently addressed bodies')
    flat=a.reshape(len(a),-1)
    for i in range(len(a)):
        if i and np.linalg.norm(flat[:i]-flat[i],axis=1).min()<=16*tolerance:
            raise ValueError('Coincident particle bodies cannot establish a unique bulk body order')
    return {'pass':True,'body_count':len(a),'maximum_position_discrepancy':error,
            'tolerance':tolerance,'states_changed':False,
            'basis':'Bulk rows checked against uniquely distinguishable, individually addressed actual native body states'}
