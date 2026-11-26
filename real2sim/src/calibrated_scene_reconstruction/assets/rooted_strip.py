"""Reparameterize an authored sheet strip at its real support boundary.

Boolean clipping can create needle elements. This construction keeps a
structured sheet and its free rim, truncating each material column at the last
solid exit. It defines a new candidate, never changes a running rest shape.
The caller must check geometric error, topology, thickness and source fidelity.
"""
import numpy as np


def exposed(points, rows, columns, signed_distance):
    p=np.asarray(points,float)
    if (type(rows) is not int or type(columns) is not int or min(rows,columns)<2
            or p.shape!=(rows*columns,3) or not np.isfinite(p).all()):
        raise ValueError('Finite structured strip required')
    grid=p.reshape(rows,columns,3);result=np.empty_like(grid);exits=[]
    for j in range(columns):
        line=grid[:,j];d=np.array([signed_distance(q) for q in line])
        if not np.isfinite(d).all() or d[-1]<=0:
            raise ValueError('Every strip column needs an exposed free rim')
        inside=np.flatnonzero(d<=0)
        if not len(inside):raise ValueError('Every strip column needs an actual embedded root')
        k=int(inside[-1]);lo,hi=0.,1.
        for _ in range(28):
            t=(lo+hi)/2;q=line[k]*(1-t)+line[k+1]*t
            if signed_distance(q)<=0:lo=t
            else:hi=t
        start=k+(lo+hi)/2;sample=np.linspace(start,rows-1,rows)
        a=np.minimum(sample.astype(int),rows-2);t=sample-a
        result[:,j]=line[a]*(1-t[:,None])+line[a+1]*t[:,None]
        exits.append(float(start/(rows-1)))
    return result.reshape(-1,3),list(range(columns)),{'column_root_fractions':exits,
        'free_rim_unchanged':bool(np.array_equal(result[-1],grid[-1])),
        'scope':'Candidate construction only; no dynamic material state or acceptance modified'}
