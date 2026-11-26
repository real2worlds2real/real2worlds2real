"""Measure declared reflection priors on actual closed source components.

Symmetry is a hypothesis supplied by semantic understanding, never an
assumption imposed on every object class. Units and source geometry stay intact.
"""
import json
import numpy as np
from scipy.optimize import linear_sum_assignment


def equivalent(actual, recorded):
    """Allow floating-point eigensolver roundoff, not a changed geometry result."""
    if isinstance(actual, dict):
        return (isinstance(recorded, dict) and actual.keys() == recorded.keys()
                and all(equivalent(value, recorded[key]) for key, value in actual.items()))
    if isinstance(actual, list):
        return (isinstance(recorded, list) and len(actual) == len(recorded)
                and all(equivalent(a, b) for a, b in zip(actual, recorded)))
    if type(actual) in (int, float) and type(recorded) in (int, float):
        return bool(np.isfinite([actual, recorded]).all()
                    and np.isclose(actual, recorded, rtol=1e-9, atol=1e-9))
    return type(actual) is type(recorded) and actual == recorded


def reflection(source,request):
    import trimesh
    names=request.get('parts');normal=np.asarray(request.get('normal'),float)
    if (not isinstance(names,list) or len(names)<2 or len(set(names))!=len(names)
            or normal.shape!=(3,) or not np.isfinite(normal).all() or not np.isclose(normal@normal,1)):
        raise ValueError('Reflection needs distinct actual components and a unit normal')
    limits={k:request.get(k) for k in ['position_error_fraction','axis_error_deg','length_error_fraction']}
    if any(type(v) not in (int,float) or not np.isfinite(v) or v<=0 for v in limits.values()):
        raise ValueError('Explicit positive structural-prior tolerances required')
    records={p['name']:p for p in json.loads(str(source['parts_json']))}
    centres=[];axes=[];lengths=[]
    for name in names:
        p=records[name];a,n,b,f=[p[k] for k in ['vertex_start','vertex_count','face_start','triangle_count']]
        mesh=trimesh.Trimesh(source['V'][a:a+n],source['F'][b:b+f]-a,process=False)
        if not mesh.is_volume:raise ValueError('Structural component must be a closed physical solid')
        values,basis=np.linalg.eigh(mesh.moment_inertia);axis=basis[:,0]
        if values[1]<=values[0]*1.1:raise ValueError('Component has no resolved longitudinal axis')
        centres.append(mesh.center_mass);axes.append(axis);lengths.append(np.ptp(mesh.vertices@axis))
    centres=np.asarray(centres);axes=np.asarray(axes);lengths=np.asarray(lengths)
    span=float(np.linalg.norm(np.ptp(centres,axis=0)))
    if span<=1e-9:raise ValueError('Distinct structural positions required')
    offset=float(centres.mean(0)@normal)
    reflected=centres-2*(centres@normal-offset)[:,None]*normal
    mirrored_axes=axes-2*(axes@normal)[:,None]*normal
    position=np.linalg.norm(reflected[:,None]-centres[None],axis=2)/span
    angle=np.degrees(np.arccos(np.clip(abs(mirrored_axes@axes.T),0,1)))
    rows,cols=linear_sum_assignment(position+angle/180)
    errors={'position_error_fraction':float(position[rows,cols].max()),
            'axis_error_deg':float(angle[rows,cols].max()),
            'length_error_fraction':float((abs(lengths[rows]-lengths[cols])/np.maximum(lengths[rows],lengths[cols])).max())}
    return {'pass':all(errors[k]<=limits[k] for k in limits),'measurements':errors,'limits':limits,
            'pairs':[[names[i],names[j]] for i,j in zip(rows,cols)],'normal':normal.tolist(),
            'plane_offset_m':offset,'span_m':span,
            'scope':'Declared reflection prior on actual source-solid mass centres, longitudinal axes and lengths; source semantics and support checks remain separate.'}
