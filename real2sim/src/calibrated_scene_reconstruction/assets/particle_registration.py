"""Strict source/native identity check, invariant to common warm-up translation.

The translation is used only to verify indices. Actual positions and velocities
remain untouched and must separately pass geometry, contact and task checks.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def native_output(source_faces,source_rest,output_faces,output_rest,output_world,
                  native_before,native_after,velocity_before,velocity_after):
    """Check solver-published USD vertex order against its independent tensor.

    A deformed sheet need not remain nearest to its authored vertices. The
    native output mesh still uses the authored face indices and material chart.
    This check requires that output, not an agent-written copy of tensor data.
    The caller must request only solver output, with no integration/state edits.
    """
    f,g=np.asarray(source_faces),np.asarray(output_faces)
    rest=np.asarray(source_rest,dtype=np.float32);other=np.asarray(output_rest,dtype=np.float32)
    a,b,c=map(lambda p:np.asarray(p,float),[output_world,native_before,native_after])
    va,vb=np.asarray(velocity_before,float),np.asarray(velocity_after,float)
    if (f.ndim!=2 or f.shape[1:]!=(3,) or f.dtype.kind not in 'iu'
            or not np.array_equal(f,g) or not np.array_equal(rest,other)):
        raise ValueError('Native output changed source topology or material chart')
    if (rest.ndim!=2 or rest.shape[1:]!=(3,) or len(rest)<3
            or any(p.shape!=rest.shape for p in [a,b,c,va,vb])
            or not all(np.isfinite(p).all() for p in [rest,a,b,c,va,vb])):
        raise ValueError('Complete finite native output, positions and velocities required')
    if not np.array_equal(b,c) or not np.array_equal(va,vb):
        raise ValueError('Publishing solver output changed actual native state')
    tolerance=8*np.finfo(np.float32).eps*max(1.,abs(a).max(),abs(b).max())
    error=float(np.linalg.norm(a-b,axis=1).max())
    if error>tolerance:
        raise ValueError('Solver USD vertex order does not match native tensor indices')
    return {'indexed_particle_order_preserved':True,'states_changed':False,
            'registration_method':'Solver-published USD mesh and native tensor, unchanged authored topology/rest chart',
            'native_output_maximum_difference_m':error,'native_output_tolerance_m':float(tolerance),
            'native_velocities_unchanged':True,'source_topology_and_material_chart_preserved':True,
            'scope':'Index correspondence only. Actual warm-up shape, contact and material validity remain separate checks.'}


def check(source,native):
    source,native=np.asarray(source,float),np.asarray(native,float)
    if (source.ndim!=2 or source.shape[1:]!=(3,) or native.shape!=source.shape
            or len(source)<3 or not np.isfinite([source,native]).all()):
        raise ValueError('Equal finite native/source particle arrays required')
    shift=native.mean(0)-source.mean(0)
    separation=float(cKDTree(source).query(source,k=2)[0][:,1].min())
    distances,indices=cKDTree(native-shift).query(source,k=2)
    distance,index=distances[:,0],indices[:,0]
    tolerance=.45*separation
    # Separated row minima give a cheap certificate of a unique assignment.
    # If rows compete for a neighbour, solve the complete assignment instead
    # of silently relabeling particles or rejecting a still-unique identity.
    # A global displacement ball, in contrast, incorrectly couples a free
    # point's normal motion to the smallest edge anywhere in the sheet.
    roundoff=8*np.finfo(np.float32).eps*max(1.,abs(source).max(),abs(native).max())
    margin=distances[:,1]-distances[:,0]
    ordered=np.array_equal(index,np.arange(len(source)))
    original=separation>0 and ordered and distance.max()<=tolerance
    certified=original or (separation>0 and ordered and margin.min()>4*roundoff)
    method='Original ordered displacement-ball certificate' if original else 'Separated ordered row minima'
    cycle_margin=None
    # Keep the established cheap path for arbitrarily large meshes. Dense
    # global certification is a bounded fallback, never an unbounded CPU/RAM
    # allocation when the source sampling itself is ambiguous.
    if not certified and separation>0 and len(source)<=512:
        cost=cdist(source,native-shift)
        if np.isfinite(cost).all():
            _,assignment=linear_sum_assignment(cost)
            if np.array_equal(assignment,np.arange(len(source))):
                # Every alternative permutation decomposes into directed
                # cycles. Certify that each cycle costs more than its changed
                # rows' float32 uncertainty. Floyd-Warshall detects any zero
                # or negative cycle after subtracting that per-row budget.
                paths=cost-np.diag(cost)[:,None]-4*roundoff
                np.fill_diagonal(paths,np.inf)
                for k in range(len(paths)):
                    paths=np.minimum(paths,paths[:,k,None]+paths[k,None,:])
                cycle_margin=float(np.diag(paths).min())
                certified=cycle_margin>0
                method='Ordered global assignment with noise-separated cycle certificate'
    if not certified:
        raise ValueError(f'Native particle/source mapping is not one-to-one: '
            f'native={len(native)},source={len(source)},unique={len(np.unique(index))},'
            f'maximum_distance={distance.max()},minimum_assignment_margin={margin.min()},'
            f'required_margin={4*roundoff},adjusted_cycle_margin={cycle_margin},'
            f'global_assignment_node_budget=512,translation={shift.tolist()}')
    return {'warmup_translation':shift.tolist(),'maximum_registered_distance':float(distance.max()),
            'minimum_source_separation':separation,'unambiguous_registration_tolerance':tolerance,
            'previous_global_distance_screen_pass':bool(distance.max()<=tolerance),
            'registration_method':method,'minimum_noise_adjusted_cycle_cost_m':cycle_margin,
            'minimum_assignment_margin_m':float(margin.min()),'required_assignment_margin_m':float(4*roundoff),
            'indexed_particle_order_preserved':True,'states_changed':False,
            'scope':'Index correspondence only; actual warm-up deformation and translation remain in all physical checks'}
