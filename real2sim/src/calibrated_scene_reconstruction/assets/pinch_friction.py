"""Finite Coulomb friction for a sheet actually between opposing pad faces.

Only measured contact geometry/velocity is accepted. No cloth position target,
grasp-node selection, remote attraction, or robot trajectory is used.
"""
import numpy as np


def bounded_static_projection(slip,masses,contact_indices,impulse_budget,dt):
    """PBD tangential correction inside a finite Coulomb impulse budget.

    The coupled sheet translation mass provides a conservative correction
    bound. Actual particle impulses are returned for the hand reaction.
    """
    correction=np.zeros_like(slip,dtype=float);ids=np.asarray(contact_indices,int)
    if not len(ids) or impulse_budget<=0:return correction,correction.copy()
    desired=-np.asarray(slip)[ids]
    effective=np.maximum(np.asarray(masses)[ids],np.sum(masses)/len(ids))
    required=float(np.sum(effective*np.linalg.norm(desired,axis=1))/dt)
    scale=min(1.,impulse_budget/max(required,1e-20))
    correction[ids]=desired*scale
    impulses=np.asarray(masses)[:,None]*correction/dt
    return correction,impulses


def pinch_impulses(points,velocities,masses,pads,normals,tangents,pad_velocity,dt,
                   coefficient=.8,radius=.00085,half_width=.006,
                   pad_stiffness=1500.,normal_force_limit=.3,tangential_slip=None):
    points=np.asarray(points,float);velocity=np.asarray(velocities,float)
    pads=np.asarray(pads,float);normal=np.asarray(normals,float)
    axis=normal[0]-normal[1];axis/=max(np.linalg.norm(axis),1e-12)
    gap=float((pads[1]-pads[0])@axis)
    empty=dict(contacts=0,normal_force_N=0.,impulse_Ns=0.,force_limit_N=0.,gap_m=gap,contact_indices=[],sliding_indices=[])
    impulses=np.zeros_like(velocity)
    if coefficient<=0 or gap<=0 or normal[0]@normal[1]>-.95:
        return impulses,empty
    pressure=min(normal_force_limit,pad_stiffness*max(0.,2*(radius+.0001)-gap))
    if pressure<=0:return impulses,empty
    hit=np.ones(len(points),bool);weights=np.ones(len(points))
    for p,n,t in zip(pads,normal,tangents):
        offset=points-p;signed=offset@n
        b=np.cross(n,t)
        hit&=(signed>=-.0002)&(signed<radius+.0002)&(abs(offset@t)<half_width)&(abs(offset@b)<half_width)
        weights*=np.maximum(radius+.0002-signed,0)
    ids=np.flatnonzero(hit)
    if not len(ids):return impulses,empty
    weights=weights[ids];weights/=weights.sum()
    relative=velocity[ids]-np.asarray(pad_velocity)
    tangent=relative-(relative@axis)[:,None]*axis
    if tangential_slip is not None:
        slip=np.asarray(tangential_slip)[ids]
        tangent+=.6*(slip-(slip@axis)[:,None]*axis)/dt
    speed=np.linalg.norm(tangent,axis=1)
    budget=coefficient*pressure*dt*weights
    size=np.minimum(np.asarray(masses)[ids]*speed,budget)
    impulses[ids]=-size[:,None]*tangent/np.maximum(speed[:,None],1e-15)
    return impulses,dict(contacts=len(ids),normal_force_N=pressure,
                         impulse_Ns=float(size.sum()),force_limit_N=coefficient*pressure,gap_m=gap,
                         contact_indices=ids.tolist(),sliding_indices=ids[np.asarray(masses)[ids]*speed>budget].tolist())
