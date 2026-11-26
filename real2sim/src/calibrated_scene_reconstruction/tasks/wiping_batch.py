"""Continuous, lifted transfers between all strokes from one observation."""
import numpy as np


def batch_legs(strokes,start,park,offset,contact_z,*,clearance=.10,wipe_speed=.025):
    strokes=np.asarray(strokes,float)
    if strokes.size==0:return []
    if strokes.ndim!=3 or strokes.shape[1:]!=(2,2) or not np.isfinite(strokes).all():
        raise ValueError('Strokes must be finite Nx2x2 table coordinates')
    if wipe_speed<=0 or clearance<=0 or (np.linalg.norm(np.diff(strokes,axis=1),axis=2)<1e-8).any():
        raise ValueError('Nonzero strokes, speed and clearance are required')
    current=np.asarray(start,float);park=np.asarray(park,float);offset=np.asarray(offset,float)
    lift=np.array([0,0,clearance]);legs=[]
    def add(first,last,label,speed,index):
        if np.linalg.norm(last-first)>1e-8:legs.append((first.copy(),last.copy(),label,speed,index))
    for i,stroke in enumerate(strokes):
        a,b=[np.r_[xy+offset,contact_z] for xy in stroke]
        add(current,a+lift,'Approach' if i==0 else 'Raised transfer',.05,i)
        add(a+lift,a,'Lower to table',.05,i)
        add(a,b,'Wipe',wipe_speed,i)
        add(b,b+lift,'Lift',.06,i)
        current=b+lift
    # Preserve the validated final return posture before moving to the camera
    # inspection position. Intermediate strokes never return to this park.
    add(current,a+lift,'Raised transfer',.05,i)
    add(a+lift,park,'Retract for next photo',.05,i)
    return legs
