"""Propose a bounded local step from recorded trials; never approve a process."""
import argparse

from calibrated_scene_reconstruction.physics.process.contract import read,write
from calibrated_scene_reconstruction.physics.process.proposal import linear_step


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--round',required=True);p.add_argument('--output',required=True)
    p.add_argument('--history-only',action='store_true',help='Read complete records from an intact started worker; this cannot certify or approve its round')
    p.add_argument('--physical-steps',help='JSON mapping each proposed parameter to its maximum step in declared physical units')
    p.add_argument('--parameters',nargs='+');p.add_argument('--trust-radius',type=float,default=.15)
    p.add_argument('--objective',choices=['training_loss','feasibility'],default='feasibility')
    a=p.parse_args()
    result=linear_step(a.round,trust_radius=a.trust_radius,parameters=a.parameters,
        history_only=a.history_only,physical_steps=read(a.physical_steps) if a.physical_steps else None,
        objective=a.objective)
    write(a.output,result)
    print({'output':a.output,'base_trial':result['base_trial'],
           'estimated_largest_normalized_fit_error':result['estimated_largest_normalized_fit_error'],
           'accepted':False},flush=True)
