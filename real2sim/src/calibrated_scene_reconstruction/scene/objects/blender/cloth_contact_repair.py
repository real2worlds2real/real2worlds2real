"""Bounded layer-clearance repair of offline geometry, never native state.

This only resolves small contact tolerances in an unapproved construction
hypothesis. Full independent authored and fresh native/task checks still apply.
"""
import numpy as np

from calibrated_scene_reconstruction.assets.cloth_clearance import measure,contact_witnesses
from calibrated_scene_reconstruction.assets.cloth_contact import self_intersections


def repair(points,faces,rest,clearance,*,maximum_displacement=None,iterations=64,device='cpu',on_iteration=None):
    original=np.asarray(points,float);points=original.copy();faces=np.asarray(faces,int)
    budget=.1*clearance if maximum_displacement is None else float(maximum_displacement)
    if not np.isfinite([clearance,budget]).all() or min(clearance,budget)<=0 or type(iterations) is not int or not 1<=iterations<=128:
        raise ValueError('Bounded explicit offline geometry-repair budget required')
    if not self_intersections(points,faces,device=device)['pass']:
        raise ValueError('Layer tolerance repair cannot resolve an intersecting construction')
    before=measure(points,faces,rest,clearance,device=device,example_limit=100,proposal_candidates=True);history=[]
    if before['minimum_gap_within_search_m'] is not None and before['minimum_gap_within_search_m']<clearance-2*budget:
        raise ValueError('Layer deficit exceeds the small offline proposal-repair budget')
    current=before
    for i in range(iterations):
        if current['pass']:break
        pairs=np.array(current['closest_violating_pairs'],int)
        witness=contact_witnesses(points,faces,pairs)
        indices=np.c_[witness['faces_a'],witness['faces_b']]
        gradient=np.concatenate([witness['weights_a'][:,:,None]*witness['normal'][:,None],
                                -witness['weights_b'][:,:,None]*witness['normal'][:,None]],axis=1)
        deficit=np.maximum(0.,clearance+2e-6-witness['gap_m'])
        correction=deficit[:,None,None]*gradient/np.maximum((gradient**2).sum((1,2))[:,None,None],1e-15)
        move=np.zeros_like(points);counts=np.zeros(len(points))
        np.add.at(move,indices.ravel(),correction.reshape(-1,3))
        np.add.at(counts,indices.ravel(),(np.linalg.norm(gradient,axis=2)>1e-12).ravel())
        direction=.9*move/np.maximum(counts[:,None],1.);fraction=1.
        for _ in range(12):
            trial=points+fraction*direction
            displacement=float(np.linalg.norm(trial-original,axis=1).max())
            if displacement<=budget and self_intersections(trial,faces,device=device)['pass']:
                points=trial;break
            fraction*=.5
        else:raise ValueError('No nonintersecting contact repair step within its geometry displacement budget')
        current=measure(points,faces,rest,clearance,device=device,example_limit=100,proposal_candidates=True)
        row={'iteration':i+1,'violating_pairs':current['violating_pairs'],'maximum_displacement_m':displacement,
             'minimum_gap_m':current['minimum_gap_within_search_m'],'proposal_pairs':len(current['closest_violating_pairs']),
             'step_fraction':fraction}
        history.append(row)
        if on_iteration is not None:on_iteration(points.copy(),row)
    if not current['pass'] or not self_intersections(points,faces,device=device)['pass']:
        raise ValueError('Bounded layer repair did not produce an admissible nonintersecting proposal')
    return points,{'proposal_only':True,'task_approved':False,'maximum_displacement_budget_m':budget,
                   'before':before,'after':current,'iterations':history,
                   'maximum_displacement_m':float(np.linalg.norm(points-original,axis=1).max()),
                   'scope':'Offline spatial clearance correction only; rest material, topology and physical estimates unchanged.'}
