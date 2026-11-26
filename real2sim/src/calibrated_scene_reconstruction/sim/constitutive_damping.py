"""Diagnostic internal viscous step *before* native elastic integration.

No positions or rest geometry are changed. This split operator must pass
native release/energy, contact and timestep controls before use in a task.
It must not follow an elastic position correction: damping that correction
can restore the strain the elastic solve just removed.
"""
import numpy as np
import torch

from calibrated_scene_reconstruction.assets.cloth_projection import _solve,rigid_projector
from .discrete_dissipation import IncrementalDissipation


def velocity_step(points,velocities,masses,dt,terms,*,linear_solver='sparse_direct'):
    terms=list(terms)
    potential=IncrementalDissipation(points,dt,terms)
    q=potential.origin;v=torch.as_tensor(velocities,dtype=q.dtype,device=q.device)
    m=torch.as_tensor(masses,dtype=q.dtype,device=q.device)
    if (v.shape!=q.shape or m.shape!=q.shape[:1] or not torch.isfinite(v).all()
            or not torch.isfinite(m).all() or bool((m<=0).any())
            or not torch.allclose(m,m[:1].expand_as(m),rtol=1e-6,atol=0)):
        raise ValueError('Finite matched states and validated uniform native masses required')
    ids,derivative,_,coefficient=potential.linearize(q)
    gradients=dt*coefficient.sqrt()[:,None,None]*derivative
    project=rigid_projector(q)
    rhs=torch.zeros_like(v)
    rate=(gradients*v[ids]).sum((1,2))
    rhs.index_add_(0,ids.reshape(-1),(-rate[:,None,None]*gradients).reshape(-1,3))
    rhs=project(rhs)
    if float(torch.linalg.vector_norm(rhs))<1e-20:
        change=torch.zeros_like(v);iterations=0;residual=0.
    else:
        change,iterations=_solve(ids,gradients,rhs,float(m[0]),iterations=1000,tolerance=1e-8,
            block_preconditioner=True,rigid_points=q,method=linear_solver)
        applied=m[:,None]*change
        rate=(gradients*change[ids]).sum((1,2))
        applied.index_add_(0,ids.reshape(-1),(rate[:,None,None]*gradients).reshape(-1,3))
        residual=float(torch.linalg.vector_norm(project(applied)-rhs)/torch.linalg.vector_norm(rhs))
        if residual>1e-6:raise ValueError('Internal viscous solve did not converge')
    after=v+change;before_energy=float(.5*(m[:,None]*v.square()).sum())
    after_energy=float(.5*(m[:,None]*after.square()).sum())
    if after_energy>before_energy+max(1e-16,1e-8*before_energy):
        raise ValueError('Internal viscous step added kinetic energy')
    impulse=m[:,None]*change
    return after,{'diagnostic_only':True,'positions_written':False,'world_drag':False,
        'kinetic_energy_before_J':before_energy,'kinetic_energy_after_J':after_energy,
        'linear_impulse_Ns':impulse.sum(0).detach().cpu().tolist(),
        'angular_impulse_Nms':torch.linalg.cross(q-q.mean(0),impulse).sum(0).detach().cpu().tolist(),
        'linear_solver_iterations':iterations,'relative_solver_residual':residual,
        'native_coupling_qualified':False}
