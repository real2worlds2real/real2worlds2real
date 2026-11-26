"""Experimental conservative hinge bending; NOT an accepted Isaac backend.

Independent natural curvature avoids changing the intrinsic membrane metric.
This is a discrete-shell hinge energy (Grinspun et al., SCA2003,
https://www.multires.caltech.edu/pubs/ds.pdf) with a physical rigidity
normalization checked on cylindrical strips. Native membrane/contact coupling,
numerical convergence and delivery integration must be validated separately.
No geometry positions, pins or targets are written by this module.
"""
import numpy as np
import torch

from calibrated_scene_reconstruction.assets.cloth_curvature import adjacency
from calibrated_scene_reconstruction.assets.cloth_projection import _solve,rigid_projector


def angles(q):
    """One signed radian dihedral for each [edge0,edge1,left,right] hinge."""
    a,b,c,d=q.unbind(1);edge=b-a
    left=torch.linalg.cross(edge,c-a);right=torch.linalg.cross(-edge,d-b)
    lengths=[torch.linalg.vector_norm(v,dim=1) for v in [edge,left,right]]
    if any(not torch.isfinite(v).all() or bool((v<=1e-14).any()) for v in lengths):
        raise ValueError('Nondegenerate finite hinge geometry required')
    tangent=edge/lengths[0][:,None];left=left/lengths[1][:,None];right=right/lengths[2][:,None]
    return torch.atan2((torch.linalg.cross(left,right)*tangent).sum(1),(left*right).sum(1))


class DiscreteBending:
    """Implicit velocity impulse from an internal elastic bending energy.

    Native membrane and contacts are NOT implemented here. This prototype
    requires a separately validated native uniform nodal mass model. Natural
    angles are construction priors, not inferred automatically from a pose.
    """
    def __init__(self,rest,faces,natural_radians,rigidity,*,device='cpu',dtype=torch.float64):
        rest=np.asarray(rest,float);faces=np.asarray(faces)
        if (rest.ndim!=2 or rest.shape[1:]!=(3,) or not np.isfinite(rest).all()
                or faces.ndim!=2 or faces.shape[1:]!=(3,) or faces.dtype.kind not in 'iu'
                or not len(faces) or faces.min()<0 or faces.max()>=len(rest)
                or not np.isfinite(rigidity) or rigidity<=0):
            raise ValueError('Finite material triangles and positive flexural rigidity required')
        pairs,edges=adjacency(faces);hinges=[]
        for pair,edge in zip(pairs,edges):
            left=[x for x in faces[pair[0]] if x not in edge]
            right=[x for x in faces[pair[1]] if x not in edge]
            if len(left)!=1 or len(right)!=1:raise ValueError('Distinct oriented adjacent faces required')
            hinges.append([*edge,left[0],right[0]])
        natural=np.asarray(natural_radians,float)
        if natural.shape!=(len(hinges),) or not np.isfinite(natural).all() or (abs(natural)>np.pi).any():
            raise ValueError('One finite natural angle in radians per actual hinge required')
        tri=rest[faces];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
        if (area<=1e-14).any():raise ValueError('Nondegenerate intrinsic material triangles required')
        square=np.sum((rest[edges[:,1]]-rest[edges[:,0]])**2,axis=1)
        self.material_length_scale=float(np.sqrt(np.median(square)))
        self.coefficient=torch.tensor(rigidity*square/area[pairs].sum(1),dtype=dtype,device=device)
        self.ids=torch.tensor(hinges,dtype=torch.long,device=device)
        self.natural=torch.tensor(natural,dtype=dtype,device=device)
        self.count=len(rest);self.dtype=dtype;self.device=torch.device(device)

    def residual(self,value):
        theta=angles(value[self.ids]);difference=theta-self.natural
        return torch.atan2(torch.sin(difference),torch.cos(difference))

    def energy(self,value):
        return .5*(self.coefficient*self.residual(value).square()).sum()

    def forces(self,value):
        q=value.detach().requires_grad_(True)
        return -torch.autograd.grad(self.energy(q),q)[0]

    def linearize(self,q,*,membrane=None,cohesion=None,viscous=None):
        local=q[self.ids].detach().requires_grad_(True)
        theta=angles(local);derivative=torch.autograd.grad(theta.sum(),local)[0]
        difference=theta.detach()-self.natural
        difference=torch.atan2(torch.sin(difference),torch.cos(difference))
        ids=self.ids;coefficient=self.coefficient
        for law in [membrane,cohesion,viscous]:
            if law is not None:
                ii,gg,rr,cc=law.linearize(q)
                ids=torch.cat([ids,ii]);derivative=torch.cat([derivative,gg])
                difference=torch.cat([difference,rr]);coefficient=torch.cat([coefficient,cc])
        return ids,derivative,difference,coefficient


    def velocity_step(self,positions,velocities,masses,dt,*,cg_iterations=1000,tolerance=1e-8,cohesion=None,membrane=None,linear_solver='cg',nonlinear_backtracking=False,nonlinear_iterations=1,on_nonlinear_iteration=None,viscosity_terms=None):
        """Linearly implicit internal impulse, with no external position goal.

        The Gauss-Newton tangent is positive semidefinite. Its rigid-motion
        nullspace conserves linear/angular momentum up to solve precision.
        This is a numerical splitting method; coupled native controls and
        timestep convergence remain mandatory before production use.
        """
        q=torch.as_tensor(positions,dtype=self.dtype,device=self.device)
        v=torch.as_tensor(velocities,dtype=self.dtype,device=self.device)
        m=torch.as_tensor(masses,dtype=self.dtype,device=self.device)
        if (q.shape!=(self.count,3) or v.shape!=q.shape or m.shape!=(self.count,)
                or not all(torch.isfinite(x).all() for x in [q,v,m]) or bool((m<=0).any())
                or not torch.allclose(m,m[:1].expand_as(m),rtol=1e-6,atol=0.)
                or not np.isfinite(dt) or dt<=0 or type(cg_iterations) is not int or not 1<=cg_iterations<=1000
                or type(nonlinear_iterations) is not int or not 1<=nonlinear_iterations<=50):
            raise ValueError('Finite native states, validated uniform masses and bounded integration controls required')
        viscous=None
        if viscosity_terms is not None:
            from .discrete_dissipation import IncrementalDissipation
            viscous=IncrementalDissipation(q,dt,viscosity_terms)
        if nonlinear_iterations>1:
            return self._nonlinear_velocity_step(q,v,m,dt,membrane=membrane,cohesion=cohesion,viscous=viscous,
                linear_solver=linear_solver,iterations=nonlinear_iterations,cg_iterations=cg_iterations,tolerance=tolerance,on_iteration=on_nonlinear_iteration)
        ids,derivative,difference,coefficient=self.linearize(q,membrane=membrane,cohesion=cohesion,viscous=viscous)
        predicted=difference+dt*(derivative*v[ids]).sum((1,2))
        rhs=torch.zeros_like(q)
        rhs.index_add_(0,ids.reshape(-1),
                       (-dt*coefficient[:,None,None]*predicted[:,None,None]*derivative).reshape(-1,3))
        gradients=dt*coefficient.sqrt()[:,None,None]*derivative
        dv,used=_solve(ids,gradients,rhs,float(m[0]),iterations=cg_iterations,tolerance=tolerance,
                       block_preconditioner=membrane is not None,rigid_points=q,method=linear_solver)
        product=(gradients*dv[ids]).sum((1,2))
        applied=m[:,None]*dv
        applied.index_add_(0,ids.reshape(-1),(product[:,None,None]*gradients).reshape(-1,3))
        relative_residual=float(torch.linalg.vector_norm(applied-rhs)/torch.linalg.vector_norm(rhs).clamp_min(1e-30))
        if relative_residual>max(1e-5,10*tolerance):
            raise ValueError(f'Internal bending solve did not converge: relative residual {relative_residual:.6g}')
        fraction=1.;objective_before=objective_after=None
        if nonlinear_backtracking:
            # Globalize the elastic update against its actual nonlinear
            # backward-Euler objective, rather than accepting a tangent step
            # that creates energy in a tightly folded configuration. A scalar
            # line search preserves the internal impulse's rigid nullspace.
            # This is still one split elastic update, not a converged nonlinear
            # contact solve or a replacement for timestep/material controls.
            def objective(change):
                trial=q+dt*(v+change)
                result=.5*(m[:,None]*change.square()).sum()
                for law in [self,membrane,cohesion,viscous]:
                    if law is not None:result=result+law.energy(trial)
                return float(result)
            with torch.no_grad():
                objective_before=objective(torch.zeros_like(dv))
                for backtrack in range(25):
                    objective_after=objective(fraction*dv)
                    if np.isfinite(objective_after) and objective_after<=objective_before:break
                    fraction*=.5
                else:raise ValueError('Internal elastic update has no energy-descending nonlinear step')
            dv=dv*fraction
        impulse=m[:,None]*dv
        report={'experimental_only':True,'linear_solver':linear_solver,'linear_solver_iterations':used,
                'cg_iterations':used if linear_solver=='cg' else 0,'relative_solver_residual':relative_residual,'linear_impulse_Ns':impulse.sum(0).detach().cpu().tolist(),
                'angular_impulse_Nms':torch.linalg.cross(q-q.mean(0),impulse).sum(0).detach().cpu().tolist(),
                'maximum_internal_velocity_change_m_s':float(torch.linalg.vector_norm(dv,dim=1).max()),
                'energy_before_J':float(self.energy(q)),'positions_written':False,'native_or_task_approval':False,
                'nonlinear_backtracking':bool(nonlinear_backtracking),'internal_step_fraction':fraction,
                'nonlinear_objective_before_J':objective_before,'nonlinear_objective_after_J':objective_after,
                'constitutive_dissipation_J':0. if viscous is None else 2*float(viscous.energy(q+dt*(v+dv)))}
        return v+dv,report

    def _nonlinear_velocity_step(self,q,v,m,dt,*,membrane,cohesion,viscous,linear_solver,iterations,cg_iterations,tolerance,on_iteration):
        """Constrained backward-Euler elastic minimization before native contacts.

        The six uniform-mass rigid modes at the beginning of the step are
        excluded, so every accepted internal update conserves momentum. Actual
        strain/contact validity and native coupling remain independent checks.
        """
        # Keep the rigid velocity and start from zero elastic deformation.
        # A ballistic guess can stretch a loaded boundary by many elements
        # before optimization, selecting an unrelated folded local minimum.
        # This is an initial guess only; the same inertial objective, external
        # momentum and nonlinear convergence requirement remain in force.
        project=rigid_projector(q);change=-project(v);rows=[];first_norm=None
        laws=[law for law in [self,membrane,cohesion,viscous] if law is not None]
        def objective(delta):
            trial=q+dt*(v+delta)
            return float(.5*(m[:,None]*delta.square()).sum()+sum(law.energy(trial) for law in laws))
        before=objective(change);maximum_residual=0.;total_iterations=0;fractions=[]
        converged=False;convergence_measure=None;step_ratio=None
        for iteration in range(iterations+1):
            trial=q+dt*(v+change)
            ids,derivative,difference,coefficient=self.linearize(trial,membrane=membrane,cohesion=cohesion,viscous=viscous)
            gradient=m[:,None]*change
            gradient.index_add_(0,ids.reshape(-1),
                (dt*coefficient[:,None,None]*difference[:,None,None]*derivative).reshape(-1,3))
            rhs=-project(gradient);norm=float(torch.linalg.vector_norm(rhs))
            if first_norm is None:first_norm=norm
            relative=norm/max(first_norm,1e-30)
            if norm<1e-16 or relative<=max(1e-6,10*tolerance):
                converged=True;convergence_measure='gradient';break
            if iteration==iterations:break
            gradients=dt*coefficient.sqrt()[:,None,None]*derivative
            direction,used=_solve(ids,gradients,rhs,float(m[0]),iterations=cg_iterations,tolerance=tolerance,
                block_preconditioner=membrane is not None,rigid_points=q,method=linear_solver)
            applied=m[:,None]*direction
            product=(gradients*direction[ids]).sum((1,2))
            applied.index_add_(0,ids.reshape(-1),(product[:,None,None]*gradients).reshape(-1,3))
            residual=float(torch.linalg.vector_norm(project(applied)-rhs)/torch.linalg.vector_norm(rhs).clamp_min(1e-30))
            if residual>max(1e-5,10*tolerance):raise ValueError('Constrained nonlinear elastic tangent did not converge')
            maximum_residual=max(maximum_residual,residual);total_iterations+=used
            step_ratio=dt*float(torch.linalg.vector_norm(direction,dim=1).max())/self.material_length_scale
            # Near equilibrium the reference force can itself be tiny; a
            # relative gradient alone then asks for an unrepresentable state.
            # Test the unscaled Newton correction in intrinsic element units,
            # not the shortened line-search step or a task acceptance bound.
            if step_ratio<=tolerance:
                converged=True;convergence_measure='material_step';break
            baseline=objective(change);fraction=1.
            with torch.no_grad():
                for backtrack in range(30):
                    proposed=change+fraction*direction;energy=objective(proposed)
                    if np.isfinite(energy) and energy<=baseline+1e-4*fraction*float((gradient*direction).sum()):break
                    fraction*=.5
                else:raise ValueError('Nonlinear elastic minimization failed its energy line search')
            change=proposed;fractions.append(fraction)
            rows.append({'iteration':iteration,'objective_J':energy,'relative_gradient_residual':relative,'step_fraction':fraction,
                         'unscaled_material_step':step_ratio})
            if on_iteration is not None:on_iteration((q+dt*(v+change)).detach().cpu().numpy(),rows[-1])
        impulse=m[:,None]*change
        return v+change,{'experimental_only':True,'linear_solver':linear_solver,
            'linear_solver_iterations':total_iterations,'cg_iterations':total_iterations if linear_solver=='cg' else 0,
            'relative_solver_residual':maximum_residual,'linear_impulse_Ns':impulse.sum(0).detach().cpu().tolist(),
            'angular_impulse_Nms':torch.linalg.cross(q-q.mean(0),impulse).sum(0).detach().cpu().tolist(),
            'maximum_internal_velocity_change_m_s':float(torch.linalg.vector_norm(change,dim=1).max()),
            'positions_written':False,'native_or_task_approval':False,'nonlinear_backtracking':True,
            'internal_step_fraction':min(fractions,default=1.),'nonlinear_objective_before_J':before,
            'nonlinear_objective_after_J':objective(change),'nonlinear_iterations':len(rows),
            'nonlinear_relative_gradient_residual':relative,'nonlinear_converged':converged,
            'nonlinear_convergence_measure':convergence_measure,'nonlinear_material_step':step_ratio,
            'nonlinear_material_step_tolerance':tolerance,'material_length_scale_m':self.material_length_scale,
            'nonlinear_trace':rows,'energy_before_J':float(self.energy(q)),
            'constitutive_dissipation_J':0. if viscous is None else 2*float(viscous.energy(q+dt*(v+change)))}
