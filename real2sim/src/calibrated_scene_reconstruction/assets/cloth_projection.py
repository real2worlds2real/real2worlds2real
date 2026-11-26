"""Global intrinsic-length geometry proposals; never simulation state writes.

Sparse Gauss-Newton steps avoid the slow long-range propagation of Jacobi
edge corrections. Actual source/contact/native acceptance stays with callers.
"""
import numpy as np
import torch

from .cloth_rest import triangle_edges


class MetricObjective:
    """Area-weighted continuum metric residuals for geometry construction.

    Narrow triangles can have small relative edge errors and substantial
    in-plane shear. Resolve both material directions explicitly, preserving
    the original chart. These rows are never native forces or a new rest pose.
    """
    def __init__(self,rest,faces,*,device='cpu'):
        from .cloth_rest import material_inverse
        rest=np.asarray(rest,float);faces=np.asarray(faces,int)
        tri=rest[faces];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
        self.faces=torch.tensor(faces,dtype=torch.long,device=device)
        self.inverse=torch.tensor(material_inverse(rest,faces),dtype=torch.float64,device=device)
        self.basis=torch.cat([-self.inverse.sum(1)[:,None,:],self.inverse],dim=1)
        self.scale=torch.tensor(np.sqrt(area),dtype=torch.float64,device=device)

    def values(self,points):
        tri=points[self.faces]
        deformation=torch.stack([tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]],dim=2)@self.inverse
        a,b=deformation[:,:,0],deformation[:,:,1]
        residual=torch.stack([.5*(a.square().sum(1)-1),.5*(b.square().sum(1)-1),
                              (a*b).sum(1)/np.sqrt(2)],dim=1)*self.scale[:,None]
        return residual,a,b

    def rows(self,points):
        residual,a,b=self.values(points);u,v=self.basis[:,:,0,None],self.basis[:,:,1,None]
        gradient=torch.stack([u*a[:,None,:],v*b[:,None,:],
                              (u*b[:,None,:]+v*a[:,None,:])/np.sqrt(2)],dim=1)*self.scale[:,None,None,None]
        return self.faces[:,None,:].expand(-1,3,-1).reshape(-1,3),gradient.reshape(-1,3,3),residual.reshape(-1)

    def energy(self,points):return self.values(points)[0].square().sum()


def rigid_projector(points):
    """Uniform-mass rigid-motion projection with one cached inertia matrix."""
    centre=points-points.mean(0)
    inertia=torch.eye(3,dtype=points.dtype,device=points.device)*centre.square().sum()-centre.T@centre
    if not torch.isfinite(inertia).all():
        raise ValueError('Finite matching internal-force geometry required')
    def project(value):
        if value.shape!=centre.shape:raise ValueError('Matching internal-force geometry required')
        linear=value-value.mean(0)
        omega=torch.linalg.solve(inertia,torch.linalg.cross(centre,linear).sum(0))
        return linear-torch.linalg.cross(omega.expand_as(centre),centre)
    return project


def _solve(indices,gradients,rhs,regularization,*,iterations,tolerance,block_preconditioner=False,rigid_points=None,method='cg'):
    """Matrix-free SPD normal equations with a diagonal preconditioner."""
    shape=rhs.shape;flat=indices.reshape(-1)
    def project(value):return value
    if rigid_points is not None:
        project=rigid_projector(rigid_points)
    if method not in {'cg','sparse_direct','cuda_direct'}:raise ValueError('Explicit supported internal linear solver required')
    if method in {'sparse_direct','cuda_direct'}:
        if method=='cuda_direct' and rhs.device.type!='cuda':raise ValueError('CUDA direct solver requires CUDA states')
        # Same SPD operator and rigid nullspace as CG. This is a numerical
        # solver choice, without stiffness scaling, position targets or pins.
        from scipy import sparse
        from scipy.sparse.linalg import splu
        ids=indices.detach().cpu().numpy();g=gradients.detach().cpu().numpy()
        columns=(ids[:,:,None]*3+np.arange(3)).reshape(-1)
        rows=np.repeat(np.arange(len(ids)),g.shape[1]*3)
        matrix=sparse.coo_matrix((g.reshape(-1),(rows,columns)),shape=(len(ids),rhs.numel())).tocsr()
        matrix.eliminate_zeros()
        normal=(matrix.T@matrix+sparse.eye(rhs.numel(),format='csc')*regularization).tocsc()
        # The normal operator is SPD. Symmetric minimum-degree ordering avoids
        # unnecessary fill from an unsymmetric column ordering on shell grids.
        basis=None
        if rigid_points is not None:
            # A nonlinear iteration's tangent need not annihilate rigid modes
            # at the beginning of the timestep. Enforce their six constraints
            # with the exact Schur complement, rather than projecting an
            # unconstrained solution after the solve.
            centre=(rigid_points-rigid_points.mean(0)).detach().cpu().numpy()
            translation=np.broadcast_to(np.eye(3),(len(centre),3,3))
            rotation=np.stack([np.cross(np.broadcast_to(e,centre.shape),centre) for e in np.eye(3)],axis=2)
            basis=np.linalg.qr(np.concatenate([translation,rotation],axis=2).reshape(-1,6),mode='reduced')[0]
        if method=='cuda_direct':
            from calibrated_scene_reconstruction.core.linalg.cudss import Factor
            normal=normal.tocsr()
            gpu=torch.sparse_csr_tensor(torch.as_tensor(normal.indptr,device=rhs.device,dtype=torch.int32),
                torch.as_tensor(normal.indices,device=rhs.device,dtype=torch.int32),
                torch.as_tensor(normal.data,device=rhs.device,dtype=rhs.dtype),size=normal.shape)
            right=project(rhs).reshape(-1)
            with Factor(gpu) as factor:
                if basis is None:value=factor.solve(right)
                else:
                    basis=torch.as_tensor(basis,device=rhs.device,dtype=rhs.dtype)
                    solved=factor.solve(torch.cat([right[:,None],basis],dim=1))
                    value=solved[:,0];inverse_basis=solved[:,1:]
                    value=value-inverse_basis@torch.linalg.solve(basis.T@inverse_basis,basis.T@value)
        else:
            factor=splu(normal,permc_spec='MMD_AT_PLUS_A',diag_pivot_thresh=0.,options={'SymmetricMode':True})
            value=factor.solve(project(rhs).detach().cpu().numpy().reshape(-1))
            if basis is not None:
                inverse_basis=factor.solve(basis)
                value-=inverse_basis@np.linalg.solve(basis.T@inverse_basis,basis.T@value)
        result=project(torch.as_tensor(value.reshape(shape),dtype=rhs.dtype,device=rhs.device))
        if not torch.isfinite(result).all():raise ValueError('Nonfinite sparse internal solve')
        return result,1
    diagonal=torch.full_like(rhs,regularization)
    diagonal.index_add_(0,flat,gradients.square().reshape(-1,3))
    if block_preconditioner:
        block=torch.eye(3,dtype=rhs.dtype,device=rhs.device)[None].repeat(len(rhs),1,1)*regularization
        local=gradients.reshape(-1,3)
        block.index_add_(0,flat,local[:,:,None]*local[:,None,:])
        inverse=torch.linalg.inv(block)
        def precondition(value):return (inverse@value[:,:,None])[:,:,0]
    else:
        def precondition(value):return value/diagonal
    def multiply(value):
        product=(gradients*value[indices]).sum((1,2))
        result=regularization*value
        result.index_add_(0,flat,(product[:,None,None]*gradients).reshape(-1,3))
        return result
    x=torch.zeros_like(rhs);r=project(rhs);z=project(precondition(r));p=z.clone();rz=(r*z).sum()
    norm=torch.linalg.vector_norm(rhs)
    if float(norm)<=1e-20:return x,0
    used=0
    for used in range(1,iterations+1):
        ap=project(multiply(p));pap=(p*ap).sum()
        if not torch.isfinite(pap) or float(pap)<=0:
            raise ValueError('Projection normal operator lost positive definiteness')
        alpha=rz/pap;x+=alpha*p;r-=alpha*ap
        if float(torch.linalg.vector_norm(r))<=tolerance*float(norm):break
        z=project(precondition(r));next_rz=(r*z).sum();p=z+(next_rz/rz)*p;rz=next_rz
    if x.shape!=shape or not torch.isfinite(x).all():raise ValueError('Nonfinite geometry projection step')
    return x,used


def project(rest,points,faces,*,iterations=100,cg_iterations=80,reference_pull=1e-7,
            maximum_step_m=.003,device='cpu',linear_contacts=None,admissible=None,on_iteration=None,
            hotspot_strain_scale=None,stop_condition=None,linear_solver='cg',objective='edges'):
    """Propose low-strain geometry with immutable intrinsic lengths.

    ``linear_contacts`` supplies local distance rows ``indices, gradients,
    signed clearance residual`` at the current geometry (positive is slack).
    Inactive inequalities add no normal stiffness. These proposal penalties never pin
    nodes in native physics. ``admissible`` independently screens each trial
    state. A rejected trial cannot replace the last valid output.
    """
    rest=np.asarray(rest,float);original=np.asarray(points,float)
    edges=triangle_edges(faces)
    if (rest.shape!=original.shape or original.ndim!=2 or original.shape[1:]!=(3,)
            or not np.isfinite(rest).all() or not np.isfinite(original).all()
            or type(iterations) is not int or not 1<=iterations<=2000
            or type(cg_iterations) is not int or not 1<=cg_iterations<=1000
            or not np.isfinite([reference_pull,maximum_step_m]).all()
            or not 0<reference_pull<=.1 or maximum_step_m<=0):
        raise ValueError('Finite matching material/state arrays and bounded projection controls required')
    if linear_solver not in {'cg','sparse_direct','cuda_direct'}:
        raise ValueError('Explicit supported projection linear solver required')
    if objective not in {'edges','triangle_metric'} or objective=='triangle_metric' and hotspot_strain_scale is not None:
        raise ValueError('Explicit supported geometry objective; edge-hotspot weighting applies only to edges')
    if admissible is not None and not admissible(original):
        raise ValueError('Geometry projection requires an independently admissible seed')
    if hotspot_strain_scale is not None and (not np.isfinite(hotspot_strain_scale)
            or not 1e-8<=hotspot_strain_scale<=.1):
        raise ValueError('Finite bounded hotspot strain scale required')
    # Double precision is useful for nearly isometric folded references.
    q=torch.tensor(original,dtype=torch.float64,device=device);target=q.clone()
    pairs=torch.tensor(edges,dtype=torch.long,device=device)
    lengths=torch.tensor(np.linalg.norm(rest[edges[:,1]]-rest[edges[:,0]],axis=1),device=device)
    if not len(lengths) or float(lengths.min())<=1e-12:raise ValueError('Positive material edge lengths required')
    base_weights=(lengths.median()/lengths).clamp(.1,100.);weights=base_weights
    continuum=MetricObjective(rest,faces,device=device) if objective=='triangle_metric' else None
    damping=1e-3;rows=[];requested_stop=False
    def energy(value):
        error=(torch.linalg.vector_norm(value[pairs[:,1]]-value[pairs[:,0]],dim=1)-lengths)*weights
        elastic=continuum.energy(value) if continuum is not None else error.square().sum()
        return float(elastic+reference_pull*(value-target).square().sum())
    for step in range(iterations):
        delta=q[pairs[:,1]]-q[pairs[:,0]];current=torch.linalg.vector_norm(delta,dim=1)
        if float(current.min())<=1e-12:raise ValueError('Collapsed geometry edge requires a new seed')
        if hotspot_strain_scale is not None:
            # Reweight scarce strained corners instead of letting a low RMS
            # hide their unchanged maximum-strain acceptance failure.
            weights=base_weights*(1+(current/lengths-1).abs()/hotspot_strain_scale).clamp(max=100.)
        direction=delta/current[:,None]*weights[:,None]
        residual=(current-lengths)*weights
        ids=pairs;grad=torch.stack([-direction,direction],1);contact_rows=None
        if continuum is not None:ids,grad,residual=continuum.rows(q)
        if linear_contacts is not None:
            contact=linear_contacts(q.detach().cpu().numpy())
            if contact is not None:
                ii,gg,rr=contact
                ii=torch.as_tensor(ii,dtype=torch.long,device=device)
                gg=torch.as_tensor(gg,dtype=q.dtype,device=device);rr=torch.as_tensor(rr,dtype=q.dtype,device=device)
                if (ii.ndim!=2 or gg.shape!=(*ii.shape,3) or rr.shape!=(len(ii),)
                        or ii.numel() and (int(ii.min())<0 or int(ii.max())>=len(q))
                        or not torch.isfinite(gg).all() or not torch.isfinite(rr).all()):
                    raise ValueError('Invalid proposal contact linearization')
                contact_rows=(ii,gg,rr)
        base_ids,base_grad,base_residual=ids,grad,residual
        used=0;change=torch.zeros_like(q)
        def linear_merit(value):
            error=base_residual+(base_grad*value[base_ids]).sum((1,2))
            total=error.square().sum()+reference_pull*(q-target+value).square().sum()+damping*value.square().sum()
            if contact_rows is not None:
                ci,cg,cr=contact_rows
                total+=(cr+(cg*value[ci]).sum((1,2))).clamp(max=0).square().sum()
            return float(total)
        # Solve the one-sided quadratic contact model. Re-evaluate the active
        # set at every inner iterate: retaining a now-positive row would turn
        # an inequality into an artificial attractive contact target.
        for active_iteration in range(8 if contact_rows is not None else 1):
            ids,grad=base_ids,base_grad
            residual=base_residual+(base_grad*change[base_ids]).sum((1,2))
            if contact_rows is not None:
                ci,cg,cr=contact_rows;gap=cr+(cg*change[ci]).sum((1,2));active=gap<0
                if bool(active.any()):
                    ii,gg,rr=ci[active],cg[active],gap[active]
                    k=max(ids.shape[1],ii.shape[1])
                    def pad(a,g):
                        return (torch.nn.functional.pad(a,(0,k-a.shape[1])),
                                torch.nn.functional.pad(g,(0,0,0,k-g.shape[1])))
                    ids,grad=pad(ids,grad);ii,gg=pad(ii,gg)
                    ids=torch.cat([ids,ii]);grad=torch.cat([grad,gg]);residual=torch.cat([residual,rr])
            rhs=-reference_pull*(q-target+change)-damping*change
            rhs.index_add_(0,ids.reshape(-1),(-residual[:,None,None]*grad).reshape(-1,3))
            increment,count=_solve(ids,grad,rhs,reference_pull+damping,iterations=cg_iterations,tolerance=1e-7,method=linear_solver)
            used+=count;before=linear_merit(change);inner_fraction=1.
            for _ in range(24):
                trial=change+inner_fraction*increment
                if linear_merit(trial)<before:
                    change=trial;break
                inner_fraction*=.5
            else:break
            if float(torch.linalg.vector_norm(inner_fraction*increment))<1e-10:break
        magnitude=float(torch.linalg.vector_norm(change,dim=1).max())
        if magnitude>maximum_step_m:change*=maximum_step_m/magnitude
        def merit(value):
            result=energy(value)
            if contact_rows is not None:
                ci,cg,cr=contact_rows
                gap=cr+(cg*(value-q)[ci]).sum((1,2))
                result+=float(gap.clamp(max=0).square().sum())
            return result
        baseline=merit(q);accepted=False;fraction=1.
        maximum_before=float((current/lengths-1).abs().max())
        for trial in range(12):
            candidate=q+fraction*change;candidate_energy=merit(candidate)
            peak=float((torch.linalg.vector_norm(candidate[pairs[:,1]]-candidate[pairs[:,0]],dim=1)/lengths-1).abs().max())
            peak_ok=hotspot_strain_scale is None or peak<=maximum_before+1e-10
            if peak_ok and candidate_energy<baseline and (admissible is None or admissible(candidate.cpu().numpy())):
                q=candidate;accepted=True;break
            fraction*=.5
        damping=max(1e-9,damping*.5) if accepted and fraction==1. else min(10.,damping*(1.5 if accepted else 10.))
        strain=torch.linalg.vector_norm(q[pairs[:,1]]-q[pairs[:,0]],dim=1)/lengths-1.
        row={'iteration':step,'accepted':accepted,'step_fraction':fraction,'cg_iterations':used,'contact_active_iterations':active_iteration+1,
             'objective':energy(q),'maximum_edge_strain':float(strain.abs().max()),
             'rms_edge_strain':float(strain.square().mean().sqrt()),'step_damping':damping,
             'maximum_displacement_m':float(torch.linalg.vector_norm(q-target,dim=1).max())}
        rows.append(row)
        if on_iteration is not None:on_iteration(q.detach().cpu().numpy(),row)
        if stop_condition is not None and stop_condition(q.detach().cpu().numpy(),row):
            requested_stop=True;break
        if accepted and row['maximum_edge_strain']<1e-6:break
        if not accepted and damping>=10.:break
    return q.detach().cpu().numpy(),{'iterations':rows,'geometry_objective':objective,'native_state_written':False,'stopped_by_requested_criterion':requested_stop,
        'material_lengths_changed':False,'source_approved':False,'task_approved':False,
        'scope':'Geometry proposal only; independent source, full triangle/contact/material and actual native checks remain required'}
