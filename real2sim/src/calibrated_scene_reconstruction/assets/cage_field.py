"""Smooth bounded source-shape proposals; never physical state control.

The field is a construction tool. Its output needs independent semantic,
source, geometry and native physical verification before becoming an asset.
"""
import numpy as np


def gradient_bound(lower,upper,offsets):
    """Conservative global Lipschitz bound, including clamped cage borders.

    Each cell's Jacobian is multi-affine, hence a convex combination of its
    corner Jacobians. The Frobenius norm bounds the operator norm. A bound
    strictly below one certifies injectivity of x + field(x); it does not
    certify straight-triangle interpolation, finite thickness or root contact.
    """
    lo,hi=np.asarray(lower,float),np.asarray(upper,float);f=np.asarray(offsets,float)
    if (lo.shape!=(3,) or hi.shape!=(3,) or (hi<=lo).any() or f.ndim!=4 or f.shape[-1]!=3
            or min(f.shape[:3])<2 or not all(np.isfinite(x).all() for x in (lo,hi,f))):
        raise ValueError('Finite nonempty cage required')
    step=(hi-lo)/(np.array(f.shape[:3])-1);maximum=0.
    for i in range(f.shape[0]-1):
        for j in range(f.shape[1]-1):
            for k in range(f.shape[2]-1):
                for a in (0,1):
                    for b in (0,1):
                        for c in (0,1):
                            J=np.stack([(f[i+1,j+b,k+c]-f[i,j+b,k+c])/step[0],
                                (f[i+a,j+1,k+c]-f[i+a,j,k+c])/step[1],
                                (f[i+a,j+b,k+1]-f[i+a,j+b,k])/step[2]],axis=1)
                            maximum=max(maximum,float(np.linalg.norm(J)))
    return maximum


def conditioned(lower,upper,offsets,*,maximum_gradient=.75):
    if not 0<maximum_gradient<1:raise ValueError('Strictly contractive construction field required')
    f=np.asarray(offsets,float);bound=gradient_bound(lower,upper,f)
    scale=min(1.,maximum_gradient/max(bound,1e-30))
    translation=f.mean(axis=(0,1,2))
    return translation+(f-translation)*scale,{'original_gradient_bound':bound,'scale':scale,
        'translation_component':translation.tolist(),
        'gradient_bound':bound*scale,'minimum_map_stretch_bound':1-bound*scale,
        'scope':'Continuous cage map only; actual mesh topology, source, finite thickness and native checks remain mandatory'}


def torch_conditioned(lower,upper,offsets,*,maximum_gradient=.75):
    """Differentiable equivalent for source construction searches only."""
    import torch
    if not 0<maximum_gradient<1:raise ValueError('Strictly contractive construction field required')
    step=(torch.as_tensor(upper,device=offsets.device,dtype=offsets.dtype)-
          torch.as_tensor(lower,device=offsets.device,dtype=offsets.dtype))/offsets.new_tensor(offsets.shape[:3]).sub(1)
    derivatives=[torch.diff(offsets,dim=k)/step[k] for k in range(3)]
    nx,ny,nz=[n-1 for n in offsets.shape[:3]];bounds=[]
    for a in (0,1):
        for b in (0,1):
            for c in (0,1):
                J=torch.stack([derivatives[0][:,b:b+ny,c:c+nz],derivatives[1][a:a+nx,:,c:c+nz],
                               derivatives[2][a:a+nx,b:b+ny,:]],dim=-1)
                bounds.append(torch.sqrt(J.square().sum((-1,-2))+1e-30).max())
    bound=torch.stack(bounds).max();scale=torch.clamp(maximum_gradient/bound.clamp_min(1e-15),max=1.)
    translation=offsets.mean((0,1,2))
    return translation+scale*(offsets-translation)


def flow(points,lower,upper,velocity,*,steps=16):
    """Compose contractive-residual maps instead of folding a large offset.

    The construction-time velocity is not a simulated material velocity.
    Every Euler map must have Lipschitz residual below one. Curved-sheet
    triangle interpolation and finite thickness still need independent checks.
    """
    if type(steps) is not int or not 1<=steps<=256:raise ValueError('Bounded construction flow steps required')
    bound=gradient_bound(lower,upper,velocity)
    if bound/steps>=.9:raise ValueError('Construction flow needs more steps to avoid folding')
    q=np.array(points,float,copy=True)
    for _ in range(steps):q=apply(q,lower,upper,np.asarray(velocity)/steps)
    return q


def torch_flow(points,lower,upper,velocity,*,steps=16,weights=None):
    """Differentiable proposal; certify its saved field with NumPy before use."""
    q=points
    for _ in range(steps):q=torch_apply(q,lower,upper,velocity/steps,weights=weights)
    return q


def apply(points,lower,upper,offsets,*,weights=None,maximum_gradient=None):
    p=np.asarray(points,float);lo=np.asarray(lower,float);hi=np.asarray(upper,float);field=np.asarray(offsets,float)
    if (p.ndim!=2 or p.shape[1:]!=(3,) or lo.shape!=(3,) or hi.shape!=(3,) or np.any(hi<=lo)
            or field.ndim!=4 or field.shape[-1]!=3 or min(field.shape[:3])<2
            or not all(np.isfinite(x).all() for x in (p,lo,hi,field))):
        raise ValueError('Finite points, nonempty3D cage and vector offsets required')
    if maximum_gradient is not None:
        if not 0<maximum_gradient<1 or gradient_bound(lo,hi,field)>maximum_gradient+1e-9:
            raise ValueError('Construction cage can fold; condition its field before reauthoring')
        if weights is not None and not np.all(np.asarray(weights)==1):
            raise ValueError('Weighted transition needs its own full-map Jacobian check; field certificate alone is insufficient')
    shape=np.array(field.shape[:3]);q=np.clip((p-lo)/(hi-lo),0,1)*(shape-1)
    index=np.minimum(np.floor(q).astype(int),shape-2);fraction=q-index;delta=np.zeros_like(p)
    for a in (0,1):
        for b in (0,1):
            for c in (0,1):
                bits=np.array([a,b,c]);weight=np.where(bits,fraction,1-fraction).prod(1)
                cell=index+bits;delta+=field[cell[:,0],cell[:,1],cell[:,2]]*weight[:,None]
    w=np.ones(len(p)) if weights is None else np.asarray(weights,float)
    if w.shape!=(len(p),) or not np.isfinite(w).all() or np.any((w<0)|(w>1)):
        raise ValueError('One construction blend weight in0..1 per vertex required')
    return p+delta*w[:,None]


def torch_apply(points,lower,upper,offsets,*,weights=None):
    """Differentiable equivalent, for source proposals only."""
    import torch
    lo=torch.as_tensor(lower,dtype=points.dtype,device=points.device);hi=torch.as_tensor(upper,dtype=points.dtype,device=points.device)
    grid=(2*(points-lo)/(hi-lo)-1)[None,None,None]
    field=offsets.permute(3,2,1,0)[None]
    delta=torch.nn.functional.grid_sample(field,grid,mode='bilinear',padding_mode='border',align_corners=True)[0,:,0,0].T
    return points+delta*(1 if weights is None else weights[:,None])
