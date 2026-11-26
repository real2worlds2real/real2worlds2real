"""Experimental Kelvin-Voigt dissipation inside the implicit shell solve.

Viscosity times are viscosity/modulus ratios in seconds, not Isaac damping
rates. This incremental potential is solved together with elastic response;
a separate velocity damping step can undo membrane corrections and add strain
energy. No positions, pins, world drag or source-pose targets are prescribed.
"""
import numpy as np
import torch


class IncrementalDissipation:
    def __init__(self,points,dt,terms):
        terms=list(terms)
        if not terms or not np.isfinite(dt) or dt<=0:
            raise ValueError('Explicit constitutive viscosity terms and positive timestep required')
        reference=terms[0][0]
        self.dtype,self.device=reference.dtype,reference.device
        self.origin=torch.as_tensor(points,dtype=self.dtype,device=self.device).detach().clone()
        if self.origin.ndim!=2 or self.origin.shape[1:]!=(3,) or not torch.isfinite(self.origin).all():
            raise ValueError('Finite material positions required')
        ids=[];gradients=[];coefficients=[]
        for law,tau in terms:
            if not np.isfinite(tau) or tau<0:raise ValueError('Nonnegative finite viscosity time in seconds required')
            if tau==0:continue
            ii,gg,_,cc=law.linearize(self.origin)
            ids.append(ii);gradients.append(gg.detach());coefficients.append(cc.detach()*tau/dt)
        self.ids=torch.cat(ids) if ids else torch.empty((0,4),dtype=torch.long,device=self.device)
        self.gradients=torch.cat(gradients) if ids else torch.empty((0,4,3),dtype=self.dtype,device=self.device)
        self.coefficient=torch.cat(coefficients) if ids else torch.empty(0,dtype=self.dtype,device=self.device)

    def residual(self,points):
        return (self.gradients*(points-self.origin)[self.ids]).sum((1,2))

    def energy(self,points):
        """Incremental dissipation potential; not stored elastic energy."""
        return .5*(self.coefficient*self.residual(points).square()).sum()

    def linearize(self,points):
        return self.ids,self.gradients,self.residual(points),self.coefficient
