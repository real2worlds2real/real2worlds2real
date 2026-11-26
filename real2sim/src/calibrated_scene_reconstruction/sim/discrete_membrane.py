"""Experimental objective plane-stress shell membrane; no native replacement.

This diagnostic provides common implicit rows for membrane/bending/interface
experiments. Native coupling, force calibration, convergence and delivery are
unvalidated. The immutable intrinsic material metric is its only reference.
"""
import numpy as np
import torch

from calibrated_scene_reconstruction.assets.cloth_rest import material_inverse


class DiscreteMembrane:
    def __init__(self,rest,faces,*,youngs_modulus,poissons_ratio,thickness,device='cpu',dtype=torch.float64):
        rest,faces=np.asarray(rest,float),np.asarray(faces)
        if (not np.isfinite([youngs_modulus,poissons_ratio,thickness]).all()
                or min(youngs_modulus,thickness)<=0 or not 0<=poissons_ratio<.5):
            raise ValueError('Positive continuum parameters and nonauxetic plane-stress model required')
        inverse=material_inverse(rest,faces);tri=rest[faces]
        area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
        lam=youngs_modulus*thickness*poissons_ratio/(1-poissons_ratio**2)
        mu=youngs_modulus*thickness/(2*(1+poissons_ratio))
        self.faces=torch.tensor(faces,dtype=torch.long,device=device)
        self.inverse=torch.tensor(inverse,dtype=dtype,device=device)
        self.coefficient=torch.tensor(area[:,None]*np.array([2*mu,2*mu,2*mu,lam]),dtype=dtype,device=device)
        self.dtype=dtype;self.device=torch.device(device)

    def residual(self,local):
        deformation=torch.stack([local[:,1]-local[:,0],local[:,2]-local[:,0]],-1)@self.inverse
        strain=.5*(deformation.transpose(1,2)@deformation-torch.eye(2,dtype=self.dtype,device=self.device))
        return torch.stack([strain[:,0,0],strain[:,1,1],2**.5*strain[:,0,1],strain[:,0,0]+strain[:,1,1]],1)

    def energy(self,points):
        return .5*(self.coefficient*self.residual(points[self.faces]).square()).sum()

    def linearize(self,points):
        local=points[self.faces].detach().requires_grad_(True);residual=self.residual(local)
        gradient=torch.stack([torch.autograd.grad(residual[:,i].sum(),local,retain_graph=i<3)[0] for i in range(4)],1)
        # Pad to the four-node hinge/interface row convention with zero weights.
        ids=torch.cat([self.faces,self.faces[:,:1]],1)[:,None].expand(-1,4,-1)
        gradient=torch.nn.functional.pad(gradient,(0,0,0,1))
        return ids.reshape(-1,4),gradient.reshape(-1,4,3),residual.detach().reshape(-1),self.coefficient.reshape(-1)
