"""Experimental finite-strength interface history; no production registration.

Isotropic bilinear traction/separation diagnostic inspired by the standard
cohesive-zone construction documented by MOOSE/INL:
https://mooseframework.inl.gov/docs/site/source/userobjects/BilinearMixedModeCohesiveZoneModel.html
This simplified point/triangle discretization is NOT the MOOSE mortar/BK law.
Native contacts still own compression. Material priors and source evidence
must justify an interface; wet appearance alone does not identify adhesion.
"""
import numpy as np
import torch


def degradation(opening,*,stiffness,peak_traction,fracture_energy):
    onset=peak_traction/stiffness;failure=2*fracture_energy/peak_traction
    if (not np.isfinite([stiffness,peak_traction,fracture_energy]).all()
            or min(stiffness,peak_traction,fracture_energy)<=0 or onset>=failure):
        raise ValueError('Positive interface priors with elastic onset below finite failure required')
    value=1-failure*(opening-onset)/(opening.clamp_min(1e-30)*(failure-onset))
    return value.clamp(0.,1.)


def local_gap(points,barycentric):
    """Objective gap in the current triangle's own frame, not a world target."""
    point,a,b,c=points.unbind(1)
    tangent=b-a;normal=torch.linalg.cross(tangent,c-a)
    size=torch.linalg.vector_norm(tangent,dim=1);area=torch.linalg.vector_norm(normal,dim=1)
    if bool((size<=1e-12).any()) or bool((area<=1e-14).any()):
        raise ValueError('Cohesive interface requires nondegenerate current triangles')
    tangent=tangent/size[:,None];normal=normal/area[:,None]
    second=torch.linalg.cross(normal,tangent)
    centre=(points[:,1:]*barycentric[:,:,None]).sum(1)
    difference=point-centre
    return torch.stack([(difference*tangent).sum(1),(difference*second).sum(1),(difference*normal).sum(1)],1)


class CohesiveInterface:
    """Breakable material interfaces, with no fixed or positional targets.

    Rest gaps encode an explicit initial contacting-interface history. They do
    not change the membrane metric or connect a sheet to any rigid body.
    Finite debonding is irreversible here; rewetting/rebonding is unimplemented.
    """
    def __init__(self,initial,indices,barycentric,area,*,stiffness,peak_traction,fracture_energy,
                 shear_ratio=1.,device='cpu',dtype=torch.float64):
        q=np.asarray(initial,float);ids=np.asarray(indices);bary=np.asarray(barycentric,float);areas=np.asarray(area,float)
        if (q.ndim!=2 or q.shape[1:]!=(3,) or ids.ndim!=2 or ids.shape[1:]!=(4,)
                or ids.dtype.kind not in 'iu' or not len(ids) or ids.min()<0 or ids.max()>=len(q)
                or any(len(set(row))!=4 for row in ids) or bary.shape!=(len(ids),3)
                or areas.shape!=(len(ids),) or (areas<=0).any()
                or not all(np.isfinite(x).all() for x in [q,bary,areas])
                or (bary<0).any() or not np.allclose(bary.sum(1),1.,atol=1e-8)
                or not np.isfinite(shear_ratio) or not 0<shear_ratio<=1):
            raise ValueError('Finite distinct material interfaces, barycentric points and tributary areas required')
        self.ids=torch.tensor(ids,dtype=torch.long,device=device)
        self.bary=torch.tensor(bary,dtype=dtype,device=device)
        self.area=torch.tensor(areas,dtype=dtype,device=device)
        self.stiffness=float(stiffness);self.peak_traction=float(peak_traction);self.fracture_energy=float(fracture_energy)
        self.shear_ratio=float(shear_ratio);self.dtype=dtype;self.device=torch.device(device)
        value=torch.tensor(q,dtype=dtype,device=device)
        self.reference=local_gap(value[self.ids],self.bary).detach()
        self.normal_sign=torch.where(self.reference[:,2]>=0,1.,-1.)
        self.maximum_opening=torch.zeros(len(ids),dtype=dtype,device=device)
        degradation(self.maximum_opening,stiffness=stiffness,peak_traction=peak_traction,fracture_energy=fracture_energy)

    def residual(self,points):
        gap=local_gap(points,self.bary)-self.reference
        return torch.cat([gap[:,:2]*self.shear_ratio**.5,(gap[:,2:]*self.normal_sign[:,None]).clamp_min(0)],1)

    def linearize(self,value,*,update_history=True):
        local=value[self.ids].detach().requires_grad_(True)
        residual=self.residual(local)
        effective=torch.linalg.vector_norm(residual.detach(),dim=1)
        history=torch.maximum(self.maximum_opening,effective)
        if update_history:self.maximum_opening=history
        fraction=degradation(history,stiffness=self.stiffness,peak_traction=self.peak_traction,fracture_energy=self.fracture_energy)
        gradients=torch.stack([torch.autograd.grad(residual[:,i].sum(),local,retain_graph=i<2)[0] for i in range(3)],1)
        ids=self.ids[:,None].expand(-1,3,-1).reshape(-1,4)
        coefficient=(self.area*self.stiffness*fraction)[:,None].expand(-1,3).reshape(-1)
        return ids,gradients.reshape(-1,4,3),residual.detach().reshape(-1),coefficient

    def report(self):
        failure=2*self.fracture_energy/self.peak_traction
        return {'experimental_only':True,'interfaces':len(self.ids),
                'debonded':int((self.maximum_opening>=failure).sum()),
                'maximum_effective_opening_m':float(self.maximum_opening.max()),
                'positions_written':False,'rigid_attachments':0,
                'fracture_energy_J_m2':self.fracture_energy,'peak_traction_Pa':self.peak_traction,
                'stiffness_Pa_m':self.stiffness,'failure_opening_m':failure,
                'native_or_task_approval':False}
