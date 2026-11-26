"""Explicit diagnostic coupling; not registered as a production backend."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .discrete_bending import DiscreteBending


def load(root,path,rest,faces):
    root=Path(root).resolve();path=Path(path).resolve()
    if not path.is_relative_to(root):raise ValueError('Diagnostic bending descriptor must belong to this capture')
    config=json.loads(path.read_text());asset=(root/config['natural_file']).resolve()
    if (config.get('method')!='experimental_discrete_hinge' or not asset.is_relative_to(root)
            or config.get('natural_sha256')!=hashlib.sha256(asset.read_bytes()).hexdigest()
            or config.get('faces_sha256')!=hashlib.sha256(np.asarray(faces,dtype=np.int64).tobytes()).hexdigest()
            or config.get('material_sha256')!=hashlib.sha256(np.asarray(rest,dtype=np.float64).tobytes()).hexdigest()
            or not config.get('source') or not config.get('basis')):
        raise ValueError('Diagnostic bending history, material or topology changed')
    with np.load(asset) as data:angles=np.asarray(data['natural_radians'],float)
    return config,angles


def install(world,view,rest,faces,natural,rigidity,*,cohesion=None,native_bending=False,membrane_parameters=None):
    """Wrap explicit World steps with internal elastic velocity impulses only.

    The runner must first disable native bending and preserve native membrane,
    self-contact and rigid contacts. This diagnostic cannot qualify delivery.
    """
    view.require_validated_force_profile()
    law=DiscreteBending(rest,faces,natural,rigidity,device='cuda:0',dtype=torch.float64)
    membrane=None
    if membrane_parameters is not None:
        if native_bending:raise ValueError('Monolithic shell cannot also run native bending')
        from .discrete_membrane import DiscreteMembrane
        membrane=DiscreteMembrane(rest,faces,**membrane_parameters,device='cuda:0',dtype=torch.float64)
    if native_bending:
        if cohesion is None:raise ValueError('Native-bending mode needs an explicit experimental interface law')
        law.coefficient.zero_()
    masses=view.get_particle_masses()[0]
    report={'experimental_only':True,'production_backend_registered':False,'task_approved':False,
        'steps':0,'positions_written':False,'fixed_nodes':0,
        'experimental_monolithic_membrane':membrane is not None,'native_membrane_disabled':membrane is not None,
        'material_viscosity_implemented':False if membrane is not None else 'native_membrane',
        'maximum_linear_impulse_Ns':0.,'maximum_angular_impulse_Nms':0.,'maximum_velocity_change_m_s':0.,
        'maximum_cg_iterations':0,'maximum_relative_solver_residual':0.,'rigidity_Nm':float(rigidity),'native_bending_disabled':not native_bending}
    original=world.step
    def step(*args,**kwargs):
        velocity,result=law.velocity_step(view.get_world_positions()[0],view.get_velocities()[0],masses,world.get_physics_dt(),cohesion=cohesion,membrane=membrane)
        view.set_velocities(velocity.to(dtype=torch.float32)[None].contiguous())
        report['steps']+=1
        report['maximum_linear_impulse_Ns']=max(report['maximum_linear_impulse_Ns'],float(np.linalg.norm(result['linear_impulse_Ns'])))
        report['maximum_angular_impulse_Nms']=max(report['maximum_angular_impulse_Nms'],float(np.linalg.norm(result['angular_impulse_Nms'])))
        report['maximum_velocity_change_m_s']=max(report['maximum_velocity_change_m_s'],result['maximum_internal_velocity_change_m_s'])
        report['maximum_cg_iterations']=max(report['maximum_cg_iterations'],result['cg_iterations'])
        report['maximum_relative_solver_residual']=max(report['maximum_relative_solver_residual'],result['relative_solver_residual'])
        if cohesion is not None:report['cohesive_interfaces']=cohesion.report()
        return original(*args,**kwargs)
    world.step=step
    return report
