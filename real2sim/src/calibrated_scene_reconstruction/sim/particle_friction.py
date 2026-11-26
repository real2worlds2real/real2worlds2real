"""Explicit particle/rigid friction ablation, requiring independent controls.

This changes every rigid contact of the selected particle material. It cannot
be called a finger-only control while the cloth touches other rigid supports.
The self-friction multiplier has no native tensor readback in this adapter:
matched self-contact response is required before a task can use the operation.
"""
import numpy as np


def coefficients(base,scale,*,epsilon=1e-6):
    if (not np.isfinite([base,scale,epsilon]).all() or min(base,scale,epsilon)<=0
            or epsilon>1e-6 or epsilon>=base or base*scale/epsilon>1e7):
        raise ValueError('Positive bounded friction and a resolved near-zero rigid coefficient required')
    return {'before_base_friction':float(base),'before_particle_friction_scale':float(scale),
            'base_friction':float(epsilon),'authored_particle_friction_scale':float(base*scale/epsilon),
            'target_particle_pair_friction':float(base*scale),
            'scale_native_readback':False,'preservation_requires_matched_self_contact_response':True,
            'affects_all_particle_rigid_contacts':True,'task_approved':False}


def ablate_rigid(material_prim,native_view):
    from pxr import PhysxSchema
    import torch
    from calibrated_scene_reconstruction.core.geometry.rigid import as_numpy
    api=PhysxSchema.PhysxPBDMaterialAPI(material_prim)
    before=as_numpy(native_view.get_friction())
    base=api.GetFrictionAttr().Get();scale=api.GetParticleFrictionScaleAttr().Get()
    if native_view.count!=1 or not np.allclose(before,base,rtol=0,atol=1e-7):
        raise ValueError('Exactly one current native particle material required')
    receipt=coefficients(base,scale)
    api.CreateFrictionAttr(receipt['base_friction'])
    api.CreateParticleFrictionScaleAttr(receipt['authored_particle_friction_scale'])
    native_view.set_friction(torch.full((1,1),receipt['base_friction'],dtype=torch.float32),
                             torch.zeros(1,dtype=torch.int32))
    actual=as_numpy(native_view.get_friction())
    if not np.allclose(actual,receipt['base_friction'],rtol=0,atol=1e-10):
        raise ValueError('Native near-zero particle base-friction update failed')
    return {**receipt,'method':'particle_base_and_self_scale','native_base_friction':actual.tolist(),
            'native_base_readback':True,'native_readback_after_step':False}
