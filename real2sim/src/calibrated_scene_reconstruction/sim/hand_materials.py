"""Read and change the actual isolated hand articulation's shape materials."""
import numpy as np


def read(view):
    raw=view.get_material_properties()
    values=raw.detach().cpu().numpy().copy() if hasattr(raw,'detach') else np.asarray(raw).copy()
    if (view.count!=1 or values.ndim!=3 or values.shape[0]!=1 or values.shape[1]<1
            or values.shape[2]!=3 or not np.isfinite(values).all() or (values<0).any()):
        raise ValueError('One actual hand articulation with finite native shape materials required')
    return raw,values


def summary(view):
    _,v=read(view)
    return {'native_readback':True,'shapes':int(v.shape[1]),'static_friction':v[0,:,0].tolist(),
            'dynamic_friction':v[0,:,1].tolist(),'restitution':v[0,:,2].tolist()}


def set_friction(view, value):
    """Tensor-property diagnostic only; does not preserve friction combine modes.

    Isaac 5.1 surface-contact controls show that this write can change effective
    friction despite a zero coefficient readback. Use author_friction for a
    causal rigid/surface zero-friction control and verify after a native step.
    """
    if not np.isfinite(value) or value<0:raise ValueError('Finite nonnegative native friction required')
    raw,before=read(view)
    data=raw.clone() if hasattr(raw,'clone') else before.copy()
    data[...,:2]=float(value)
    if hasattr(raw,'detach'):
        import torch
        indices=torch.zeros(1,dtype=torch.int32,device='cpu')
    else:indices=np.zeros(1,dtype=np.uint32)
    view.set_material_properties(data,indices)
    _,after=read(view)
    passed=bool(np.allclose(after[...,:2],value,rtol=0.,atol=1e-7)
                and np.allclose(after[...,2],before[...,2],rtol=0.,atol=1e-7))
    if not passed:raise ValueError('Native hand friction write/readback failed or changed restitution')
    return {'pass':True,'native_readback':True,'requested_friction':float(value),
            'before':before.tolist(),'after':after.tolist(),'restitution_preserved':True}


def author_friction(material_prim, value):
    """Edit the existing USD material, retaining its pair-combination policy.

    Changes propagate at the next physics step. This receipt is deliberately
    not a native readback or proof of zero effective contact friction.
    """
    from pxr import UsdPhysics, PhysxSchema
    if not np.isfinite(value) or value<0:
        raise ValueError('Finite nonnegative friction required')
    api=UsdPhysics.MaterialAPI(material_prim)
    if not api:
        raise ValueError('Existing rigid physics material required')
    combine=PhysxSchema.PhysxMaterialAPI(material_prim).GetFrictionCombineModeAttr().Get()
    before=[api.GetStaticFrictionAttr().Get(),api.GetDynamicFrictionAttr().Get()]
    api.GetStaticFrictionAttr().Set(float(value))
    api.GetDynamicFrictionAttr().Set(float(value))
    return {'method':'existing_usd_material','combine_mode':combine,'before':before,
            'requested_friction':float(value),'native_readback_after_step':False}
