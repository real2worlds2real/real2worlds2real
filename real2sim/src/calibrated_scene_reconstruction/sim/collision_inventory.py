"""Account for active collision shapes by their actual physical source owner."""
import numpy as np
from pxr import Sdf,UsdPhysics,PhysxSchema


def source_groups(stage,owners,*,excluded_roots=()):
    from .contact_carrier import OWNER as contact_owner,verify as verify_contact
    from .rigid_partition import OWNER as partition_owner,verify as verify_partition
    paths={name:Sdf.Path(path) for name,path in owners.items()}
    excluded=list(map(Sdf.Path,excluded_roots));groups={name:[] for name in paths}
    verified=set()
    for prim in stage.Traverse():
        if (not prim.HasAPI(UsdPhysics.CollisionAPI)
                or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False
                or any(prim.GetPath().HasPrefix(p) for p in excluded)):
            continue
        body=prim
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):body=body.GetParent()
        alias=None
        if body:
            alias=body.GetCustomDataByKey(contact_owner)
            if alias:
                if alias not in verified:verify_contact(stage,alias);verified.add(alias)
            else:
                alias=body.GetCustomDataByKey(partition_owner)
                if alias and alias not in verified:verify_partition(stage,alias);verified.add(alias)
        target=Sdf.Path(alias) if alias else prim.GetPath()
        matches=[name for name,path in paths.items() if target.HasPrefix(path)]
        if len(matches)!=1:raise ValueError('Active collision shape has an unknown or ambiguous source owner: '+str(prim.GetPath()))
        margin=PhysxSchema.PhysxCollisionAPI(prim).GetContactOffsetAttr().Get()
        if margin is None or not np.isfinite(margin) or margin<0:
            raise ValueError('Explicit finite source collision margin required: '+str(prim.GetPath()))
        groups[matches[0]].append({'path':str(prim.GetPath()),
                                  'body_path':str(body.GetPath()) if body else str(prim.GetPath()),
                                  'contact_offset_m':float(margin)})
    if any(not rows for rows in groups.values()):raise ValueError('Every source owner requires an active collision shape')
    return {name:{'colliders':rows,'maximum_contact_offset_m':max(r['contact_offset_m'] for r in rows)}
            for name,rows in groups.items()}
