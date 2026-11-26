"""Explicit diagnostic contact isolation across actual compound colliders."""


def rigid_partners(stage,roots,*,exclude_roots=()):
    """Disable actual CollisionAPI descendants, leaving flexible owners alone.

    A rigid part's visible surface need not be its collision geometry. The
    caller supplies the diagnostic stage scope and explicit flexible exclusions.
    Scanning that scope also catches welded collision carriers that are siblings
    of their visual owner. Explicit rigid-only roots need no exclusions.
    """
    from pxr import Sdf,Usd,UsdPhysics
    if not roots or len(set(map(str,roots)))!=len(roots):
        raise ValueError('Distinct explicit rigid partner roots required')
    colliders={};counts={}
    excluded=[]
    for path in exclude_roots:
        if not stage.GetPrimAtPath(path):raise ValueError('Excluded flexible owner missing: '+str(path))
        excluded.append(Sdf.Path(path))
    for path in roots:
        root=stage.GetPrimAtPath(path)
        if not root or not root.IsValid():raise ValueError('Rigid contact root missing: '+str(path))
        found=[prim for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.CollisionAPI)
               and not any(prim.GetPath().HasPrefix(path) for path in excluded)]
        if not found:raise ValueError('Rigid contact root has no actual colliders: '+str(path))
        counts[str(path)]=len(found)
        for prim in found:colliders[str(prim.GetPath())]=prim
    before={path:UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            for path,prim in colliders.items()}
    for prim in colliders.values():
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
    after={path:UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
           for path,prim in colliders.items()}
    if any(value is not False for value in after.values()):
        raise ValueError('Rigid contact ablation did not read back disabled colliders')
    return {'diagnostic_only':True,'task_approved':False,'rigid_collision_geometry_changed':False,
            'colliders_per_root':counts,'excluded_flexible_roots':list(map(str,excluded)),
            'before_enabled':before,'disabled_readback':after}
