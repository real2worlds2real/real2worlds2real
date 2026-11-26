"""Read solver-published USD cloth geometry without stepping or setting state."""
import numpy as np


def read(prim,view,part):
    from pxr import UsdGeom,PhysxSchema
    from omni.physx import get_physx_interface
    import carb.settings
    from calibrated_scene_reconstruction.core.geometry.rigid import as_numpy
    from calibrated_scene_reconstruction.assets.particle_registration import native_output
    before=as_numpy(view.get_world_positions())[0].copy()
    velocity=as_numpy(view.get_velocities())[0].copy()
    # This is PhysX's native output path (also used by its cloth API tests),
    # not a tensor-to-USD copy authored by this reconstruction pipeline.
    settings=carb.settings.get_settings()
    keys=['/physics/updateToUsd','/physics/updateParticlesToUsd']
    from omni.physxfabric import get_physx_fabric_interface
    from omni.physxfabric.bindings import _physxFabric as fabric_settings
    fabric_enabled=bool(settings.get(fabric_settings.SETTING_FABRIC_ENABLED))
    if fabric_enabled:keys.append(fabric_settings.SETTING_FABRIC_UPDATE_POINTS)
    previous={key:settings.get(key) for key in keys}
    try:
        for key in keys:settings.set_bool(key,True)
        if fabric_enabled:
            from isaacsim.core.api import SimulationContext
            import omni.usd
            import usdrt
            context=SimulationContext.instance();fabric=get_physx_fabric_interface()
            # Same native output refresh as SimulationContext.render(), with
            # no app update or simulation step. Read Fabric directly: copying
            # the whole stage to USD can resend articulation states through
            # setters unsupported by the native direct-GPU API.
            fabric.force_update(context.get_physics_dt(),context.current_time)
            usd_context=omni.usd.get_context()
            if usd_context.get_stage()!=prim.GetStage():raise ValueError('Current native stage required')
            stage=usdrt.Usd.Stage.Attach(usd_context.get_stage_id())
            rt_prim=stage.GetPrimAtPath(str(prim.GetPath()))
            points=np.asarray(usdrt.UsdGeom.Mesh(rt_prim).GetPointsAttr().Get()).copy()
            transform=usdrt.Rt.Xformable(rt_prim).GetFabricHierarchyWorldMatrixAttr()
            matrix=np.asarray(transform.Get()) if transform else np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
        else:
            get_physx_interface().update_transformations(True,True,True,False)
            points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get()).copy()
            matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
    finally:
        for key,value in previous.items():
            if value is None:settings.destroy_item(key)
            else:settings.set(key,value)
    mesh=UsdGeom.Mesh(prim)
    counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    if not np.all(counts==3):raise ValueError('Native cloth output must retain triangular topology')
    faces=np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
    world=points@matrix[:3,:3]+matrix[3,:3]
    rest=np.asarray(PhysxSchema.PhysxParticleClothAPI(prim).GetRestPointsAttr().Get())
    after=as_numpy(view.get_world_positions())[0].copy()
    velocity_after=as_numpy(view.get_velocities())[0].copy()
    try:
        report=native_output(part['F'],part['cloth_rest']['points'],faces,rest,world,
                             before,after,velocity,velocity_after)
    except ValueError as error:
        error.native_output={'output_world':world,'native_before':before,'native_after':after,
                             'velocity_before':velocity,'velocity_after':velocity_after}
        raise
    restored=all(settings.get(k)==v for k,v in previous.items())
    if not restored:raise ValueError('Native output settings were not restored')
    return {**report,'output_settings_restored':True,'fabric_output':fabric_enabled,
            'previous_output_settings':previous}
