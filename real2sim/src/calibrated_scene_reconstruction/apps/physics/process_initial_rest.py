"""Source-stationary time-zero conditioning after free preparation settling."""
import numpy as np
from calibrated_scene_reconstruction.physics.process.contract import require
from calibrated_scene_reconstruction.physics.process.initial_motion import stationary,validate_actors


def quiesce(stage,actors,simulation,stage_id,*,encode_path=None):
    """Set one initial state; native contacts/forces may wake bodies afterwards.

    This is not a per-step velocity clamp, joint or kinematic lock. Only a
    source-reviewed stationary interval may opt in. Jointed/volumetric systems
    still need their model-owned initializer through the existing protocol.
    """
    import omni.physx
    from pxr import UsdGeom,UsdPhysics
    if encode_path is None:
        from pxr import Sdf,PhysicsSchemaTools
        encode_path=lambda path:PhysicsSchemaTools.sdfPathToInt(Sdf.Path(path))
    validate_actors(stage,actors)
    interface=omni.physx.get_physx_interface()
    def native_poses():
        values={}
        for k,v in actors.items():
            if v['kind']!='rigid':continue
            p=interface.get_rigidbody_transformation(v['path'])
            require(p['ret_val'],'Missing native rigid state during initial conditioning: '+k)
            values[k]=np.r_[p['position'],p['rotation']]
        return values
    native_before=native_poses()
    cache=UsdGeom.XformCache();before={k:np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(v['path'])))
                                     for k,v in actors.items()}
    # Authoring rigid USD velocities on an already simulated mixed system can
    # re-parse rounded USD poses and invalidate sleeping contact manifolds.
    # Native put_to_sleep clears rigid velocities directly without a USD edit.
    soft={k:v for k,v in actors.items() if v['kind']!='rigid'}
    audit=stationary(stage,soft) if soft else {'mode':'source_stationary','pass':True,'bodies':{}}
    if soft:simulation.flush_changes()
    sleeping={}
    for name,item in actors.items():
        if item['kind']=='rigid':
            api=UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(item['path']))
            audit['bodies'][name]={'kind':'rigid','inherited':{
                'velocity':np.asarray(api.GetVelocityAttr().Get(),float).tolist(),
                'angularVelocity':np.asarray(api.GetAngularVelocityAttr().Get(),float).tolist()},
                'initialization':'native_put_to_sleep'}
            body=encode_path(item['path']);simulation.put_to_sleep(stage_id,body)
            sleeping[name]=bool(simulation.is_sleeping(stage_id,body))
    after=UsdGeom.XformCache();unchanged=all(np.array_equal(M,np.array(after.GetLocalToWorldTransform(
        stage.GetPrimAtPath(actors[k]['path'])))) for k,M in before.items())
    native_after=native_poses()
    native_unchanged=all(np.array_equal(v,native_after[k]) for k,v in native_before.items())
    require(sleeping and all(sleeping.values()) and unchanged and native_unchanged,
            'Native source-stationary time-zero conditioning failed')
    interface.update_transformations(False,True,True)
    # Sleep is used only to initialize velocity. Resume ordinary contact
    # solving immediately: GPU scenes otherwise omit persistent reports for
    # pairs put to sleep externally, even during subsequent force wake-up.
    awake={}
    for name,item in actors.items():
        if item['kind']=='rigid':
            body=encode_path(item['path']);simulation.wake_up(stage_id,body)
            awake[name]=not bool(simulation.is_sleeping(stage_id,body))
    require(all(awake.values()),'Native responses must resume free contact solving after velocity initialization')
    return {'pass':True,'phase':'after_settling_before_source_time_zero','before':audit,
            'native_sleeping':sleeping,'native_awake_for_contacts':awake,
            'poses_unchanged':unchanged,'native_poses_unchanged':native_unchanged,
            'scope':'Source-conditioned initial velocities reset after preparation using native rigid sleep, then wake-up resumes normal free contact solving. No force, pose lock or runtime clamp added. PBD velocity updates flushed; subsequent free source/contact checks remain required.'}
