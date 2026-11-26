"""Author and verify the same native simulation profile in checks and USD/Z."""
import sys

import numpy as np
from pxr import PhysxSchema, UsdPhysics, Vt


def configure_initialization(world,profile):
    """Optional real small-step reset; never edits states or assay step size.

    Dense source-indexed cloth can deform during normal reset before its
    tensor indices can be registered. Explicit small initialization steps keep
    that registration unambiguous; all resulting positions/velocities survive.
    Existing profiles without this option retain their original reset.
    """
    hz=(profile or {}).get('initialization_hz')
    if hz is None:return
    if getattr(world,'_r2s2r_initialization_hz',None) is not None:
        if world._r2s2r_initialization_hz!=hz:raise ValueError('Conflicting reset initialization profiles')
        return
    original=world.reset
    def reset(*args,**kwargs):
        dt=world.get_physics_dt()
        try:
            world.set_simulation_dt(physics_dt=1/hz)
            return original(*args,**kwargs)
        finally:
            world.set_simulation_dt(physics_dt=dt)
    world.reset=reset;world._r2s2r_initialization_hz=hz


def apply_rigid_iterations(stage,profile):
    """Apply declared task-object iterations after their bodies are authored."""
    values=(profile or {}).get('rigid_iterations')
    if values is None:return
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith('/World/Objects/') and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            api=PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            api.CreateSolverPositionIterationCountAttr(values[0])
            api.CreateSolverVelocityIterationCountAttr(values[1])


def apply(stage, profile, *, flexible=False):
    if profile is None:
        return
    if flexible and profile['friction_type'] != 'patch':
        raise ValueError('Native GPU flexible simulation requires patch friction')
    scenes = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
    if len(scenes) != 1:
        raise ValueError('One physics scene required for an explicit simulation profile')
    api = PhysxSchema.PhysxSceneAPI.Apply(scenes[0])
    api.CreateTimeStepsPerSecondAttr(profile['physics_hz'])
    api.CreateSolverTypeAttr(profile.get('solver', 'TGS'))
    if 'external_forces_every_iteration' in profile:
        every_iteration = profile['external_forces_every_iteration']
        if type(every_iteration) is not bool:
            raise ValueError('external_forces_every_iteration must be an explicit boolean')
        if every_iteration and profile.get('solver', 'TGS') != 'TGS':
            raise ValueError('external_forces_every_iteration=true requires TGS')
        # An explicit common profile has the same native force integration
        # with or without flexible objects. Keep legacy defaults when absent.
        api.CreateEnableExternalForcesEveryIterationAttr(every_iteration)
    elif flexible:
        api.CreateEnableExternalForcesEveryIterationAttr(profile.get('solver', 'TGS') == 'TGS')
    api.CreateFrictionTypeAttr(profile['friction_type'])
    if 'gpu_max_deformable_surface_contacts' in profile:
        api.CreateGpuMaxDeformableSurfaceContactsAttr(profile['gpu_max_deformable_surface_contacts'])
    if 'gpu_found_lost_aggregate_pairs_capacity' in profile:
        api.CreateGpuFoundLostAggregatePairsCapacityAttr(profile['gpu_found_lost_aggregate_pairs_capacity'])
    if 'gpu_dynamics' in profile:
        if flexible and not profile['gpu_dynamics']:
            raise ValueError('Flexible simulation cannot disable required GPU dynamics')
        api.CreateEnableGPUDynamicsAttr(profile['gpu_dynamics'])
        api.CreateBroadphaseTypeAttr('GPU' if profile['gpu_dynamics'] else 'MBP')
    # USD-only authoring need not import/start Isaac. When an existing Isaac
    # World owns this stage, sync its clock as well as the authored USD value.
    module = sys.modules.get('isaacsim.core.api')
    world = module.World.instance() if module is not None else None
    if world is not None and world.stage == stage:
        if 'gpu_dynamics' in profile:
            context=world.get_physics_context()
            context.enable_gpu_dynamics(profile['gpu_dynamics'])
            context.set_broadphase_type('GPU' if profile['gpu_dynamics'] else 'MBP')
        world.set_simulation_dt(physics_dt=1/profile['physics_hz'])
        if not np.isclose(world.get_physics_dt(), 1/profile['physics_hz'], rtol=1e-8):
            raise ValueError('Native runtime clock did not adopt the scene simulation profile')
        configure_initialization(world,profile)
    metadata=dict(profile)
    if 'rigid_iterations' in metadata:
        metadata['rigid_iterations']=Vt.IntArray(metadata['rigid_iterations'])
    stage.GetRootLayer().customLayerData = {**dict(stage.GetRootLayer().customLayerData),
                                          'r2s2r_simulation': metadata,
                                          'r2s2r_flexible_simulation': bool(flexible)}


def audit(stage, profile):
    if profile is None:
        return None
    scenes = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)]
    if len(scenes) != 1:
        raise ValueError('Delivered asset must have one physics scene')
    api = PhysxSchema.PhysxSceneAPI(scenes[0])
    actual = {'physics_hz': api.GetTimeStepsPerSecondAttr().Get(),
              'friction_type': str(api.GetFrictionTypeAttr().Get()),
              'solver': str(api.GetSolverTypeAttr().Get())}
    if 'gpu_max_deformable_surface_contacts' in profile:
        capacity=api.GetGpuMaxDeformableSurfaceContactsAttr().Get()
        if capacity!=profile['gpu_max_deformable_surface_contacts']:
            raise ValueError('Native deformable surface contact capacity differs from checked profile')
        actual['gpu_max_deformable_surface_contacts']=capacity
    if 'gpu_found_lost_aggregate_pairs_capacity' in profile:
        capacity=api.GetGpuFoundLostAggregatePairsCapacityAttr().Get()
        if capacity!=profile['gpu_found_lost_aggregate_pairs_capacity']:
            raise ValueError('Native aggregate pair capacity differs from checked profile')
        actual['gpu_found_lost_aggregate_pairs_capacity']=capacity
    if 'initialization_hz' in profile:
        stored=stage.GetRootLayer().customLayerData.get('r2s2r_simulation',{})
        if stored.get('initialization_hz')!=profile['initialization_hz']:
            raise ValueError('Serialized initialization profile differs from the checked source')
        actual['initialization_hz']=profile['initialization_hz']
    if (actual['physics_hz'] != profile['physics_hz'] or actual['friction_type'] != profile['friction_type']
            or actual['solver'] != profile.get('solver', 'TGS')):
        raise ValueError('Actual simulation profile differs from the checked source: '+str(actual))
    actual['external_forces_every_iteration'] = bool(api.GetEnableExternalForcesEveryIterationAttr().Get())
    if 'external_forces_every_iteration' in profile:
        if actual['external_forces_every_iteration'] != profile['external_forces_every_iteration']:
            raise ValueError('Simulation external-force setting differs from the checked profile')
    elif stage.GetRootLayer().customLayerData.get('r2s2r_flexible_simulation', False):
        if actual['external_forces_every_iteration'] != (actual['solver'] == 'TGS'):
            raise ValueError('Flexible simulation external-force setting differs from the checked solver')
    if 'gpu_dynamics' in profile:
        actual['gpu_dynamics']=bool(api.GetEnableGPUDynamicsAttr().Get())
        actual['broadphase']=str(api.GetBroadphaseTypeAttr().Get())
        if (actual['gpu_dynamics']!=profile['gpu_dynamics']
                or actual['broadphase']!=('GPU' if profile['gpu_dynamics'] else 'MBP')):
            raise ValueError('Native dynamics backend differs from checked profile')
    if profile.get('rigid_iterations') is not None:
        bodies={}
        for prim in stage.Traverse():
            if str(prim.GetPath()).startswith('/World/Objects/') and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                body=PhysxSchema.PhysxRigidBodyAPI(prim)
                values=[body.GetSolverPositionIterationCountAttr().Get(),body.GetSolverVelocityIterationCountAttr().Get()]
                if values!=profile['rigid_iterations']:
                    raise ValueError('Rigid solver iterations differ from checked profile: '+str(prim.GetPath()))
                bodies[str(prim.GetPath())]=values
        actual['rigid_iterations']=bodies
    from .solver_diagnostics import cautions
    actual['numerical_cautions'] = cautions(profile)
    return actual
