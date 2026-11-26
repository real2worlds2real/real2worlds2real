"""Experimental internal shell law over Isaac's particle integration/contact.

Diagnostic only. Material dissipation and delivery/reload registration remain
unqualified. No runtime position writes, fixed nodes or external shape targets.
"""
import numpy as np


class DeferredParticleContacts:
    """Suspend gravity/particle contacts only during native reset warmup.

    Keep rigid colliders enabled throughout: toggling a live dynamic SDF mesh's
    simulation flag can invalidate its native shape. Pair filtering isolates
    the cloth without changing the rigid representation.
    """
    def __init__(self,stage,cloth,*,material_rest=None):
        from pxr import UsdPhysics,PhysxSchema,UsdGeom,Vt
        from .experimental_startup import DeferredStartup,FilteredPairs
        entries=[];partners=[]
        for prim in stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):entries.append((UsdPhysics.Scene(prim).GetGravityMagnitudeAttr(),0.))
            elif prim.GetPath()!=cloth.GetPath() and prim.HasAPI(UsdPhysics.CollisionAPI):
                partners.append(str(prim.GetPath()))
        system=stage.GetPrimAtPath(PhysxSchema.PhysxParticleAPI(cloth).GetParticleSystemRel().GetTargets()[0])
        filters=FilteredPairs(UsdPhysics.FilteredPairsAPI.Apply(system).CreateFilteredPairsRel())
        entries.append((filters,list(dict.fromkeys(filters.Get()+partners))))
        self.initial_pose=None
        if material_rest is None:
            entries.append((PhysxSchema.PhysxParticleAPI(cloth).GetSelfCollisionAttr(),False))
        else:
            from .experimental_startup import InitialMaterialPose
            mesh=UsdGeom.Mesh(cloth);target=np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
            rest=np.asarray(material_rest,dtype=float)
            if rest.shape!=target.shape or not np.isfinite(rest).all():
                raise ValueError('Material cooking requires corresponding finite intrinsic nodes')
            # Translation is immaterial to the rest-distance filter. Keep the
            # cooked chart near the object to retain floating-point resolution.
            cooked=rest-rest.mean(axis=0)+target.mean(axis=0)
            matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(cloth))
            if not np.allclose(matrix[:3,:3]@matrix[:3,:3].T,np.eye(3),atol=1e-6):
                raise ValueError('Material contact cooking requires an unscaled metric transform')
            self.initial_pose=InitialMaterialPose(cooked@matrix[:3,:3]+matrix[3,:3],target@matrix[:3,:3]+matrix[3,:3])
            if not PhysxSchema.PhysxParticleAPI(cloth).GetSelfCollisionAttr().Get():
                raise ValueError('Material cooking must retain actual self-contact')
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(cooked.astype('float32')))
        self.startup=DeferredStartup(entries);self.startup.suspend()

    def author_initial_pose(self,world,path,*,native_reset_time_s):
        if self.initial_pose is None:return None
        import omni.physics.tensors as tensors
        owner=tensors.create_simulation_view('torch');owner.set_subspace_roots('/')
        view=owner.create_particle_cloth_view(path)
        if view.count!=1:raise ValueError('One native material body required for initial authoring')
        report=self.initial_pose.apply(view,time_s=world.current_time-native_reset_time_s)
        report['native_reset_clock_origin_s']=native_reset_time_s
        return report

    def restore(self):
        return self.startup.restore()


def install(world,stage,path,part,parameters,*,linear_solver='sparse_direct',prediction_output=None,nonlinear_iterations=12,dissipation=None):
    import omni.physics.tensors as tensors
    import torch
    from pxr import UsdGeom,UsdShade,PhysxSchema
    from calibrated_scene_reconstruction.assets.cloth_rest import area
    from calibrated_scene_reconstruction.assets.cloth_curvature import adjacency,measured
    from .discrete_bending import DiscreteBending
    from .discrete_membrane import DiscreteMembrane
    from calibrated_scene_reconstruction.core.geometry.rigid import as_numpy
    if world.get_physics_context().get_solver_type()!='PGS':raise ValueError('Diagnostic particle shell requires validated PGS')
    if dissipation is not None and (set(dissipation)!={'bending_viscosity_time_s','membrane_viscosity_time_s'}
            or not all(np.isfinite(v) and 0<=v<=1 for v in dissipation.values())):
        raise ValueError('Explicit diagnostic viscosity times in seconds required; native damping rates cannot be reused')
    times=dissipation or {'bending_viscosity_time_s':0.,'membrane_viscosity_time_s':0.}
    rest=part['cloth_rest']
    if set(rest)&{'bend_memory','bend_angles'}:raise ValueError('Unqualified material history cannot be reinterpreted by the diagnostic shell')
    owner=tensors.create_simulation_view('torch');owner.set_subspace_roots('/')
    view=owner.create_particle_cloth_view(path)
    if view.count!=1:raise ValueError('One native particle body required')
    if np.any(as_numpy(view.get_spring_stiffness())!=0) or np.any(as_numpy(view.get_spring_damping())!=0):
        raise ValueError('Disable and verify native elasticity before installing internal shell forces')
    current=as_numpy(view.get_positions())[0].reshape(-1,3)
    prim=stage.GetPrimAtPath(path);matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
    system_path=PhysxSchema.PhysxParticleAPI(prim).GetParticleSystemRel().GetTargets()[0]
    system=PhysxSchema.PhysxParticleSystem.Get(stage,system_path)
    material=UsdShade.MaterialBindingAPI(system.GetPrim()).ComputeBoundMaterial('physics')[0]
    material_view=owner.create_particle_material_view(str(material.GetPath()))
    native_friction=as_numpy(material_view.get_friction()).copy()
    if material_view.count!=1 or not np.allclose(native_friction,parameters['dynamic_friction'],rtol=0,atol=1e-7):
        raise ValueError('Actual native particle friction differs from the physical estimate')
    if not np.allclose(matrix[:3,:3]@matrix[:3,:3].T,np.eye(3),atol=1e-6):raise ValueError('Unscaled metric cloth transform required')
    expected=np.asarray(part['V'])@matrix[:3,:3]+matrix[3,:3]
    if current.shape!=expected.shape or not np.allclose(current,expected,rtol=0,atol=1e-6):
        raise ValueError('Native material-node order or initial state differs from the authored model')
    masses=as_numpy(view.get_masses())[0].copy();target=parameters['areal_density']*area(rest['points'],part['F'])
    if not np.isclose(masses.sum(),target,rtol=1e-4):raise ValueError('Actual native cloth mass differs from the physical estimate')
    natural=np.zeros(len(adjacency(part['F'])[0]))
    if rest.get('bend_reference'):natural=np.radians(measured(rest['bend_reference']['points'],part['F'])[1])
    law=DiscreteBending(rest['points'],part['F'],natural,parameters['flexural_rigidity'],device='cuda:0')
    membrane=DiscreteMembrane(rest['points'],part['F'],**{k:parameters[k] for k in ['youngs_modulus','poissons_ratio','thickness']},device='cuda:0')
    index=torch.zeros(1,dtype=torch.int32,device='cuda:0')
    report={'experimental_only':True,'production_backend_registered':False,'task_approved':False,
        'native_integration_and_contacts':'Isaac Sim particle cloth','native_springs_disabled':True,
        'internal_material_law':'Discrete shell membrane and bending energy; implicit internal velocity impulses',
        'linear_solver':linear_solver,'positions_written':False,'fixed_material_nodes':0,'steps':0,
        'node_count':len(current),'actual_native_mass_kg':float(masses.sum()),'native_mass_readback':True,
        'native_friction':native_friction.tolist(),'native_friction_readback':True,
        'contact_profile':{'particle_contact_offset_m':system.GetParticleContactOffsetAttr().Get(),
            'rigid_rest_offset_m':system.GetRestOffsetAttr().Get(),'solid_rest_offset_m':system.GetSolidRestOffsetAttr().Get(),
            'position_iterations':system.GetSolverPositionIterationCountAttr().Get(),
            'max_neighborhood':system.GetMaxNeighborhoodAttr().Get()},
        'nonlinear_energy_backtracking':True,'nonlinear_iteration_budget':nonlinear_iterations,
        'nonlinear_converged':True,'maximum_nonlinear_gradient_residual':0.,'minimum_internal_step_fraction':1.,
        'maximum_relative_solver_residual':0.,'maximum_internal_linear_impulse_Ns':0.,
        'maximum_internal_angular_impulse_Nms':0.,'maximum_internal_velocity_change_m_s':0.,
        'material_dissipation_implemented':dissipation is not None,
        'diagnostic_dissipation_hypothesis':dissipation,'viscous_work_J':0.,
        'unapplied_material_dissipation':{k:parameters[k] for k in ['elasticity_damping','bend_damping','linear_damping'] if k in parameters}}
    original=world.step
    def step(*args,**kwargs):
        positions=view.get_positions()[0].reshape(-1,3);velocities=view.get_velocities()[0].reshape(-1,3)
        def progress(points,row):
            if prediction_output is not None:
                import json
                from pathlib import Path
                Path(prediction_output).with_name('nonlinear_progress.json').write_text(json.dumps({'native_step':report['steps'],**row,'task_approved':False},indent=2)+'\n')
            print('SHELL_NONLINEAR_ITERATION',report['steps'],row['iteration'],row['objective_J'],row['relative_gradient_residual'],flush=True)
        update,control=law.velocity_step(positions,velocities,masses,world.get_physics_dt(),membrane=membrane,
            linear_solver=linear_solver,tolerance=1e-7,nonlinear_backtracking=True,nonlinear_iterations=nonlinear_iterations,
            on_nonlinear_iteration=progress,viscosity_terms=None if dissipation is None else
            [(law,times['bending_viscosity_time_s']),(membrane,times['membrane_viscosity_time_s'])])
        report['viscous_work_J']+=control['constitutive_dissipation_J']
        if report['steps']==0 and prediction_output is not None:
            np.savez_compressed(prediction_output,points=as_numpy(positions),
                velocities_before=as_numpy(velocities),velocities_after=as_numpy(update),
                predicted_points=as_numpy(positions)+world.get_physics_dt()*as_numpy(update))
            import json
            from pathlib import Path
            Path(prediction_output).with_suffix('.json').write_text(json.dumps(control,indent=2)+'\n')
        report['nonlinear_converged']=report['nonlinear_converged'] and control.get('nonlinear_converged',False)
        report['maximum_nonlinear_gradient_residual']=max(report['maximum_nonlinear_gradient_residual'],control.get('nonlinear_relative_gradient_residual',float('inf')))
        if not report['nonlinear_converged']:raise ValueError('Unconverged nonlinear internal shell update; no native step accepted')
        view.set_velocities(update.to(dtype=torch.float32).reshape(1,-1).contiguous(),index)
        report['steps']+=1
        report['minimum_internal_step_fraction']=min(report['minimum_internal_step_fraction'],control['internal_step_fraction'])
        for name in ['relative_solver_residual','internal_velocity_change_m_s']:
            key='maximum_'+name;source=name if name=='relative_solver_residual' else 'maximum_'+name
            report[key]=max(report[key],control[source])
        report['maximum_internal_linear_impulse_Ns']=max(report['maximum_internal_linear_impulse_Ns'],float(np.linalg.norm(control['linear_impulse_Ns'])))
        report['maximum_internal_angular_impulse_Nms']=max(report['maximum_internal_angular_impulse_Nms'],float(np.linalg.norm(control['angular_impulse_Nms'])))
        return original(*args,**kwargs)
    world.step=step
    return report
