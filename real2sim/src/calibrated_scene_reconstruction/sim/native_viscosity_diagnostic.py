"""Shared diagnostic native-elastic viscosity adapter with matched transport proof.

Never grants source, task, material-identification or delivery approval.
"""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install(world,mass_view,*,proof_path,session_dir,out,faces,rest,material,cloth,
            solver,physics_hz,position_iterations,linear_solver='sparse_direct'):
    if linear_solver not in {'sparse_direct','cuda_direct'}:
        raise ValueError('A residual-checked direct solver is required by this diagnostic adapter')
    out=Path(out)
    proof=Path(proof_path).resolve()
    if not proof.is_relative_to(Path(session_dir).resolve()):raise ValueError('Transport proof must belong to the current capture')
    receipt=json.loads(proof.read_text());transport=receipt.get('experimental_shell',{})
    from calibrated_scene_reconstruction.sim import constitutive_damping,discrete_bending,discrete_membrane,discrete_dissipation
    from calibrated_scene_reconstruction.assets import cloth_projection,cloth_curvature
    from calibrated_scene_reconstruction.sim import surface_cloth
    modules=[constitutive_damping,discrete_bending,discrete_membrane,discrete_dissipation,cloth_projection]
    settings=receipt.get('material_control',{}).get('settings',{})
    from pxr import PhysxSchema
    collision=PhysxSchema.PhysxCollisionAPI(mass_view.prim)
    actual={'rest_offset':collision.GetRestOffsetAttr().Get(),
            'contact_offset':collision.GetContactOffsetAttr().Get(),
            'self_collision_filter_distance':mass_view.prim.GetAttribute('physxDeformableBody:selfCollisionFilterDistance').Get(),
            'solver_position_iterations':mass_view.prim.GetAttribute('physxDeformableBody:solverPositionIterationCount').Get()}
    if any(actual[k] is None or not np.isclose(actual[k],cloth[k],rtol=1e-6,atol=1e-10) for k in actual):
        raise ValueError('Actual authored surface contact profile differs from the requested diagnostic profile')
    (out/'viscosity_contact_readback.json').write_text(json.dumps({
        'scope':'Stage attributes after material priors and native initialization; not a tensor-level contact-parameter readback',
        'settings':actual},indent=2)+'\n')
    if (not receipt.get('pass') or receipt.get('case')!='collision' or not transport.get('native_elasticity_preserved')
            or receipt.get('solver')!=solver or not np.isclose(receipt.get('physics_dt_s',0),1/physics_hz)
            or settings.get('solver_position_iterations')!=position_iterations
            or actual['solver_position_iterations']!=position_iterations
            or receipt.get('source_sha256')!=sha(surface_cloth.__file__)
            or receipt.get('probe_sha256')!=sha('apps/scene/check_surface_cloth.py')
            or any(transport.get('sources_sha256',{}).get(Path(m.__file__).name)!=sha(m.__file__) for m in modules)
            or any(not np.isclose(settings.get(k,0),material[k]) for k in
                ['youngs_modulus','poissons_ratio','thickness','flexural_rigidity','areal_density'])
            or any(not np.isclose(settings.get(k,0),actual[k],rtol=1e-6,atol=1e-10) for k in
                ['rest_offset','contact_offset','self_collision_filter_distance'])):
        raise ValueError('Current matching native-elastic transport control is required')
    with np.load(proof.parent/'momentum.npz') as data:
        if not np.array_equal(data['faces'],faces) or not np.array_equal(data['rest'],np.asarray(rest['points'])):
            raise ValueError('Transport proof uses a different intrinsic mesh')
        baseline=np.array([0,0,-.2*receipt['rigid_mass_readback_kg']])
        error=np.linalg.norm(data['trace']-baseline,axis=1)/np.linalg.norm(baseline)
        if not np.isfinite(error).all() or error.max()>=.05:raise ValueError('Raw native transport does not conserve momentum')
    shutil.copy2(proof,out/'viscosity_transport_proof.json')
    shutil.copy2(proof.parent/'momentum.npz',out/'viscosity_transport_raw.npz')
    for module in modules:shutil.copy2(module.__file__,out/('viscosity_'+Path(module.__file__).name))
    natural=np.zeros(len(cloth_curvature.adjacency(faces)[0]))
    if rest.get('bend_reference'):natural=np.radians(cloth_curvature.measured(rest['bend_reference']['points'],faces)[1])
    bend=discrete_bending.DiscreteBending(rest['points'],faces,natural,material['flexural_rigidity'],device='cuda:0')
    membrane=discrete_membrane.DiscreteMembrane(rest['points'],faces,
        **{k:material[k] for k in ['youngs_modulus','poissons_ratio','thickness']},device='cuda:0')
    times=transport['viscosity_times_s'];terms=[(bend,times['bending']),(membrane,times['membrane'])]
    native_viscosity_report={'diagnostic_only':True,'task_approved':False,'native_elasticity_preserved':True,
        'linear_solver':linear_solver,
        'authored_contact_readback':actual,
        'viscosity_times_s':times,'transport_proof_sha256':sha(proof),'steps':0,'positions_written':False,
        'maximum_internal_linear_impulse_Ns':0.,'maximum_internal_angular_impulse_Nms':0.,
        'rigid_contact_or_self_contact_qualified':False,'source_material_identified':False}
    original_step=world.step
    def viscous_native_step(*step_args,**step_kwargs):
        import torch
        velocity,control=constitutive_damping.velocity_step(mass_view.get_world_positions()[0],mass_view.get_velocities()[0],
            mass_view.get_particle_masses()[0],world.get_physics_dt(),terms,linear_solver=linear_solver)
        mass_view.set_velocities(velocity.to(dtype=torch.float32)[None].contiguous())
        native_viscosity_report['steps']+=1
        for key in ['linear_impulse_Ns','angular_impulse_Nms']:
            name='maximum_internal_'+key
            native_viscosity_report[name]=max(native_viscosity_report[name],float(np.linalg.norm(control[key])))
        return original_step(*step_args,**step_kwargs)
    world.step=viscous_native_step
    shutil.copy2(__file__,out/'viscosity_adapter.py')
    return native_viscosity_report
