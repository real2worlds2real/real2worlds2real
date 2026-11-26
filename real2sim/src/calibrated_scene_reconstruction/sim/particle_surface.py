"""Source-indexed native particle cloth; masses are actual native readback."""
import numpy as np
from calibrated_scene_reconstruction.core.geometry.rigid import as_numpy


def reset_and_register(world,stage,records,*,initial_dt):
    """Use a bounded real warm-up step for strict particle index registration.

    Isaac5.1 GPU cloth positions cannot be read until integration has run.
    Ordinary reset steps can already deform a fine mesh enough to make nearest
    correspondence ambiguous. Use microsecond initialization steps, retaining
    their actual displacement and velocities. Restore the requested task step
    before any assay. No state, mass, contact, gravity or material is edited.
    """
    dt=world.get_physics_dt()
    if not np.isfinite(initial_dt) or not 1/20000<=initial_dt<=dt:
        raise ValueError('Explicit bounded registration timestep required')
    try:
        world.set_simulation_dt(physics_dt=initial_dt)
        world.reset()
        views={name:ParticleView(stage,path,part) for name,path,part in records}
        for view in views.values():
            view.index_registration.update(initialization_dt_s=initial_dt,task_dt_s=dt,
                registration_phase='Real small-step reset; actual warm-up state retained')
        return views
    finally:
        world.set_simulation_dt(physics_dt=dt)


class ParticleView:
    def contact_profile(self, parameters):
        from calibrated_scene_reconstruction.sim.particle_contact_binding import read
        return read(self, parameters)

    mass_source='Actual native Isaac5.1 particle masses; no runtime mass correction'
    backend='particle'
    def __init__(self,stage,path,part):
        from isaacsim.core.prims import ClothPrim
        from pxr import UsdGeom
        from calibrated_scene_reconstruction.assets.particle_registration import check
        import torch
        self.prim=stage.GetPrimAtPath(path);self.stage=stage;self.rest=np.array(part['cloth_rest']['points']);self.faces=np.array(part['F'])
        self.view=ClothPrim(path,name=str(path).replace('/','_'),reset_xform_properties=False);self.view.initialize()
        matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(self.prim))
        source=np.asarray(part['V'])@matrix[:3,:3]+matrix[3,:3]
        native=as_numpy(self.view.get_world_positions())[0]
        try:
            self.index_registration=check(source,native)
        except ValueError as cause:
            try:
                from calibrated_scene_reconstruction.sim.particle_registration_output import read
                self.index_registration=read(self.prim,self.view,part)
                self.index_registration['authored_nearest_assignment_failure']=str(cause)
                self.index_registration['maximum_authored_displacement_m']=float(np.linalg.norm(native-source,axis=1).max())
            except (ValueError,RuntimeError) as output_error:
                error=ValueError(f'{path}: {cause}; native_output={output_error}; '
                    f'source_shape={source.shape},native_shape={native.shape},'
                    f'nonfinite_native={int((~np.isfinite(native)).sum())}')
                error.particle_state={'source':source,'native':native,'path':str(path)}
                error.native_output=getattr(output_error,'native_output',None)
                raise error from output_error
        self.index_registration['stage_meters_per_unit']=float(UsdGeom.GetStageMetersPerUnit(stage))
        index=np.arange(len(source))
        self.order=torch.as_tensor(index,device='cuda:0',dtype=torch.long)
        self.inverse=torch.as_tensor(np.argsort(index),device='cuda:0',dtype=torch.long)
    def get_world_positions(self):return self.view.get_world_positions()[:,self.order]
    def get_velocities(self):return self.view.get_velocities()[:,self.order]
    def get_particle_masses(self):return self.view.get_particle_masses()[:,self.order]
    def set_velocities(self,value):self.view.set_velocities(value[:,self.inverse].contiguous())
    def restore_setup_state(self,positions,velocities):
        import torch
        p=torch.as_tensor(positions,dtype=torch.float32,device='cuda:0');v=torch.as_tensor(velocities,dtype=torch.float32,device='cuda:0')
        self.view.set_world_positions(p[:,self.inverse].contiguous());self.view.set_velocities(v[:,self.inverse].contiguous())
