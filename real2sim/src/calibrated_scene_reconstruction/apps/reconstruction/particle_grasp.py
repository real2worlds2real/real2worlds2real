"""Native PBD hand-control intervention; never applies forces to cloth nodes."""
import numpy as np


class ParticleGraspControl:
    def __init__(self, view, parameters, proof, *, physics_hz, solver):
        from pxr import PhysxSchema, UsdShade
        import omni.physics.tensors as tensors
        from calibrated_scene_reconstruction.scene.objects.particle_contact_binding import read
        from calibrated_scene_reconstruction.scene.objects.particle_hand_proof import bind
        self.proof = bind(proof, read(view, parameters), physics_hz=physics_hz, solver=solver)
        system = PhysxSchema.PhysxParticleAPI(view.prim).GetParticleSystemRel().GetTargets()[0]
        prim = view.stage.GetPrimAtPath(system)
        self.material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')[0].GetPrim()
        self.owner = tensors.create_simulation_view('torch')
        self.owner.set_subspace_roots('/')
        self.native = self.owner.create_particle_material_view(str(self.material.GetPath()))
        self.receipt = None

    def switch(self, time):
        from calibrated_scene_reconstruction.sim.particle_friction import ablate_rigid
        if self.receipt is not None:
            raise ValueError('Particle friction intervention can only occur once')
        self.receipt = {**ablate_rigid(self.material, self.native), 'time_s': float(time)}
        return self.receipt

    def observe_after_step(self):
        from calibrated_scene_reconstruction.core.geometry.rigid import as_numpy
        if self.receipt is None:
            raise ValueError('No particle friction intervention has occurred')
        actual = as_numpy(self.native.get_friction())
        if not np.allclose(actual, self.receipt['base_friction'], rtol=0, atol=1e-10):
            raise ValueError('Native near-zero particle friction did not persist')
        self.receipt.update(native_readback_after_step=True, native_base_friction=actual.tolist())
        return True
