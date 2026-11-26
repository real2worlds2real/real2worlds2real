"""Native fluid with conservative split cloth contact and finite-rate wetting.

Separate native systems permit high-friction cloth and low-friction water.
Their collision impulses are solved from actual particle states, in both
directions. There is no prescribed cloth position or water animation.
"""
import numpy as np
from pxr import PhysxSchema,Sdf,Vt,UsdGeom,UsdShade,Gf
from calibrated_scene_reconstruction.sim.wipe_puddle import WipePuddle
from calibrated_scene_reconstruction.assets.wetting import ContactAbsorption


class ProgressivePuddle(WipePuddle):
    def __init__(self,stage,scene_path,center,table_z,bounds,seed=17,appearance=None):
        optics=dict(frosting_roughness=.025)
        if appearance:optics.update(appearance)
        super().__init__(stage,scene_path,center,table_z,bounds,seed=seed,appearance=optics,
                         spacing=.0016,layers=1,footprint_scale=.85)
        system=PhysxSchema.PhysxParticleSystem.Get(stage,self.system_path)
        system.CreateSolidRestOffsetAttr(.0011)
        system.CreateSolverPositionIterationCountAttr(64)
        PhysxSchema.PhysxParticleIsosurfaceAPI.Get(stage,self.system_path).GetIsosurfaceEnabledAttr().Set(False)
        # Particle-system material is shared by fluid and cloth in PhysX107.3.
        mat=PhysxSchema.PhysxPBDMaterialAPI.Get(stage,'/World/Looks/WipingWater')
        mat.CreateFrictionAttr(.12)
        mat.CreateCohesionAttr(.04)
        mat.CreateSurfaceTensionAttr(.03)
        mat.CreateViscosityAttr(.000017)
        # Prepare a resting puddle rather than a freshly injected lattice with
        # numerical relaxation velocities. Removed before sensing/interacting.
        mat.CreateDampingAttr(25.)
        system.CreateMaxVelocityAttr(.04)
        self.activated=False
        self.particle_mass=1000*self.spacing**3
        self.model=ContactAbsorption(self.initial_count,self.particle_mass,seed,
                                     minimum_dwell_s=.04,mean_extra_dwell_s=.04)
        self.events=[];self.clock=0.;self.reference=None;self.contact_events=[]
        self.air=UsdGeom.PointInstancer.Define(stage,'/World/WipingAirborneWater')
        sphere=UsdGeom.Sphere.Define(stage,'/World/WipingAirborneWater/Drop')
        sphere.CreateRadiusAttr(self.rest)
        UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(UsdShade.Material.Get(stage,'/World/Looks/WipingWater'))
        self.air.CreatePrototypesRel().SetTargets([sphere.GetPath()])
        self.air.CreateProtoIndicesAttr([]);self.air.CreatePositionsAttr([])
        self.description=dict(model='Native fluid + native independent 2D cloth + conservative split contact + finite absorption',
            particle_mass_kg=self.particle_mass,initial_water_mass_kg=self.initial_count*self.particle_mass,
            minimum_full_contact_dwell_s=.04,mean_extra_full_contact_dwell_s=.04,
            maximum_mass_rate_kg_s=self.model.rate,capacity_kg=self.model.capacity,
            water_rigid_friction=.12,cloth_rigid_friction=.8,cohesion=.04,surface_tension=.03,viscosity=.000017,
            contact_solver='Double-sided triangle-surface contact; barycentric equal/opposite cloth reaction impulses',
            prescribed_water_animation=False,contact_distance_m=.0022,
            wet_mass_model='Capacity reservoir; uptake does not change mechanical cloth mass or inertia. Pore-scale wicking unresolved',
            hardware_material_calibrated=False)
        self.description['initialization']='Velocity-damped settling; damping removed and native max velocity restored before interactive sensing'
        self.description['material_profile']='Effective thin-film PBD parameters; not a measured SI water/wet-wipe calibration'
        self.description['split_contact_separation_speed_limit_m_s']=.003
        self.description['surface_rendering']='Particle-volume heightfield only for particles near and inside the tabletop; native-position spheres elsewhere. Render reconstruction only.'
        self.description['uptake_contact_distance_m']=.0025

    def set_reference(self):self.reference=self.state()[0].copy()

    def activate(self):
        import omni.physx
        PhysxSchema.PhysxPBDMaterialAPI.Get(self.stage,'/World/Looks/WipingWater').GetDampingAttr().Set(0.)
        PhysxSchema.PhysxParticleSystem.Get(self.stage,self.system_path).GetMaxVelocityAttr().Set(.6)
        self.activated=True;self.description['interactive_damping']=0.
        omni.physx.get_physx_simulation_interface().flush_changes()

    def update_surface(self):
        from calibrated_scene_reconstruction.assets.fluid_surface import particle_heightfield
        p,_=self.state()
        film=(p[:,2]>=self.z-.002)&(p[:,2]<=self.z+.010)&((p[:,:2]>=self.bounds[0])&(p[:,:2]<=self.bounds[1])).all(1)
        high=~film
        vertices,faces,volume=particle_heightfield(p[film],self.z,self.spacing)
        self.air.GetPositionsAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(p[high])))
        self.air.GetProtoIndicesAttr().Set(Vt.IntArray([0]*int(high.sum())))
        if len(faces):
            self.surface.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(vertices))
            self.surface.GetFaceVertexCountsAttr().Set(Vt.IntArray([3]*len(faces)))
            self.surface.GetFaceVertexIndicesAttr().Set(Vt.IntArray(faces.ravel().tolist()))
            self.surface.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(self.normals(vertices,faces).astype(np.float32)))
            self.surface.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.array([vertices.min(0),vertices.max(0)],np.float32)))
            self.surface.MakeVisible()
        else:self.surface.MakeInvisible()
        self.description['last_film_particle_volume_m3']=volume

    def absorb_from_cloth(self,cloth,dt,enabled=True):
        import omni.physx
        if self.empty:return 0
        self.clock+=dt
        p,v=self.state();q=cloth.points()
        from calibrated_scene_reconstruction.assets.wet_contact import contact_impulses,sheet_contact_geometry
        geometry=sheet_contact_geometry(p,q,cloth.faces)
        new_v,cloth_v,contact=contact_impulses(p,v,self.particle_mass,q,cloth.velocities(),cloth.masses(),dt,
                                             faces=cloth.faces,geometry=geometry)
        if contact['impulse_Ns']>0:
            self.particles.GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(new_v.astype(np.float32)))
            cloth.apply_contact_velocities(cloth_v)
            v=new_v
            self.contact_events.append(dict(t=self.clock,**contact))
        distance=geometry[3]
        weight=np.clip((.0025-distance)/.001,0,1)
        if not enabled:
            if contact['impulse_Ns']>0:omni.physx.get_physx_simulation_interface().flush_changes()
            return 0
        remove=self.model.advance(weight,dt)
        if not remove.any():
            if contact['impulse_Ns']>0:omni.physx.get_physx_simulation_interface().flush_changes()
            return 0
        mass=int(remove.sum())*self.particle_mass
        momentum=self.particle_mass*v[remove].sum(0)
        cloth.accept_water(mass,momentum)
        self.absorbed+=int(remove.sum())
        self.events.append(dict(t=self.clock,removed=int(remove.sum()),remaining=int((~remove).sum()),
                                absorbed_mass_kg=self.model.absorbed_mass,
                                maximum_contact_distance_m=float(distance[remove].max()),
                                maximum_particle_speed_m_s=float(np.linalg.norm(v,axis=1).max())))
        if remove.all():
            # Keep the shared system alive: it still owns every cloth node.
            PhysxSchema.PhysxParticleAPI(self.particles.GetPrim()).CreateParticleEnabledAttr(False)
            self.surface.MakeInvisible();self.empty=True
            omni.physx.get_physx_simulation_interface().flush_changes()
            return int(remove.sum())
        keep=~remove
        values=Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(p[keep]))
        with Sdf.ChangeBlock():
            sim=self.pset.GetSimulationPointsAttr()
            if sim.HasAuthoredValue():sim.Set(values)
            self.particles.GetPointsAttr().Set(values)
            self.particles.GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(v[keep])))
            self.particles.GetWidthsAttr().Set(Vt.FloatArray.FromNumpy(np.full(keep.sum(),2*self.rest,np.float32)))
        omni.physx.get_physx_simulation_interface().flush_changes()
        return int(remove.sum())
