"""Native two-dimensional particle cloth, held only by physical finger contact.

No rigid patch, attachment, kinematic node, prescribed runtime position or
external hand-following force is used. All mesh vertices are dynamic particles.
"""
import numpy as np
from scipy.spatial import cKDTree
from pxr import PhysxSchema,Usd,UsdGeom,UsdPhysics,Vt,Sdf

from calibrated_scene_reconstruction.assets.cloth_rest import triangulate_grid
from calibrated_scene_reconstruction.assets.cloth2d_geometry import center_sheet
from calibrated_scene_reconstruction.sim.usd import add_mesh,color_material


class Cloth2DWipe:
    @staticmethod
    def grip_height(*_):return .0175

    def __init__(self,stage,grip,table_z,yaw=0.,particle_system=None,absorption_mass_in_dynamics=False,**_):
        from calibrated_scene_reconstruction.sim.mechanism_stage import add_deformable
        self.stage=stage;self.path='/World/WetWipe';self.table_z=table_z
        self.absorption_mass_in_dynamics=absorption_mass_in_dynamics
        data=center_sheet(grip_height=self.grip_height())
        c,s=np.cos(yaw),np.sin(yaw);R=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        points=data['points']@R.T+np.asarray(grip)
        faces,rest,meshing=triangulate_grid(points,data['material'],data['rows'],data['columns'])
        self.faces=faces;self.grasp_nodes=data['grasp_nodes'];self.rest=data['material']
        self.part=dict(name='wet_wipe',kind='cloth',V=points,F=faces,cloth_rest=rest)
        self.mesh=add_mesh(stage,self.path,points,faces,data['material'][:,:2]/.12+.5,
                           color_material(stage,'/World/Looks/Cloth2D',[.93,.94,.91],roughness=.97))
        self.mesh.CreateDoubleSidedAttr(True)
        self.settings=dict(mass=.002,contact_offset=.0014,rest_offset=.00085,solid_rest_offset=.0011,
                           stretch_stiffness=1200.,shear_stiffness=400.,bend_stiffness=.002,
                           damping=.02,friction=.8,solver_position_iterations=128,max_depenetration_velocity=.50)
        add_deformable(stage,self.mesh,self.part,self.settings,None)
        api=PhysxSchema.PhysxParticleAPI(self.mesh.GetPrim())
        self.system_path=api.GetParticleSystemRel().GetTargets()[0]
        PhysxSchema.PhysxParticleSystem.Get(stage,self.system_path).CreateMaxVelocityAttr(.8)
        if particle_system is not None:
            old=self.system_path;self.system_path=particle_system
            api.GetParticleSystemRel().SetTargets([particle_system]);stage.RemovePrim(old)
        api.CreateParticleGroupAttr(1)
        self.description=dict(backend='native 2D particle cloth',nodes=len(points),triangles=len(faces),
            width_m=.12,length_m=.12,mass_kg=.002,grip_location='center',grip_material_fraction=[.5,.5],
            hand_attachments=False,kinematic_nodes=0,rigid_patches=0,arbitrary_hand_following_forces=False,
            cloth_pose_following=False,prescribed_cloth_positions_after_initialization=False,
            full_2d_stretch_shear_and_bending=True,shared_fluid_system=particle_system is not None,
            material=self.settings,meshing=meshing,
            limitation='Native particle-shell discretization with uncalibrated wet-wipe material parameters')
        self.description['absorbed_water_changes_mechanical_mass']=absorption_mass_in_dynamics
        self.description['absorption_reservoir']='Absorbed water tracked as a capacity reservoir; mechanical cloth mass held constant in this prototype' if not absorption_mass_in_dynamics else 'Water mass enters native cloth inertia'

    def initialize(self):
        import carb
        self.usd_readback=not carb.settings.get_settings().get_as_bool('/physics/suppressReadback')
        if self.usd_readback:
            mass=float(UsdPhysics.MassAPI(self.mesh.GetPrim()).GetMassAttr().Get())
            self.native_masses=np.full(len(self.rest),mass/len(self.rest))
            self.wet_mass_kg=getattr(self,'wet_mass_kg',0.)
            self.maximum_absorption_momentum_error=getattr(self,'maximum_absorption_momentum_error',0.)
            self.description['state_readback']='Native PhysX USD particle readback; uniform nodal mass derived from authored total mass'
            self.audit_independence()
            return
        import omni.physics.tensors as tensors
        self.owner=tensors.create_simulation_view('torch');self.owner.set_subspace_roots('/')
        self.view=self.owner.create_particle_cloth_view(self.path)
        masses=self.view.get_masses()[0].detach().cpu().numpy()
        if (masses<=0).any() or abs(float(masses.sum())-(.002+getattr(self,'wet_mass_kg',0.)))>.00002:
            raise RuntimeError('Cloth needs finite positive native node masses totaling 2 g')
        self.native_masses=masses.copy()
        self.wet_mass_kg=getattr(self,'wet_mass_kg',0.)
        self.maximum_absorption_momentum_error=getattr(self,'maximum_absorption_momentum_error',0.)
        self.audit_independence()

    def points(self):
        if self.usd_readback:
            import omni.physx
            omni.physx.get_physx_interface().update_transformations(False,True,True,False)
            points=np.asarray(self.mesh.GetPointsAttr().Get(),np.float32).copy()
            # Native cloth recentres its mesh and translates the prim. USD
            # vertices are local; the material/robot geometry uses world space.
            T=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(self.mesh.GetPrim())).T
            points=points@T[:3,:3].T+T[:3,3]
            if not np.isfinite(points).all():raise RuntimeError('Invalid native USD cloth positions')
            return points
        points=self.view.get_positions()[0].reshape(-1,3).detach().cpu().numpy().copy()
        if not np.isfinite(points).all():raise RuntimeError('Invalid native 2D cloth state')
        return points

    def velocities(self):
        if self.usd_readback:
            import omni.physx
            omni.physx.get_physx_interface().update_transformations(False,True,True,False)
            return np.asarray(self.mesh.GetVelocitiesAttr().Get(),np.float32).copy()
        return self.view.get_velocities()[0].reshape(-1,3).detach().cpu().numpy().copy()

    def masses(self):
        if self.usd_readback:
            return np.full(len(self.rest),float(UsdPhysics.MassAPI(self.mesh.GetPrim()).GetMassAttr().Get())/len(self.rest))
        return self.view.get_masses()[0].detach().cpu().numpy().copy()

    def apply_contact_velocities(self,velocities):
        """Apply the equal/opposite impulse from the split fluid contact solve."""
        values=np.asarray(velocities,np.float32)
        if self.usd_readback:
            self.mesh.GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(values))
        else:
            import torch
            self.view.set_velocities(torch.as_tensor(values.reshape(1,-1),device='cuda:0'))

    def apply_contact_state(self,points,velocities):
        """Commit a force-limited PBD contact correction, never a hand pose."""
        if not self.usd_readback:raise RuntimeError('Split PBD contact requires explicit USD state mode')
        T=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(self.mesh.GetPrim())).T
        local=(np.asarray(points)-T[:3,3])@np.linalg.inv(T[:3,:3]).T
        with Sdf.ChangeBlock():
            self.mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(local.astype(np.float32)))
            self.mesh.GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(velocities,np.float32)))

    def accept_water(self,mass,momentum):
        if not self.absorption_mass_in_dynamics:
            # Task-level uptake reservoir. Do not recook native cloth mass at
            # every sink event: that changes the SDK contact solution mid-grasp.
            self.wet_mass_kg+=mass
            self.description['absorbed_water_mass_kg']=self.wet_mass_kg
            self.description['absorption_momentum_transfer']=False
            return
        import torch
        from calibrated_scene_reconstruction.assets.wetting import transfer_water_mass
        old_mass=self.masses()
        velocity=self.velocities()
        new_mass,new_velocity=transfer_water_mass(old_mass,velocity,mass,momentum)
        expected=(old_mass[:,None]*velocity).sum(0)+momentum
        measured=(new_mass[:,None]*new_velocity).sum(0)
        self.maximum_absorption_momentum_error=max(self.maximum_absorption_momentum_error,float(np.linalg.norm(expected-measured)))
        if self.usd_readback:
            UsdPhysics.MassAPI(self.mesh.GetPrim()).GetMassAttr().Set(float(new_mass.sum()))
            self.mesh.GetVelocitiesAttr().Set(Vt.Vec3fArray.FromNumpy(new_velocity.astype(np.float32)))
        else:
            index=torch.zeros(1,dtype=torch.int32,device='cuda:0')
            self.view.set_masses(torch.as_tensor(new_mass[None],dtype=torch.float32,device='cuda:0'),index)
            self.view.set_velocities(torch.as_tensor(new_velocity.reshape(1,-1),dtype=torch.float32,device='cuda:0'),index)
        self.wet_mass_kg+=mass
        self.description['absorbed_water_mass_kg']=self.wet_mass_kg
        self.description['absorption_momentum_transfer']=True

    def grip_center(self):return self.points()[self.grasp_nodes].mean(0)

    def finger_proximity(self,robot,pinch):
        pads=[]
        for name in ['thumb','index']:
            p,R=robot.link_pose(pinch[name]['link']);pads.append(p+R@np.asarray(pinch[name]['pad_local']))
        distances=cKDTree(self.points()[self.grasp_nodes]).query(pads)[0]
        distances=np.maximum(distances-self.settings['rest_offset'],0)
        return dict(same_material_patch=True,maximum_pad_distance_m=float(distances.max()),
                    pad_to_material_patch_m=distances.tolist())

    def update_visual(self):
        points=self.points();tri=points[self.faces]
        face=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normals=np.zeros_like(points)
        for i in range(3):np.add.at(normals,self.faces[:,i],face)
        normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
        T=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(self.mesh.GetPrim())).T
        normals=normals@T[:3,:3]
        self.mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))

    def contact_samples(self,table_z,height=.006):
        points=self.points();return points[points[:,2]<table_z+height]

    def audit_independence(self):
        for prim in self.stage.Traverse():
            if 'Attachment' in prim.GetTypeName():raise RuntimeError('Attachments are forbidden')
            if str(prim.GetPath()).startswith(self.path) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError('No rigid panels or rigid grip patch allowed')
        prim=self.mesh.GetPrim()
        assert prim.HasAPI(PhysxSchema.PhysxParticleClothAPI)
        return dict(attachments=0,kinematic_nodes=0,rigid_patches=0,all_nodes_dynamic=True)
