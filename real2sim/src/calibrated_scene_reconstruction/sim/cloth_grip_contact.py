"""Two-way bounded pad friction supplement for native thin particle cloth.

Native collisions supply the normal contact. Tangential static friction uses
an explicit compliant-pad preload estimate, with equal/opposite articulation
forces and torques. Opening the jaws removes contact; friction zero removes
every supplemental impulse. No permanent constraint or attachment is created.
"""
import numpy as np
from calibrated_scene_reconstruction.assets.pinch_friction import pinch_impulses,bounded_static_projection


class ClothGripContact:
    def __init__(self,robot,cloth,pinch,hz):
        self.robot,self.cloth,self.pinch,self.hz=robot,cloth,pinch,hz
        self.coefficient=.8;self.events=[];self.previous=None;self.clock=0.
        self.anchors=np.full((len(cloth.rest),2),np.nan)
        self.description=dict(model='Two-way bounded Coulomb pad friction',
            estimated_normal_force_limit_N=.3,pad_stiffness_N_m=1500.,
            coefficient=.8,requires_opposing_geometric_contact=True,
            prescribed_cloth_pose_or_trajectory=False,hand_reaction_forces=True,
            force_limited_tangential_contact_projections=True,
            normal_force_is_compliance_estimate_not_sensor_measurement=True,
            static_friction_memory='Tangential contact anchors released on opening/loss of contact or Coulomb sliding')
        cloth.description.pop('external_cloth_forces',None)
        cloth.description['external_contact_impulses']=True
        cloth.description['arbitrary_hand_following_forces']=False
        cloth.description.pop('prescribed_cloth_positions_after_initialization',None)
        cloth.description['contact_position_projections']=True
        cloth.description['prescribed_cloth_pose_or_trajectory']=False

    def advance(self,dt):
        import torch,omni.physx
        pads=[];normals=[];tangents=[];indices=[]
        for finger in ['thumb','index']:
            info=self.pinch[finger];p,R=self.robot.link_pose(info['link'])
            n=np.asarray(info['normal_local'],float);n/=np.linalg.norm(n)
            t=np.array([1.,0,0]);t-=n*(t@n);t/=np.linalg.norm(t)
            pads.append(p+R@info['pad_local']);normals.append(R@n);tangents.append(R@t)
            indices.append(self.robot.body_names.index(info['link']))
        pads=np.array(pads);self.clock+=dt
        self.description['coefficient']=self.coefficient
        speed=np.zeros(3) if self.previous is None else (pads-self.previous).mean(0)/dt
        self.previous=pads.copy()
        points=self.cloth.points();velocity=self.cloth.velocities();masses=self.cloth.masses()
        axis=np.asarray(normals[0])-normals[1];axis/=np.linalg.norm(axis)
        a=np.asarray(tangents[0]).copy();a-=axis*(a@axis);a/=np.linalg.norm(a)
        basis=np.column_stack([a,np.cross(axis,a)])
        coordinates=(points-pads.mean(0))@basis
        slip=np.where(np.isfinite(self.anchors),coordinates-self.anchors,0)@basis.T
        impulses,event=pinch_impulses(points,velocity,masses,pads,normals,tangents,speed,dt,
                                       coefficient=self.coefficient,radius=self.cloth.settings['rest_offset'])
        ids=event.pop('contact_indices');sliding=event.pop('sliding_indices')
        active=np.zeros(len(points),bool);active[ids]=True;self.anchors[~active]=np.nan
        new=active&~np.isfinite(self.anchors).all(1);self.anchors[new]=coordinates[new]
        self.anchors[sliding]=coordinates[sliding]
        sticking=[i for i in ids if i not in set(sliding)]
        budget=max(0.,event['force_limit_N']*dt-event['impulse_Ns'])
        correction,position_impulse=bounded_static_projection(slip,masses,sticking,budget,dt)
        # A sticking contact already has pad-relative tangential velocity zero
        # from the Coulomb velocity solve. Do not add the same positional slip
        # again to its velocity: that would cause an artificial oscillation.
        event['position_projection_max_m']=float(np.linalg.norm(correction,axis=1).max())
        event['position_correction_impulse_bound_Ns']=float(np.linalg.norm(position_impulse,axis=1).sum())
        moving=(np.linalg.norm(correction,axis=1)+1e-10<np.linalg.norm(slip,axis=1))&active
        self.anchors[moving]=(coordinates+correction@basis)[moving]
        event['impulse_Ns']=float(np.linalg.norm(impulses,axis=1).sum())
        if event['impulse_Ns']<=0:return
        self.cloth.apply_contact_state(points+correction,velocity+impulses/masses[:,None])
        omni.physx.get_physx_simulation_interface().flush_changes()
        self.robot._ensure_views()
        view=self.robot.view._physics_view
        ref=view.get_link_transforms();shape=(1,view.max_links,3)
        force=np.zeros(shape,np.float32);torque=force.copy();position=force.copy()
        total=impulses.sum(0);reaction=-total/2
        for i,p in zip(indices,pads):
            force[0,i]=reaction*self.hz
            torque[0,i]=np.cross(points-p,-impulses/2).sum(0)*self.hz
            position[0,i]=p
        convert=lambda a:torch.as_tensor(a,dtype=torch.float32,device=ref.device)
        view.apply_forces_and_torques_at_position(convert(force),convert(torque),convert(position),
                    torch.zeros(1,dtype=torch.int32,device=ref.device),True)
        self.events.append(dict(t=self.clock,**event,
            computed_linear_momentum_error_Ns=float(np.linalg.norm(total+2*reaction))))
