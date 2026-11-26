"""Finite rigid contact fixture from source fingertip collision geometry.

This tests an asset's contact interface, not robot joint control or reach.
Only the two rigid tools receive forces. Cloth has no attachment/force API.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


class FingertipFixture:
    def __init__(self,stage,hand,model,position,rotation,out,*,force_limit_N=2.,initial_gap_m=.018,
                 collision_approximation='convexHull',sdf_resolution=512):
        from pxr import Gf,PhysxSchema,UsdGeom,UsdPhysics,UsdShade,Vt
        from calibrated_scene_reconstruction.sim.hand_fixture import HandFixture
        from calibrated_scene_reconstruction.sim.hand_collision import colliders
        from calibrated_scene_reconstruction.sim.physics import physics_material
        from calibrated_scene_reconstruction.sim.usd import set_pose
        if collision_approximation not in {'convexHull','sdf'}:raise ValueError('Explicit supported source-tip collision representation required')
        if not np.isfinite(force_limit_N) or not 0<force_limit_N<=5:
            raise ValueError('Explicit bounded contact-fixture force required')
        if not np.isfinite(initial_gap_m) or not 0<initial_gap_m<=.05:
            raise ValueError('Finite initial fingertip opening within50mm required')
        self.path='/World/ContactFixture';self.rows=[];self.paths=[];self.names=[]
        self.contacts=[];self.geometry=[];self.mass=[];self.force_limits=[];self.torque_limits=[]
        closed={j.name:float(hand['closed_state'].get(j.name,0.)) for j in model.movable}
        donor=HandFixture(stage,hand['asset'],model,closed,[0,0,0],np.eye(3),
            path='/World/ContactGeometrySource',finger_effort_Nm=hand['finger_effort_limit_Nm'],
            joint_friction_prior=hand.get('joint_friction_prior'))
        source=colliders(stage,donor.path);frames=model.transforms(closed)
        by_child={j.child:j for j in model.joints}
        self.friction=.8
        self.contact_offset_m=.0005
        material=physics_material(stage,self.path+'/Physics',self.friction,self.friction,0.)
        PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim()).CreateFrictionCombineModeAttr('multiply')
        self.material=material.GetPrim()
        global_r=np.asarray(hand['R']);global_p=np.asarray(hand['p'])
        for finger in ['thumb','index']:
            contact=hand[finger];link=contact['link'];group={link};node=link
            while node in by_child and by_child[node].kind=='fixed':
                node=by_child[node].parent;group.add(node)
            pieces=[row for row in source if row['link'] in group]
            if not pieces:raise ValueError('Actual source fingertip colliders required')
            masses=[UsdPhysics.MassAPI(stage.GetPrimAtPath(donor.path+'/'+n)).GetMassAttr().Get()
                    for n in {row['link'] for row in pieces}]
            if any(m is None or not np.isfinite(m) or m<=0 for m in masses):
                raise ValueError('Explicit positive source fingertip masses required')
            mass=float(sum(masses));name=link;path=self.path+'/'+name
            body=UsdGeom.Xform.Define(stage,path).GetPrim()
            quaternion=Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]]
            outward=np.array([-initial_gap_m/2 if finger=='thumb' else initial_gap_m/2,0,0])
            set_pose(body,np.asarray(position)+np.asarray(rotation)@outward,quaternion)
            UsdPhysics.RigidBodyAPI.Apply(body)
            native=PhysxSchema.PhysxRigidBodyAPI.Apply(body)
            native.CreateDisableGravityAttr(True)
            native.CreateSolverPositionIterationCountAttr(128);native.CreateSolverVelocityIterationCountAttr(8)
            UsdPhysics.MassAPI.Apply(body).CreateMassAttr(mass)
            all_v=[];all_f=[]
            for i,row in enumerate(pieces):
                tf=frames[row['link']]
                vertices=(row['V']@tf[:3,:3].T+tf[:3,3])@global_r.T+global_p
                mesh=UsdGeom.Mesh.Define(stage,path+'/collision_'+str(i))
                mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype('float32')))
                mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3]*len(row['F'])))
                mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(row['F'].ravel().tolist()))
                UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
                UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
                if collision_approximation=='sdf':
                    from calibrated_scene_reconstruction.sim.convex_cooking import sdf_from_declared_hull
                    sdf_from_declared_hull(mesh.GetPrim(),sdf_resolution)
                collision=PhysxSchema.PhysxCollisionAPI.Apply(mesh.GetPrim())
                collision.CreateContactOffsetAttr(self.contact_offset_m);collision.CreateRestOffsetAttr(0.)
                UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material,
                    UsdShade.Tokens.strongerThanDescendants,'physics')
                self.rows.append({'link':name,'V':vertices,'F':row['F']})
                all_f.append(row['F']+sum(len(v) for v in all_v));all_v.append(vertices)
            mesh_path=Path(out)/('source_'+finger+'_tip.npz')
            np.savez_compressed(mesh_path,V=np.concatenate(all_v),F=np.concatenate(all_f))
            cp=frames[link][:3,:3]@np.asarray(contact['pad_local'])+frames[link][:3,3]
            self.contacts.append(global_r@cp+global_p)
            # Conservative source joint-torque envelope, from CAD contact
            # Jacobians. This is a fixture bound, not an actuator identification.
            bounds=[];efforts=[];child=link
            while child!=model.root:
                joint=by_child[child];child=joint.parent
                if joint.kind=='fixed':continue
                efforts.append(donor.description['joint_effort_limits_Nm'][joint.name])
                eps=1e-6;test={**closed,joint.name:closed[joint.name]+eps}
                tf=model.transforms(test)[link]
                displaced=tf[:3,:3]@np.asarray(contact['pad_local'])+tf[:3,3]
                lever=float(np.linalg.norm((displaced-cp)/eps))
                if lever>1e-8:bounds.append(donor.description['joint_effort_limits_Nm'][joint.name]/lever)
            limit=min(float(force_limit_N),.25*min(bounds))
            self.force_limits.append(limit);self.mass.append(mass)
            self.torque_limits.append(min(.03,.25*min(efforts)))
            self.paths.append(path);self.names.append(name)
            self.geometry.append({'name':'source_'+finger+'_tip','body':name,'mesh':str(mesh_path.resolve()),
                'sha256':hashlib.sha256(mesh_path.read_bytes()).hexdigest(),'source_links':sorted(group),
                'mass_kg':mass,'force_limit_N':limit,'color':[.28,.31,.34]})
        self.source={'asset':hand['asset'],'sha256':{str(Path(p).resolve()):hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in model.dependencies},'closed_state':closed,'geometry_only':True,
            'robot_joint_control_test':False,'cloth_forces':False,'cloth_attachments':False,
            'collision_approximation':collision_approximation,
            'collision_contact_offset_m':self.contact_offset_m,'collision_rest_offset_m':0.,
            'sdf_resolution':sdf_resolution if collision_approximation=='sdf' else None,
            'scope':'Actual source fingertip and fixed adjacent distal colliders/masses. Two independently actuated finite-force rigid contact tools; not a full robot/hand controller.'}
        stage.RemovePrim(donor.path)
        self.contacts=np.asarray(self.contacts)
        (Path(out)/'fixture_geometry.json').write_text(json.dumps({'source':self.source,'parts':self.geometry},indent=2)+'\n')
        self.held=None;self.tick=0;self.peak=None

    def initialize(self,hz):
        from isaacsim.core.prims import RigidPrim
        from calibrated_scene_reconstruction.core.geometry.rigid import as_numpy
        self.view=RigidPrim(self.paths,reset_xform_properties=False,name='finite_source_tips')
        self.view.initialize()
        if list(self.view.prim_paths)!=self.paths:raise ValueError('Native source fingertip order changed')
        self.stride=max(1,round(hz/240));self.dt=self.stride/hz
        self.native_mass=as_numpy(self.view.get_masses())
        if not np.allclose(self.native_mass,self.mass,rtol=1e-5):raise ValueError('Native source fingertip masses differ')
        inertia=as_numpy(self.view.get_inertias()).reshape(-1,3,3)
        self.inertia=np.linalg.eigvalsh(inertia).min(1)
        if (self.inertia<=0).any():raise ValueError('Positive native compound fingertip inertia required')

    def control(self,centre,rotation,gap,*,normal_force_limit_N=None,normal_preload_N=None):
        import torch
        if normal_force_limit_N is not None and (not np.isfinite(normal_force_limit_N) or normal_force_limit_N<=0):
            raise ValueError('Positive finite optional closing-axis force limit required')
        if normal_preload_N is not None and (not np.isfinite(normal_preload_N) or normal_preload_N<=0
                or normal_force_limit_N is None or normal_preload_N>normal_force_limit_N):
            raise ValueError('Explicit preload requires a positive closing-force cap at least as large')
        if self.held is None or self.tick%self.stride==0:
            p,q=self.view.get_world_poses(clone=False);velocity=self.view.get_velocities(clone=False)
            target=np.tile(centre,(2,1));target+=np.array([[-gap/2,0,0],[gap/2,0,0]])@np.asarray(rotation).T
            goal=torch.as_tensor(target,dtype=p.dtype,device=p.device)
            scalar=Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]]
            wanted=torch.as_tensor(scalar[None],dtype=p.dtype,device=p.device).expand_as(q)
            tw,tv=wanted[:,:1],wanted[:,1:];aw,av=q[:,:1],q[:,1:]
            w=tw*aw+(tv*av).sum(-1,keepdim=True);v=aw*tv-tw*av-torch.linalg.cross(tv,av)
            sign=torch.where(w<0,-1.,1.);w,v=w*sign,v*sign
            length=torch.linalg.vector_norm(v,dim=1,keepdim=True)
            error=v*(2*torch.atan2(length,w)/length.clamp_min(1e-12))
            m=torch.as_tensor(self.mass,dtype=p.dtype,device=p.device)[:,None]
            inertia=torch.as_tensor(self.inertia,dtype=p.dtype,device=p.device)[:,None]
            k=5000.;c=2*torch.sqrt(k*m);kr=1.;cr=2*torch.sqrt(kr*inertia);dt=self.dt
            force=(k*(goal-p-dt*velocity[:,:3])-c*velocity[:,:3])/(1+(c*dt+k*dt*dt)/m)
            if normal_force_limit_N is not None:
                axis=torch.as_tensor(np.asarray(rotation)[:,0],dtype=p.dtype,device=p.device)
                normal=force@axis
                force+=(normal.clamp(-normal_force_limit_N,normal_force_limit_N)-normal)[:,None]*axis
            torque=(kr*(error-dt*velocity[:,3:])-cr*velocity[:,3:])/(1+(cr*dt+kr*dt*dt)/inertia)
            limits=torch.as_tensor(self.force_limits,dtype=p.dtype,device=p.device)[:,None]
            if normal_preload_N is not None:
                from calibrated_scene_reconstruction.sim.pinch_load import apply
                axis=torch.as_tensor(np.asarray(rotation)[:,0],dtype=p.dtype,device=p.device)
                # Implicit-equivalent damping for the sample/hold interval.
                # Continuous critical damping explodes on gram-scale tools
                # when sampled at240Hz, even at a small bounded force.
                normal_damping=c/(1+c*dt/m)
                force=apply(force,velocity[:,:3],axis,normal_preload_N,normal_damping,
                    limits.clamp_max(normal_force_limit_N))
            force*=torch.minimum(torch.ones_like(limits),limits/torch.linalg.vector_norm(force,dim=1,keepdim=True).clamp_min(1e-12))
            torque_limits=torch.as_tensor(self.torque_limits,dtype=p.dtype,device=p.device)[:,None]
            torque*=torch.minimum(torch.ones_like(torque_limits),torque_limits/torch.linalg.vector_norm(torque,dim=1,keepdim=True).clamp_min(1e-12))
            self.held=(force,torque)
            peak=torch.stack([torch.linalg.vector_norm(force,dim=1),torch.linalg.vector_norm(torque,dim=1)],1)
            self.peak=peak if self.peak is None else torch.maximum(self.peak,peak)
        self.view.apply_forces_and_torques_at_pos(forces=self.held[0],torques=self.held[1],is_global=True)
        self.tick+=1

    def poses(self):
        from calibrated_scene_reconstruction.core.geometry.rigid import as_numpy
        p,q=map(as_numpy,self.view.get_world_poses())
        return np.c_[p,q[:,[1,2,3,0]]]
