"""Renderer-free execution using the PhysX shipped inside Isaac Sim 5.1.

This is an optional worker backend for native scene models. The standard Isaac
World backend remains available for robot/sensor factories that need it.
"""
import numpy as np

from calibrated_scene_reconstruction.core.records import require


class Context:
    def __init__(self,prim):
        from pxr import PhysxSchema
        self.schema=PhysxSchema.PhysxSceneAPI.Apply(prim)

    def enable_fabric(self,value):
        require(value is False,'Physics-only worker requires USD readback')

    def enable_gpu_dynamics(self,value):self.schema.CreateEnableGPUDynamicsAttr(bool(value))
    def set_broadphase_type(self,value):self.schema.CreateBroadphaseTypeAttr(value)
    def set_solver_type(self,value):self.schema.CreateSolverTypeAttr(value)


class World:
    _instance=None
    raw_physx=True

    @classmethod
    def clear_instance(cls):
        if cls._instance is not None:cls._instance.clear()
        cls._instance=None

    def __init__(self,*,stage_units_in_meters=1.,physics_dt=1/120,rendering_dt=None,backend='numpy',device='cpu'):
        import omni.usd
        import omni.physx
        from pxr import UsdGeom,UsdPhysics,Gf
        require(backend=='numpy' and device=='cpu','Raw PhysX exposes source-ordered CPU readback, not a tensor wrapper')
        self.stage=omni.usd.get_context().get_stage();self.dt=physics_dt;self.time=0.;self.attached=False
        UsdGeom.SetStageMetersPerUnit(self.stage,stage_units_in_meters);UsdGeom.SetStageUpAxis(self.stage,UsdGeom.Tokens.z)
        scene=UsdPhysics.Scene.Define(self.stage,'/physicsScene')
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));scene.CreateGravityMagnitudeAttr(9.81)
        self.context=Context(scene.GetPrim());self.api=omni.physx.get_physx_simulation_interface()
        self.interface=omni.physx.get_physx_interface();self.contacts=[];self.subscription=None
        self.contact_cache={};self.contact_seen=set();self.contact_unknown=[];self.cached_contacts=0;self.contact_events=0
        self.contact_unknown_examples={}
        World._instance=self

    def get_physics_context(self):return self.context
    def get_physics_dt(self):return self.dt

    def reset(self):
        import omni.usd
        require(self.api.attach_stage(omni.usd.get_context().get_stage_id()),'Native PhysX stage attachment failed')
        self.attached=True;self.time=0.

    def step(self,render=False):
        require(not render,'Use the renderer-backed worker for rendering')
        self.contacts=[]
        self.contact_seen=set();self.contact_unknown=[];self.cached_contacts=0
        self.api.simulate(self.dt,self.time);self.api.fetch_results()
        self.interface.update_transformations(False,True,True)
        for key,item in self.contact_cache.items():
            signature=self._relative_pose(key[:2])
            if key in self.contact_seen:
                item['signature']=signature;self.contacts.extend(item['distances'])
            elif signature is not None and item.get('signature') is not None and np.array_equal(signature,item['signature']):
                self.contacts.extend(item['distances']);self.cached_contacts+=1
            else:
                self.contact_unknown.append(key)
                examples=getattr(self,'contact_unknown_examples',None)
                if examples is not None and key not in examples and len(examples)<16:
                    old=item.get('signature')
                    examples[key]={'t':self.time,'actors':self._actor_paths(key[:2]),
                        'previous_signature':None if old is None else old.tolist(),
                        'current_signature':None if signature is None else signature.tolist()}
                    if getattr(self,'contact_diagnostic_path',None):
                        from calibrated_scene_reconstruction.core.records import write
                        path=self.contact_diagnostic_path
                        write(path.with_name(f'unknown_contact_{len(examples)-1:04d}.json'),examples[key])
        self.time+=self.dt

    def stop(self):pass

    def clear(self):
        self.subscription=None
        if self.attached:self.api.detach_stage()
        self.attached=False

    def contact_monitor(self):
        from omni.physx.bindings._physx import ContactEventType
        def receive(headers,data):
            for header in headers:
                key=(header.actor0,header.actor1,header.collider0,header.collider1)
                if header.type==ContactEventType.CONTACT_LOST:
                    self.contact_cache.pop(key,None);continue
                start,count=header.contact_data_offset,header.num_contact_data
                if count:
                    self.contact_cache[key]={'distances':[float(data[i].separation) for i in range(start,start+count)]}
                    self.contact_seen.add(key);self.contact_events+=1
        self.subscription=self.api.subscribe_contact_report_events(receive)

    def _relative_pose(self,actors):
        from pxr import PhysicsSchemaTools,UsdGeom
        matrices=[]
        for actor in actors:
            path=str(PhysicsSchemaTools.intToSdfPath(actor));prim=self.stage.GetPrimAtPath(path)
            if not prim or any(x in str(prim.GetMetadata('apiSchemas')) for x in ['Particle','Deformable']):return None
            value=self.interface.get_rigidbody_transformation(path)
            if value['ret_val']:
                from scipy.spatial.transform import Rotation
                M=np.eye(4);M[:3,:3]=Rotation.from_quat(np.asarray(value['rotation'])).as_matrix();M[:3,3]=value['position']
            else:M=np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.)).T
            matrices.append(M)
        return np.linalg.inv(matrices[0])@matrices[1]

    @staticmethod
    def _actor_paths(actors):
        from pxr import PhysicsSchemaTools
        return [str(PhysicsSchemaTools.intToSdfPath(actor)) for actor in actors]


class SingleRigidPrim:
    def __init__(self,path,reset_xform_properties=False):
        self.path=path

    def initialize(self):pass

    def get_world_pose(self):
        import omni.physx
        value=omni.physx.get_physx_interface().get_rigidbody_transformation(self.path)
        require(value['ret_val'],'Missing native rigid transformation: '+self.path)
        p=np.array(value['position'],float);q=np.array(value['rotation'],float)
        return p,q[[3,0,1,2]]
