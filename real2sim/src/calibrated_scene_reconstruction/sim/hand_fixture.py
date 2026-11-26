"""A real articulated hand on a finite wrist-wrench test fixture.

The source hand geometry and joint limits are retained. Only the wrist and
finger motors are actuated; this class has no cloth/node/attachment API.
"""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def axis_properties(prim):
    """Capture effective source properties before a per-axis API can override them."""
    active=prim.HasAPI('PhysxJointAxisAPI','angular')
    result={}
    for name,default in [('armature',0.),('maxJointVelocity',1e6)]:
        key=('physxJointAxis:angular:' if active else 'physxJoint:')+name
        attr=prim.GetAttribute(key);value=attr.Get() if attr else None
        result[name]=float(default if value is None else value)
    return result


def preserve_axis_properties(prim,values):
    from pxr import Sdf
    for name,value in values.items():
        prim.CreateAttribute('physxJointAxis:angular:'+name,Sdf.ValueTypeNames.Float).Set(float(value))


def effort_budget(requested, source_limit):
    """A fixture budget may reduce a source motor limit, never enlarge it."""
    if not np.isfinite(requested) or requested <= 0:
        raise ValueError('Positive finite fixture effort required')
    if source_limit is None or source_limit == np.inf:
        return float(requested)  # Unknown hardware limit; retain an explicit finite fixture budget.
    if not np.isfinite(source_limit) or source_limit <= 0:
        raise ValueError('Cannot replace a nonpositive or invalid source motor limit')
    return float(min(requested, source_limit))


def tensor_wrist_wrench(position, quaternion, velocity, target_position, target_quaternion,
                        feedback_dt=0., mass_kg=1., minimum_inertia_kg_m2=1.):
    """Same finite PD wrench, with scalar-first quaternions, entirely on device."""
    import torch
    tw, tv = target_quaternion[..., :1], target_quaternion[..., 1:]
    aw, av = quaternion[..., :1], quaternion[..., 1:]
    w = tw * aw + (tv * av).sum(-1, keepdim=True)
    v = aw * tv - tw * av - torch.linalg.cross(tv, av)
    sign = torch.where(w < 0, -1., 1.)
    w, v = w * sign, v * sign
    length = torch.linalg.vector_norm(v, dim=-1, keepdim=True)
    rotvec = v * (2 * torch.atan2(length, w) / length.clamp_min(1e-12))
    # Backward-Euler PD prediction avoids an explicit derivative feedback
    # instability when the controller period exceeds a small palm inertia's
    # damping time. This changes the finite controller, never body inertia.
    force = (1000. * (target_position - position - feedback_dt * velocity[..., :3])
             - 80. * velocity[..., :3]) / (1 + (80. * feedback_dt + 1000. * feedback_dt**2) / mass_kg)
    torque = (20. * (rotvec - feedback_dt * velocity[..., 3:])
              - 2. * velocity[..., 3:]) / (1 + (2. * feedback_dt + 20. * feedback_dt**2) / minimum_inertia_kg_m2)
    force = force * (100. / torch.linalg.vector_norm(force, dim=-1, keepdim=True).clamp_min(100.))
    torque = torque * (5. / torch.linalg.vector_norm(torque, dim=-1, keepdim=True).clamp_min(5.))
    return force, torque


class HandFixture:
    def __init__(self, stage, asset, model, state, position, rotation, *, path='/World/Hand',
                 friction=.8, finger_effort_Nm=.8, contact_offset_m=.0005, joint_friction_compensation=False,
                 joint_friction_prior=None, control_hz=240.):
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade
        from calibrated_scene_reconstruction.sim.physics import physics_material

        self.stage, self.path, self.model = stage, path, model
        self.position, self.rotation = np.asarray(position, float), np.asarray(rotation, float)
        self.state = dict(state)
        if not np.isfinite(control_hz) or not 120 <= control_hz <= 7680:
            raise ValueError('Bounded finite hand controller frequency required')
        self.control_hz = float(control_hz)
        if joint_friction_prior is not None:
            f=joint_friction_prior
            if (joint_friction_compensation or not f.get('basis')
                    or not np.isfinite([f['static_Nm'],f['dynamic_Nm'],f['viscous_Nms_rad']]).all()
                    or not 0<=f['dynamic_Nm']<=f['static_Nm']<finger_effort_Nm or f['viscous_Nms_rad']<0):
                raise ValueError('Explicit SI actuator-friction fixture prior, below the motor budget, required')
        if (self.position.shape != (3,) or self.rotation.shape != (3, 3)
                or not np.isfinite(self.position).all() or not np.isfinite(self.rotation).all()
                or not np.allclose(self.rotation.T@self.rotation, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(self.rotation), 1.)
                or not np.isfinite([friction, finger_effort_Nm, contact_offset_m]).all()
                or friction < 0 or min(finger_effort_Nm, contact_offset_m) <= 0):
            raise ValueError('Finite rigid wrist transform, friction and bounded motor/contact settings required')
        prim = stage.DefinePrim(path, 'Xform');prim.GetReferences().AddReference(str(Path(asset).resolve()))
        disabled_graphs=[str(child.GetPath()) for child in Usd.PrimRange(prim) if child.GetTypeName()=='OmniGraph']
        for name in disabled_graphs:
            graph=stage.GetPrimAtPath(name)
            if graph:graph.SetActive(False)
        def pose(prim, p, r):
            q = Rotation.from_matrix(r).as_quat()
            x = UsdGeom.Xformable(prim);x.ClearXformOpOrder()
            x.AddTranslateOp(opSuffix='fixture').Set(Gf.Vec3d(*np.asarray(p, float)))
            x.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble,opSuffix='fixture').Set(
                Gf.Quatd(float(q[3]), Gf.Vec3d(*q[:3])))
        pose(prim, self.position, self.rotation)
        transforms = model.transforms(state)
        bodies, joints, colliders = {}, {}, []
        source_joint_friction={};source_axis_properties={};source_efforts={};joint_efforts={}
        for p in list(Usd.PrimRange(prim)):
            if p.IsInstance():p.SetInstanceable(False)
        # PhysX107.3 deformable beta combines against the rigid partner's
        # dynamic-friction mode. Multiplication permits a finger-only zero
        # control while leaving cloth/container and cloth/self friction intact.
        material = physics_material(stage, path+'/FingerPhysics', friction, friction, 0.,friction_combine='multiply')
        for p in list(Usd.PrimRange(prim)):
            if p.HasAPI(UsdPhysics.ArticulationRootAPI):p.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            if p.IsA(UsdPhysics.Joint):
                joint = UsdPhysics.Joint(p)
                if not joint.GetBody0Rel().GetTargets() or not joint.GetBody1Rel().GetTargets():
                    p.SetActive(False)  # Replace the asset's world mount with a finite wrist wrench.
                elif p.IsA(UsdPhysics.RevoluteJoint):
                    name=p.GetName();joints[name]=str(p.GetPath())
                    source_limit=UsdPhysics.DriveAPI(p,'angular').GetMaxForceAttr().Get()
                    budget=effort_budget(finger_effort_Nm,source_limit)
                    source_efforts[name]=float(source_limit) if source_limit is not None and np.isfinite(source_limit) else None
                    joint_efforts[name]=budget
                    if joint_friction_prior and joint_friction_prior['static_Nm']>=budget:
                        raise ValueError('Actuator friction prior exceeds the source-capped motor budget: '+name)
                    source_axis_properties[name]=axis_properties(p)
                    api=PhysxSchema.PhysxJointAPI(p)
                    source_joint_friction[name]=api.GetJointFrictionAttr().Get() if api else 0.
                    if joint_friction_compensation or joint_friction_prior is not None:
                        PhysxSchema.PhysxJointAPI.Apply(p).CreateJointFrictionAttr(0.)
                    if joint_friction_prior is not None:
                        from pxr import Sdf
                        if not p.ApplyAPI('PhysxJointAxisAPI','angular'):
                            raise RuntimeError('Native SI per-axis joint friction unavailable')
                        preserve_axis_properties(p,source_axis_properties[name])
                        for label,value in [('staticFrictionEffort',f['static_Nm']),('dynamicFrictionEffort',f['dynamic_Nm']),
                                            ('viscousFrictionCoefficient',f['viscous_Nms_rad']*np.pi/180)]:
                            p.CreateAttribute('physxJointAxis:angular:'+label,Sdf.ValueTypeNames.Float).Set(float(value))
                    drive=UsdPhysics.DriveAPI.Apply(p,'angular')
                    # Scale gains with the finite effort budget. Keeping the
                    # high-effort damping on a low-force pinch limits free
                    # finger speed before any contact and mimics a false stop.
                    drive.CreateTypeAttr('force');drive.CreateStiffnessAttr(20.*budget*np.pi/180)
                    drive.CreateDampingAttr(2.*budget*np.pi/180);drive.CreateMaxForceAttr(budget)
                    drive.CreateTargetPositionAttr(float(np.degrees(state.get(name,0.))))
                    PhysxSchema.JointStateAPI.Apply(p,'angular').CreatePositionAttr(float(np.degrees(state.get(name,0.))))
                    PhysxSchema.JointStateAPI(p,'angular').CreateVelocityAttr(0.)
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                name=p.GetName()
                if name not in transforms:raise ValueError('Hand source has an unmodeled rigid link: '+name)
                tf=transforms[name];pose(p,tf[:3,3],tf[:3,:3]);bodies[name]=str(p.GetPath())
                body=UsdPhysics.RigidBodyAPI(p);body.CreateVelocityAttr(Gf.Vec3f(0));body.CreateAngularVelocityAttr(Gf.Vec3f(0))
                native=PhysxSchema.PhysxRigidBodyAPI.Apply(p)
                native.CreateDisableGravityAttr(True)  # Explicit active hand gravity compensation only; cloth gravity stays on.
                native.CreateSolverPositionIterationCountAttr(128);native.CreateSolverVelocityIterationCountAttr(8)
            if p.HasAPI(UsdPhysics.CollisionAPI):
                collision=PhysxSchema.PhysxCollisionAPI.Apply(p)
                collision.CreateRestOffsetAttr(0.);collision.CreateContactOffsetAttr(contact_offset_m)
                UsdShade.MaterialBindingAPI.Apply(p).Bind(material,UsdShade.Tokens.strongerThanDescendants,'physics')
                colliders.append(str(p.GetPath()))
        self.palm_path=bodies[model.root]
        root=prim;UsdPhysics.ArticulationRootAPI.Apply(root)
        art=PhysxSchema.PhysxArticulationAPI.Apply(root);art.CreateEnabledSelfCollisionsAttr(True)
        art.CreateSolverPositionIterationCountAttr(128);art.CreateSolverVelocityIterationCountAttr(8)
        art.CreateSleepThresholdAttr(0.)  # Direct-GPU motor target writes must not leave an idle hand asleep.
        # Known CAD adjacency: the thumb metacarpal nearly touches the palm.
        # Preserve all nonadjacent finger/finger and hand/object collisions.
        thumb=stage.GetPrimAtPath(path+'/right_thumb_MC')
        if thumb:
            pairs=UsdPhysics.FilteredPairsAPI.Apply(thumb).CreateFilteredPairsRel()
            pairs.SetTargets(list(set(pairs.GetTargets()+[stage.GetPrimAtPath(self.palm_path).GetPath()])))
        if not colliders or len(joints) != len(model.movable):
            raise ValueError('Actual hand colliders and every movable joint are required')
        self.description={'asset':str(Path(asset).resolve()),'root':model.root,'colliders':colliders,
            'joint_limits_rad':{j.name:[j.lower,j.upper] for j in model.movable},
            'friction':friction,'finger_effort_limit_Nm':finger_effort_Nm,
            'friction_combine_mode':'multiply',
            'finger_stiffness_Nm_rad':20.*finger_effort_Nm,'finger_damping_Nms_rad':2.*finger_effort_Nm,
            'source_joint_effort_limits_Nm':source_efforts,'joint_effort_limits_Nm':joint_efforts,
            'joint_gain_policy':'Each joint uses min(requested fixture effort, finite source drive limit); stiffness=20*effort and damping=2*effort. Scalar gain fields are requested maxima.',
            'collision_rest_offset_m':0.,'collision_contact_offset_m':contact_offset_m,
            'hand_gravity_compensation':True,'cloth_gravity_disabled':False,
            'source_joint_friction':source_joint_friction,'joint_friction_compensation':bool(joint_friction_compensation),
            'source_joint_axis_properties':source_axis_properties,
            'joint_friction_prior':joint_friction_prior,
            'passive_hand_dynamics_calibrated':False,
            'hand_sleep_threshold':0.,'disabled_source_control_graphs':disabled_graphs,
            'wrist_force_limit_N':100.,'wrist_torque_limit_Nm':5.,
            'positions_written_after_reset':False,'cloth_actuation':False}

    def initialize(self):
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
        from isaacsim.core.api import SimulationContext
        self.art=SingleArticulation(self.path,name='acceptance_hand');self.art.initialize()
        self.palm=SingleRigidPrim(self.palm_path,name='acceptance_wrist');self.palm.initialize()
        self.names=list(self.art.dof_names)
        if set(self.names) != {j.name for j in self.model.movable}:
            raise ValueError('Native hand DOFs do not match source CAD')
        self.peak_force=self.peak_torque=0.
        self._peak_wrench = None
        self._target_rotation = None
        dt = SimulationContext.instance().get_physics_dt()
        self._control_stride = max(1, round(1 / (self.control_hz * dt)))
        self._control_tick = 0; self._held_wrench = None
        from calibrated_scene_reconstruction.core.geometry.rigid import as_numpy
        actual_efforts=as_numpy(self.art.get_articulation_controller().get_max_efforts()).ravel()
        expected_efforts=[self.description['joint_effort_limits_Nm'][name] for name in self.names]
        if actual_efforts.shape!=(len(self.names),) or not np.allclose(actual_efforts,expected_efforts,rtol=1e-5,atol=1e-8):
            raise ValueError('Native motor effort limits differ from the source-capped fixture')
        self.description['native_joint_effort_limits_Nm']=dict(zip(self.names,actual_efforts.tolist()))
        self.description['source_capped_effort_readback']=True
        self._palm_mass = float(as_numpy(self.palm._rigid_prim_view.get_masses())[0])
        inertia = as_numpy(self.palm._rigid_prim_view.get_inertias())[0].reshape(3,3)
        self._palm_inertia = float(np.linalg.eigvalsh(inertia).min())
        if min(self._palm_mass,self._palm_inertia) <= 0:
            raise ValueError('Native positive palm mass and principal inertia required')
        self._feedback_dt = dt * self._control_stride
        self.description.update(control_hz=1 / (dt * self._control_stride),
                                wrench_zero_order_hold=True, native_motor_drives_every_physics_step=True,
                                wrist_controller='Mass-aware backward-Euler finite PD',
                                native_palm_mass_kg=self._palm_mass, native_minimum_palm_inertia_kg_m2=self._palm_inertia)

    def control(self, position, rotation, state):
        from calibrated_scene_reconstruction.core.geometry.rigid import as_numpy,rot
        view=self.palm._rigid_prim_view
        tick=self._control_tick;self._control_tick+=1
        if self._held_wrench is not None and tick % self._control_stride:
            force, torque = self._held_wrench
            view.apply_forces_and_torques_at_pos(forces=force, torques=torque, is_global=True)
            return
        position,rotation=np.asarray(position,float),np.asarray(rotation,float)
        requested={j.name:float(state.get(j.name,0.)) for j in self.model.movable}
        if any(not np.isfinite(requested[j.name]) or not j.lower-1e-7<=requested[j.name]<=j.upper+1e-7
               for j in self.model.movable):raise ValueError('Finger drive target outside actual joint limits')
        art_view=self.art._articulation_view
        targets=art_view._backend_utils.convert(np.asarray([[requested[n] for n in self.names]],np.float32),device=art_view._device)
        art_view.set_joint_position_targets(targets)
        if view._backend == 'torch':
            import torch
            p, q = view.get_world_poses(clone=False)
            velocity = view.get_velocities(clone=False)
            if self._target_rotation is None or not np.array_equal(rotation, self._target_rotation):
                xyzw = Rotation.from_matrix(rotation).as_quat()
                self._target_quaternion = torch.as_tensor(xyzw[[3,0,1,2]][None], dtype=p.dtype, device=p.device)
                self._target_rotation = rotation.copy()
            target = torch.as_tensor(position[None], dtype=p.dtype, device=p.device)
            force, torque = tensor_wrist_wrench(p, q, velocity, target, self._target_quaternion,
                                               self._feedback_dt,self._palm_mass,self._palm_inertia)
            self._held_wrench = force, torque
            view.apply_forces_and_torques_at_pos(forces=force, torques=torque, is_global=True)
            peak = torch.stack([torch.linalg.vector_norm(force), torch.linalg.vector_norm(torque)])
            self._peak_wrench = peak if self._peak_wrench is None else torch.maximum(self._peak_wrench, peak)
            return
        p,q=[as_numpy(v) for v in self.palm.get_world_pose()]
        velocity,angular=as_numpy(self.palm.get_linear_velocity()),as_numpy(self.palm.get_angular_velocity())
        dt=self._feedback_dt
        force=(1000.*(position-p-dt*velocity)-80.*velocity)/(1+(80.*dt+1000.*dt**2)/self._palm_mass)
        torque=(20.*(Rotation.from_matrix(rotation@rot(q).T).as_rotvec()-dt*angular)-2.*angular)/(1+(2.*dt+20.*dt**2)/self._palm_inertia)
        force*=min(1.,100./max(np.linalg.norm(force),1e-12));torque*=min(1.,5./max(np.linalg.norm(torque),1e-12))
        view.apply_forces_and_torques_at_pos(
            forces=view._backend_utils.convert(np.asarray([force],np.float32),device=view._device),
            torques=view._backend_utils.convert(np.asarray([torque],np.float32),device=view._device),is_global=True)
        self.peak_force=max(self.peak_force,float(np.linalg.norm(force)));self.peak_torque=max(self.peak_torque,float(np.linalg.norm(torque)))

    def report(self):
        if self._peak_wrench is not None:
            self.peak_force, self.peak_torque = self._peak_wrench.detach().cpu().tolist()
        return {**self.description,'peak_wrist_force_N':self.peak_force,'peak_wrist_torque_Nm':self.peak_torque}
