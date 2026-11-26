"""Tools for a task check (apps/scene/task_check.py runs configs/<name>/agent/task_check.py, the agent's own, written for the
session's task): measurements only -- what counts as the task done is the agent's script's to say.

A check script defines `check(ctx)` and returns {"pass": bool, "what": str, ...}. `ctx` (Ctx below) holds the built
twin (stage, world, the scene's objects) and these tools:

    ctx.body(name)                      the object's rigid body (pose, velocities)
    ctx.pose(name) -> (p, R)            its world position and rotation matrix
    ctx.points(name) -> (N, 3)          its surface points in the world, as it is now
    ctx.place(name, p, R)               set it (velocities zeroed)
    ctx.kinematic_fixture(name)        temporary hold; release restores original CCD/dynamic state
    ctx.step(seconds)                   PhysX for that long
    ctx.opening(name, link=None) -> {xy, rim_z, depth, radius} | None   opening in its physical owner's current frame
    ctx.hold_over(target, xy, z, R)     the target set with its lowest point at height z over xy, rotation R
    ctx.tilt(name, R0) -> deg           how far it turned from rotation R0
    ctx.apply_nodal_forces(name, indices, forces, force_limit_N=...)  bounded PBD cloth forces, not poses
    ctx.image(look, eye, path)          a render of the twin, saved under qa/
"""
import numpy as np








class Ctx:
    def kinematic_fixture(self, name):
        from calibrated_scene_reconstruction.sim.rigid_fixture import KinematicFixture
        return KinematicFixture(self.body(name).prim)

    @staticmethod
    def native_particle_profile(view, parameters):
        from calibrated_scene_reconstruction.sim.particle_contact_binding import read
        return read(view, parameters)

    def native_collision_profile(self, prim, *, particle=False):
        from pxr import PhysxSchema
        if particle:
            system = PhysxSchema.PhysxParticleAPI(prim).GetParticleSystemRel().GetTargets()[0]
            rest = PhysxSchema.PhysxParticleSystem(self.stage.GetPrimAtPath(system)).GetRestOffsetAttr().Get()
        else:
            rest = PhysxSchema.PhysxCollisionAPI(prim).GetRestOffsetAttr().Get()
        return {'rest_offset': rest,
                'sdf_resolution': PhysxSchema.PhysxSDFMeshCollisionAPI(prim).GetSdfResolutionAttr().Get()}

    def __init__(self, s, ts, world, stage, render):
        from isaacsim.core.prims import SingleRigidPrim
        from pxr import UsdPhysics

        from calibrated_scene_reconstruction.assets import kits

        self.s, self.ts, self.world, self.stage, self._render = s, ts, world, stage, render
        self.sg = s.scene_graph
        self.task = s.config("task.json", {}) or {}
        self.kits = kits
        self._bodies = {}
        self._partitions = {}
        self._contact_carrier_checks = {}
        self._joint_sources = {}
        self._links, self._soft, self._cloth = {}, {}, {}
        for n, p in ts.objects.items():
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                b = SingleRigidPrim(str(p.GetPath()),reset_xform_properties=False)
                b.initialize()
                self._bodies[n] = b
        self._local = {}
        for n in ts.objects:
            fx = (self.sg["dynamic_objects"].get(n) or {}).get("fix") or {}
            if fx.get("model"):
                from calibrated_scene_reconstruction.assets.metric_mesh import asset
                k = asset(fx)
                self._joint_sources[n] = k.get('joints',[])
                self._local[n] = np.concatenate([np.asarray(q["V"], float) for q in k["parts"]])
                if k.get('links'):
                    self._links[n] = []
                    for part in k['parts']:
                        key = f"{n}/{part['name']}"
                        self._links[n].append(key)
                        self._local[key] = np.asarray(part['V'], float)
                        path = f"{ts.objects[n].GetPath()}/{part['name']}"
                        if part['kind'] == 'rigid':
                            body = SingleRigidPrim(path, name=key.replace('/', '_'),reset_xform_properties=False)
                            body.initialize()
                            self._bodies[key] = body
                            from calibrated_scene_reconstruction.sim.rigid_partition import verify
                            partition=verify(stage,path)
                            if partition:
                                from isaacsim.core.prims import RigidPrim
                                frames=RigidPrim(partition['carriers'],reset_xform_properties=False,
                                                 name=key.replace('/','_')+'_bond_frames')
                                frames.initialize()
                                if list(frames.prim_paths)!=partition['carriers']:raise ValueError('Numerical rigid frame order changed')
                                self._partitions[key]={'view':frames,'metadata':partition}
                            from calibrated_scene_reconstruction.sim.contact_carrier import verify as verify_contact_carrier
                            carrier=verify_contact_carrier(stage,path)
                            if carrier:
                                import json
                                from isaacsim.core.prims import RigidPrim
                                from calibrated_scene_reconstruction.sim.source_moments import KEY as moments_key
                                record=json.loads(stage.GetPrimAtPath(path).GetCustomDataByKey(moments_key))
                                frames=RigidPrim([carrier['carrier']],reset_xform_properties=False,
                                                 name=key.replace('/','_')+'_contact_frame');frames.initialize()
                                self._partitions[key]={'view':frames,'metadata':{
                                    'source_mass_kg':carrier['source_mass_kg'],
                                    'source_centre':record['centre'],'source_inertia_per_kg':record['inertia_per_kg'],
                                    'component_centres':[record['centre'],record['centre']]}}
                                self._contact_carrier_checks[key]={'last_time_s':None,'samples':0,
                                    'maximum_translation_error_m':0.,'maximum_rotation_error':0.}
                                self.rigid_mass_properties(key)
                        elif part['kind'] in {'deformable', 'rope'}:
                            if part.get('backend','legacy')=='volume_fem':
                                from calibrated_scene_reconstruction.sim.volume_body import VolumeView
                                view=VolumeView(stage,path+'/surface',part)
                            else:
                                from isaacsim.core.prims import DeformablePrim
                                view = DeformablePrim(path+'/surface', name=key.replace('/', '_'))
                                view.initialize()
                            self._soft[key] = view
                        elif part['kind'] == 'cloth':
                            if part.get('backend','particle')=='surface_fem':
                                from calibrated_scene_reconstruction.sim.surface_cloth import SurfaceView
                                view=SurfaceView(stage,path+'/surface',part)
                            elif part.get('cloth_rest') is not None:
                                from calibrated_scene_reconstruction.sim.particle_surface import ParticleView
                                view=ParticleView(stage,path+'/surface',part)
                            else:
                                # Legacy auto-cloth has no explicit source
                                # material chart. Preserve its existing path.
                                from isaacsim.core.prims import ClothPrim
                                view=ClothPrim(path+'/surface',name=key.replace('/','_'));view.initialize()
                            self._cloth[key] = view
                    if len(self._links[n]) == 1 and self._links[n][0] in self._bodies:
                        self._bodies[n] = self._bodies[self._links[n][0]]
            elif fx.get("kit"):
                k = kits.build(fx["kit"], fx.get("params", {}))
                self._local[n] = np.concatenate([np.asarray(q["V"], float) for q in k["parts"]])

    def initial_articulation_state(self):
        """Call immediately after native reset, before any fixture or settling."""
        from calibrated_scene_reconstruction.assets.articulation_initial_state import measure
        objects={}
        for obj,joints in self._joint_sources.items():
            names={j[key] for j in joints for key in ['parent','child']}
            objects[obj]=measure(joints,{name:self.pose(obj+'/'+name) for name in names})
        return {'pass':all(r['pass'] for r in objects.values()),'objects':objects,
                'time_s':float(self.world.current_time)}

    def body(self, name):
        if name.endswith('/model') and name[:-6] in self._bodies:
            name = name[:-6]
        if name not in self._bodies:
            raise ValueError(f'{name}: choose a rigid link as object/link; flexible bodies use nodal state')
        return self._bodies[name]

    def pose(self, name):
        p, q = (as_numpy(v) for v in self.body(name).get_world_pose())
        R=rot(q)
        state=getattr(self,'_contact_carrier_checks',{}).get(name)
        if state is not None:
            now=float(self.world.current_time)
            if state['last_time_s'] is None or now-state['last_time_s']>=.025-1e-9 or now<state['last_time_s']:
                positions,orientations=map(as_numpy,self._partitions[name]['view'].get_world_poses())
                position_error=float(np.linalg.norm(positions[0]-p));rotation_error=float(np.linalg.norm(rot(orientations[0])-R))
                state.update(last_time_s=now,samples=state['samples']+1,
                    maximum_translation_error_m=max(state['maximum_translation_error_m'],position_error),
                    maximum_rotation_error=max(state['maximum_rotation_error'],rotation_error))
                if position_error>1e-5 or rotation_error>1e-4:
                    raise ValueError('Native fixed collision carrier diverged from its physical source owner: '+name)
        return p,R

    def points(self, name):
        if name.endswith('/model') and name not in self._local and name[:-6] in self._local:
            name = name[:-6]
        if name in self._links:
            return np.concatenate([self.points(key) for key in self._links[name]])
        if name in self._soft or name in self._cloth:
            return self.nodes(name)
        p, R = self.pose(name)
        return self._local[name] @ R.T + p

    def nodes(self, name):
        """Actual simulator FEM nodes, not the rest mesh or animated appearance."""
        if name not in self._soft and name not in self._cloth:
            raise ValueError(f'{name}: no native deformable solver view')
        value = (self._soft[name].get_simulation_mesh_nodal_positions() if name in self._soft else
                 self._cloth[name].get_world_positions())
        points = as_numpy(value)[0]
        if not len(points) or not np.isfinite(points).all():
            raise ValueError('Native deformable state unavailable or nonfinite')
        return points

    def apply_nodal_forces(self, name, indices, forces, *, force_limit_N):
        """Apply one finite external impulse F*dt to selected native cloth nodes.

        Isaac 5.1 exposes particle velocities rather than arbitrary per-particle
        forces. Preserve current velocities, use native masses or the validated
        native surface-mass model (explicitly reported below), and
        leave positions, constraints and all other nodes to the native solver.
        This is not a kinematic grasp or a prescribed cloth trajectory.
        """
        if name not in self._cloth:
            raise ValueError('Nodal-force adapter supports native particle/surface cloth only; no rigid substitute')
        ids, force = np.asarray(indices), np.asarray(forces, float)
        if (ids.ndim != 1 or not len(ids) or ids.dtype.kind not in 'iu' or len(set(ids.tolist())) != len(ids)
                or force.shape != (len(ids), 3) or not np.isfinite(force).all()):
            raise ValueError('Finite per-node forces and unique integer cloth indices required')
        if (isinstance(force_limit_N, bool) or not np.isfinite(force_limit_N) or force_limit_N <= 0
                or np.any(np.linalg.norm(force, axis=1) > force_limit_N*(1+1e-12))):
            raise ValueError('Nodal forces exceed their explicit finite force limit')
        view = self._cloth[name]
        if hasattr(view,'require_validated_force_profile'):
            view.require_validated_force_profile()
        original = view.get_velocities()
        masses = view.get_particle_masses()
        dt = float(self.world.get_physics_dt())
        if not np.isfinite(dt) or dt<=0:raise ValueError('Finite positive physics timestep required')
        if hasattr(original,'detach'):
            import torch
            velocities=original.clone()
            masses=torch.as_tensor(masses,device=original.device)
            if (velocities.ndim!=3 or velocities.shape[0]!=1 or velocities.shape[2]!=3
                    or masses.shape!=velocities.shape[:2] or ids.min()<0 or ids.max()>=velocities.shape[1]):
                raise ValueError('Native particle velocity/mass state is unavailable or indices are out of bounds')
            selected=torch.as_tensor(ids,dtype=torch.long,device=original.device)
            mass=masses[0,selected]
            # Preserve the former float64 impulse arithmetic and one final
            # cast, without transferring the full sheet to CPU and back.
            impulse=torch.as_tensor(force,dtype=torch.float64,device=original.device)*dt/mass.double()[:,None]
            velocities[0,selected]=(velocities[0,selected].double()+impulse).to(original.dtype)
            # Collect every original guard and the receipt in one host read.
            # Repeated bool(tensor)/float(tensor) calls otherwise synchronize
            # the shared GPU separately at each native physics step. No state
            # is written until all checks, including pinned masses, pass.
            checks=torch.stack([torch.isfinite(original).all(),torch.isfinite(mass).all(),
                                (mass>0).all(),torch.isfinite(velocities).all()]).double()
            receipt=torch.cat([checks,mass.double().sum().reshape(1)]).detach().cpu().numpy()
            if not receipt[0]:raise ValueError('Native particle velocity/mass state is unavailable or indices are out of bounds')
            if not (receipt[1] and receipt[2]):
                raise ValueError('Finite positive node masses required; pinned nodes cannot be forced')
            if not receipt[3]:raise ValueError('Nonfinite nodal impulse refused')
            view.set_velocities(velocities)
            selected_mass=float(receipt[4])
            return {'method':'bounded external impulse F*dt on native nodal state','prescribed_motion':False,
                    'mass_source':getattr(view,'mass_source','Native particle mass readback'),
                    'dt_s':dt,'force_limit_per_node_N':float(force_limit_N),'total_force_N':force.sum(0).tolist(),
                    'node_count':len(ids),'selected_mass_kg':selected_mass}
        velocities = as_numpy(original).copy()
        masses = as_numpy(masses)
        if (velocities.ndim != 3 or velocities.shape[0] != 1 or velocities.shape[2] != 3
                or masses.shape != velocities.shape[:2] or ids.min() < 0 or ids.max() >= velocities.shape[1]
                or not np.isfinite(velocities).all()):
            raise ValueError('Native particle velocity/mass state is unavailable or indices are out of bounds')
        mass = masses[0, ids]
        if not np.isfinite(mass).all() or (mass <= 0).any() or not np.isfinite(dt) or dt <= 0:
            raise ValueError('Finite positive node masses and physics timestep required; pinned nodes cannot be forced')
        velocities[0, ids] += force*dt/mass[:, None]
        if not np.isfinite(velocities).all():
            raise ValueError('Nonfinite nodal impulse refused')
        if hasattr(original, 'detach'):
            import torch
            value = torch.as_tensor(velocities, dtype=original.dtype, device=original.device)
        else:
            value = velocities
        view.set_velocities(value)
        return {'method': 'bounded external impulse F*dt on native nodal state', 'prescribed_motion': False,
                'mass_source':getattr(view,'mass_source','Native particle mass readback'),
                'dt_s': dt, 'force_limit_per_node_N': float(force_limit_N),
                'total_force_N': force.sum(0).tolist(), 'node_count': len(ids),
                'selected_mass_kg': float(mass.sum())}

    def node_patch_state(self,name,indices):
        """Read an actual finite material patch without copying a whole sheet."""
        ids=np.asarray(indices)
        if (name not in self._cloth or ids.ndim!=1 or not len(ids) or ids.dtype.kind not in 'iu'
                or ids.min()<0 or len(np.unique(ids))!=len(ids)):
            raise ValueError('Native cloth and distinct nonnegative patch indices required')
        view=self._cloth[name];points=view.get_world_positions();velocity=view.get_velocities()
        if points.shape!=velocity.shape or len(points.shape)!=3 or points.shape[0]!=1 or points.shape[2]!=3 or ids.max()>=points.shape[1]:
            raise ValueError('Actual corresponding native patch positions and velocities required')
        if hasattr(points,'detach'):
            import torch
            index=torch.as_tensor(ids,dtype=torch.long,device=points.device)
        else:index=ids
        p,v=as_numpy(points[0,index]),as_numpy(velocity[0,index])
        if not np.isfinite(p).all() or not np.isfinite(v).all():raise ValueError('Nonfinite native patch state')
        return p,v

    def apply_joint_effort(self, assembly, parent, child, effort, *, effort_limit):
        """Bounded native articulation effort, including the direct GPU pipeline.

        Changing USD drive targets while this GPU pipeline is running is not a
        supported actuation API. Positive effort follows the model joint axis;
        units are Nm for a revolute joint and N for a prismatic joint.
        """
        from isaacsim.core.prims import SingleArticulation
        from pxr import UsdPhysics
        from calibrated_scene_reconstruction.assets.mechanisms import joint_components
        from calibrated_scene_reconstruction.assets.metric_mesh import asset
        if (isinstance(effort, bool) or isinstance(effort_limit, bool)
                or not np.isfinite([effort, effort_limit]).all() or effort_limit <= 0
                or abs(effort) > effort_limit*(1+1e-12)):
            raise ValueError('Joint effort must lie within an explicit finite positive limit')
        if not hasattr(self, '_joint_handles'):
            self._joint_handles, self._joint_views = {}, {}
        key = (assembly, parent, child)
        if key not in self._joint_handles:
            kit = asset(self.sg['dynamic_objects'][assembly]['fix'])
            matches = [(i, j) for i, j in enumerate(kit.get('joints', []))
                       if j['parent'] == parent and j['child'] == child and j['type'] != 'fixed']
            if len(matches) != 1:
                raise ValueError('Effort needs exactly one declared moving parent/child joint')
            index, joint = matches[0]
            root = self.ts.objects[assembly]
            if root.HasAPI(UsdPhysics.ArticulationRootAPI):
                path = str(root.GetPath())
            else:
                components = joint_components({p['name']: {'kind': p['kind']} for p in kit['parts']}, kit['joints'])
                component = next(c for c in components if parent in c['links'] and child in c['links'])
                path = str(root.GetPath())+'/'+component['root']
            if path not in self._joint_views:
                view = SingleArticulation(path)
                view.initialize()
                self._joint_views[path] = (view, np.zeros(view.num_dof, dtype=np.float32))
            view, commands = self._joint_views[path]
            name = f'joint_{index}'
            if name not in view.dof_names:
                raise ValueError('Native articulation has no matching authored joint: '+name)
            self._joint_handles[key] = (path, view.dof_names.index(name), joint['type'])
        path, index, kind = self._joint_handles[key]
        view, commands = self._joint_views[path]
        commands[index] = effort
        values = commands.copy()
        if self.world.backend == 'torch':
            import torch
            values = torch.as_tensor(values, dtype=torch.float32, device=self.world.device)
        view.set_joint_efforts(values)
        readback=float(as_numpy(view.get_applied_joint_efforts())[index])
        if not np.isfinite(readback) or not np.isclose(readback,effort,rtol=1e-6,atol=1e-10):
            raise RuntimeError('Native joint actuation command did not match its readback')
        return {'effort': float(effort), 'effort_limit': float(effort_limit),
                'native_command_readback':readback,
                'unit': 'N*m' if kind == 'revolute' else 'N', 'method': 'native articulation tensor actuation',
                'positions_prescribed': False}

    def clear_joint_efforts(self):
        for view, commands in getattr(self, '_joint_views', {}).values():
            commands.fill(0.)
            values = commands.copy()
            if self.world.backend == 'torch':
                import torch
                values = torch.as_tensor(values, dtype=torch.float32, device=self.world.device)
            view.set_joint_efforts(values)

    def place(self, name, p, R):
        if getattr(self,'_assay_active',False):
            raise ValueError('Body placement is assay setup only; active trials require native forces')
        b = self.body(name)
        def array(value):
            if self.world.backend == 'torch':
                import torch
                return torch.as_tensor(value, dtype=torch.float32, device=self.world.device)
            return np.asarray(value, float)
        b.set_world_pose(array(p), array(quat(np.asarray(R, float))))
        # The separate velocity setters are unsupported in the GPU pipeline.
        # Use the native combined six-component state for an actual reset.
        b._rigid_prim_view.set_velocities(array(np.zeros((1,6))))

    def capture_assay_state(self):
        """Exact independent-trial initial state, including native soft nodes."""
        if getattr(self,'_assay_active',False):raise ValueError('Cannot replace trial setup during native motion')
        return {'rigid':{name:tuple(value.copy() for value in self.pose(name)) for name in self._bodies},
                'partitions':{name:tuple(as_numpy(v).copy() for v in data['view'].get_world_poses())
                              for name,data in self._partitions.items()},
                'surface':{name:{'positions':as_numpy(view.get_world_positions()).copy(),
                                 'velocities':as_numpy(view.get_velocities()).copy()}
                           for name,view in self._cloth.items()}}

    def restore_assay_state(self,state):
        """Restore a captured setup, never follow a target during a trial."""
        if getattr(self,'_assay_active',False):raise ValueError('Native assay states cannot be prescribed during motion')
        for name,(p,R) in state['rigid'].items():self.place(name,p,R)
        for name,(positions,orientations) in state.get('partitions',{}).items():
            self._place_partition(name,positions,orientations)
        for name,values in state['surface'].items():
            view=self._cloth[name]
            if not hasattr(view,'restore_setup_state'):raise ValueError('This flexible backend lacks qualified independent-trial setup restoration')
            view.restore_setup_state(values['positions'],values['velocities'])

    def place_assay_assembly(self,obj,p,R,state,*,anchor='model'):
        """One rigid setup transform for every real rigid and flexible part."""
        if getattr(self,'_assay_active',False):raise ValueError('Assembly placement is setup only')
        p=np.asarray(p,float);R=np.asarray(R,float)
        if p.shape!=(3,) or R.shape!=(3,3) or not np.isfinite(p).all() or not np.allclose(R.T@R,np.eye(3),atol=1e-7) or not np.isclose(np.linalg.det(R),1):
            raise ValueError('Finite proper rigid setup transform required')
        p0,R0=state['rigid'][obj+'/'+anchor];delta=R@R0.T
        for name,(position,rotation) in state['rigid'].items():
            if name.startswith(obj+'/'):self.place(name,p+delta@(position-p0),delta@rotation)
        for name,(positions,orientations) in state.get('partitions',{}).items():
            if name.startswith(obj+'/'):
                self._place_partition(name,(positions-p0)@delta.T+p,
                                      np.array([quat(delta@rot(q)) for q in orientations]))
        for name,values in state['surface'].items():
            if name.startswith(obj+'/'):
                points=(values['positions']-p0)@delta.T+p
                self._cloth[name].restore_setup_state(points,values['velocities']@delta.T)

    def _place_partition(self,name,positions,orientations):
        if getattr(self,'_assay_active',False):raise ValueError('Numerical rigid placement is setup only')
        view=self._partitions[name]['view']
        if self.world.backend=='torch':
            import torch
            positions=torch.as_tensor(positions,dtype=torch.float32,device=self.world.device)
            orientations=torch.as_tensor(orientations,dtype=torch.float32,device=self.world.device)
            zeros=torch.zeros((len(positions),6),device=self.world.device)
        else:zeros=np.zeros((len(positions),6),dtype=np.float32)
        view.set_world_poses(positions,orientations);view.set_velocities(zeros)

    def rigid_mass_properties(self,name):
        """Actual aggregate native mass/inertia, including equivalent weld frames."""
        native=self.body(name)._rigid_prim_view
        mass=float(as_numpy(native.get_masses()).sum())
        inertia=as_numpy(native.get_inertias()).reshape(-1,3,3).sum(0)
        if name in self._partitions:
            from calibrated_scene_reconstruction.assets.rigid_partition import parallel_axis
            info=self._partitions[name]['metadata'];view=self._partitions[name]['view']
            masses=np.r_[mass,as_numpy(view.get_masses()).reshape(-1)]
            mass+=float(as_numpy(view.get_masses()).sum())
            inertia+=as_numpy(view.get_inertias()).reshape(-1,3,3).sum(0)
            inertia+=sum(kg*parallel_axis(np.array(c)-np.array(info['source_centre']))
                         for kg,c in zip(masses,info['component_centres']))
            expected=np.array(info['source_inertia_per_kg'])*mass
            relative_error=float(np.linalg.norm(inertia-expected,ord=2)/np.linalg.norm(expected,ord=2))
            if not np.isclose(mass,info['source_mass_kg'],rtol=1e-4) or relative_error>1e-4:
                raise ValueError('Native numerical partition changed the physical source mass or full inertia: '
                                 f'mass={mass:g}, expected_mass={info["source_mass_kg"]:g}, '
                                 f'relative_tensor_error={relative_error:g}')
        return mass,inertia

    def step(self, seconds):
        for _ in range(int(round(seconds / self.world.get_physics_dt()))):
            self.world.step(render=False)

    def opening(self, name, *, link=None):
        fx = self.sg["dynamic_objects"][name].get("fix") or {}
        op = fx.get("opening") if fx.get("model") else (
            self.kits.opening(fx.get("kit"), fx.get("params", {})) if fx.get("kit") else None)
        if op is None:
            return None
        p, R = self.pose(name if link is None else name+'/'+link)
        c = p + R @ np.array([op["xy"][0], op["xy"][1], op["rim_z"]])
        return {**op, "xy": c[:2].tolist(), "rim_z": float(c[2])}

    def hold_over(self, target, xy, z, R):
        V = self._local[target] @ np.asarray(R, float).T
        lo, hi = V.min(0), V.max(0)
        self.place(target, [xy[0] - (lo[0] + hi[0]) / 2, xy[1] - (lo[1] + hi[1]) / 2, z - lo[2]], R)

    def tilt(self, name, R0):
        _, R = self.pose(name)
        up = R @ (np.asarray(R0).T @ np.array([0.0, 0.0, 1.0]))
        return float(np.degrees(np.arccos(np.clip(up[2], -1, 1))))

    def image(self, look, eye, path="qa/task_check.jpg"):
        return self._render(np.asarray(look, float), np.asarray(eye, float), self.s / path)


from calibrated_scene_reconstruction.core.geometry.rigid import as_numpy, rot, quat  # compatibility exports
