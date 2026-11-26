"""Independent wet-wipe sheet held only by native finger contact.

No attachment, fixed joint, prescribed nodes, pose following or cloth forces.
The initial raised edge is a prepared grasp presentation, not a pickup policy.
"""
import numpy as np
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from calibrated_scene_reconstruction.assets.cloth_rest import triangulate_grid
from calibrated_scene_reconstruction.sim.surface_cloth import SurfaceView, author
from calibrated_scene_reconstruction.sim.usd import add_mesh, color_material


def preload_targets(plan, overlap=.001, target_gap=None):
    """Finite-drive overtravel target; actual contact, never a finger teleport."""
    if 'cloth2d_motor_targets' in plan:
        return dict(plan['cloth2d_motor_targets']),dict(effort_limited=True,
            finger_positions_prescribed_during_execution=False,
            **plan['cloth2d_motor_target_basis'])
    from calibrated_scene_reconstruction.robots.articulated_model import from_usd
    model = from_usd(plan['asset'], 10)
    closed = dict(plan['closed_state'])
    transforms = model.transforms(closed)
    a = transforms[plan['thumb']['link']]
    b = transforms[plan['index']['link']]
    pa = a[:3, :3] @ plan['thumb']['pad_local'] + a[:3, 3]
    pb = b[:3, :3] @ plan['index']['pad_local'] + b[:3, 3]
    axis = (pb - pa) / np.linalg.norm(pb - pa)

    def gap(state):
        tf = model.transforms(state)
        pts = [tf[plan[f]['link']][:3, :3] @ plan[f]['pad_local'] + tf[plan[f]['link']][:3, 3]
               for f in ['thumb', 'index']]
        return float((pts[1] - pts[0]) @ axis)

    names = list(closed)
    gradient = []
    for name in names:
        altered = {**closed, name: closed[name] + 1e-5}
        gradient.append((gap(altered) - gap(closed)) / 1e-5)
    gradient = np.asarray(gradient)
    desired_gap = -overlap if target_gap is None else float(target_gap)
    delta = (desired_gap - gap(closed)) * gradient / np.dot(gradient, gradient)
    limits = {j.name: (j.lower, j.upper) for j in model.movable}
    target = {name: float(np.clip(closed[name] + np.clip(d, -.12, .12), *limits[name]))
              for name, d in zip(names, delta)}
    return target, {'geometric_closed_gap_m': gap(closed), 'unobstructed_motor_target_gap_m': gap(target),
                    'effort_limited': True, 'finger_positions_prescribed_during_execution': False}


def configure_pinch(stage, effort=.4, friction=1.2, stiffness=40., damping=1.,contact_plan=None):
    """Finite finger motor/contact hypotheses, retaining CAD limits and inertia."""
    from calibrated_scene_reconstruction.sim.hand_fixture import axis_properties, preserve_axis_properties
    from calibrated_scene_reconstruction.sim.physics import physics_material
    material = physics_material(stage, '/World/Looks/WipeFingerPhysics', friction, friction, 0.,
                                friction_combine='multiply')
    original_friction = {}
    root = stage.GetPrimAtPath('/World/Robot')
    for prim in list(Usd.PrimRange(root)):
        if '/right_' in str(prim.GetPath()) and prim.IsInstance():
            prim.SetInstanceable(False)
    collision_count = 0
    bodies = [p for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    all_bodies = [p.GetPath() for p in bodies]
    for body in bodies:
        if not body.GetName().startswith('right_'):
            # Enable native finger/finger contact without introducing unrelated
            # whole-robot self-collision changes to the reconstruction prototype.
            UsdPhysics.FilteredPairsAPI.Apply(body).CreateFilteredPairsRel().SetTargets(
                [p for p in all_bodies if p != body.GetPath()])
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            articulation = PhysxSchema.PhysxArticulationAPI.Apply(prim)
            articulation.CreateEnabledSelfCollisionsAttr(True)
            articulation.CreateSolverPositionIterationCountAttr(32)
            articulation.CreateSolverVelocityIterationCountAttr(4)
    thumb_mc = stage.GetPrimAtPath('/World/Robot/right_thumb_MC')
    UsdPhysics.FilteredPairsAPI.Apply(thumb_mc).CreateFilteredPairsRel().SetTargets(
        [stage.GetPrimAtPath('/World/Robot/right_hand_C_MC').GetPath()])
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdPhysics.RevoluteJoint) and prim.GetName().startswith('right_'):
            old = axis_properties(prim)
            api = PhysxSchema.PhysxJointAPI.Apply(prim)
            original_friction[prim.GetName()] = api.GetJointFrictionAttr().Get()
            api.CreateJointFrictionAttr(0.)
            prim.ApplyAPI('PhysxJointAxisAPI', 'angular')
            preserve_axis_properties(prim, old)
            for name, value in [('staticFrictionEffort', .005), ('dynamicFrictionEffort', .003),
                                ('viscousFrictionCoefficient', .0002 * np.pi / 180)]:
                prim.CreateAttribute('physxJointAxis:angular:' + name, Sdf.ValueTypeNames.Float).Set(float(value))
            drive = UsdPhysics.DriveAPI.Apply(prim, 'angular')
            drive.CreateTypeAttr('force')
            drive.CreateStiffnessAttr(stiffness * np.pi / 180)
            drive.CreateDampingAttr(damping * np.pi / 180)
            drive.CreateMaxForceAttr(effort)
        if prim.HasAPI(UsdPhysics.CollisionAPI) and '/right_' in str(prim.GetPath()):
            collision_count += 1
            api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            api.CreateRestOffsetAttr(0.)
            api.CreateContactOffsetAttr(.0003)
            if any('/' + n + '/' in str(prim.GetPath()) for n in ['right_thumb_elastomer', 'right_index_elastomer']):
                # The default GPU hull is capped at 64 vertices. Use the real
                # pad surface's SDF for a submillimetre cloth pinch.
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('sdf')
                PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim).CreateSdfResolutionAttr(256)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants, 'physics')
    if collision_count < 10:
        raise RuntimeError('Native Sharpa finger collision shapes missing')
    if contact_plan is not None:
        from calibrated_scene_reconstruction.sim.usd import set_pose,quat_from_matrix
        for finger in ['thumb','index']:
            info=contact_plan[finger];body=stage.GetPrimAtPath('/World/Robot/'+info['link'])
            for child in Usd.PrimRange(body):
                if child.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr().Set(False)
            normal=np.asarray(info['normal_local'],float);normal/=np.linalg.norm(normal)
            tangent=np.array([1.,0,0]);tangent-=normal*(tangent@normal);tangent/=np.linalg.norm(tangent)
            R=np.column_stack([tangent,np.cross(normal,tangent),normal])
            pad=UsdGeom.Cube.Define(stage,str(body.GetPath())+'/WipeContactPad')
            pad.CreateSizeAttr(1.)
            set_pose(pad.GetPrim(),np.asarray(info['pad_local'])-.001*normal,quat_from_matrix(R))
            UsdGeom.Xformable(pad).AddScaleOp().Set(Gf.Vec3f(.012,.012,.002))
            pad.MakeInvisible()
            UsdPhysics.CollisionAPI.Apply(pad.GetPrim()).CreateCollisionEnabledAttr(True)
            api=PhysxSchema.PhysxCollisionAPI.Apply(pad.GetPrim())
            api.CreateRestOffsetAttr(0.);api.CreateContactOffsetAttr(.0003)
            UsdShade.MaterialBindingAPI.Apply(pad.GetPrim()).Bind(material,UsdShade.Tokens.strongerThanDescendants,'physics')
    return {'effort_Nm': effort, 'stiffness_Nm_rad': stiffness, 'damping_Nms_rad': damping,
            'solver_position_iterations': 32, 'solver_velocity_iterations': 4,
            'finger_friction': friction, 'combine': 'multiply', 'collision_shapes': collision_count,
            'finger_contact_model':'12mm tangent-plane rigid contact proxies at CAD pad points' if contact_plan is not None else 'Source curved pad SDF',
            'legacy_joint_friction': original_friction, 'joint_friction_assumption_Nm': [.005, .003],
            'hardware_dynamics_calibrated': False}


class WetWipe:
    @staticmethod
    def grip_height(backend='particle'):
        return (.0012 if backend == 'particle' else .0006) + .008 + .020 * 2 / np.pi - .004

    def __init__(self, stage, grip, table_z, yaw=0., width=.060, length=.090, backend='particle'):
        self.stage = stage
        self.path = '/World/WetWipe'
        self.grip_initial = np.asarray(grip, float)
        self.backend = backend
        u = np.linspace(-width / 2, width / 2, 17)
        v = np.unique(np.r_[np.linspace(0, .008, 5), np.linspace(.008, .028, 11), np.linspace(.028, length, 22)])
        U, V = np.meshgrid(u, v)
        rest = np.c_[U.ravel(), V.ravel(), np.zeros(U.size)]
        s = V.ravel()
        radius = .020 * 2 / np.pi
        angle = -np.pi / 2 + np.clip(s - .008, 0, .020) / radius
        x = radius * (np.sin(angle) + 1) + np.maximum(s - .028, 0)
        z_top = table_z + (.0012 if backend == 'particle' else .0006) + .008 + radius
        z = z_top - np.minimum(s, .008) - radius * np.cos(angle)
        local = np.c_[x, U.ravel(), z - (table_z + self.grip_height(backend))]
        c, sn = np.cos(yaw), np.sin(yaw)
        self.rotation = np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1.]])
        points = local @ self.rotation.T + grip
        faces, material, meshing = triangulate_grid(points, rest, len(v), len(u))
        self.part = {'name': 'wet_wipe', 'kind': 'cloth', 'backend': backend,
                     'V': points, 'F': faces, 'cloth_rest': material}
        self.faces = faces
        self.grasp_nodes = np.flatnonzero((np.abs(rest[:, 0]) < .006) & (rest[:, 1] >= .002) & (rest[:, 1] <= .006))
        self.mesh = add_mesh(stage, self.path, points, faces, rest[:, :2] / [width, length],
                              color_material(stage, '/World/Looks/WetWipe', [.93, .94, .90], roughness=.95))
        self.settings = {'youngs_modulus': 3e5, 'poissons_ratio': .2, 'thickness': .0005,
                         'flexural_rigidity': 1e-6, 'elasticity_damping': .05, 'linear_damping': .05,
                         'friction': .55, 'areal_density': .3, 'contact_offset': .0005,
                         'rest_offset': .00025, 'self_collision_filter_distance': .0005,
                         'solver_position_iterations': 128, 'collision_pair_updates': 16,
                         'collision_iteration_multiplier': 2, 'max_depenetration_velocity': .15}
        if backend == 'surface_fem':
            author(stage, self.mesh, self.part, self.settings)
        elif backend == 'particle':
            from calibrated_scene_reconstruction.sim.mechanism_stage import add_deformable
            self.settings = {'areal_density': .3, 'contact_offset': .0013, 'rest_offset': .0008,
                             'solid_rest_offset': .0008, 'stretch_stiffness': 12000., 'shear_stiffness': 4000.,
                             'bend_stiffness': .008, 'damping': .05, 'friction': .7,
                             'solver_position_iterations': 128, 'max_depenetration_velocity': .15}
            add_deformable(stage, self.mesh, self.part, self.settings, None)
        else:
            raise ValueError('Unknown native cloth backend')
        self.description = {'backend': 'native ' + backend, 'nodes': len(points), 'triangles': len(faces),
                            'width_m': width, 'length_m': length, 'material': self.settings, 'meshing': meshing,
                            'attachments': False, 'kinematic_nodes': 0, 'cloth_pose_following': False,
                            'cloth_forces': False, 'prepared_initial_edge': True,
                            'positions_written_after_initialization': False}

    def initialize(self):
        if self.backend == 'surface_fem':
            self.view = SurfaceView(self.stage, self.path, self.part)
        else:
            import omni.physics.tensors as tensors
            self.owner = tensors.create_simulation_view('torch')
            self.owner.set_subspace_roots('/')
            self.view = self.owner.create_particle_cloth_view(self.path)
            masses = self.view.get_masses()[0].cpu().numpy()
            if (masses <= 0).any():
                raise RuntimeError('Every cloth node must have positive finite mass')

    def points(self):
        values = (self.view.get_world_positions()[0] if self.backend == 'surface_fem' else
                  self.view.get_positions()[0].reshape(-1, 3))
        points = values.detach().cpu().numpy().copy()
        if not np.isfinite(points).all():
            raise RuntimeError('Non-finite native cloth positions')
        return points

    def contact_samples(self, table_z, height=.004):
        points = self.points()
        tri = points[self.faces]
        near = tri[:, :, 2].min(1) < table_z + height
        tri = tri[near]
        if not len(tri):
            return np.empty((0, 3))
        # Source triangles, not an assumed hand/TCP rectangle. 2 mm material
        # spacing and barycentric interior/edge witnesses bound sampling gaps.
        sample = np.concatenate([tri[:, i] for i in range(3)] + [tri.mean(1)] +
                                [(tri[:, i] + tri[:, (i + 1) % 3]) / 2 for i in range(3)])
        return sample[sample[:, 2] < table_z + height]

    def grip_center(self):
        return self.points()[self.grasp_nodes].mean(0)
