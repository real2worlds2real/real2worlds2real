"""Compliant wet-wipe strip, held exclusively by native friction contact.

Short dynamic shell segments approximate bending with passive material springs.
A continuous render skin is reconstructed from their measured poses. It never
sets physics positions or constrains anything to the hand. This is a discrete
strip model, not continuum FEM cloth or a model of fine cross-width wrinkles.
"""
import numpy as np
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade, Vt
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation

from calibrated_scene_reconstruction.sim.hinged_wipe import HingedWipe
from calibrated_scene_reconstruction.sim.physics import physics_material
from calibrated_scene_reconstruction.sim.usd import color_material, quat_from_matrix, set_pose
from calibrated_scene_reconstruction.assets.wipe_geometry import soft_strip_rest


class SoftWipe(HingedWipe):
    def __init__(self, stage, grip, table_z, yaw=0., grip_location='center', **_):
        import trimesh
        self.stage, self.path, self.table_z = stage, '/World/WetWipe', table_z
        self.grip_location = grip_location
        stage.DefinePrim(self.path, 'Xform')
        grip = np.asarray(grip, float)
        yaw_R = Rotation.from_euler('z', yaw).as_matrix()
        rest_shape = soft_strip_rest(grip_location, self.grip_height())
        self.lengths, self.thickness = rest_shape['lengths'], rest_shape['thickness']
        self.grip_panel = rest_shape['grip_panel']
        total_length = float(self.lengths.sum())
        self.sizes = [np.array([.05, t, length]) for t, length in zip(self.thickness, self.lengths)]
        boundary = rest_shape['boundary']
        knots = rest_shape['knots'] @ yaw_R.T + grip
        rotations = [yaw_R @ R for R in rest_shape['rotations']]
        centers = (knots[:-1] + knots[1:]) / 2
        self.grip_local = rotations[self.grip_panel].T @ (grip - centers[self.grip_panel])
        self.meshes = [trimesh.creation.box(extents=s) for s in self.sizes]
        material = physics_material(stage, '/World/Looks/SoftWipePhysics', 1., .8, 0.)
        visual = color_material(stage, '/World/Looks/SoftWetWipe', [.92, .935, .905], roughness=.98)
        vertices, faces = [], []
        for i, (size, p, R, mesh) in enumerate(zip(self.sizes, centers, rotations, self.meshes)):
            actor = UsdGeom.Xform.Define(stage, self.path + '/Panel' + str(i))
            set_pose(actor.GetPrim(), p, quat_from_matrix(R))
            cube = UsdGeom.Cube.Define(stage, str(actor.GetPath()) + '/Geometry')
            cube.CreateSizeAttr(1.)
            set_pose(cube.GetPrim(), [0, 0, 0], scale=size)
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            collision = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
            collision.CreateContactOffsetAttr(.0002)
            collision.CreateRestOffsetAttr(0.)
            UsdPhysics.RigidBodyAPI.Apply(actor.GetPrim()).CreateKinematicEnabledAttr(False)
            UsdPhysics.MassAPI.Apply(actor.GetPrim()).CreateMassAttr(float(.002 * self.lengths[i] / total_length))
            body = PhysxSchema.PhysxRigidBodyAPI.Apply(actor.GetPrim())
            body.CreateSolverPositionIterationCountAttr(64)
            body.CreateSolverVelocityIterationCountAttr(4)
            body.CreateLinearDampingAttr(.01)
            body.CreateAngularDampingAttr(.005)
            body.CreateSleepThresholdAttr(0.)
            UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material, UsdShade.Tokens.strongerThanDescendants, 'physics')
            cube.MakeInvisible()
            vertices.append(np.asarray(mesh.vertices) @ R.T + p)
            faces.append(np.asarray(mesh.faces) + i * 8)
        self.faces = np.concatenate(faces)
        rest = np.vstack([*vertices, grip[None]])
        self.grasp_nodes = np.array([len(rest) - 1])
        self.part = {'V': rest, 'F': self.faces, 'cloth_rest': {'points': rest.tolist()}}
        for i, anchor in enumerate(knots[1:-1]):
            joint = (UsdPhysics.Joint.Define(stage, self.path + '/MaterialBend' + str(i))
                     if grip_location == 'center' else
                     UsdPhysics.RevoluteJoint.Define(stage, self.path + '/MaterialBend' + str(i)))
            joint.CreateBody0Rel().SetTargets([self.path + '/Panel' + str(i)])
            joint.CreateBody1Rel().SetTargets([self.path + '/Panel' + str(i + 1)])
            joint.CreateLocalPos0Attr(Gf.Vec3f(*(rotations[i].T @ (anchor - centers[i]))))
            joint.CreateLocalPos1Attr(Gf.Vec3f(*(rotations[i + 1].T @ (anchor - centers[i + 1]))))
            # Identical material frames prefer a flat sheet. The initial fold
            # at the desk is an initial condition, not a retained 90deg crease.
            joint.CreateLocalRot0Attr(Gf.Quatf(1.))
            joint.CreateLocalRot1Attr(Gf.Quatf(1.))
            if grip_location == 'center':
                for axis in ['transX', 'transY', 'transZ', 'rotY']:
                    limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
                    limit.CreateLowAttr(0.)
                    limit.CreateHighAttr(0.)
                for axis, bound in [('rotX', 170.), ('rotZ', 60.)]:
                    limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
                    limit.CreateLowAttr(-bound)
                    limit.CreateHighAttr(bound)
                drives = [('rotX', 1e-5), ('rotZ', 2e-6)]
            else:
                joint.CreateAxisAttr('X')
                joint.CreateLowerLimitAttr(-170.)
                joint.CreateUpperLimitAttr(170.)
                drives = [('angular', 1e-5)]
            for axis, stiffness in drives:
                drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), axis)
                drive.CreateTypeAttr('force')
                drive.CreateStiffnessAttr(stiffness * np.pi / 180)
                drive.CreateDampingAttr(1e-6 * np.pi / 180)
                drive.CreateMaxForceAttr(.0002)
                drive.CreateTargetPositionAttr(0.)
        self.skin = UsdGeom.Mesh.Define(stage, self.path + '/ContinuousSkin')
        self.skin.CreateSubdivisionSchemeAttr('none')
        self.skin.CreateDoubleSidedAttr(True)
        self.skin.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        UsdShade.MaterialBindingAPI.Apply(self.skin.GetPrim()).Bind(visual)
        self.material_knots = boundary
        self.render_s = np.linspace(0., total_length, 193 if grip_location == 'center' else 129)
        self.width_s = np.linspace(-.025, .025, 5)
        rows, cols = len(self.render_s), len(self.width_s)
        front = []
        for j in range(rows - 1):
            for k in range(cols - 1):
                a = j * cols + k
                front.extend([[a, a + 1, a + cols], [a + 1, a + cols + 1, a + cols]])
        n = rows * cols
        skin_faces = np.r_[front, np.asarray(front)[:, ::-1] + n]
        # Stitch the two visible surfaces at the sheet's thin perimeter.
        perimeter = list(range(cols)) + [j * cols + cols - 1 for j in range(1, rows)]
        perimeter += list(range(n - 2, n - cols - 1, -1)) + [j * cols for j in range(rows - 2, 0, -1)]
        for a, b in zip(perimeter, perimeter[1:] + perimeter[:1]):
            skin_faces = np.vstack([skin_faces, [a, b, a + n], [b, b + n, a + n]])
        self.skin.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(skin_faces)))
        self.skin.CreateFaceVertexIndicesAttr(Vt.IntArray(skin_faces.ravel().tolist()))
        self._update_skin([(p, R, s) for p, R, s in zip(centers, rotations, self.sizes)])
        self.initial_centers = centers.copy()
        self.description = {'backend': 'native compliant segmented shell strip', 'panels': len(self.sizes),
                            'mass_kg': .002, 'folded_grip_thickness_m': .003, 'sheet_thickness_m': .0007,
                            'bend_stiffness_Nm_rad': 1e-5, 'bend_damping_Nms_rad': 1e-6,
                            'twist_stiffness_Nm_rad': 2e-6 if grip_location == 'center' else None,
                            'hand_attachments': False, 'cloth_pose_following': False,
                            'external_cloth_forces': False, 'prescribed_cloth_positions_after_initialization': False,
                            'continuous_skin_from_measured_physics': True,
                            'grip_location': grip_location, 'sheet_length_m': total_length,
                            'grip_material_fraction': float(rest_shape['material_grip_s'] / total_length),
                            'free_material_lengths_m': [float(boundary[self.grip_panel]),
                                                        float(total_length - boundary[self.grip_panel + 1])],
                            'limitation': 'Discrete bending strip; no continuum in-plane stretch or fine cross-width wrinkles'}

    def initialize(self):
        from isaacsim.core.prims import RigidPrim
        self.panel_paths = [self.path + '/Panel' + str(i) for i in range(len(self.sizes))]
        self.panel_view = RigidPrim(self.panel_paths, reset_xform_properties=False, name='soft_wipe_segments')
        self.panel_view.initialize()
        native_paths = list(self.panel_view._physics_view.prim_paths)
        self.panel_order = [native_paths.index(p) for p in self.panel_paths]
        # Compare simultaneous independent native reads. World.reset can take
        # a settling step, so authored rest positions are not a valid oracle.
        from isaacsim.core.api import SimulationContext
        initial = np.array([p for p, _, _ in self.boxes()])
        singles = []
        for path in self.panel_paths:
            view = SimulationContext.instance().physics_sim_view.create_rigid_body_view(path)
            pose = view.get_transforms()
            pose = pose.detach().cpu().numpy() if hasattr(pose, 'detach') else np.asarray(pose)
            singles.append(pose[0, :3])
        error = float(np.max(np.linalg.norm(initial - np.asarray(singles), axis=1)))
        if error > 1e-6:
            raise RuntimeError(f'Soft wipe batch/individual native pose mismatch: {error} m')
        self.description['native_batch_vs_individual_pose_error_m'] = error
        self.audit_independence()

    def boxes(self):
        if getattr(self.panel_view, '_physics_view', None) is None:
            from isaacsim.core.api import SimulationContext
            self.panel_view.initialize(SimulationContext.instance().physics_sim_view)
            native_paths = list(self.panel_view._physics_view.prim_paths)
            self.panel_order = [native_paths.index(p) for p in self.panel_paths]
        p, q = self.panel_view.get_world_poses()
        p = p.detach().cpu().numpy() if hasattr(p, 'detach') else np.asarray(p)
        q = q.detach().cpu().numpy() if hasattr(q, 'detach') else np.asarray(q)
        p, q = p[self.panel_order], q[self.panel_order]
        rotations = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
        return list(zip(p, rotations, self.sizes))

    def _update_skin(self, boxes):
        starts = np.array([p + R[:, 2] * s[2] / 2 for p, R, s in boxes])
        ends = np.array([p - R[:, 2] * s[2] / 2 for p, R, s in boxes])
        points = np.vstack([starts[0], (ends[:-1] + starts[1:]) / 2, ends[-1]])
        # Shape-preserving interpolation avoids smoothing through the desk.
        curve = PchipInterpolator(self.material_knots, points, axis=0)
        center = curve(self.render_s)
        tangent = curve(self.render_s, 1)
        bad = np.linalg.norm(tangent, axis=1) < 1e-8
        tangent[bad] = np.gradient(center, self.render_s, axis=0)[bad]
        tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
        axes = np.array([R[:, 0] for _, R, _ in boxes])
        if self.grip_location == 'center':
            mids = (self.material_knots[:-1] + self.material_knots[1:]) / 2
            width = np.array([np.interp(self.render_s, mids, axes[:, k]) for k in range(3)]).T
            width -= (width * tangent).sum(1, keepdims=True) * tangent
            width /= np.maximum(np.linalg.norm(width, axis=1, keepdims=True), 1e-9)
        else:
            width = np.tile(axes.mean(0), (len(self.render_s), 1))
            width /= np.linalg.norm(width, axis=1, keepdims=True)
        normal = np.cross(width, tangent)
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
        thickness = np.interp(self.render_s, (self.material_knots[:-1] + self.material_knots[1:]) / 2, self.thickness)
        surface = center[:, None] + self.width_s[None, :, None] * width[:, None]
        skins = [surface + sign * thickness[:, None, None] * normal[:, None] / 2 for sign in [1, -1]]
        vertices = np.concatenate([s.reshape(-1, 3) for s in skins])
        # Submillimetre interpolation overshoot is a render-only correction.
        correction = max(0., float(self.table_z + .00005 - vertices[:, 2].min()))
        self.max_skin_floor_correction_m = max(getattr(self, 'max_skin_floor_correction_m', 0.), correction)
        if hasattr(self, 'description'):
            self.description['max_render_floor_correction_m'] = self.max_skin_floor_correction_m
        vertices[:, 2] = np.maximum(vertices[:, 2], self.table_z + .00005)
        normals = np.concatenate([np.repeat(normal, len(self.width_s), axis=0),
                                  -np.repeat(normal, len(self.width_s), axis=0)])
        self.skin.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
        self.skin.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
        self.skin.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.array([vertices.min(0), vertices.max(0)], np.float32)))

    def update_visual(self):
        self._update_skin(self.boxes())
