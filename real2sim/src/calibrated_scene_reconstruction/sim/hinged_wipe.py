"""Coarse folded-wipe mechanics: independent thin panels with passive hinges.

This is an explicit material approximation, not continuum cloth. All hand-wipe
interaction is native contact/friction. No joint, attachment or pose follower
connects the wipe to the hand. Internal hinges only connect wipe panels.
"""
import numpy as np
from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics, UsdShade
from scipy.spatial.transform import Rotation

from calibrated_scene_reconstruction.sim.usd import color_material, set_pose, quat_from_matrix
from calibrated_scene_reconstruction.sim.physics import physics_material


class HingedWipe:
    @staticmethod
    def grip_height(*_):
        return .0175

    def __init__(self, stage, grip, table_z, yaw=0., **_):
        import trimesh
        self.stage = stage
        self.path = '/World/WetWipe'
        stage.DefinePrim(self.path, 'Xform')
        grip = np.asarray(grip)
        rotation = Rotation.from_euler('z', yaw).as_matrix()
        self.sizes = [np.array([.05, .003, .022]), np.array([.05, .034, .003]), np.array([.05, .034, .003])]
        centers = [np.array([grip[0], grip[1], table_z + .0125]),
                   np.array([grip[0], grip[1] - .017, table_z + .0015]),
                   np.array([grip[0], grip[1] - .051, table_z + .0015])]
        centers = [grip + rotation @ (p - grip) for p in centers]
        self.grip_local = rotation.T @ (grip - centers[0])
        visual = color_material(stage, '/World/Looks/FoldedWipe', [.91, .92, .88], roughness=.95)
        material = physics_material(stage, '/World/Looks/FoldedWipePhysics', 1.0, .8, 0.)
        self.meshes = [trimesh.creation.box(extents=size) for size in self.sizes]
        rest = []
        faces = []
        for i, (size, center, mesh) in enumerate(zip(self.sizes, centers, self.meshes)):
            path = self.path + '/Panel' + str(i)
            actor = UsdGeom.Xform.Define(stage, path)
            set_pose(actor.GetPrim(), center, quat_from_matrix(rotation))
            cube = UsdGeom.Cube.Define(stage, path + '/Geometry')
            cube.CreateSizeAttr(1.)
            set_pose(cube.GetPrim(), [0, 0, 0], scale=size)
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            collision = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
            collision.CreateContactOffsetAttr(.0003)
            collision.CreateRestOffsetAttr(0.)
            UsdPhysics.RigidBodyAPI.Apply(actor.GetPrim()).CreateKinematicEnabledAttr(False)
            UsdPhysics.MassAPI.Apply(actor.GetPrim()).CreateMassAttr(.002 / 3)
            body = PhysxSchema.PhysxRigidBodyAPI.Apply(actor.GetPrim())
            body.CreateSolverPositionIterationCountAttr(32)
            body.CreateSolverVelocityIterationCountAttr(4)
            body.CreateLinearDampingAttr(.01)
            body.CreateAngularDampingAttr(.02)
            body.CreateSleepThresholdAttr(0.)
            UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(visual)
            UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material, UsdShade.Tokens.strongerThanDescendants, 'physics')
            rest.append(np.asarray(mesh.vertices) @ rotation.T + center)
            faces.append(np.asarray(mesh.faces) + i * 8)
        self.faces = np.concatenate(faces)
        rest = np.concatenate(rest)
        # One material witness, not a simulated node or constraint.
        rest = np.vstack([rest, grip])
        self.grasp_nodes = np.array([len(rest) - 1])
        self.part = {'V': rest, 'F': self.faces, 'cloth_rest': {'points': rest.tolist()}}
        anchors = [[grip[0], grip[1], table_z + .0015], [grip[0], grip[1] - .034, table_z + .0015]]
        anchors = [grip + rotation @ (np.asarray(p) - grip) for p in anchors]
        for i, anchor in enumerate(anchors):
            joint = UsdPhysics.RevoluteJoint.Define(stage, self.path + '/MaterialHinge' + str(i))
            joint.CreateBody0Rel().SetTargets([self.path + '/Panel' + str(i)])
            joint.CreateBody1Rel().SetTargets([self.path + '/Panel' + str(i + 1)])
            joint.CreateAxisAttr('X')
            joint.CreateLocalPos0Attr(Gf.Vec3f(*(rotation.T @ (np.asarray(anchor) - centers[i]))))
            joint.CreateLocalPos1Attr(Gf.Vec3f(*(rotation.T @ (np.asarray(anchor) - centers[i + 1]))))
            joint.CreateLocalRot0Attr(Gf.Quatf(1.))
            joint.CreateLocalRot1Attr(Gf.Quatf(1.))
            joint.CreateLowerLimitAttr(-170.)
            joint.CreateUpperLimitAttr(170.)
            drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), 'angular')
            drive.CreateStiffnessAttr(.0002 * np.pi / 180)
            drive.CreateDampingAttr(.00005 * np.pi / 180)
            drive.CreateMaxForceAttr(.002)
            drive.CreateTargetPositionAttr(0.)
        self.description = {'backend': 'native rigid panels + passive internal material hinges', 'panels': 3,
                            'mass_kg': .002, 'thickness_m': .003, 'hand_attachments': False,
                            'cloth_pose_following': False, 'external_cloth_forces': False,
                            'prescribed_cloth_positions_after_initialization': False,
                            'limitation': 'Coarse folded strip; no in-plane wrinkles or continuum fabric dynamics'}

    def initialize(self):
        from isaacsim.core.prims import SingleRigidPrim
        self.bodies = []
        for i in range(len(self.sizes)):
            body = SingleRigidPrim(self.path + '/Panel' + str(i), name='wet_wipe_panel_' + str(i))
            body.initialize()
            self.bodies.append(body)
        self.audit_independence()

    def audit_independence(self):
        from pxr import Usd
        for prim in self.stage.Traverse():
            if 'Attachment' in prim.GetTypeName():
                raise RuntimeError('Attachments are forbidden in this physical grasp demo')
            if prim.IsA(UsdPhysics.Joint):
                joint = UsdPhysics.Joint(prim)
                targets = [str(p) for p in joint.GetBody0Rel().GetTargets() + joint.GetBody1Rel().GetTargets()]
                wipe = any(p.startswith(self.path + '/') for p in targets)
                if wipe and any(not p.startswith(self.path + '/') for p in targets):
                    raise RuntimeError('A joint connects the wipe to an external object')
        for prim in Usd.PrimRange(self.stage.GetPrimAtPath(self.path)):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                if UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get():
                    raise RuntimeError('Every wipe panel must remain dynamic')
                if UsdPhysics.MassAPI(prim).GetMassAttr().Get() <= 0:
                    raise RuntimeError('Every wipe panel must have positive mass')
        return {'external_joints': 0, 'attachments': 0, 'kinematic_bodies': 0, 'positive_panel_masses': True}

    def boxes(self):
        result = []
        for body, size in zip(self.bodies, self.sizes):
            # Isaac 5.1 invalidates rigid views when an unrelated exhausted
            # particle prim is deleted. Rebind without moving the wipe.
            view = body._rigid_prim_view
            if getattr(view, '_physics_view', None) is None:
                from isaacsim.core.api import SimulationContext
                view._physics_view = SimulationContext.instance().physics_sim_view.create_rigid_body_view(body.prim_path)
            p, q = body.get_world_pose()
            p = p.detach().cpu().numpy() if hasattr(p, 'detach') else np.asarray(p)
            q = q.detach().cpu().numpy() if hasattr(q, 'detach') else np.asarray(q)
            result.append((p, Rotation.from_quat([*q[1:], q[0]]).as_matrix(), size))
        return result

    def points(self):
        boxes = self.boxes()
        vertices = [np.asarray(mesh.vertices) @ R.T + p for mesh, (p, R, _) in zip(self.meshes, boxes)]
        grip = boxes[getattr(self, 'grip_panel', 0)]
        vertices.append((grip[0] + grip[1] @ self.grip_local)[None])
        return np.concatenate(vertices)

    def grip_center(self):
        return self.points()[-1]

    def finger_proximity(self, robot, pinch):
        pads = []
        for name in ['thumb', 'index']:
            p, R = robot.link_pose(pinch[name]['link'])
            pads.append(p + R @ np.asarray(pinch[name]['pad_local']))
        distances = []
        for p, R, size in self.boxes():
            local = (np.asarray(pads) - p) @ R
            outside = np.abs(local) - size / 2
            distances.append(np.linalg.norm(np.maximum(outside, 0), axis=1))
        distances = np.asarray(distances)
        selected = int(np.argmin(distances.max(1)))
        return {'same_panel_index': selected, 'pad_to_panel_m': distances[selected].tolist(),
                'maximum_pad_distance_m': float(distances[selected].max())}
