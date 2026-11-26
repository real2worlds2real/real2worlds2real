"""Metric CAD kinematics for visual calibration without a joint readout.

USD joint frames and limits are robot geometry, not a measured capture pose.
Revolute limits are authored in degrees; this module exposes radians.
"""
from dataclasses import dataclass, field
import re

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class Joint:
    name: str
    parent: str
    child: str
    frame0: np.ndarray
    frame1_inverse: np.ndarray
    axis: np.ndarray
    kind: str
    lower: float = 0.
    upper: float = 0.


@dataclass
class ArticulatedModel:
    root: str
    joints: list
    surfaces: dict
    source: str = ''
    floor_vertices: dict = field(default_factory=dict)
    dependencies: tuple = ()
    normals: dict = field(default_factory=dict)
    triangles: dict = field(default_factory=dict)

    @property
    def movable(self):
        return [j for j in self.joints if j.kind != 'fixed']

    def fixed_root(self):
        """Rigid base geometry provides a scale seed independent of articulation."""
        fixed = {self.root}
        joints = []
        for joint in self.joints:
            if joint.kind == 'fixed' and joint.parent in fixed:
                fixed.add(joint.child)
                joints.append(joint)
        return ArticulatedModel(self.root, joints, {name: points for name, points in self.surfaces.items()
                                                  if name in fixed}, self.source,
                                {name: points for name, points in self.floor_vertices.items() if name in fixed},
                                self.dependencies, {name: values for name, values in self.normals.items() if name in fixed},
                                {name: values for name, values in self.triangles.items() if name in fixed})

    def transforms(self, state=None):
        """Forward kinematics relative to the robot root; missing joints use zero.

        Zero is an explicit hypothesis, never a hardware state. Callers must
        report unestimated joints instead of claiming a fully observed pose.
        """
        state = state or {}
        unknown = set(state) - {j.name for j in self.movable}
        if unknown:
            raise ValueError(f'Unknown CAD joints: {sorted(unknown)}')
        transforms = {self.root: np.eye(4)}
        for joint in self.joints:
            if joint.parent not in transforms:
                raise ValueError('CAD joints must form a connected topological tree')
            motion = np.eye(4)
            value = float(state.get(joint.name, 0.))
            if not np.isfinite(value):
                raise ValueError('Finite visual joint hypotheses required')
            if joint.kind == 'revolute':
                motion[:3, :3] = Rotation.from_rotvec(joint.axis * value).as_matrix()
            elif joint.kind == 'prismatic':
                motion[:3, 3] = joint.axis * value
            elif joint.kind != 'fixed':
                raise ValueError(f'Unsupported visual joint kind: {joint.kind}')
            transforms[joint.child] = (transforms[joint.parent] @ joint.frame0
                                       @ motion @ joint.frame1_inverse)
        return transforms

    def points(self, state=None, exclude_links=(), per_link=None):
        transforms = self.transforms(state)
        points, labels, names = [], [], []
        for name, local in self.surfaces.items():
            if any(re.search(pattern, name, re.I) for pattern in exclude_links):
                continue
            if per_link and len(local) > per_link:
                local = local[np.linspace(0, len(local)-1, per_link).astype(int)]
            transform = transforms[name]
            points.append(local @ transform[:3, :3].T + transform[:3, 3])
            labels.append(np.full(len(local), len(names), np.int32))
            names.append(name)
        if not points:
            raise ValueError('No CAD visual surfaces selected')
        return np.concatenate(points), np.concatenate(labels), names

    def floor_height(self):
        """Lowest fixed-root surface; articulation cannot change this floor datum."""
        fixed = {self.root}
        for joint in self.joints:
            if joint.kind == 'fixed' and joint.parent in fixed:
                fixed.add(joint.child)
        transforms = self.transforms()
        values = []
        for name in fixed:
            if name in self.surfaces:
                transform = transforms[name]
                points = self.floor_vertices.get(name, self.surfaces[name])
                values.append((points @ transform[:3, :3].T + transform[:3, 3])[:, 2].min())
        if not values:
            raise ValueError('Visual calibration needs a fixed-root floor-contact surface')
        return float(min(values))


def from_usd(path, points_per_link=800):
    """Read legal robot asset geometry and joint frames without importing Isaac.

    Joint convention follows OpenUSD UsdPhysicsJoint localPos/localRot0/1.
    https://openusd.org/release/api/class_usd_physics_joint.html
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(path))
    if stage is None or not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.):
        raise ValueError('Robot CAD must open and use metre units')
    cache = UsdGeom.XformCache()
    bodies, joints = {}, []

    def frame(position, quaternion):
        result = np.eye(4)
        result[:3, 3] = np.asarray(position, float)
        result[:3, :3] = Rotation.from_quat([*quaternion.GetImaginary(), quaternion.GetReal()]).as_matrix()
        return result

    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        schema = UsdPhysics.Joint(prim)
        if not schema.GetJointEnabledAttr().Get():
            continue
        parent, child = schema.GetBody0Rel().GetTargets(), schema.GetBody1Rel().GetTargets()
        if not parent or not child:  # a world mount is outside root-relative kinematics
            continue
        for body in (parent[0], child[0]):
            name = stage.GetPrimAtPath(body).GetName()
            if name in bodies and bodies[name] != body:
                raise ValueError('Robot CAD body names must be unique')
            bodies[name] = body
        kind, axis, lower, upper = 'fixed', np.zeros(3), 0., 0.
        if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
            kind = 'revolute' if prim.IsA(UsdPhysics.RevoluteJoint) else 'prismatic'
            axis = np.eye(3)['XYZ'.index(str(prim.GetAttribute('physics:axis').Get()))]
            lower, upper = [float(prim.GetAttribute('physics:'+key+'Limit').Get()) for key in ('lower', 'upper')]
            if kind == 'revolute':
                lower, upper = np.radians([lower, upper])
        elif not prim.IsA(UsdPhysics.FixedJoint):
            raise ValueError(f'Unsupported CAD joint type: {prim.GetTypeName()}')
        joints.append(Joint(prim.GetName(), stage.GetPrimAtPath(parent[0]).GetName(),
                            stage.GetPrimAtPath(child[0]).GetName(),
                            frame(schema.GetLocalPos0Attr().Get(), schema.GetLocalRot0Attr().Get()),
                            np.linalg.inv(frame(schema.GetLocalPos1Attr().Get(), schema.GetLocalRot1Attr().Get())),
                            axis, kind, float(lower), float(upper)))
    roots = set(bodies) - {j.child for j in joints}
    if len(roots) != 1:
        raise ValueError(f'Expected one connected robot root, found {sorted(roots)}')
    root = roots.pop()
    ordered, seen = [], {root}
    while joints:
        ready = [j for j in joints if j.parent in seen]
        if not ready:
            raise ValueError('Disconnected or cyclic CAD joint graph')
        for joint in ready:
            if joint.child in seen:
                raise ValueError('Multiple CAD parents are unsupported')
            ordered.append(joint)
            seen.add(joint.child)
            joints.remove(joint)
    by_path = {str(path): name for name, path in bodies.items()}
    surfaces, triangles, floor_vertices, normals = {}, {}, {}, {}
    for prim in Usd.PrimRange(stage.GetDefaultPrim(), Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh) or 'collision' in str(prim.GetPath()).lower():
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        parent = prim.GetParent()
        while parent and str(parent.GetPath()) not in by_path:
            parent = parent.GetParent()
        if not parent:
            continue
        name = by_path[str(parent.GetPath())]
        matrix = np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(parent)).T) @ np.asarray(
            cache.GetLocalToWorldTransform(prim)).T
        points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(), float)
        points = points @ matrix[:3, :3].T + matrix[:3, 3]
        indices = np.asarray(UsdGeom.Mesh(prim).GetFaceVertexIndicesAttr().Get(), int)
        counts = np.asarray(UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get(), int)
        faces, start = [], 0
        for count in counts:
            polygon = indices[start:start+count]
            faces.extend((polygon[0], polygon[i], polygon[i+1]) for i in range(1, count-1))
            start += count
        if not faces:
            continue
        triangles.setdefault(name, []).append(points[np.asarray(faces)])
        floor_vertices.setdefault(name, []).append(points)
    rng = np.random.default_rng(0)
    for name, parts in triangles.items():
        triangles_link = np.concatenate(parts)
        area = np.linalg.norm(np.cross(triangles_link[:, 1]-triangles_link[:, 0],
                                       triangles_link[:, 2]-triangles_link[:, 0]), axis=1)
        valid = area > 1e-14
        triangles_link, area = triangles_link[valid], area[valid]
        triangles[name] = triangles_link
        if not len(area):
            continue
        chosen = triangles_link[rng.choice(len(area), points_per_link, p=area/area.sum())]
        normal = np.cross(chosen[:, 1]-chosen[:, 0], chosen[:, 2]-chosen[:, 0])
        normals[name] = normal/np.linalg.norm(normal, axis=1, keepdims=True)
        barycentric = rng.random((points_per_link, 2))
        flip = barycentric.sum(1) > 1
        barycentric[flip] = 1-barycentric[flip]
        surfaces[name] = (chosen[:, 0]+barycentric[:, :1]*(chosen[:, 1]-chosen[:, 0])+
                          barycentric[:, 1:]*(chosen[:, 2]-chosen[:, 0]))
        floor_vertices[name] = np.concatenate(floor_vertices[name])
    dependencies = tuple(sorted(str(layer.realPath) for layer in stage.GetUsedLayers() if layer.realPath))
    return ArticulatedModel(root, ordered, surfaces, str(path), floor_vertices, dependencies, normals, triangles)
