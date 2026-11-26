"""Usable planar task support from actual collision triangles, including holes."""
import numpy as np


class Tabletop:
    def __init__(self, vertices, faces, height, tolerance=.0005):
        vertices, faces = np.asarray(vertices, float), np.asarray(faces, int)
        if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or faces.ndim != 2 or faces.shape[1:] != (3,)
                or not np.isfinite(vertices).all() or not np.isfinite([height, tolerance]).all()
                or tolerance <= 0 or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError('Finite actual support triangles and a top plane required')
        triangles = vertices[faces]
        normal = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        norm = np.linalg.norm(normal, axis=1)
        keep = (abs(triangles[:, :, 2]-height).max(1) <= tolerance) & (abs(normal[:, 2]) >= .98*norm) & (norm > 1e-12)
        self.triangles = triangles[keep, :, :2]
        if not len(self.triangles):
            raise ValueError('No actual collision surface at the task support height')
        edges = {}
        for triangle in self.triangles:
            for i, j in [(0, 1), (1, 2), (2, 0)]:
                key = tuple(sorted(tuple(p) for p in np.round(triangle[[i, j]], 8)))
                edges[key] = edges.get(key, 0)+1
        self.boundary = np.array([edge for edge, count in edges.items() if count == 1])
        if not len(self.boundary):
            raise ValueError('Support top has no finite boundary')
        self.height = float(height)

    def contains(self, points):
        points = np.asarray(points, float)
        if points.ndim != 2 or points.shape[1:] != (2,) or not np.isfinite(points).all():
            raise ValueError('Finite planar test points required')
        result = np.zeros(len(points), bool)
        for triangle in self.triangles:
            a = triangle
            b = np.roll(triangle, -1, axis=0)
            v = points[:, None]-a
            d = b-a
            cross = d[None, :, 0]*v[:, :, 1]-d[None, :, 1]*v[:, :, 0]
            result |= (cross >= -1e-10).all(1) | (cross <= 1e-10).all(1)
        return result

    def clearance(self, points):
        points = np.asarray(points, float)
        a, b = self.boundary[:, 0], self.boundary[:, 1]
        delta = b-a
        t = np.sum((points[:, None]-a)*delta, axis=2)/np.sum(delta*delta, axis=1)
        closest = a+np.clip(t, 0, 1)[:, :, None]*delta
        distance = np.linalg.norm(points[:, None]-closest, axis=2).min(1)
        return np.where(self.contains(points), distance, -distance)

    def stroke(self, avoid_xy, *, avoid_radius, clearance, length):
        """Choose a nearby clear stroke, not a corner of a rotated mesh's AABB."""
        if not np.isfinite([avoid_radius, clearance, length]).all() or min(avoid_radius, clearance, length) <= 0:
            raise ValueError('Positive physical footprint and stroke budgets required')
        points = self.triangles.reshape(-1, 2)
        lo, hi = points.min(0), points.max(0)
        candidates = np.array([[x, y] for x in np.linspace(lo[0], hi[0], 31) for y in np.linspace(lo[1], hi[1], 31)])
        distance = np.linalg.norm(candidates-avoid_xy, axis=1)
        candidates = candidates[(self.clearance(candidates) >= clearance) & (distance >= avoid_radius+clearance)]
        candidates = sorted(candidates, key=lambda p: np.linalg.norm(p-avoid_xy))
        for start in candidates:
            for angle in np.arange(8)*np.pi/4:
                end = start+length*np.array([np.cos(angle), np.sin(angle)])
                path = np.linspace(start, end, 21)
                if ((self.clearance(path) >= clearance).all()
                        and (np.linalg.norm(path-avoid_xy, axis=1) >= avoid_radius+clearance).all()):
                    return np.r_[start, self.height], np.r_[end, self.height]
        raise ValueError('No clear wiping stroke on the actual task support mesh')


def from_stage(ctx, name, height):
    from pxr import UsdGeom, UsdPhysics
    roots = [f'/World/Support/{name}', f'/World/Room/{name}', f'/World/Static/{name}', f'/World/Room/collider_{name}']
    vertices, faces, offset = [], [], 0
    cache = UsdGeom.XformCache()
    for prim in ctx.stage.Traverse():
        path = str(prim.GetPath())
        if (not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI)
                or not any(path == root or path.startswith(root+'/') for root in roots)):
            continue
        mesh = UsdGeom.Mesh(prim)
        if np.any(np.asarray(mesh.GetFaceVertexCountsAttr().Get()) != 3):
            raise ValueError('Actual task support collider must be triangulated')
        v = np.asarray(mesh.GetPointsAttr().Get(), float)
        world = (np.c_[v, np.ones(len(v))]@np.asarray(cache.GetLocalToWorldTransform(prim)))[:, :3]
        vertices.append(world);faces.append(np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)+offset)
        offset += len(v)
    if not vertices:
        raise ValueError('No actual collision mesh for task support '+name)
    return Tabletop(np.concatenate(vertices), np.concatenate(faces), height)
