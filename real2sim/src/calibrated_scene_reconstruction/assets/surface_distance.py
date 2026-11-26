"""Signed distance to actual closed rigid triangles for native-state checks.

Negative is inside. This queries the authored surface, not a bounding box or
the PhysX voxel approximation. It does not certify unsampled cloth interiors.
"""
import numpy as np
import warp as wp

from calibrated_scene_reconstruction.assets.metric_mesh import geometry


@wp.kernel
def _distance(mesh: wp.uint64, points: wp.array(dtype=wp.vec3),
              distances: wp.array(dtype=float), hits: wp.array(dtype=int)):
    index = wp.tid()
    query = wp.mesh_query_point_sign_winding_number(mesh, points[index], 10000.)
    hits[index] = int(query.result)
    if query.result:
        closest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        distances[index] = query.sign*wp.length(points[index]-closest)


@wp.kernel
def _project(mesh: wp.uint64, points: wp.array(dtype=wp.vec3), clearance: float,
             output: wp.array(dtype=wp.vec3), hits: wp.array(dtype=int)):
    index = wp.tid()
    point = points[index]
    query = wp.mesh_query_point_sign_winding_number(mesh, point, 10000.)
    hits[index] = int(query.result)
    output[index] = point
    if query.result:
        closest = wp.mesh_eval_position(mesh, query.face, query.u, query.v)
        delta = point-closest
        distance = wp.length(delta)
        if query.sign*distance < clearance:
            normal = wp.mesh_eval_face_normal(mesh, query.face)
            if distance > 1.0e-10:
                normal = query.sign*delta/distance
            output[index] = closest+clearance*normal


class SignedSurface:
    def __init__(self, vertices, faces, *, device='cuda:0'):
        vertices, faces = np.asarray(vertices, float), np.asarray(faces)
        geometry(vertices, faces, max_extent_m=10., min_extent_m=1e-5, require_bottom_origin=False)
        wp.init()
        self.device = device
        self.mesh = wp.Mesh(points=wp.array(vertices.astype(np.float32), dtype=wp.vec3, device=device),
                            indices=wp.array(faces.astype(np.int32).ravel(), dtype=int, device=device),
                            support_winding_number=True)
        self.bounds = np.array([vertices.min(0),vertices.max(0)])
        self.vertex_shape=vertices.shape
        self.count = 0

    def update(self, vertices):
        """Refit unchanged topology to actual native solid nodes.

        This performs no native writes and grants no validity certificate.
        The caller must independently reject inverted volume cells.
        """
        points=np.asarray(vertices,np.float32)
        if points.shape!=self.vertex_shape or not np.isfinite(points).all():
            raise ValueError('Updated solid must preserve finite indexed vertex dimensions')
        self.mesh.points.assign(points)
        self.mesh.refit()
        self.bounds=np.array([points.min(0),points.max(0)])

    def __call__(self, points):
        points = np.asarray(points, np.float32)
        if points.ndim != 2 or points.shape[1:] != (3,) or not len(points) or not np.isfinite(points).all():
            raise ValueError('Signed-surface queries need finite (N,3) local points')
        if len(points) != self.count:
            self.count = len(points)
            self.points = wp.empty(self.count, dtype=wp.vec3, device=self.device)
            self.distances = wp.empty(self.count, dtype=float, device=self.device)
            self.hits = wp.empty(self.count, dtype=int, device=self.device)
        self.points.assign(points)
        wp.launch(_distance, dim=self.count,
                  inputs=[self.mesh.id, self.points, self.distances, self.hits], device=self.device)
        if not self.hits.numpy().all():
            raise ValueError('Actual rigid surface query failed; cannot assume no penetration')
        distances = self.distances.numpy()
        if not np.isfinite(distances).all():
            raise ValueError('Nonfinite rigid-surface distances')
        return distances

    def max_penetration(self, points):
        points = np.asarray(points, float)
        if points.ndim != 2 or points.shape[1:] != (3,) or not len(points) or not np.isfinite(points).all():
            raise ValueError('Penetration queries need finite local points')
        inside_box = ((points >= self.bounds[0]) & (points <= self.bounds[1])).all(1)
        if not inside_box.any():
            return 0.
        # The box only excludes points that cannot be inside a closed surface;
        # cavity/wall decisions still query the actual triangles.
        return float(max(0., -self(points[inside_box]).min()))

    def project_out(self, points, clearance):
        """Geometry initialization only: move points outside actual solid walls.

        No native node state is written. A projected model candidate must still
        pass independent material, contact, self-collision and task checks.
        """
        points = np.asarray(points, np.float32)
        if (points.ndim != 2 or points.shape[1:] != (3,) or not len(points)
                or not np.isfinite(points).all() or not np.isfinite(clearance) or clearance <= 0):
            raise ValueError('Finite geometry points and positive projection clearance required')
        source = wp.array(points, dtype=wp.vec3, device=self.device)
        output = wp.empty(len(points), dtype=wp.vec3, device=self.device)
        hits = wp.empty(len(points), dtype=int, device=self.device)
        wp.launch(_project, dim=len(points), inputs=[self.mesh.id, source, float(clearance), output, hits], device=self.device)
        if not hits.numpy().all():
            raise ValueError('Actual surface projection failed')
        result = output.numpy()
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite surface projection')
        return result


def triangle_samples(points, faces):
    """Include edge midpoints and face centres, not only particle vertices."""
    points, faces = np.asarray(points, float), np.asarray(faces)
    if (points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all()
            or faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces) or faces.dtype.kind not in 'iu'
            or faces.min() < 0 or faces.max() >= len(points)):
        raise ValueError('Finite surface points and valid triangles required')
    triangles = points[faces]
    return np.concatenate([points, triangles.mean(1),
                           (triangles[:, 0]+triangles[:, 1])/2,
                           (triangles[:, 1]+triangles[:, 2])/2,
                           (triangles[:, 2]+triangles[:, 0])/2])
