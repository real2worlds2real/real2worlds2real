"""Visible part silhouettes from measured native triangles, with self occlusion."""
import numpy as np
import warp as wp


@wp.kernel
def _visible(mesh: wp.uint64, face_map: wp.array(dtype=int), origin: wp.vec3,
             inverse: wp.mat33, rotation: wp.mat33, width: int, near: float,
             result: wp.array(dtype=int)):
    index = wp.tid()
    x, y = index % width, index // width
    # Calibrated OpenCV image coordinates put pixel centres at integers.
    # The GPU rasterizer compensates its half-pixel sample convention in its
    # projection matrix; adding half a pixel here would shift only CPU rays.
    direction = wp.normalize(rotation*(inverse*wp.vec3(float(x), float(y), 1.0)))
    hit = wp.mesh_query_ray(mesh, origin+near*direction, direction, 10000.0)
    if hit.result:
        result[index] = face_map[hit.face]+1


class VisibleParts:
    """One immutable actual state; rigid and flexible parts occlude each other."""
    def __init__(self, vertices, faces, target_faces, *, device='cpu',transmitting_faces=None):
        vertices, faces = np.asarray(vertices, float), np.asarray(faces)
        target = np.asarray(target_faces)
        if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices)
                or not np.isfinite(vertices).all() or faces.ndim != 2 or faces.shape[1:] != (3,)
                or faces.dtype.kind not in 'iu' or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices)
                or target.shape != (len(faces),) or target.dtype.kind != 'b'):
            raise ValueError('Finite native vertices, valid triangles and a target-face mask required')
        wp.init()
        transmitting=(np.zeros(len(faces),bool) if transmitting_faces is None else np.asarray(transmitting_faces))
        if (transmitting.shape!=(len(faces),) or transmitting.dtype.kind!='b'
                or (transmitting&target).any() or transmitting.all()):
            raise ValueError('Optical transmitters must be distinct from the observed physical surface')
        kept=np.flatnonzero(~transmitting)
        self.origin = vertices.mean(0)
        self.device = device
        self.mesh = wp.Mesh(points=wp.array((vertices-self.origin).astype(np.float32), dtype=wp.vec3, device=device),
                            indices=wp.array(faces[kept].astype(np.int32).ravel(), dtype=int, device=device))
        self.face_map=wp.array(kept.astype(np.int32),dtype=int,device=device)
        self.targets = wp.array(target.astype(np.int32), dtype=int, device=device)
        self.target_faces = target.copy()
        self.faces = faces.copy()

    def face_image(self, K, R_world_cam, centre, window, shape, *, near=.05):
        K, rotation, centre = np.asarray(K, float), np.asarray(R_world_cam, float), np.asarray(centre, float)
        if (K.shape != (3, 3) or rotation.shape != (3, 3) or centre.shape != (3,)
                or not all(np.isfinite(v).all() for v in [K, rotation, centre])
                or len(shape) != 2 or min(shape) < 1 or not np.isfinite(near) or near <= 0):
            raise ValueError('Finite calibrated camera and positive image/near plane required')
        x0, y0, _, _, scale = window
        crop = np.array([[scale, 0, -x0*scale], [0, scale, -y0*scale], [0, 0, 1.]])
        inverse = np.linalg.inv(crop@K)
        result = wp.zeros(int(np.prod(shape)), dtype=int, device=self.device)
        wp.launch(_visible, dim=int(np.prod(shape)),
                  inputs=[self.mesh.id, self.face_map, wp.vec3(*(centre-self.origin)),
                          wp.mat33(*inverse.ravel()), wp.mat33(*rotation.ravel()),
                          int(shape[1]), float(near), result], device=self.device)
        return result.numpy().reshape(shape)-1

    def mask(self, *args, **kwargs):
        ids = self.face_image(*args, **kwargs)
        result = ids >= 0
        result[result] &= self.target_faces[ids[result]]
        return result

    def visible_vertices(self, *args, domain=None, **kwargs):
        ids = self.face_image(*args, **kwargs)
        if domain is not None:
            if np.asarray(domain).shape != ids.shape or np.asarray(domain).dtype.kind != 'b':
                raise ValueError('Visibility domain must be a matching independent source mask')
            ids = ids[domain]
        ids = np.unique(ids[ids >= 0])
        ids = ids[self.target_faces[ids]]
        return np.unique(self.faces[ids])
