"""Rasterize audited surfaces that cross the captured camera's near plane."""
import numpy as np

from calibrated_scene_reconstruction.scene.background.mesh_textures import triangle_pixels


def clip_near(polygon, near):
    """Clip in camera space before dividing by depth; retain polygon winding."""
    polygon = np.asarray(polygon, float)
    if (polygon.ndim != 2 or polygon.shape[1] != 3 or len(polygon) < 3
            or not np.isfinite(polygon).all() or not np.isfinite(near) or near <= 0):
        raise ValueError('Finite camera-space polygon and positive near plane required')
    output = []
    for before, after in zip(np.roll(polygon, 1, axis=0), polygon):
        a, b = before[2] >= near, after[2] >= near
        if a != b:
            t = (near-before[2])/(after[2]-before[2])
            output.append(before+t*(after-before))
        if b:
            output.append(after)
    return np.asarray(output, float).reshape(-1, 3)


def depth_buffer(vertices, faces, K, R, centre, shape, near=.05):
    """Perspective-correct nearest depth, including near-crossing large solids."""
    pc = (np.asarray(vertices)-centre)@R
    depth = np.full(shape, np.inf, np.float32)
    for face in faces:
        polygon = clip_near(pc[face], near)
        for i in range(1, len(polygon)-1):
            triangle = polygon[[0, i, i+1]]
            projected = triangle@K.T
            uv = projected[:, :2]/projected[:, 2, None]
            for yy, xx, bary in triangle_pixels(uv, shape):
                value = 1/(bary/triangle[:, 2]).sum(1)
                depth[yy, xx] = np.minimum(depth[yy, xx], value)
    return depth
