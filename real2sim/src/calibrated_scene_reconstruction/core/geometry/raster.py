"""Triangle coverage in image coordinates, shared by scene and physics evidence."""
import numpy as np

from .pixel_coverage import triangle_mask


def legacy_triangle_mask(uv, faces, shape):
    """Historical conservative drawing, for old-result diagnostics only.

    cv2's polygon drawing expands triangle boundaries beyond pixel centres.
    Current geometric evidence must use ``triangle_mask`` instead.
    """
    import cv2

    image = np.zeros(shape, np.uint8)
    points = np.round(np.asarray(uv) * 16).astype(np.int32)
    # Filling them together with fillPoly would cancel overlapping triangles.
    for tri in points[np.asarray(faces)]:
        cv2.fillConvexPoly(image, tri, 1, lineType=cv2.LINE_8, shift=4)
    return image.astype(bool)


def camera_triangle_mask(vertices, faces, K, shape, *, near=.05):
    """Clip triangles to the camera near plane before perspective projection.

    A large support can extend behind a close-up camera. Its visible triangles
    still contribute; one hidden corner must not erase the entire support.
    Vertices are in camera coordinates and K already includes crop/resize.
    """
    return _camera_triangle_mask(vertices, faces, K, shape, near=near, rasterizer=triangle_mask)


def legacy_camera_triangle_mask(vertices, faces, K, shape, *, near=.05):
    """Explicit reproduction of historical conservative source diagnostics."""
    return _camera_triangle_mask(vertices, faces, K, shape, near=near, rasterizer=legacy_triangle_mask)


def _camera_triangle_mask(vertices, faces, K, shape, *, near, rasterizer):
    vertices, faces, K = np.asarray(vertices), np.asarray(faces), np.asarray(K)
    if not np.isfinite(near) or near <= 0:
        raise ValueError('A positive finite near plane is required')
    front = vertices[:, 2] >= near
    if front.all():
        projected = vertices @ K.T
        return rasterizer(projected[:, :2]/projected[:, 2, None], faces, shape)
    visible = front[faces]
    full = faces[visible.all(axis=1)]
    triangles = list(vertices[full])
    for triangle in vertices[faces[visible.any(axis=1) & ~visible.all(axis=1)]]:
        polygon = []
        for a, b in zip(triangle, np.roll(triangle, -1, axis=0)):
            inside_a, inside_b = a[2] >= near, b[2] >= near
            if inside_a:
                polygon.append(a)
            if inside_a != inside_b:
                polygon.append(a + (b-a)*((near-a[2])/(b[2]-a[2])))
        triangles.extend([polygon[0], polygon[i], polygon[i+1]] for i in range(1, len(polygon)-1))
    if not triangles:
        return np.zeros(shape, bool)
    clipped = np.asarray(triangles).reshape(-1, 3)
    projected = clipped @ K.T
    return rasterizer(projected[:, :2]/projected[:, 2, None],
                      np.arange(len(clipped)).reshape(-1, 3), shape)
