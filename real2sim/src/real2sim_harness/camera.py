from __future__ import annotations

from typing import Any


def solve_pnp(
    object_points: list[list[float]],
    image_points: list[list[float]],
    *,
    width: int,
    height: int,
    fx: float,
    fy: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
    distortion: list[float] | None = None,
) -> dict[str, Any]:
    """Estimate a Blender camera from metric 3D points and image pixels."""
    import cv2
    import numpy as np

    if len(object_points) != len(image_points) or len(object_points) < 4:
        raise ValueError("solvePnP needs at least four paired 3D and 2D points")
    if width <= 0 or height <= 0 or fx <= 0 or (fy is not None and fy <= 0):
        raise ValueError("Image dimensions and focal lengths must be positive")
    fy = fx if fy is None else fy
    cx = width / 2 if cx is None else cx
    cy = height / 2 if cy is None else cy
    points_3d = np.asarray(object_points, dtype=np.float64)
    points_2d = np.asarray(image_points, dtype=np.float64)
    camera_matrix = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    coefficients = np.asarray(distortion or [], dtype=np.float64)
    if (points_3d.shape != (len(object_points), 3) or points_2d.shape != (len(image_points), 2)
            or not np.isfinite(points_3d).all() or not np.isfinite(points_2d).all()
            or not np.isfinite(camera_matrix).all() or not np.isfinite(coefficients).all()
            or coefficients.ndim != 1 or len(coefficients) not in {0, 4, 5, 8, 12, 14}
            or np.linalg.matrix_rank(points_3d - points_3d.mean(axis=0)) < 2):
        raise ValueError('Invalid or collinear correspondences, intrinsics or distortion')
    ok, rvec, tvec = cv2.solvePnP(
        points_3d,
        points_2d,
        camera_matrix,
        coefficients,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("OpenCV solvePnP did not converge")
    rotation, _ = cv2.Rodrigues(rvec)
    if not np.isfinite(tvec).all() or np.any((points_3d @ rotation.T + tvec[:, 0])[:, 2] <= 0):
        raise ValueError('Camera solution has nonfinite values or points behind the camera')
    world_to_cv = np.eye(4)
    world_to_cv[:3, :3] = rotation
    world_to_cv[:3, 3] = tvec[:, 0]
    cv_to_blender_camera = np.diag([1.0, -1.0, -1.0, 1.0])
    camera_to_world = np.linalg.inv(cv_to_blender_camera @ world_to_cv)
    projected, _ = cv2.projectPoints(
        points_3d, rvec, tvec, camera_matrix, coefficients
    )
    residual = projected[:, 0, :] - points_2d
    errors = np.linalg.norm(residual, axis=1)
    sensor_width_mm = 36.0
    if not 1 / 200 <= fx / fy <= 200:
        raise ValueError('Focal-length ratio is outside Blender pixel-aspect limits')
    return {
        "schema_version": 1,
        "source": "opencv_pnp",
        "image_size": [width, height],
        "intrinsics": {
            "fx": float(fx),
            "fy": float(fy),
            "cx": float(cx),
            "cy": float(cy),
            "distortion": coefficients.tolist(),
        },
        "blender": {
            "matrix_world": camera_to_world.tolist(),
            "sensor_width_mm": sensor_width_mm,
            "lens_mm": float(fx * sensor_width_mm / width),
            "shift_x": float((width / 2 - cx) / width),
            "shift_y": float((cy - height / 2) / width * fx / fy),
            "pixel_aspect_x": float(max(1, fy / fx)),
            "pixel_aspect_y": float(max(1, fx / fy)),
            "sensor_fit": "HORIZONTAL",
        },
        "world_to_camera_opencv": world_to_cv[:3].tolist(),
        "reprojection": {
            "mean_px": float(errors.mean()),
            "max_px": float(errors.max()),
            "per_point_px": errors.tolist(),
        },
    }
