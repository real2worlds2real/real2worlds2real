"""Calibrated scene views in the existing OpenCV observation convention."""

from __future__ import annotations

import numpy as np

from .posed_observations import camera_spec


def calibrated_view(record):
    if set(record) - {"image_size", "intrinsics", "world_to_camera", "source", "distortion"}:
        raise ValueError("Unknown calibrated camera fields")
    c = camera_spec(record)
    k, rotation, eye = c["K"], c["pose"][:3, :3], c["center"]
    if abs(k[0, 1]) > 1e-10 or abs(k[1, 0]) > 1e-10:
        raise ValueError("Scene renderer requires zero-skew pinhole intrinsics")
    calibration = {
        "image_size": [c["width"], c["height"]],
        "intrinsics": k.tolist(),
        "world_to_camera": c["pose"].tolist(),
        "source": c["source"],
    }
    return {
        "eye_m": eye.tolist(),
        "target_m": (eye + rotation[2]).tolist(),
        "up": (-rotation[1]).tolist(),
        "horizontal_fov_deg": float(np.degrees(2 * np.arctan(c["width"] / (2 * k[0, 0])))),
        "calibration": calibration,
    }


def raster(calibration, width=None, height=None):
    """Preserve the input raster by default; explicit resizing scales K, never crops."""
    c = camera_spec(calibration)
    w, h = c["width"], c["height"]
    if any(v is not None and (type(v) is not int or not 8 <= v <= 4096) for v in (width, height)):
        raise ValueError("Camera output dimensions must be integers in 8..4096")
    if width is None and height is not None:
        width = round(w * height / h)
    if height is None and width is not None:
        height = round(h * width / w)
    width = w if width is None else width
    height = h if height is None else height
    if any(type(v) is not int or not 8 <= v <= 4096 for v in (width, height)):
        raise ValueError("Camera output dimensions must be integers in 8..4096")
    k = np.diag([width / w, height / h, 1.0]) @ c["K"]
    return width, height, k


def author_intrinsics(camera, k, width, height):
    """Portable USD projection; runtime RTX additionally needs its native lens API."""
    focal = 18.0
    camera.CreateFocalLengthAttr(focal)
    camera.CreateHorizontalApertureAttr(float(width * focal / k[0, 0]))
    camera.CreateVerticalApertureAttr(float(height * focal / k[1, 1]))
    camera.CreateHorizontalApertureOffsetAttr(float((width / 2 - k[0, 2]) * focal / k[0, 0]))
    camera.CreateVerticalApertureOffsetAttr(float((k[1, 2] - height / 2) * focal / k[1, 1]))


def configure_rtx(camera, calibration, width=None, height=None):
    """Apply native OpenCV projection, retain/read back the effective calibration."""
    from pxr import Gf

    w, h, k = raster(calibration, width, height)
    prim = camera.GetPrim()
    schema = "OmniLensDistortionOpenCvPinholeAPI"
    if not prim.ApplyAPI(schema):
        raise ValueError("Renderer does not expose the native OpenCV pinhole camera API")
    prim.GetAttribute("omni:lensdistortion:model").Set("opencvPinhole")
    prefix = "omni:lensdistortion:opencvPinhole:"
    values = {
        "cx": float(k[0, 2]),
        "cy": float(k[1, 2]),
        "fx": float(k[0, 0]),
        "fy": float(k[1, 1]),
        "imageSize": Gf.Vec2i(w, h),
    }
    for key in ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4"):
        values[key] = 0.0
    for key, value in values.items():
        attr = prim.GetAttribute(prefix + key)
        if not attr or not attr.Set(value) or not np.allclose(attr.Get(), value):
            raise ValueError("Camera calibration readback failed: " + key)
    # RTX's native lens mapping uses a centered square-pixel base projection.
    # Keep the calibrated projection in its native attributes, without applying
    # non-square pixels a second time through the base aperture ratio.
    base_k = np.array([[k[0, 0], 0.0, w / 2], [0.0, k[0, 0], h / 2], [0.0, 0.0, 1.0]])
    author_intrinsics(camera, base_k, w, h)
    return {
        "model": prim.GetAttribute("omni:lensdistortion:model").Get(),
        "image_size": [w, h],
        "intrinsics": k.tolist(),
        "source": calibration["source"],
        "world_to_camera": calibration["world_to_camera"],
        "visual_acceptance": False,
    }


def render_plan(calibration, width=None, height=None):
    """Map square-pixel RTX samples to the requested calibrated image rays.

    Oversampling covers the entire requested frustum; no missing border is filled
    or scene geometry moved. Keep raw images alongside resampled evidence.
    """
    w, h, k = raster(calibration, width, height)
    focal = float(max(k[0, 0], k[1, 1]))
    sx, sy = focal / k[0, 0], focal / k[1, 1]
    rw, rh = int(np.ceil(w * sx)) + 4, int(np.ceil(h * sy)) + 4
    rw += rw % 2
    rh += rh % 2
    if max(rw, rh) > 4096:
        raise ValueError(
            "Calibrated sensor oversampling exceeds 4096 pixels; request a smaller output"
        )
    rk = np.array([[focal, 0, 2 + sx * k[0, 2]], [0, focal, 2 + sy * k[1, 2]], [0, 0, 1.0]])
    sensor = {**calibration, "image_size": [rw, rh], "intrinsics": rk.tolist()}
    return {
        "output_size": [w, h],
        "output_intrinsics": k.tolist(),
        "sensor": sensor,
        "pixel_mapping": "same pinhole rays; linear sampling at pixel centers",
        "scale_xy": [float(sx), float(sy)],
        "offset_xy": [2.0, 2.0],
    }


def resample_rgb(image, plan):
    import cv2

    w, h = plan["output_size"]
    expected = tuple(reversed(plan["sensor"]["image_size"]))
    if image.shape[:2] != expected:
        raise ValueError("Raw camera raster differs from retained sensor calibration")
    y, x = np.mgrid[:h, :w].astype(np.float32)
    sx, sy = plan["scale_xy"]
    ox, oy = plan["offset_xy"]
    return cv2.remap(image, (x + 0.5) * sx + ox - 0.5, (y + 0.5) * sy + oy - 0.5, cv2.INTER_LINEAR)
