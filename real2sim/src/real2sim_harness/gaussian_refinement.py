"""Depth-tested propagation of restored views; estimated geometry stays explicit."""

from __future__ import annotations

import cv2
import numpy as np

from .posed_observations import project


def valid_depth(depth, camera, *, discontinuity_fraction=0.1):
    depth = np.asarray(depth, dtype=float)
    if (
        depth.shape != (camera["height"], camera["width"])
        or not np.isfinite(depth).all()
        or (depth < 0).any()
    ):
        raise ValueError(
            "Use finite nonnegative camera-Z depth matching its camera; zero means unknown"
        )
    valid = depth > 0
    low = cv2.erode(depth.astype(np.float32), np.ones((3, 3), np.uint8))
    high = cv2.dilate(depth.astype(np.float32), np.ones((3, 3), np.uint8))
    valid &= (high - low) <= np.maximum(depth * discontinuity_fraction, 1e-6)
    return depth, valid


def warp_rgb(
    source_rgb,
    source_depth,
    source_camera,
    target_depth,
    target_camera,
    *,
    absolute_tolerance_m=0.02,
    relative_tolerance=0.02,
):
    """Backward warp using both depth maps; reject disocclusion and depth edges."""
    if (
        not np.isfinite(absolute_tolerance_m)
        or absolute_tolerance_m <= 0
        or not np.isfinite(relative_tolerance)
        or not 0 <= relative_tolerance <= 0.2
    ):
        raise ValueError(
            "Use positive absolute depth tolerance and relative tolerance from 0 to .2"
        )
    source_depth, source_valid = valid_depth(source_depth, source_camera)
    target_depth, target_valid = valid_depth(target_depth, target_camera)
    source_rgb = np.asarray(source_rgb)
    if source_rgb.shape != (*source_depth.shape, 3) or source_rgb.dtype != np.uint8:
        raise ValueError("Source RGB must be an uint8 image matching source depth")
    y, x = np.indices(target_depth.shape)
    rays = (
        np.stack([x, y, np.ones_like(x)], axis=-1).reshape(-1, 3)
        @ np.linalg.inv(target_camera["K"]).T
    )
    points_camera = rays * target_depth.reshape(-1, 1)
    world = (points_camera - target_camera["pose"][:3, 3]) @ target_camera["pose"][:3, :3]
    uv, projected_z, in_frame = project(world, source_camera)
    map_x = uv[:, 0].reshape(target_depth.shape).astype(np.float32)
    map_y = uv[:, 1].reshape(target_depth.shape).astype(np.float32)
    sampled_z = cv2.remap(
        source_depth.astype(np.float32),
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    sampled_valid = (
        cv2.remap(
            source_valid.astype(np.uint8),
            map_x,
            map_y,
            cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        > 0
    )
    z = projected_z.reshape(target_depth.shape)
    valid = target_valid & in_frame.reshape(target_depth.shape) & sampled_valid
    valid &= np.abs(sampled_z - z) <= absolute_tolerance_m + relative_tolerance * z
    warped = cv2.remap(
        source_rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    warped[~valid] = 0
    return warped, valid


def propagate_anchors(anchors, target_depth, target_camera, **tolerances):
    """Keep the earliest consistent anchor; generated views need not agree independently."""
    result = np.zeros((*target_depth.shape, 3), np.uint8)
    coverage = np.zeros(target_depth.shape, bool)
    contributions = []
    for anchor in anchors:
        rgb, valid = warp_rgb(
            anchor["rgb"],
            anchor["depth"],
            anchor["camera"],
            target_depth,
            target_camera,
            **tolerances,
        )
        new = valid & ~coverage
        result[new] = rgb[new]
        coverage |= valid
        contributions.append(int(new.sum()))
    return (
        result,
        coverage,
        {
            "covered_pixels": int(coverage.sum()),
            "coverage_fraction": float(coverage.mean()),
            "new_pixels_per_anchor": contributions,
            "geometry_source": "supplied depth; Gaussian-rendered depths are model estimates, not real measurements",
        },
    )
