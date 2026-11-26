"""Shared observed-image geometry; poses are OpenCV world-to-camera in metres."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .artifacts import sha256_file
from .measurement import rigid_transform


def camera_spec(row):
    pose = rigid_transform(row["world_to_camera"])
    intrinsic = np.asarray(row["intrinsics"], dtype=float)
    size = row["image_size"]
    if (
        len(size) != 2
        or any(type(v) is not int or not 8 <= v <= 4096 for v in size)
        or intrinsic.shape != (3, 3)
        or not np.isfinite(intrinsic).all()
        or not np.allclose(intrinsic[2], [0, 0, 1])
        or intrinsic[0, 0] <= 0
        or intrinsic[1, 1] <= 0
        or np.abs(intrinsic).max() > 1e7
        or not np.allclose(np.asarray(row.get("distortion", [0]), dtype=float), 0)
    ):
        raise ValueError("Use undistorted 8..4096 pixel images with pinhole intrinsics")
    if not isinstance(row.get("source"), str) or not row["source"].strip():
        raise ValueError("Each camera needs observation/registration provenance in source")
    return {
        "pose": pose,
        "K": intrinsic,
        "width": size[0],
        "height": size[1],
        "source": row["source"],
        "center": -pose[:3, :3].T @ pose[:3, 3],
    }


def file_path(base, value):
    if not isinstance(value, str) or not value:
        raise ValueError("Supply a file path")
    path = (Path(base) / value).resolve()
    if not path.is_file():
        raise ValueError("Missing input: " + path.name)
    return path


def load_views(request, base, *, required=(), min_views=1):
    if request.get("schema_version") != 1:
        raise ValueError("Use schema_version=1")
    frame = request.get("frame", {})
    if (
        not isinstance(frame.get("source"), str)
        or not frame["source"].strip()
        or type(frame.get("metric_verified")) is not bool
    ):
        raise ValueError(
            "Describe the common coordinate frame/scale source and metric_verified status"
        )
    rows = request.get("views", [])
    if not isinstance(rows, list) or not min_views <= len(rows) <= 100:
        raise ValueError(f"Supply {min_views}..100 posed observations")
    views, hashes, total_pixels = [], {}, 0
    for row in rows:
        view = camera_spec(row)
        total_pixels += view["width"] * view["height"]
        if total_pixels > 32_000_000:
            raise ValueError(
                "Observation budget exceeds 32 million pixels; select or consistently downsample views"
            )
        for key in set(required) | ({"image", "mask", "ignore", "depth"} & set(row)):
            path = file_path(base, row.get(key))
            hashes[str(path)] = sha256_file(path)
            if key == "depth":
                pixels = np.load(path, allow_pickle=False)
                if pixels.ndim != 2 or not np.isfinite(pixels).all() or (pixels < 0).any():
                    raise ValueError(
                        "Depth is a finite nonnegative .npy camera-Z map in metres; zero means unknown"
                    )
            else:
                pixels = cv2.imread(
                    str(path), cv2.IMREAD_COLOR if key == "image" else cv2.IMREAD_GRAYSCALE
                )
                if pixels is None:
                    raise ValueError("Cannot decode observation: " + path.name)
                if key == "image":
                    pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)
            if pixels.shape[:2] != (view["height"], view["width"]):
                raise ValueError(
                    "Camera/image dimensions disagree; update intrinsics when resizing/cropping"
                )
            view[key] = pixels
            view[key + "_path"] = str(path)
        views.append(view)
    return views, hashes


def project(points, camera):
    xyz = np.asarray(points) @ camera["pose"][:3, :3].T + camera["pose"][:3, 3]
    projected = xyz @ camera["K"].T
    uv = projected[:, :2] / np.maximum(xyz[:, 2:3], 1e-12)
    valid = (
        (xyz[:, 2] > 1e-5)
        & (uv[:, 0] >= -0.5)
        & (uv[:, 0] < camera["width"] - 0.5)
        & (uv[:, 1] >= -0.5)
        & (uv[:, 1] < camera["height"] - 0.5)
    )
    return uv, xyz[:, 2], valid


def mesh_arrays(vertices, triangles, uv=None):
    vertices, triangles = np.asarray(vertices, dtype=float), np.asarray(triangles)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not 3 <= len(vertices) <= 300_000
        or not np.isfinite(vertices).all()
        or triangles.ndim != 2
        or triangles.shape[1] != 3
        or not 1 <= len(triangles) <= 100_000
        or triangles.dtype.kind not in "iu"
        or triangles.min() < 0
        or triangles.max() >= len(vertices)
    ):
        raise ValueError(
            "Use finite Nx3 vertices and indexed integer triangle faces within the mesh budget"
        )
    result = {"vertices": vertices, "triangles": triangles.astype(np.int32)}
    if uv is not None:
        uv = np.asarray(uv, dtype=float)
        if (
            uv.shape != (len(triangles), 3, 2)
            or not np.isfinite(uv).all()
            or (uv < 0).any()
            or (uv > 1).any()
        ):
            raise ValueError("UVs must be per-triangle corners, shape Fx3x2, in [0,1]")
        result["uv"] = uv
    return result


def load_mesh(path, *, require_uv=False):
    with np.load(path, allow_pickle=False) as data:
        if {"V", "F"} <= set(data.files) and not {"vertices", "triangles"}.intersection(data.files):
            mesh = mesh_arrays(data["V"], data["F"], data.get("UV"))
        else:
            mesh = mesh_arrays(data["vertices"], data["triangles"], data.get("uv"))
    if require_uv and "uv" not in mesh:
        raise ValueError("Texture baking requires an existing nonoverlapping UV atlas")
    return mesh


def barycentric_pixels(triangle, width, height, *, offset=0.5):
    """Pixel-center samples with barycentric coordinates, in top-left image coordinates."""
    if not np.isfinite(triangle).all():
        raise ValueError("Projection exceeded the supported numeric range")
    low = np.maximum(np.floor(triangle.min(axis=0)), 0)
    high = np.minimum(np.ceil(triangle.max(axis=0)), [width - 1, height - 1])
    if np.any(high < low):
        return None
    low, high = low.astype(int), high.astype(int)
    x, y = np.meshgrid(np.arange(low[0], high[0] + 1), np.arange(low[1], high[1] + 1))
    a, b, c = triangle
    den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(den) < 1e-12:
        return None
    alpha = ((b[1] - c[1]) * (x + offset - c[0]) + (c[0] - b[0]) * (y + offset - c[1])) / den
    beta = ((c[1] - a[1]) * (x + offset - c[0]) + (a[0] - c[0]) * (y + offset - c[1])) / den
    weights = np.stack([alpha, beta, 1 - alpha - beta], axis=-1)
    inside = (weights >= -1e-8).all(axis=-1)
    return y[inside], x[inside], weights[inside]


def render_depth(mesh, camera):
    """Double-sided triangle Z-buffer with perspective depth and near-plane clipping."""
    xyz = mesh["vertices"] @ camera["pose"][:3, :3].T + camera["pose"][:3, 3]
    depth = np.full((camera["height"], camera["width"]), np.inf)
    near = 1e-5
    for indices in mesh["triangles"]:
        polygon = list(xyz[indices])
        clipped = []
        for start, end in zip(polygon, polygon[1:] + polygon[:1], strict=True):
            if start[2] >= near:
                clipped.append(start)
            if (start[2] >= near) != (end[2] >= near):
                clipped.append(start + (end - start) * ((near - start[2]) / (end[2] - start[2])))
        for i in range(1, len(clipped) - 1):
            triangle = np.array([clipped[0], clipped[i], clipped[i + 1]])
            pix = triangle @ camera["K"].T
            projected = pix[:, :2] / triangle[:, 2:3]
            samples = barycentric_pixels(projected, camera["width"], camera["height"], offset=0)
            if samples is None:
                continue
            y, x, weights = samples
            z = 1 / np.maximum((weights / triangle[:, 2]).sum(axis=1), 1e-12)
            depth[y, x] = np.minimum(depth[y, x], z)
    return depth


def write_report(output, result):
    from .artifacts import atomic_json

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "report.json", result)
    return result


def load_request(path):
    path = Path(path).resolve()
    return json.loads(path.read_text()), path.parent
