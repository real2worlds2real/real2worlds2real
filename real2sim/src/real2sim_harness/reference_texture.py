"""Visibility-aware projection of posed observed photographs into an existing UV atlas."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .artifacts import sha256_file
from .posed_observations import (
    barycentric_pixels,
    file_path,
    load_mesh,
    load_views,
    project,
    render_depth,
    write_report,
)


def unwrap(mesh):
    """Use the mature xatlas backend, retaining the original geometry/face order."""
    try:
        import xatlas
    except ImportError as error:
        raise ValueError(
            'Automatic UV needs the texture extra: pip install -e ".[texture]"; existing UVs need no xatlas'
        ) from error
    mapping, faces, coordinates = xatlas.parametrize(
        mesh["vertices"].astype(np.float32), mesh["triangles"].astype(np.uint32)
    )
    mapped = np.asarray(mapping)[faces]
    if np.array_equal(mapped, mesh["triangles"]):
        uv = np.asarray(coordinates)[faces]
    else:
        from collections import defaultdict, deque

        original = defaultdict(deque)
        for i, triangle in enumerate(mesh["triangles"]):
            original[tuple(sorted(triangle.tolist()))].append(i)
        uv = np.empty((len(mesh["triangles"]), 3, 2), dtype=float)
        for triangle, chart in zip(mapped, np.asarray(coordinates)[faces], strict=True):
            key = tuple(sorted(triangle.tolist()))
            if not original[key]:
                raise ValueError(
                    "UV backend changed surface topology; use explicit Blender UV preparation"
                )
            i = original[key].popleft()
            for j, vertex in enumerate(mesh["triangles"][i]):
                locations = np.flatnonzero(triangle == vertex)
                if len(locations) != 1:
                    raise ValueError("Remove degenerate faces before UV preparation")
                uv[i, j] = chart[locations[0]]
    return {**mesh, "uv": uv}


def complete_atlas(rgb, known):
    """Optional push/pull completion; observed coverage is retained separately."""
    if not known.any():
        return None
    levels = [(rgb.astype(np.float32), known.astype(np.float32))]
    while min(levels[-1][1].shape) > 1:
        image, weight = levels[-1]
        size = (max(1, (image.shape[1] + 1) // 2), max(1, (image.shape[0] + 1) // 2))
        w = cv2.resize(weight, size, interpolation=cv2.INTER_AREA)
        weighted = cv2.resize(image * weight[:, :, None], size, interpolation=cv2.INTER_AREA)
        levels.append((weighted / np.maximum(w[:, :, None], 1e-12), w))
    result = levels[-1][0]
    for image, weight in reversed(levels[:-1]):
        upsampled = cv2.resize(result, image.shape[1::-1], interpolation=cv2.INTER_LINEAR)
        result = np.where((weight > 0)[:, :, None], image, upsampled)
    return np.rint(np.clip(result, 0, 255)).astype(np.uint8)


def srgb_to_linear(rgb):
    encoded = rgb / 255.0
    return np.where(encoded <= 0.04045, encoded / 12.92, ((encoded + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(rgb):
    return (
        np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.maximum(rgb, 0) ** (1 / 2.4) - 0.055)
        * 255
    )


def atlas_samples(mesh, resolution):
    points = np.zeros((resolution, resolution, 3), np.float32)
    normals = np.zeros_like(points)
    occupied = np.zeros((resolution, resolution), bool)
    for indices, uv in zip(mesh["triangles"], mesh["uv"], strict=True):
        triangle = mesh["vertices"][indices]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        length = np.linalg.norm(normal)
        if length <= 1e-12:
            continue
        pixel_triangle = uv * [resolution, -resolution] + [0, resolution]
        samples = barycentric_pixels(pixel_triangle, resolution, resolution)
        if samples is None:
            continue
        y, x, weights = samples
        values = weights @ triangle
        previous = occupied[y, x]
        if previous.any() and np.any(
            np.linalg.norm(points[y[previous], x[previous]] - values[previous], axis=1) > 1e-5
        ):
            raise ValueError(
                "Overlapping UV islands map different surface positions to the same texel; unwrap first"
            )
        points[y, x] = values
        normals[y, x] = normal / length
        occupied[y, x] = True
    if not occupied.any():
        raise ValueError("No nondegenerate UV triangles cover atlas texels")
    return points[occupied], normals[occupied], occupied


def bake(request, base, output):
    views, hashes = load_views(request, base, required=("image",))
    path = file_path(base, request["mesh"])
    hashes[str(path)] = sha256_file(path)
    mesh = load_mesh(path)
    uv_mode = request.get("uv_mode", "auto")
    if uv_mode not in ("auto", "existing", "unwrap"):
        raise ValueError("uv_mode must be auto, existing or unwrap")
    generated_uv = uv_mode == "unwrap" or ("uv" not in mesh and uv_mode == "auto")
    if generated_uv:
        mesh = unwrap(mesh)
    elif "uv" not in mesh:
        raise ValueError("No existing UV atlas; choose uv_mode=auto or prepare UVs in Blender")
    resolution = request.get("resolution", 512)
    erosion = request.get("mask_erode_px", 1)
    tolerance = float(request.get("visibility_tolerance_m", 0.003))
    if (
        type(resolution) is not int
        or not 32 <= resolution <= 2048
        or type(erosion) is not int
        or not 0 <= erosion <= 16
        or not np.isfinite(tolerance)
        or tolerance <= 0
    ):
        raise ValueError(
            "Use resolution=32..2048, mask_erode_px=0..16 and positive visibility tolerance"
        )
    points, normals, occupied = atlas_samples(mesh, resolution)
    zbuffers, masks, sharpness = [], [], []
    for view in views:
        zbuffers.append(render_depth(mesh, view))
        mask = view.get("mask", np.ones((view["height"], view["width"]), np.uint8)) > 0
        if erosion:
            mask = cv2.erode(mask.astype(np.uint8), np.ones((2 * erosion + 1,) * 2, np.uint8)) > 0
        if "ignore" in view:
            mask &= view["ignore"] == 0
        masks.append(mask)
        gray = cv2.cvtColor(view["image"], cv2.COLOR_RGB2GRAY)
        # Evaluate the target region, not a sharp but irrelevant background.
        region = np.isfinite(zbuffers[-1]) & mask
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        sharpness.append(float(np.var(laplacian[region])) if region.any() else 0.0)
    sharpness = np.clip(np.asarray(sharpness) / max(max(sharpness), 1), 0.1, 1)
    result = np.zeros((len(points), 3), np.float32)
    counts = np.zeros(len(points), np.uint16)
    disagreements = np.zeros(len(points), np.float32)
    color_threshold = float(request.get("color_outlier_threshold", 80))
    if not 1 <= color_threshold <= 442:
        raise ValueError("color_outlier_threshold must be an RGB distance from 1 to 442")
    per_view = np.zeros(len(views), int)
    # cv2.remap requires each image/map dimension to be strictly below
    # SHRT_MAX (32767), including this one-row batch of atlas samples.
    batch_size = 32766
    for start in range(0, len(points), batch_size):
        xyz, norm = points[start : start + batch_size], normals[start : start + batch_size]
        colors = np.zeros((len(views), len(xyz), 3), np.float32)
        weights = np.zeros((len(views), len(xyz)), np.float32)
        for i, view in enumerate(views):
            uv, z, valid = project(xyz, view)
            pixels = np.rint(np.clip(uv, [0, 0], [view["width"] - 1, view["height"] - 1])).astype(
                int
            )
            x, y = pixels.T
            nearest = zbuffers[i][y, x]
            # Compare at the same discrete image ray. Query points on a sloped
            # face can differ substantially in Z from the nearest pixel center.
            camera_normals = norm @ view["pose"][:3, :3].T
            camera_points = xyz @ view["pose"][:3, :3].T + view["pose"][:3, 3]
            rays_at_pixels = np.column_stack([x, y, np.ones(len(x))]) @ np.linalg.inv(view["K"]).T
            denominator = np.einsum("ij,ij->i", camera_normals, rays_at_pixels)
            plane_offset = np.einsum("ij,ij->i", camera_normals, camera_points)
            expected_z = np.divide(
                plane_offset,
                denominator,
                out=np.full_like(z, np.inf),
                where=np.abs(denominator) > 1e-10,
            )
            valid &= np.isfinite(nearest) & np.isfinite(expected_z) & (expected_z > 0)
            valid &= np.abs(expected_z - nearest) <= tolerance
            valid &= masks[i][y, x]
            if "depth" in view:
                observed = view["depth"][y, x]
                # Unknown depth supplies no positive visibility evidence; mesh self-visibility still applies.
                valid &= (observed == 0) | (expected_z <= observed + tolerance)
            rays = view["center"] - xyz
            lengths = np.linalg.norm(rays, axis=1)
            rays /= np.maximum(lengths[:, None], 1e-12)
            cosine = np.einsum("ij,ij->i", norm, rays)
            if request.get("double_sided", False):
                cosine = np.abs(cosine)
            valid &= cosine > 0.05
            colors[i] = cv2.remap(
                view["image"],
                uv[:, 0].astype(np.float32).reshape(1, -1),
                uv[:, 1].astype(np.float32).reshape(1, -1),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            ).reshape(-1, 3)
            weights[i] = valid * np.maximum(cosine, 0) ** 2 * sharpness[i]
            per_view[i] += int(valid.sum())
        observed = weights > 0
        # Weighted per-channel median keeps a sharp real observation instead of averaging unrelated occluders.
        median = np.zeros((len(xyz), 3), np.float32)
        for channel in range(3):
            order = np.argsort(colors[:, :, channel], axis=0)
            ordered_weights = np.take_along_axis(weights, order, axis=0)
            crossing = np.cumsum(ordered_weights, axis=0) >= weights.sum(axis=0)[None, :] / 2
            indices = np.argmax(crossing, axis=0)
            median[:, channel] = np.take_along_axis(colors[:, :, channel], order, axis=0)[
                indices, np.arange(len(xyz))
            ]
        residual = np.linalg.norm(colors - median[None, :, :], axis=2)
        weights *= residual <= color_threshold
        totals = weights.sum(axis=0)
        blended = (srgb_to_linear(colors) * weights[:, :, None]).sum(axis=0) / np.maximum(
            totals[:, None], 1e-12
        )
        result[start : start + len(xyz)] = linear_to_srgb(blended)
        counts[start : start + len(xyz)] = (weights > 0).sum(axis=0)
        disagreements[start : start + len(xyz)] = (residual * observed).sum(axis=0) / np.maximum(
            observed.sum(axis=0), 1
        )
    texture = np.zeros((resolution, resolution, 4), np.uint8)
    texture[occupied, :3] = np.rint(np.clip(result, 0, 255)).astype(np.uint8)
    texture[occupied, 3] = np.where(counts > 0, 255, 0)
    coverage = np.zeros((resolution, resolution), np.uint16)
    coverage[occupied] = counts
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(texture).save(output / "texture.png")
    completed = None
    if request.get("fill_unseen", False):
        completed = complete_atlas(texture[:, :, :3], coverage > 0)
        if completed is not None:
            Image.fromarray(completed).save(output / "texture_completed.png")
    # Unseen texels stay transparent. Filling them would be generated appearance, not observed texture.
    Image.fromarray(np.where(coverage > 0, 255, 0).astype(np.uint8)).save(output / "coverage.png")
    np.savez_compressed(
        output / "coverage.npz",
        view_count=coverage,
        occupied=occupied,
        observed_color_disagreement=disagreements,
    )
    np.savez_compressed(output / "textured_mesh.npz", **mesh)
    return write_report(
        output,
        {
            "schema_version": 1,
            "ok": True,
            "operation": "reference_texture",
            "frame": request["frame"],
            "input_hashes": hashes,
            "texture": "texture.png",
            "mesh": "textured_mesh.npz",
            "uv_backend": "xatlas" if generated_uv else "supplied_uv",
            "completed_texture": "texture_completed.png" if completed is not None else None,
            "completion_is_inferred": completed is not None,
            "preview_images": ["texture.png", "coverage.png"],
            "surface_texels": len(points),
            "observed_texels": int((counts > 0).sum()),
            "observed_fraction": float((counts > 0).mean()),
            "per_view_visible_texels": per_view.tolist(),
            "limits": [
                "Requires correct poses, geometry and a nonoverlapping UV atlas.",
                "Self-occlusion is tested against this mesh; external occlusion needs masks/ignore/depth evidence.",
                "Photographs include lighting and reflections; this is not recovered intrinsic albedo or calibrated PBR.",
                "Unobserved texels remain transparent, not evidence-backed hidden textures.",
                "Review RGB seams/coverage and import into editable source; this does not certify physics.",
            ],
        },
    )
