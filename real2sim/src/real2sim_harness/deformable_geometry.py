"""Geometry evidence shared by thin shells and volumetric deformables."""

from __future__ import annotations

import numpy as np


def surface_mesh(vertices, triangles, *, closed=False):
    points = np.asarray(vertices, dtype=float)
    raw_faces = np.asarray(triangles)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("Surface vertices must be finite 3D points")
    if (
        raw_faces.ndim != 2
        or raw_faces.shape[1] != 3
        or not len(raw_faces)
        or not np.issubdtype(raw_faces.dtype, np.integer)
    ):
        raise ValueError("Use a nonempty integer triangle table")
    if raw_faces.min() < 0 or raw_faces.max() >= len(points):
        raise ValueError("Triangle index outside the vertex table")
    faces = raw_faces.astype(np.int32)
    if len(np.unique(np.sort(faces, axis=1), axis=0)) != len(faces):
        raise ValueError("Duplicate surface triangle")
    tri = points[faces]
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    scale = max(float(np.linalg.norm(np.ptp(points, axis=0))), 1e-9)
    if np.any(area2 <= scale * scale * 1e-12):
        raise ValueError("Degenerate surface triangle")
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    canonical = np.sort(edges, axis=1)
    unique, inverse, counts = np.unique(canonical, axis=0, return_inverse=True, return_counts=True)
    if (counts > 2).any() or (closed and (counts != 2).any()):
        raise ValueError("Volume controls need a closed manifold; shell edges must be manifold")
    votes = np.bincount(inverse, weights=np.where(edges[:, 0] < edges[:, 1], 1, -1))
    if np.any(votes[counts == 2] != 0):
        raise ValueError("Adjacent triangles have inconsistent winding")
    return (
        points,
        faces,
        {
            "vertices": len(points),
            "triangles": len(faces),
            "boundary_edges": int((counts == 1).sum()),
            "area_m2": float(area2.sum() / 2),
            "edges": unique,
        },
    )


def tetrahedra(vertices, elements):
    points = np.asarray(vertices, dtype=float)
    raw = np.asarray(elements)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not np.isfinite(points).all()
        or raw.ndim != 2
        or raw.shape[1] != 4
        or not len(raw)
        or not np.issubdtype(raw.dtype, np.integer)
    ):
        raise ValueError("Use finite vertices and integer tetrahedra")
    if raw.min() < 0 or raw.max() >= len(points):
        raise ValueError("Tetrahedron index outside the vertex table")
    indices = raw.astype(np.int32).copy()
    if len(np.unique(np.sort(indices, axis=1), axis=0)) != len(indices):
        raise ValueError("Duplicate tetrahedron")
    q = points[indices]
    matrices = np.transpose(q[:, 1:] - q[:, :1], (0, 2, 1))
    determinant = np.linalg.det(matrices)
    threshold = max(float(np.linalg.norm(np.ptp(points, axis=0))), 1e-9) ** 3 * 1e-13
    if (np.abs(determinant) <= threshold).any():
        raise ValueError("Degenerate tetrahedron")
    negative = determinant < 0
    indices[negative, :2] = indices[negative, :2][:, ::-1]
    q = points[indices]
    matrices = np.transpose(q[:, 1:] - q[:, :1], (0, 2, 1))
    return points, indices, np.linalg.inv(matrices), np.linalg.det(matrices) / 6


def shell_bending_coefficients(vertices, edge_indices, rigidity_nm):
    """Convert plate rigidity to Newton's stiffness per unit hinge length.

    The barycentric dual width is one third of the adjacent triangle heights.
    Newton multiplies this coefficient by the rest edge length internally.
    """
    points, edges = np.asarray(vertices, dtype=float), np.asarray(edge_indices, dtype=int)
    if not np.isfinite(rigidity_nm) or rigidity_nm < 0 or edges.ndim != 2 or edges.shape[1] != 4:
        raise ValueError("Use finite nonnegative plate rigidity and Newton hinge connectivity")
    result = np.zeros(len(edges))
    valid = (edges >= 0).all(axis=1)
    active = edges[valid]
    if not len(active):
        return result
    if active.max() >= len(points):
        raise ValueError("Hinge index outside the vertex table")
    opposite0, opposite1, start, end = (points[active[:, i]] for i in range(4))
    edge = end - start
    length = np.linalg.norm(edge, axis=1)
    double_area = np.linalg.norm(np.cross(edge, opposite0 - start), axis=1) + np.linalg.norm(
        np.cross(edge, opposite1 - start), axis=1
    )
    if (length < 1e-12).any() or (double_area < 1e-18).any():
        raise ValueError("Degenerate bending hinge")
    result[valid] = rigidity_nm / (double_area / (3 * length))
    return result


def boundary_triangles(elements):
    tet = np.asarray(elements, dtype=np.int32)
    faces = np.concatenate(
        [tet[:, [1, 2, 3]], tet[:, [0, 3, 2]], tet[:, [0, 1, 3]], tet[:, [0, 2, 1]]]
    )
    _, inverse, counts = np.unique(
        np.sort(faces, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    if (counts > 2).any():
        raise ValueError("Nonmanifold tetrahedral face")
    return faces[counts[inverse] == 1]


def tetrahedralize(
    vertices, triangles, *, max_volume_m3, quality=1.4, max_elements=100000, max_surface_edge_m=None
):
    """Tetrahedralize an explicit closed PLC; no primitive-specific shape recipe."""
    from meshpy import tet

    points, faces, _ = surface_mesh(vertices, triangles, closed=True)
    if max_surface_edge_m is not None:
        points, faces = refine_surface(points, faces, max_edge_m=max_surface_edge_m)
    if (
        len(points) > 100000
        or len(faces) > 200000
        or type(max_elements) is not int
        or not 1 <= max_elements <= 200000
    ):
        raise ValueError("Use a bounded tetrahedralization budget")
    if not np.isfinite(max_volume_m3) or max_volume_m3 <= 0 or not 1 <= quality <= 3:
        raise ValueError("Use a positive maximum tet volume and quality from 1 to 3")
    mesh = tet.MeshInfo()
    mesh.set_points(points.tolist())
    mesh.set_facets(faces.tolist())
    result = tet.build(mesh, options=tet.Options(f"pq{quality}"), max_volume=max_volume_m3)
    if not 0 < len(result.elements) <= max_elements:
        raise ValueError(
            f"Tetrahedralization produced {len(result.elements)} elements, outside the 1..{max_elements} budget; reconsider boundary density and mesh quality"
        )
    points, indices, inverse, volumes = tetrahedra(result.points, result.elements)
    boundary = boundary_triangles(indices)
    surface_mesh(points, boundary, closed=True)
    return {
        "vertices": points,
        "tetrahedra": indices,
        "triangles": boundary,
        "rest_inverse": inverse,
        "rest_volumes": volumes,
    }


def refine_surface(vertices, triangles, *, max_edge_m, adaptive=False):
    """Conforming midpoint refinement preserves authored facets and source IDs."""
    points, faces, _ = surface_mesh(vertices, triangles)
    if not np.isfinite(max_edge_m) or max_edge_m <= 0:
        raise ValueError("Choose a positive finite surface edge size")
    if type(adaptive) is not bool:
        raise ValueError("adaptive must be boolean")
    for _ in range(24):
        edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        unique, inverse = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True)
        lengths = np.linalg.norm(points[unique[:, 0]] - points[unique[:, 1]], axis=1)
        if lengths.max() <= max_edge_m:
            return points, faces
        if adaptive:
            points, faces = _refine_marked_edges(points, faces, unique, inverse, lengths > max_edge_m)
            continue
        if len(points) + len(unique) > 100000 or len(faces) * 4 > 200000:
            raise ValueError("Surface refinement exceeds the control mesh budget")
        midpoint = len(points) + inverse.reshape(3, len(faces)).T
        ab, bc, ca = midpoint.T
        a, b, c = faces.T
        faces = np.concatenate(
            [
                np.column_stack([a, ab, ca]),
                np.column_stack([ab, b, bc]),
                np.column_stack([ca, bc, c]),
                np.column_stack([ab, bc, ca]),
            ]
        )
        points = np.concatenate([points, points[unique].mean(axis=1)])
    raise ValueError("Surface refinement did not meet the requested edge size")


def _refine_marked_edges(points, faces, edges, inverse, marked):
    """Split shared edges consistently, leaving unmarked regions untouched."""
    if len(points) + int(marked.sum()) > 100000:
        raise ValueError("Surface refinement exceeds the control mesh budget")
    mapping = np.full(len(edges), -1, dtype=np.int64)
    mapping[marked] = len(points) + np.arange(marked.sum())
    expanded = np.concatenate([points, points[edges[marked]].mean(axis=1)])
    midpoint = mapping[inverse].reshape(3, len(faces)).T
    result = []
    for vertices, mids in zip(faces, midpoint, strict=True):
        count = int((mids >= 0).sum())
        a, b, c = vertices
        ab, bc, ca = mids
        if count == 0:
            result.append([a, b, c])
        elif count == 3:
            result.extend([[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]])
        elif count == 1:
            if bc >= 0:
                a, b, c, ab = b, c, a, bc
            elif ca >= 0:
                a, b, c, ab = c, a, b, ca
            result.extend([[a, ab, c], [ab, b, c]])
        else:
            # Rotate so b is the common vertex of the two split edges.
            if ab < 0:
                a, b, c, ab, bc = b, c, a, bc, ca
            elif bc < 0:
                a, b, c, ab, bc = c, a, b, ca, ab
            result.append([ab, b, bc])
            if np.linalg.norm(expanded[a] - expanded[bc]) <= np.linalg.norm(expanded[ab] - expanded[c]):
                result.extend([[a, ab, bc], [a, bc, c]])
            else:
                result.extend([[a, ab, c], [ab, bc, c]])
    if len(result) > 200000:
        raise ValueError("Surface refinement exceeds the control mesh budget")
    return expanded, np.asarray(result, dtype=np.int32)


def solid_metrics(rest, elements, states):
    points, indices, inverse, volumes = tetrahedra(rest, elements)
    positions = np.asarray(states, dtype=float)
    if (
        positions.ndim != 3
        or positions.shape[1:] != points.shape
        or not np.isfinite(positions).all()
    ):
        raise ValueError("Retained solid states must match the finite control mesh")
    minimum, maximum, stretch, inverted = np.inf, -np.inf, 0.0, 0
    volume_ratios = []
    for position in positions:
        q = position[indices]
        gradient = np.transpose(q[:, 1:] - q[:, :1], (0, 2, 1)) @ inverse
        jacobian = np.linalg.det(gradient)
        singular = np.linalg.svd(gradient, compute_uv=False)
        minimum, maximum = min(minimum, float(jacobian.min())), max(maximum, float(jacobian.max()))
        stretch = max(stretch, float(singular.max()))
        inverted += int((jacobian <= 0).sum())
        volume_ratios.append(float(np.dot(volumes, jacobian) / volumes.sum()))
    return {
        "minimum_jacobian": minimum,
        "maximum_jacobian": maximum,
        "inverted_element_samples": inverted,
        "maximum_principal_stretch": stretch,
        "volume_ratio_min": min(volume_ratios),
        "volume_ratio_max": max(volume_ratios),
        "rest_volume_m3": float(volumes.sum()),
        "frames_checked": len(positions),
    }


def shell_metrics(rest, triangles, states):
    points, faces, info = surface_mesh(rest, triangles)
    positions = np.asarray(states, dtype=float)
    if (
        positions.ndim != 3
        or positions.shape[1:] != points.shape
        or not np.isfinite(positions).all()
    ):
        raise ValueError("Retained shell states must match the finite control mesh")
    edges = info["edges"]
    lengths = np.linalg.norm(points[edges[:, 1]] - points[edges[:, 0]], axis=1)
    tri = points[faces]
    areas = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    edge_ratio = (
        np.linalg.norm(positions[:, edges[:, 1]] - positions[:, edges[:, 0]], axis=2) / lengths
    )
    tri = positions[:, faces]
    area_ratio = (
        np.linalg.norm(np.cross(tri[:, :, 1] - tri[:, :, 0], tri[:, :, 2] - tri[:, :, 0]), axis=2)
        / areas
    )
    worst_frame, worst_edge = np.unravel_index(np.argmax(edge_ratio), edge_ratio.shape)
    return {
        "minimum_area_ratio": float(area_ratio.min()),
        "maximum_area_ratio": float(area_ratio.max()),
        "maximum_edge_stretch": float(edge_ratio.max()),
        "edge_stretch_p95": float(np.quantile(edge_ratio, 0.95)),
        "worst_edge": {
            "frame": int(worst_frame),
            "indices": edges[worst_edge].tolist(),
            "positions_m": positions[worst_frame, edges[worst_edge]].tolist(),
        },
        "frames_checked": len(positions),
        "scope": "Intrinsic stretch and collapse; changed world-space normals alone are not inversions.",
    }


def audit_sphere_contacts(
    positions,
    triangles,
    body_poses,
    probes,
    *,
    closed=False,
    tolerance_m=0.001,
    surface_thickness_m=0.0,
):
    """Check triangle surfaces against physical spheres, beyond particle-only contacts."""
    from .surface_binding import closest_weights

    if not np.isfinite(surface_thickness_m) or surface_thickness_m < 0:
        raise ValueError("Use a finite nonnegative shell thickness")

    faces = np.asarray(triangles, dtype=np.int32)
    worst, witnesses, checked, unchecked = np.inf, [], 0, []
    for body, spec in enumerate(probes):
        if not spec.get("collision", True):
            continue
        if spec["shape"] != "sphere":
            unchecked.append(spec["id"])
            continue
        for frame, points in enumerate(positions):
            tri = points[faces]
            center = body_poses[frame, body, :3]
            _, distances = closest_weights(tri, np.broadcast_to(center, (len(faces), 3)))
            distance = float(np.sqrt(distances.min()))
            if closed:
                relative = tri - center
                a, b, c = relative[:, 0], relative[:, 1], relative[:, 2]
                lengths = np.linalg.norm(relative, axis=2)
                denominator = (
                    np.prod(lengths, axis=1)
                    + (a * b).sum(1) * lengths[:, 2]
                    + (b * c).sum(1) * lengths[:, 0]
                    + (c * a).sum(1) * lengths[:, 1]
                )
                angle = 2 * np.arctan2((a * np.cross(b, c)).sum(1), denominator)
                if abs(angle.sum()) > 2 * np.pi:
                    distance = -distance
            clearance = distance - spec["radius_m"] - surface_thickness_m / 2
            checked += 1
            worst = min(worst, clearance)
            if clearance < -tolerance_m and len(witnesses) < 16:
                witnesses.append({"frame": frame, "probe": spec["id"], "clearance_m": clearance})
    return {
        "ok": not witnesses,
        "sphere_frame_checks": checked,
        "minimum_surface_clearance_m": float(worst) if checked else None,
        "witnesses": witnesses,
        "unchecked_non_sphere_probes": unchecked,
        "surface_thickness_m": surface_thickness_m,
        "scope": "Triangle-sphere checks at retained timestamps, with solid containment; not self-contact or continuous-time certification.",
    }


def _inside_closed_surface(triangles, point):
    relative = triangles - point
    a, b, c = relative[:, 0], relative[:, 1], relative[:, 2]
    lengths = np.linalg.norm(relative, axis=2)
    denominator = (
        np.prod(lengths, axis=1)
        + (a * b).sum(1) * lengths[:, 2]
        + (b * c).sum(1) * lengths[:, 0]
        + (c * a).sum(1) * lengths[:, 1]
    )
    angle = 2 * np.arctan2((a * np.cross(b, c)).sum(1), denominator)
    return abs(angle.sum()) > 2 * np.pi


def triangle_box_separation(triangles, center, half_size):
    """Normalized separating-axis margin; negative means triangle/box overlap.

    Positive values are conservative separation bounds, not exact Euclidean
    distances. Negative values are SAT escape margins, not calibrated contact depth.
    """
    tri = np.asarray(triangles, dtype=float) - np.asarray(center)
    edges = np.roll(tri, -1, axis=1) - tri
    basis = np.broadcast_to(np.eye(3), (len(tri), 3, 3))
    normal = np.cross(edges[:, 0], edges[:, 1])[:, None, :]
    cross = np.cross(edges[:, :, None, :], np.eye(3)[None, None, :, :]).reshape(len(tri), 9, 3)
    axes = np.concatenate([basis, normal, cross], axis=1)
    length = np.linalg.norm(axes, axis=2)
    unit = axes / np.maximum(length[:, :, None], 1e-20)
    projected = np.einsum("mvi,mai->mav", tri, unit)
    extent = np.abs(unit) @ np.asarray(half_size)
    gap = np.maximum(projected.min(axis=2) - extent, -extent - projected.max(axis=2))
    gap[length < 1e-12] = -np.inf
    return gap.max(axis=1)


def audit_box_contacts(
    positions,
    triangles,
    body_poses,
    probes,
    *,
    closed=False,
    tolerance_m=0.001,
    surface_thickness_m=0.0,
):
    """Retained triangle-vs-axis-aligned-box checks with optional shell skin.

    AABBs prune distant triangles; the full 13-axis triangle-box SAT catches
    walls crossed between particles. It does not certify continuous-time contact.
    """
    if not np.isfinite(surface_thickness_m) or surface_thickness_m < 0:
        raise ValueError("Use finite nonnegative shell thickness")
    faces = np.asarray(triangles, dtype=np.int32)
    worst = np.inf
    checked = 0
    witnesses = []
    for body, spec in enumerate(probes):
        if spec["shape"] != "box" or not spec.get("collision", True):
            continue
        half = np.asarray(spec["size_m"]) / 2
        skin = surface_thickness_m / 2
        for frame, points in enumerate(positions):
            pose = np.asarray(body_poses[frame, body])
            if np.linalg.norm(pose[3:6]) > 1e-6 or abs(abs(pose[6]) - 1) > 1e-6:
                raise ValueError("Axis-aligned box audit cannot silently ignore probe rotation")
            center = pose[:3]
            tri = np.asarray(points)[faces]
            low, high = tri.min(axis=1), tri.max(axis=1)
            delta = np.maximum(np.maximum(low - (center + half), (center - half) - high), 0)
            bounds = np.linalg.norm(delta, axis=1)
            near = bounds <= skin + tolerance_m
            clearance = bounds - skin
            if near.any():
                clearance[near] = triangle_box_separation(tri[near], center, half) - skin
            k = int(clearance.argmin())
            value = float(clearance[k])
            checked += 1
            contained = bool(closed and _inside_closed_surface(tri, center))
            if contained:
                value = min(value, -float(half.min()))
            worst = min(worst, value)
            if (value < -tolerance_m or contained) and len(witnesses) < 16:
                witnesses.append(
                    {
                        "frame": frame,
                        "probe": spec["id"],
                        "triangle": k,
                        "sat_clearance_bound_m": value,
                        "box_center_inside_solid": contained,
                    }
                )
    return {
        "ok": not witnesses,
        "box_frame_checks": checked,
        "minimum_surface_clearance_bound_m": float(worst) if checked else None,
        "witnesses": witnesses,
        "surface_thickness_m": surface_thickness_m,
        "scope": "Retained triangle/axis-aligned-box SAT with conservative skin offset and closed-solid center containment. Not exact Euclidean depth, self-contact or continuous-time certification.",
    }
