"""Backend-independent controls and task measurements for shell/solid diagnostics."""

from __future__ import annotations

import numpy as np

from .deformable_geometry import boundary_triangles, surface_mesh, tetrahedra


def validate_behavior(c):
    if c.get("schema_version") != 1 or c.get("representation") not in {"surface", "volume"}:
        raise ValueError("Use schema 1 and a surface or volume representation")
    frames, fps = c.get("frames", 120), c.get("fps", 24)
    if (
        type(frames) is not int
        or not 2 <= frames <= 1440
        or type(fps) is not int
        or not 1 <= fps <= 120
    ):
        raise ValueError("Choose 2..1440 frames and 1..120 integer fps")
    geometry = c["geometry"]
    points, faces, _ = surface_mesh(
        geometry["vertices"], geometry["triangles"], closed=c["representation"] == "volume"
    )
    if len(points) > 100000 or len(faces) > 200000:
        raise ValueError("Control mesh exceeds the particle/triangle budget")
    if len(points) * frames > 20_000_000:
        raise ValueError("Retained vertex-frame budget exceeded; reduce mesh size or duration")
    if c["representation"] == "surface" and len(np.unique(faces)) != len(points):
        raise ValueError("A shell cannot have unused control vertices")
    if c["representation"] == "volume":
        _, elements, _, _ = tetrahedra(points, geometry["tetrahedra"])
        if len(elements) > 200000 or len(np.unique(elements)) != len(points):
            raise ValueError("Use a bounded volume mesh without unused particles")
        expected = {tuple(sorted(f)) for f in boundary_triangles(elements)}
        if expected != {tuple(sorted(f)) for f in faces}:
            raise ValueError("Visual boundary must match the tetrahedral boundary")
    solver = c.get("solver", {})
    if type(solver.get("full_surface_contact", False)) is not bool:
        raise ValueError("Full-surface contact must be boolean")
    if type(solver.get("self_contact", True)) is not bool:
        raise ValueError("Self contact must be boolean")
    for key in ("self_contact_margin_m", "self_contact_gap_m"):
        if key in solver:
            value = solver[key]
            if (
                isinstance(value, bool)
                or not np.isscalar(value)
                or not np.isfinite(value)
                or value < 0
                or (key == "self_contact_margin_m" and value == 0)
            ):
                raise ValueError("Use a positive self-contact margin and nonnegative detection gap")
    review = c.get("review", {})
    for key, default in (
        ("maximum_stretch", 1.5),
        ("floor_tolerance_m", 0.003),
        ("contact_tolerance_m", 0.001),
    ):
        value = review.get(key, default)
        if not np.isfinite(value) or value < 0 or (key == "maximum_stretch" and value < 1):
            raise ValueError("Use finite review bounds and maximum stretch of at least one")
    for key, default, limit in (("substeps", 12, 128), ("iterations", 16, 64)):
        value = solver.get(key, default)
        if type(value) is not int or not 1 <= value <= limit:
            raise ValueError(f"Invalid solver {key} budget")
    owners = set()
    for grasp in c.get("grasps", []):
        indices = _indices(grasp["indices"], len(points))
        if owners.intersection(indices):
            raise ValueError("A control vertex cannot have conflicting grasps")
        owners.update(indices)
        release = grasp.get("release_seconds")
        if release is not None and (not np.isfinite(release) or release < 0):
            raise ValueError("Release time must be finite and nonnegative")
    names = set()
    for probe in c.get("probes", []):
        if not isinstance(probe.get("id"), str) or not probe["id"] or probe["id"] in names:
            raise ValueError("Probe identifiers must be unique nonempty strings")
        names.add(probe["id"])
        position = np.asarray(probe["position_m"], dtype=float)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("Probe position must be a finite 3D vector")
        size = np.asarray(
            probe.get("size_m") if probe["shape"] == "box" else [probe.get("radius_m")], dtype=float
        )
        if (
            probe["shape"] not in {"box", "sphere"}
            or size.shape != ((3,) if probe["shape"] == "box" else (1,))
            or not np.isfinite(size).all()
            or (size <= 0).any()
        ):
            raise ValueError("Use a box/sphere with positive finite dimensions")
    for control in c.get("probes", []) + c.get("grasps", []):
        keys = np.asarray(control["keyframes"], dtype=float)
        if (
            keys.ndim != 2
            or keys.shape[1] != 4
            or len(keys) < 2
            or not np.isfinite(keys).all()
            or not np.allclose(keys[0], [0, 0, 0, 0])
            or (np.diff(keys[:, 0]) <= 0).any()
        ):
            raise ValueError("Control keys must be ordered [seconds,dx,dy,dz], initially zero")
    names = set()
    for measurement in c.get("measurements", []):
        if (
            not isinstance(measurement.get("id"), str)
            or not measurement["id"]
            or measurement["id"] in names
        ):
            raise ValueError("Measurement identifiers must be unique nonempty strings")
        names.add(measurement["id"])
        _indices(measurement["indices"], len(points))
        axis = np.asarray(measurement["axis"], dtype=float)
        if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-12:
            raise ValueError("Measurement axis must be nonzero and finite")
        for key in (
            "baseline_window_seconds",
            "response_window_seconds",
            "recovery_window_seconds",
        ):
            if key not in measurement and key == "recovery_window_seconds":
                continue
            window = np.asarray(measurement[key], dtype=float)
            if (
                window.shape != (2,)
                or not np.isfinite(window).all()
                or window[0] < 0
                or window[0] > window[1]
                or window[1] > (frames - 1) / fps
                or not (
                    (np.arange(frames) / fps >= window[0]) & (np.arange(frames) / fps <= window[1])
                ).any()
            ):
                raise ValueError("Measurement window must contain retained timestamps")
        for key in ("minimum_response_m", "maximum_residual_m"):
            if key in measurement and (not np.isfinite(measurement[key]) or measurement[key] < 0):
                raise ValueError("Measurement thresholds must be finite and nonnegative")
        for key in ("minimum_final_point_projection_m", "maximum_final_point_projection_m"):
            if key in measurement and not np.isfinite(measurement[key]):
                raise ValueError("Final region projection bounds must be finite")
        if measurement["baseline_window_seconds"][1] > measurement["response_window_seconds"][0]:
            raise ValueError("Baseline must precede the response window")
        if (
            "recovery_window_seconds" in measurement
            and measurement["recovery_window_seconds"][0]
            < measurement["response_window_seconds"][1]
        ):
            raise ValueError("Recovery must follow the response window")
        if "maximum_residual_m" in measurement and "recovery_window_seconds" not in measurement:
            raise ValueError("A residual gate needs a recovery window")
    return c


def _indices(raw, count):
    ids = np.asarray(raw)
    if (
        ids.ndim != 1
        or not len(ids)
        or not np.issubdtype(ids.dtype, np.integer)
        or ids.min() < 0
        or ids.max() >= count
        or len(np.unique(ids)) != len(ids)
    ):
        raise ValueError("Use unique integer control vertex indices")
    return ids.tolist()


def measure_response(positions, times, specifications):
    """Measure signed regional motion relative to a settled, explicitly chosen baseline."""
    points, times = np.asarray(positions), np.asarray(times)
    rows = []
    for spec in specifications:
        axis = np.asarray(spec["axis"], dtype=float)
        axis /= np.linalg.norm(axis)
        trace = points[:, spec["indices"]].mean(axis=1) @ axis

        def window(key, spec=spec, trace=trace):
            lo, hi = spec[key]
            selected = trace[(times >= lo) & (times <= hi)]
            if not len(selected):
                raise ValueError("Measurement window has no retained timestamps")
            return selected

        baseline = float(window("baseline_window_seconds").mean())
        response = float((window("response_window_seconds") - baseline).max())
        residual = (
            float(abs(window("recovery_window_seconds").mean() - baseline))
            if "recovery_window_seconds" in spec
            else None
        )
        failures = []
        if "minimum_response_m" in spec and response < spec["minimum_response_m"]:
            failures.append("insufficient_response")
        if residual is not None and residual > spec.get("maximum_residual_m", np.inf):
            failures.append("excessive_residual")
        final_points = points[-1, spec["indices"]] @ axis
        lower, upper = float(final_points.min()), float(final_points.max())
        if lower < spec.get("minimum_final_point_projection_m", -np.inf):
            failures.append("region_below_final_projection_bound")
        if upper > spec.get("maximum_final_point_projection_m", np.inf):
            failures.append("region_above_final_projection_bound")
        rows.append(
            {
                "id": spec["id"],
                "ok": not failures,
                "failures": failures,
                "peak_response_m": response,
                "final_point_projection_bounds_m": [lower, upper],
                "recovery_residual_m": residual,
                "scope": "Regional projected centroid motion; not force calibration or a contact certificate.",
            }
        )
    return rows
