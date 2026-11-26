"""Temporal diagnostics supplement geometric and finite-state simulation checks."""

from __future__ import annotations

import numpy as np


def surface_trajectory_metrics(positions, times, *, quiet_start=None, speed_tolerance=None):
    points, times = np.asarray(positions), np.asarray(times, dtype=float)
    if (
        points.ndim != 3
        or points.shape[-1] != 3
        or len(points) != len(times)
        or len(times) < 3
        or not np.isfinite(points).all()
        or not np.isfinite(times).all()
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError(
            "A finite, ordered particle trajectory with at least three samples is required"
        )
    if speed_tolerance is not None and (not np.isfinite(speed_tolerance) or speed_tolerance <= 0):
        raise ValueError("Quiet-hold speed tolerance must be positive")
    dt = np.diff(times)
    velocity = np.diff(points, axis=0) / dt[:, None, None]
    speed = np.linalg.norm(velocity, axis=-1)
    acceleration = np.diff(velocity, axis=0) / ((dt[1:] + dt[:-1]) / 2)[:, None, None]
    result = {
        "speed_p95_m_s": float(np.quantile(speed, 0.95)),
        "speed_max_m_s": float(speed.max()),
        "acceleration_p95_m_s2": float(np.quantile(np.linalg.norm(acceleration, axis=-1), 0.95)),
        "scope": "Temporal motion diagnostics; visual realism and physical calibration require separate evidence.",
    }
    if quiet_start is not None:
        if not np.isfinite(quiet_start):
            raise ValueError("Quiet-hold start time must be finite")
        selected = times[:-1] >= quiet_start
        if selected.sum() < 2:
            raise ValueError("The quiet-hold window needs at least two sampled velocity intervals")
        value = float(np.quantile(speed[selected], 0.95))
        result["quiet_hold"] = {
            "start_seconds": float(quiet_start),
            "speed_p95_m_s": value,
            "speed_tolerance_m_s": speed_tolerance,
            "ok": None if speed_tolerance is None else value <= speed_tolerance,
            "assumption": "Caller declares an unforced hold with stationary grasp constraints.",
        }
    elif speed_tolerance is not None:
        raise ValueError("A quiet speed limit requires an explicit unforced-hold window")
    return result
