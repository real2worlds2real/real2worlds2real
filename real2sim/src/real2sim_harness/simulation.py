"""Backend-independent metrics for exported asset verification."""

from __future__ import annotations

import math
from typing import Any


def check_joint_motion(
    *,
    name: str,
    initial: list[float],
    target: list[float],
    final: list[float],
    joint_index: int,
    peak_other_drift: float,
    tracking_tolerance: float = 0.04,
    drift_tolerance: float = 0.03,
) -> dict[str, Any]:
    finite = all(math.isfinite(x) for row in (initial, target, final) for x in row)
    requested = abs(target[joint_index] - initial[joint_index])
    displacement = abs(final[joint_index] - initial[joint_index])
    error = abs(final[joint_index] - target[joint_index])
    failures = []
    if not finite:
        failures.append("non_finite_state")
    if error > tracking_tolerance or not math.isfinite(error):
        failures.append("target_not_reached")
    if requested > 0.01 and displacement < requested * 0.5:
        failures.append("joint_did_not_move")
    if peak_other_drift > drift_tolerance or not math.isfinite(peak_other_drift):
        failures.append("uncommanded_joint_drift")
    return {
        "name": name, "ok": not failures, "failures": failures,
        "initial": initial, "target": target, "final": final,
        "tracking_error_rad": error, "displacement_rad": displacement,
        "peak_other_drift_rad": peak_other_drift,
        "tracking_tolerance_rad": tracking_tolerance,
        "drift_tolerance_rad": drift_tolerance,
    }


def check_surface_motion(
    *,
    finite: bool,
    min_z: float,
    ground_z: float,
    pin_count: int,
    pin_error_max: float,
    displacement_max: float,
    edge_ratio_p95: float,
    edge_ratio_min: float,
    pin_tolerance: float = 0.001,
    penetration_tolerance: float = 0.01,
) -> dict[str, Any]:
    failures = []
    if not finite:
        failures.append("non_finite_state")
    if not math.isfinite(min_z) or min_z < ground_z - penetration_tolerance:
        failures.append("ground_penetration")
    if pin_count and (not math.isfinite(pin_error_max) or pin_error_max > pin_tolerance):
        failures.append("pins_not_enforced")
    if not math.isfinite(edge_ratio_p95) or edge_ratio_p95 > 1.5:
        failures.append("excessive_stretch")
    if not math.isfinite(edge_ratio_min) or edge_ratio_min < 0.05:
        failures.append("collapsed_edges")
    if not math.isfinite(displacement_max) or displacement_max < 0.001:
        failures.append("no_motion")
    return {
        "ok": not failures, "failures": failures, "finite": finite,
        "min_z_m": min_z, "ground_z_m": ground_z,
        "pin_count": pin_count, "pin_error_max_m": pin_error_max,
        "displacement_max_m": displacement_max,
        "edge_length_ratio_p95": edge_ratio_p95,
        "edge_length_ratio_min": edge_ratio_min,
        "pin_tolerance_m": pin_tolerance,
        "penetration_tolerance_m": penetration_tolerance,
    }
