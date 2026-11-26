"""Optional bridges to motion-fitting functions chosen by the external agent."""
from __future__ import annotations

import copy

import numpy as np

from .artifacts import atomic_json
from .physics_inference import _digest, _track


def select_window(observations, start_s, end_s, *, track_ids=None):
    """Select a half-open time window without re-timing, smoothing or interpolation."""
    if not np.isfinite([start_s, end_s]).all() or start_s >= end_s:
        raise ValueError("Choose a finite increasing time interval")
    result = copy.deepcopy(observations)
    result["tracks"] = []
    selected = track_ids if track_ids is not None else [t["id"] for t in observations["tracks"]]
    if not selected or len(set(selected)) != len(selected) or set(selected) - {t["id"] for t in observations["tracks"]}:
        raise ValueError("Select existing unique track IDs")
    for row in observations["tracks"]:
        if row["id"] not in selected:
            continue
        t, _, _ = _track(row)
        indices = np.flatnonzero((t >= start_s) & (t < end_s))
        if len(indices) < 2:
            raise ValueError("Selected track needs at least two samples in the interval")
        new = copy.deepcopy(row)
        for key in ("time_s", "values", "sample_state", "frame_indices"):
            if key in row:
                new[key] = [row[key][int(i)] for i in indices]
        result["tracks"].append(new)
    result["events"] = {name: value for name, value in observations.get("events", {}).items()
                        if value is None or start_s <= value < end_s}
    result["selection"] = {"source_sha256": _digest(observations), "start_s": start_s,
                           "end_s_exclusive": end_s, "time_rebased": False}
    return result


def estimate_coulomb_slide(observations, track_id, interval, fitter, *, fitter_source,
                           assumptions, gravity_m_s2=9.81, relative_scale_std=None,
                           relative_clock_std=None, output=None):
    """Use an operator-supplied fitter compatible with calibrated_scene_reconstruction fit_slide.

    This is a conditional initial estimate, not a mandatory motion route. The
    callable is provided by native Python, never imported from task JSON.
    """
    required = ("horizontal_support", "free_response", "translation_dominant", "physical_clock")
    if any(not isinstance(assumptions.get(key), str) or not assumptions[key].strip() for key in required):
        raise ValueError("State evidence for horizontal support, free response, translation and physical timing")
    if not isinstance(fitter_source, str) or not fitter_source.strip() or not callable(fitter):
        raise ValueError("Supply the chosen fitter callable and source provenance")
    gravity = float(gravity_m_s2)
    if not np.isfinite(gravity) or gravity <= 0:
        raise ValueError("Gravity must be positive")
    for value in (relative_scale_std, relative_clock_std):
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError("Relative scale/clock uncertainties must be nonnegative or unknown")
    selected = select_window(observations, *interval, track_ids=[track_id])
    row = selected["tracks"][0]
    t, xy, valid = _track(row)
    if row["units"] != "m" or xy.shape[1] != 2 or not observations.get("metric_evidence", {}).get("source"):
        raise ValueError("Sliding fit requires justified two-coordinate metric-plane measurements")
    if valid.sum() < 8:
        raise ValueError("Need at least eight observed samples for the selected slide fitter")
    t, xy = t[valid], xy[valid]
    fit = fitter(t.copy(), xy.copy())
    p, covariance = np.asarray(fit["p"], float), np.asarray(fit["cov"], float)
    if p.shape != (5,) or covariance.shape != (5, 5) or not np.isfinite(p).all() or not np.isfinite(covariance).all():
        raise ValueError("Fitter must return finite (x0,y0,heading,v0,deceleration) and covariance")
    if not np.allclose(covariance, covariance.T) or np.linalg.eigvalsh(covariance).min() < -1e-10:
        raise ValueError("Fit covariance must be symmetric positive semidefinite")
    a, speed = float(p[4]), float(p[3])
    if a <= 0 or speed <= 0:
        raise ValueError("Fit does not describe a decelerating forward free slide")
    mu = a / gravity
    stderr_fit = float(np.sqrt(max(covariance[4, 4], 0)) / gravity)
    scale_term = None if relative_scale_std is None else mu * relative_scale_std
    clock_term = None if relative_clock_std is None else 2 * mu * relative_clock_std
    complete_uncertainty = scale_term is not None and clock_term is not None
    total = float(np.sqrt(stderr_fit**2 + scale_term**2 + clock_term**2)) if complete_uncertainty else None
    # Preserve alternative-model evidence from the upstream implementation rather
    # than silently treating any fitted deceleration as friction.
    diagnostics = {}
    for key in ("rms", "rms_halted", "rms_struck", "rms_along", "stop_time", "distance"):
        value = fit.get(key)
        diagnostics[key] = float(value) if value is not None and np.isfinite(value) else None
    diagnostics["stopped"] = bool(fit.get("stopped", False))
    warnings = ["Conditional Coulomb-slide estimate; validate on the target simulator and another motion when available."]
    if not complete_uncertainty:
        warnings.append("Scale or playback-rate uncertainty is unspecified; fit covariance is not total uncertainty.")
    if diagnostics["rms_halted"] is not None and diagnostics["rms"] is not None and diagnostics["rms_halted"] < diagnostics["rms"]:
        warnings.append("An abrupt-stop model fits better; inspect obstacle contact before interpreting friction.")
    if (diagnostics["rms_struck"] is not None and diagnostics["rms_along"] is not None
            and diagnostics["rms_struck"] < .6 * diagnostics["rms_along"]):
        warnings.append("A piecewise-speed model fits better; inspect collisions or external intervention.")
    report = {"ok": True, "acceptance": False, "status": "conditional_initial_estimate",
              "fitter_source": fitter_source, "observations_sha256": _digest(observations),
              "selection": selected["selection"], "assumptions": assumptions,
              "metric_evidence": copy.deepcopy(observations["metric_evidence"]),
              "parameter": {"name": "effective_kinetic_friction", "value": mu,
                            "stderr_fit": stderr_fit, "stderr_scale": scale_term,
                            "stderr_clock": clock_term, "stderr_combined": total},
              "initial_state_estimate": {"time_s": float(t[0]), "xy_m": p[:2].tolist(),
                                         "heading_rad": float(p[2]), "speed_m_s": speed},
              "fit_parameters": p.tolist(), "fit_covariance": covariance.tolist(),
              "diagnostics": diagnostics, "warnings": warnings,
              "not_estimated": ["mass", "static_friction", "restitution"]}
    if output is not None:
        atomic_json(output, report)
    return report
