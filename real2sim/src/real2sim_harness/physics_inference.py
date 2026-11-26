"""Optional evidence comparison and agent-owned physics experiments.

No model, simulator, optimizer or parameter proposal is started implicitly.
Native external agents supply trial callbacks under their own execution grants.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .artifacts import atomic_json, sha256_file


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _track(row):
    t = np.asarray(row["time_s"], float)
    x = np.asarray(row["values"], float)
    if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Track times must be finite and strictly increasing")
    if x.ndim != 2 or x.shape[0] != len(t) or x.shape[1] < 1:
        raise ValueError("Track values must be [time, coordinates]")
    state = row.get("sample_state", ["observed"] * len(t))
    if len(state) != len(t) or set(state) - {"observed", "missing", "interpolated", "predicted"}:
        raise ValueError("Invalid sample_state")
    measured = np.array([s == "observed" for s in state])
    if not np.isfinite(x[measured]).all():
        raise ValueError("Observed samples must be finite; label absent samples missing")
    for key in ("id", "units", "frame", "source"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise ValueError(f"Track needs {key}")
    return t, x, measured


def _sample(t, x, valid, query, max_gap):
    """Interpolate adjacent available samples only; never hold/extrapolate ends."""
    result = np.full((len(query), x.shape[1]), np.nan)
    for i, q in enumerate(query):
        right = int(np.searchsorted(t, q))
        if right < len(t) and abs(t[right] - q) < 1e-9:
            if valid[right]:
                result[i] = x[right]
        elif 0 < right < len(t):
            left = right - 1
            if valid[left] and valid[right] and t[right] - t[left] <= max_gap:
                alpha = (q - t[left]) / (t[right] - t[left])
                result[i] = (1 - alpha) * x[left] + alpha * x[right]
    return result


def compare(observations, rollout, *, max_gap_s, time_offset_s=0.0):
    """Compare registered tracks/events; retain residuals, coverage and failures.

    time_offset_s maps simulator time to observation time by addition. It must
    come from an independently justified synchronization, not optimized warping.
    Explicit per-track scales normalize different units for optional search.
    """
    gap = _positive(max_gap_s, "max_gap_s")
    offset = float(time_offset_s)
    if not np.isfinite(offset):
        raise ValueError("time_offset_s must be finite")
    if observations.get("schema_version") != 1 or rollout.get("schema_version") != 1:
        raise ValueError("Use observation and rollout schema_version 1")
    if not observations.get("clock_source") or not rollout.get("clock_source"):
        raise ValueError("Both documents need clock_source")
    simulated = {}
    for row in rollout.get("tracks", []):
        _track(row)
        if row["id"] in simulated:
            raise ValueError("Duplicate simulated track id")
        simulated[row["id"]] = row
    results, scores, seen = [], [], set()
    for row in observations.get("tracks", []):
        t, x, measured = _track(row)
        key = row["id"]
        if key in seen:
            raise ValueError("Duplicate observed track id")
        seen.add(key)
        scale = _positive(row["residual_scale"], "residual_scale")
        sim = simulated.get(key)
        sampled = np.full_like(x, np.nan)
        if sim is not None:
            if row.get("observable") != sim.get("observable"):
                raise ValueError("Match the same observable; silhouette centroid is not body center")
            ts, xs, valid = _track(sim)
            if (row["units"], row["frame"], x.shape[1]) != (sim["units"], sim["frame"], xs.shape[1]):
                raise ValueError("Track units, registered frame and dimensions must match")
            sampled = _sample(ts + offset, xs, valid, t, gap)
        usable = measured & np.isfinite(sampled).all(axis=1)
        differences = sampled[usable] - x[usable]
        distances = np.linalg.norm(differences, axis=1)
        n, count = int(measured.sum()), int(usable.sum())
        complete = n > 0 and count == n
        normalized = distances / scale
        # Dimensionless Huber loss; no fixed conversion between pixels and metres.
        loss = float(np.mean(np.where(normalized <= 1, .5 * normalized**2, normalized - .5))) if count else None
        if complete:
            scores.append(loss)
        indices = np.flatnonzero(usable)
        worst = indices[np.argsort(distances)[-5:][::-1]] if count else []
        residuals = [None] * len(t)
        vectors = [None] * len(t)
        for index, distance in zip(indices, distances, strict=True):
            residuals[int(index)] = float(distance)
        for index, difference in zip(indices, differences, strict=True):
            vectors[int(index)] = difference.tolist()
        results.append({
            "id": key, "units": row["units"], "frame": row["frame"],
            "observed_samples": n, "compared_samples": count,
            "excluded_observation_samples": len(t) - n,
            "coverage": count / n if n else 0.0, "complete": complete,
            "mean_distance": float(distances.mean()) if count else None,
            "rms_distance": float(np.sqrt(np.mean(distances**2))) if count else None,
            "max_distance": float(distances.max()) if count else None,
            "axis_bias": differences.mean(axis=0).tolist() if count else None,
            "axis_rmse": np.sqrt(np.mean(differences**2, axis=0)).tolist() if count else None,
            "residual_vector": vectors, "residual_convention": "simulation minus observation",
            "normalized_huber": loss, "time_s": t.tolist(),
            "distance": residuals, "worst_time_s": [float(t[i]) for i in worst],
        })
    if not results:
        raise ValueError("Provide at least one observed track")
    event_results = []
    for name, value in observations.get("events", {}).items():
        other = rollout.get("events", {}).get(name)
        if value is None:
            event_results.append({"name": name, "status": "unobserved", "error_s": None})
            continue
        value = float(value)
        if not np.isfinite(value) or (other is not None and not np.isfinite(float(other))):
            raise ValueError("Event times must be finite or null")
        event_results.append({"name": name, "status": "compared" if other is not None else "missing_in_rollout",
                              "error_s": abs(float(other) + offset - value) if other is not None else None})
    complete = all(row["complete"] for row in results) and all(e["status"] != "missing_in_rollout" for e in event_results)
    return {"schema_version": 1, "ok": True, "complete": complete,
            "score": float(np.mean(scores)) if complete else None,
            "score_definition": "mean normalized track Huber; event errors reported separately",
            "tracks": results, "events": event_results,
            "time_offset_s": offset, "max_gap_s": gap,
            "observation_sha256": _digest(observations), "rollout_sha256": _digest(rollout),
            "acceptance": False}


def compare_files(request, base, output):
    base, output = Path(base), Path(output)
    paths = {name: (base / request[name]).resolve() for name in ("observations", "rollout")}
    documents = {name: json.loads(path.read_text()) for name, path in paths.items()}
    report = compare(**documents, max_gap_s=request["max_gap_s"],
                     time_offset_s=request.get("time_offset_s", 0.0))
    report["inputs"] = {name: {"sha256": sha256_file(path)} for name, path in paths.items()}
    report["preview_images"] = []
    atomic_json(output / "report.json", report)
    return report


class ParameterSpace:
    """Agent-selected named bounds; no built-in material or object parameter list."""

    def __init__(self, parameters):
        self.parameters = json.loads(json.dumps(parameters, allow_nan=False))
        if not self.parameters:
            raise ValueError("Select at least one parameter")
        names = set()
        for row in self.parameters:
            for key in ("name", "units", "target", "rationale"):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"Parameter needs {key}")
            if row["name"] in names:
                raise ValueError("Duplicate parameter name")
            names.add(row["name"])
            lo, hi = np.asarray(row["bounds"], float)
            if not np.isfinite([lo, hi]).all() or lo >= hi or (row.get("log", False) and lo <= 0):
                raise ValueError("Invalid parameter bounds")

    def values(self, unit):
        u = np.asarray(unit, float)
        if u.shape != (len(self.parameters),) or not np.isfinite(u).all() or np.any((u < 0) | (u > 1)):
            raise ValueError("Unit candidate must be finite and in [0,1]")
        values = {}
        for row, x in zip(self.parameters, u, strict=True):
            lo, hi = row["bounds"]
            value = np.exp(np.log(lo) + x * np.log(hi / lo)) if row.get("log", False) else lo + x * (hi - lo)
            values[row["name"]] = float(np.clip(value, lo, hi))
        return values

    def validate(self, values):
        if set(values) != {p["name"] for p in self.parameters}:
            raise ValueError("Candidate must supply exactly the selected parameters")
        result = {}
        for row in self.parameters:
            value = float(values[row["name"]])
            if not np.isfinite(value) or not row["bounds"][0] <= value <= row["bounds"][1]:
                raise ValueError("Candidate outside parameter bounds")
            result[row["name"]] = value
        return result


def run_trial(experiment, parameters, candidate, runner, output):
    """Execute one agent-proposed candidate through a caller-owned reset/run callback.

    runner(request, trial_directory) returns a rollout document plus receipt.
    This is a native Python API, not an execution permission or an asset acceptor.
    """
    space = ParameterSpace(parameters)
    values = space.validate(candidate)
    # The experiment includes observations, initialization, control/boundary
    # conditions, source hashes and numerical solver settings, all caller-owned.
    for key in ("observations", "initial_state", "controls", "sources", "solver", "comparison"):
        if key not in experiment:
            raise ValueError(f"Experiment needs {key}")
    if not experiment["sources"]:
        raise ValueError("Retain source fingerprints in experiment.sources")
    frozen = json.loads(json.dumps(experiment, allow_nan=False))
    fingerprint = _digest(frozen)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    request = {"schema_version": 1, "experiment": frozen, "experiment_sha256": fingerprint,
               "parameters": space.parameters, "candidate": values}
    atomic_json(output / "request.json", request)
    request_digest = sha256_file(output / "request.json")
    try:
        returned = runner(json.loads(json.dumps(request)), output)
        if sha256_file(output / "request.json") != request_digest:
            raise ValueError("Runner changed the retained request")
        receipt = returned["receipt"]
        if receipt.get("experiment_sha256") != fingerprint or receipt.get("reset") is not True:
            raise ValueError("Runner must confirm reset and the same experiment")
        applied = receipt.get("applied_parameters", {})
        if set(applied) != set(values) or any(
            not np.isfinite(applied[name]) or not np.isclose(applied[name], value, rtol=1e-6, atol=1e-12)
            for name, value in values.items()
        ):
            raise ValueError("Runner applied parameters differ from requested candidate")
        if not receipt.get("backend") or not receipt.get("version"):
            raise ValueError("Runner must report backend and version")
        if frozen.get("target_backend") and receipt["backend"] != frozen["target_backend"]:
            raise ValueError("Experiment ran on a different backend from the selected target asset")
        # Source strings/receipts are declarations: adapters must produce these
        # from effective runtime settings, not merely echo desired values.
        atomic_json(output / "rollout.json", returned["rollout"])
        atomic_json(output / "receipt.json", receipt)
        report = compare(frozen["observations"], returned["rollout"], **frozen["comparison"])
        report.update(candidate=values, applied_parameters=applied, experiment_sha256=fingerprint)
    except Exception as exc:
        atomic_json(output / "report.json", {"ok": False, "acceptance": False,
                    "candidate": values, "experiment_sha256": fingerprint,
                    "error": f"{type(exc).__name__}: {exc}"})
        raise
    atomic_json(output / "report.json", report)
    return report


def search(experiment, parameters, runner, optimizer, output, *, generations):
    """Optional ask/tell optimizer bridge (including calibrated_scene_reconstruction CMAES/CEM).

    No parallel resource allocation. The agent explicitly selects optimizer and
    budget. Direct run_trial calls do not depend on this helper.
    """
    if type(generations) is not int or generations < 1:
        raise ValueError("Choose a positive generation budget")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    space = ParameterSpace(parameters)
    best, history = None, []
    for generation in range(generations):
        raw = np.asarray(optimizer.ask(), float)
        if raw.ndim != 2 or raw.shape[1] != len(parameters) or not len(raw) or not np.isfinite(raw).all():
            raise ValueError("Optimizer must propose a finite [population,parameters] array")
        losses = []
        for i, point in enumerate(raw):
            unit = np.clip(point, 0, 1)
            candidate = space.values(unit)
            path = output / f"generation-{generation:03d}-trial-{i:03d}"
            report = run_trial(experiment, parameters, candidate, runner, path)
            if not report["complete"]:
                raise ValueError("Incomplete rollout evidence; inspect trial instead of optimizing missing data")
            score = report["score"]
            losses.append(score + 1000 * float(np.sum((point - unit)**2)))
            row = {"candidate": candidate, "score": score, "trial": path.name}
            history.append(row)
            if best is None or score < best["score"]:
                best = row
            atomic_json(output / "search.json", {"best": best, "history": history, "acceptance": False})
        optimizer.tell(raw, np.asarray(losses))
    return {"best": best, "history": history, "acceptance": False}
