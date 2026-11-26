"""World-bound CPU behavior cloning for explicitly ordered dense vector data.

The numerical learner generalizes the normalized ridge/MLP objective in
peg_worker.train. It does not reuse Peg feature engineering, action scaling,
success criteria, a teacher, or task-specific privileged state. RGB encoders,
action chunks and closed-loop evaluation are outside this adapter's scope.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from astra_e2e.oversight.evidence import dataset_manifest, digest, file_hash, unchanged
from astra_e2e.oversight.policy_bridge import verify_binding


FORMAT = "astra_dense_bc_v1"
METHODS = ("ridge_bc", "mlp_bc")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("Non-finite JSON")))


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _controller(root):
    from astra_e2e.oversight.controller import Controller
    return Controller(root)


def _dimensions(fields):
    if not fields or len({field["name"] for field in fields}) != len(fields):
        raise ValueError("Dense BC requires distinct ordered vector fields")
    dimensions = 0
    for field in fields:
        shape = field["shape"]
        if (not isinstance(shape, list) or len(shape) != 1 or type(shape[0]) is not int
                or not 1 <= shape[0] <= 1024 or field["dtype"] not in ("float32", "float64")):
            raise ValueError("Dense BC supports float vector I/O only, not images or action chunks")
        dimensions += shape[0]
    if dimensions > 1024:
        raise ValueError("Dense BC dimension limit exceeded")
    return dimensions


def _context(binding_path, request_path):
    binding = verify_binding(binding_path)
    request_path = Path(request_path).resolve(strict=True)
    request = read(request_path)
    if (request.get("schema") != "astra_bound_job_v1" or request.get("operation") != "train"
            or request.get("module") != "train" or request.get("simulation_only") is not True
            or request.get("binding_sha256") != binding["binding_sha256"]
            or request.get("method") not in METHODS or not request.get("job_id")):
        raise ValueError("Expected a current world-bound dense BC training request")
    controller = _controller(binding["oversight_run"])
    # The parent owns the writer lock during a job. Reading immutable history
    # here avoids deadlocking the worker; current inputs are checked again later.
    plan, state = controller._load()
    if state.get("holdout_consumed"):
        raise ValueError("Sealed holdout has been consumed; no further training in this experiment")
    train = state["modules"]["train"]
    data = state["modules"]["data"]
    if train["status"] != "running" or not train["attempts"] or not controller._current("data", plan, state):
        raise ValueError("Training requires a running owned attempt and current accepted data")
    attempt = train["attempts"][-1]
    if attempt["inputs"].get(str(request_path)) != file_hash(request_path) or not unchanged(attempt["inputs"]):
        raise ValueError("Training request is not pinned by the current attempt")
    expected = {name: state["modules"][name]["accepted"] for name in plan["modules"]["train"]["depends_on"]}
    if request.get("dependencies") != expected or any(attempt["dependencies"].get(name) != item["id"]
                                                      for name, item in expected.items()):
        raise ValueError("Training dependencies differ from the current accepted data")
    review_path = Path(data["attempts"][-1]["directory"]) / "review.json"
    accepted = data["accepted"]
    if accepted["files"].get(str(review_path.resolve())) != file_hash(review_path):
        raise ValueError("Data review is not part of the accepted evidence")
    review = read(review_path)
    manifest_path = Path(review["dataset_manifest"]).resolve(strict=True)
    if accepted["files"].get(str(manifest_path)) != file_hash(manifest_path):
        raise ValueError("Dataset manifest is not part of current accepted data")
    return binding, request, plan["contract"], accepted, manifest_path


def _dataset(binding, spec, accepted, manifest_path):
    manifest = read(manifest_path)
    io = binding["policy_io"]
    expected = {"format": FORMAT, "world_sha256": binding["world_sha256"], "task_id": binding["task_id"],
                "policy_io_sha256": digest(io),
                "observation_fields": [field["name"] for field in io["observations"]],
                "action_fields": [field["name"] for field in io["actions"]],
                "action_alignment": io["action_alignment"]}
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("Dataset world, task, dense layout or policy I/O does not match the binding")
    obs_dim, action_dim = _dimensions(io["observations"]), _dimensions(io["actions"])
    count, files = dataset_manifest(spec, manifest_path)
    if count < 1 or any(accepted["files"].get(path) != sha for path, sha in files.items()):
        raise ValueError("Every episode must belong to the current accepted data evidence")
    rows, labels, ids = [], [], []
    frames = 0
    for episode in manifest["episodes"]:
        if episode["split"] != "train":
            continue
        with np.load(episode["artifact"], allow_pickle=False) as archive:
            if not {"observations", "actions"} <= set(archive.files):
                raise ValueError("Dense episode requires observations and expert command actions")
            x, y = archive["observations"], archive["actions"]
            if (x.ndim != 2 or y.ndim != 2 or x.shape[1] != obs_dim or y.shape[1] != action_dim
                    or len(x) != len(y) or not len(x) or x.dtype.kind != "f" or y.dtype.kind != "f"
                    or not np.isfinite(x).all() or not np.isfinite(y).all()):
                raise ValueError("Invalid dense episode dimensions, floating dtype or finite labels")
            frames += len(x)
            if frames * (obs_dim + action_dim) > 16_000_000:
                raise ValueError("Dense CPU adapter data budget exceeded")
            rows.append(x.astype(np.float64))
            labels.append(y.astype(np.float64))
            ids.append(episode["id"])
    return np.concatenate(rows), np.concatenate(labels), ids


def _parameters(method, parameters, manifest_path):
    allowed = {"dataset_manifest", "seed", "ridge_alpha"} if method == "ridge_bc" else {
        "dataset_manifest", "seed", "hidden_dim", "updates", "learning_rate"}
    if not isinstance(parameters, dict) or set(parameters) - allowed:
        raise ValueError("Unsupported dense BC parameters")
    if "dataset_manifest" in parameters and Path(parameters["dataset_manifest"]).resolve() != manifest_path:
        raise ValueError("Dataset override differs from the accepted data review")
    result = {"seed": parameters.get("seed", 0)}
    if type(result["seed"]) is not int or not 0 <= result["seed"] < 2**32:
        raise ValueError("Invalid training seed")
    if method == "ridge_bc":
        alpha = parameters.get("ridge_alpha", 0.001)
        if isinstance(alpha, bool) or not isinstance(alpha, (float, int)) or not math.isfinite(alpha) or not 1e-8 <= alpha <= 100:
            raise ValueError("ridge_alpha must be finite and in [1e-8, 100]")
        result["ridge_alpha"] = alpha
    else:
        for key, default, lo, hi in (("hidden_dim", 24, 1, 512), ("updates", 500, 1, 5000)):
            value = parameters.get(key, default)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError("Invalid bounded parameter: " + key)
            result[key] = value
        rate = parameters.get("learning_rate", 0.009)
        if isinstance(rate, bool) or not isinstance(rate, (float, int)) or not math.isfinite(rate) or not 1e-6 <= rate <= 0.1:
            raise ValueError("Invalid bounded learning_rate")
        result["learning_rate"] = rate
    return result


def fit(x, y, method, parameters):
    """Fit real weights; statistics use training rows only, actions retain units."""
    mean, scale = x.mean(0), np.maximum(x.std(0), 1e-6)
    xb = np.column_stack([(x - mean) / scale, np.ones(len(x))])
    if method == "ridge_bc":
        reg = np.eye(xb.shape[1]) * parameters["ridge_alpha"]
        weights = {"w": np.linalg.solve(xb.T @ xb + reg, xb.T @ y)}
        updates = 0
    else:
        rng = np.random.default_rng(parameters["seed"])
        w1 = rng.normal(0, 0.4, (xb.shape[1], parameters["hidden_dim"]))
        w2 = rng.normal(0, 0.15, (parameters["hidden_dim"] + 1, y.shape[1]))
        weights = {"w1": w1, "w2": w2}
        moment, variance = [np.zeros_like(w1), np.zeros_like(w2)], [np.zeros_like(w1), np.zeros_like(w2)]
        updates = parameters["updates"]
        for step in range(1, updates + 1):
            hidden = np.tanh(xb @ w1)
            hb = np.column_stack([hidden, np.ones(len(x))])
            error = (hb @ w2 - y) / len(y)
            grads = [xb.T @ ((error @ w2[:-1].T) * (1 - hidden**2)), hb.T @ error]
            for i, (weight, grad) in enumerate(zip((w1, w2), grads)):
                moment[i] = 0.9 * moment[i] + 0.1 * grad
                variance[i] = 0.999 * variance[i] + 0.001 * grad**2
                weight -= parameters["learning_rate"] * (moment[i] / (1 - 0.9**step)) / (
                    np.sqrt(variance[i] / (1 - 0.999**step)) + 1e-8)
    values = {"mean": mean, "scale": scale, **weights}
    if not all(np.isfinite(array).all() for array in values.values()):
        raise ValueError("Training produced non-finite policy weights")
    return values, updates


def predict(policy, observations):
    observations = np.asarray(observations, dtype=float)
    single = observations.ndim == 1
    x = np.atleast_2d(observations)
    if x.shape[1] != len(policy["mean"]) or not np.isfinite(x).all():
        raise ValueError("Prediction observation contract mismatch")
    xb = np.column_stack([(x - policy["mean"]) / policy["scale"], np.ones(len(x))])
    if "w" in policy:
        action = xb @ policy["w"]
    else:
        action = np.column_stack([np.tanh(xb @ policy["w1"]), np.ones(len(x))]) @ policy["w2"]
    return action[0] if single else action


def train_bound(binding_path, request_path, result_path):
    binding, request, spec, accepted, manifest_path = _context(binding_path, request_path)
    result_path = Path(result_path).resolve()
    if result_path.parent != Path(request_path).resolve().parent or result_path.exists():
        raise ValueError("Write a new result beside the owned job request")
    parameters = _parameters(request["method"], request["parameters"], manifest_path)
    x, y, episode_ids = _dataset(binding, spec, accepted, manifest_path)
    params, updates = fit(x, y, request["method"], parameters)
    metrics = {"method": request["method"], "frames": len(x), "episodes": len(episode_ids),
               "training_mse": float(np.mean((predict(params, x) - y)**2)),
               "optimizer_updates": updates, "validation_used_for_fit": False,
               "task_success_evaluated": False, "physical_actions_executed": False}
    metadata = {"schema": FORMAT, "method": request["method"], "parameters": parameters,
                "binding_sha256": binding["binding_sha256"], "world_sha256": binding["world_sha256"],
                "task_id": binding["task_id"], "backend_id": binding["backend_id"],
                "request_sha256": digest(request), "policy_io": binding["policy_io"],
                "dataset_manifest": str(manifest_path), "dataset_sha256": file_hash(manifest_path),
                "data_acceptance_id": accepted["id"], "training_episode_ids": episode_ids,
                "implementation_sha256": file_hash(__file__),
                "observation_transform": "train_mean_std_floor_1e-6",
                "action_transform": "identity_in_declared_units",
                "scope": "Dense vector BC; no task-specific features, visual encoder, chunks or closed-loop result."}
    # Recheck source/acceptance revisions before making a checkpoint claim.
    current = _context(binding_path, request_path)
    if current[:2] != (binding, request) or current[3] != accepted:
        raise ValueError("Training inputs changed while fitting")
    output = result_path.parent
    checkpoint, io_path, metrics_path = output / "policy.npz", output / "policy_io.json", output / "training_metrics.json"
    with checkpoint.open("xb") as stream:
        np.savez(stream, metadata_json=np.asarray(json.dumps(metadata, allow_nan=False)), **params)
    write_new(io_path, binding["policy_io"])
    write_new(metrics_path, metrics)
    paths = {"policy": str(checkpoint), "policy_io": str(io_path), "metrics": str(metrics_path)}
    artifacts = [{"id": key, "kind": key, "sha256": file_hash(path),
                  "revision_sha256": binding["world_sha256"], "contract_sha256": binding["contract_sha256"],
                  "scope": "training"} for key, path in paths.items()]
    receipt = {"schema": "astra_bound_policy_receipt_v1", "status": "completed", "operation": "train",
               "facts": {**metrics, "oversight": {
                   "binding_sha256": binding["binding_sha256"], "task_id": binding["task_id"],
                   "backend_id": binding["backend_id"], "module": "train", "request_sha256": digest(request),
                   "policy_io_ref": "policy_io", "dataset_manifest_ref": None,
                   "checkpoint_ref": "policy", "evaluation_ref": None}}, "artifacts": artifacts}
    receipt_path = output / "receipt.json"
    write_new(receipt_path, receipt)
    result = {"receipt": str(receipt_path), "artifact_paths": paths}
    write_new(result_path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(train_bound(args.binding, args.request, args.result), allow_nan=False))


if __name__ == "__main__":
    main()
