"""Cross-task development regression and reversible shared backend releases.

Only already accepted development evaluations can qualify a release. This
registry selects backend versions; it never rewrites earlier runs, consumes a
holdout, or transfers a task's old acceptance to a new scene.
"""
from contextlib import contextmanager
from pathlib import Path
import re
import time

from ..oversight.controller import Controller
from ..oversight.evidence import digest, evaluate_episodes, file_hash
from ..oversight.policy_bridge import BINDING_SCHEMA, IO_FIELDS, verify_binding
from .backends import load_backend, read, write_new


SCHEMA = "astra_shared_backend_release_v1"


def _directory(root, name):
    if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", name):
        raise ValueError("Release name must be a single managed identifier")
    return Path(root).resolve() / name


@contextmanager
def _lock(directory):
    import fcntl
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _history(directory):
    rows, previous = [], None
    for path in sorted((directory / "history").glob("*.json")):
        row = read(path)
        payload = {key: value for key, value in row.items() if key != "sha256"}
        if digest(payload) != row.get("sha256") or row.get("previous_event") != previous:
            raise ValueError("Shared release history integrity failure")
        previous = row["sha256"]
        rows.append(row)
    return rows


def _version(directory, identity):
    if not isinstance(identity, str) or not re.fullmatch(r"[a-f0-9]{64}", identity):
        raise ValueError("Invalid release version identity")
    value = read(directory / "versions" / (identity + ".json"))
    payload = {key: item for key, item in value.items() if key != "sha256"}
    if digest(payload) != identity or value.get("sha256") != identity or value.get("schema") != SCHEMA:
        raise ValueError("Shared release version integrity failure")
    return value


def _append(directory, rows, action, active, reason):
    payload = {"previous_event": rows[-1]["sha256"] if rows else None,
               "action": action, "active": active, "reason": reason, "time": time.time()}
    row = {**payload, "sha256": digest(payload)}
    write_new(directory / "history" / f"{len(rows):06d}.json", row)
    return row


def _json_inputs(attempt):
    # Inputs are already hash-pinned by Controller. Small JSON manifests are the
    # protocol carriers, not arbitrary dataset payloads or model weight files.
    values = []
    for raw in attempt["inputs"]:
        path = Path(raw)
        if path.suffix == ".json" and path.stat().st_size <= 4 * 1024 * 1024:
            try:
                value = read(path)
            except (ValueError, UnicodeError):
                continue
            if isinstance(value, dict):
                values.append((str(path), value))
    return values


def _validation(run, task_id, *, forbid_holdout):
    ctrl = Controller(run)
    with ctrl.lock():
        plan, state = ctrl._load()
        if not ctrl._current("validation", plan, state):
            raise ValueError("Cross-task regression requires current accepted development validation")
        if forbid_holdout and state["holdout_consumed"]:
            raise ValueError("Choose release versions before consuming the sealed holdout")
        module = state["modules"]["validation"]
        attempt = module["attempts"][-1]
        report_path = Path(attempt["directory"]) / "review.json"
        report = read(report_path)
        if report.get("module") != "validation" or report.get("plan_sha256") != state["plan_sha256"]:
            raise ValueError("Validation review belongs to a different module or plan")
        inputs = _json_inputs(attempt)
        requests = [value for _, value in inputs if value.get("schema") == "astra_bound_job_v1"
                    and value.get("module") == "validation" and value.get("operation") == "develop"]
        if len(requests) != 1:
            raise ValueError("Development acceptance must pin one executed bound-backend request")
        request = requests[0]
        bindings = [(path, value) for path, value in inputs if value.get("schema") == BINDING_SCHEMA
                    and value.get("binding_sha256") == request.get("binding_sha256")]
        if len(bindings) != 1:
            raise ValueError("Development request lacks its exact world binding")
        binding_path, binding = bindings[0]
        # verify_binding is read-only and can run under this controller lock.
        verify_binding(binding)
        if binding["task_id"] != task_id or Path(binding["oversight_run"]).resolve() != ctrl.root:
            raise ValueError("Regression task or bound run differs from the actual validation")
        packages = [(path, value) for path, value in inputs if value.get("sha256") == request.get("backend_sha256")
                    and "manifest" in value and "files" in value]
        if len(packages) != 1:
            raise ValueError("Validation request did not pin the registered backend package")
        backend_path, _ = packages[0]
        backend = load_backend(backend_path)
        if any(backend["manifest"][key] != binding[key] for key in ("task_id", "backend_id")):
            raise ValueError("Validated backend does not match its world/task")
        receipts = []
        for path in report["artifacts"]:
            if Path(path).suffix != ".json" or Path(path).stat().st_size > 4 * 1024 * 1024:
                continue
            try:
                value = read(path)
            except (ValueError, UnicodeError):
                continue
            if isinstance(value, dict) and value.get("schema") == "astra_bound_policy_receipt_v1":
                receipts.append(value)
        matching = [receipt for receipt in receipts if receipt.get("status") == "completed"
                    and receipt.get("operation") == "develop"
                    and receipt.get("facts", {}).get("oversight", {}).get("request_sha256") == digest(request)]
        if len(matching) != 1:
            raise ValueError("Accepted validation must retain its exact completed worker receipt")
        info = matching[0]["facts"]["oversight"]
        if any(info.get(key) != binding[key] for key in ("binding_sha256", "task_id", "backend_id")) or info.get("module") != "validation":
            raise ValueError("Accepted worker receipt does not match its executed task/world")
        evaluated = evaluate_episodes(plan["contract"], "validation", report["evaluation"])
        if not evaluated["pass"]:
            raise ValueError("Development evaluation did not satisfy its frozen task success gate")
        spec = plan["contract"]
        frozen_path = Path(binding["native_session"]) / "qa" / "independent_freeze.json"
        frozen = read(frozen_path)
        if not frozen.get("files"):
            raise ValueError("Native world comparison needs its nonempty frozen file manifest")
        world_files = sorted((row["base"], row["path"].replace("\\", "/"), row["sha256"])
                             for row in frozen["files"])
        conditions = {
            "task": spec["task"], "policy_io": {key: spec["policy"][key] for key in IO_FIELDS},
            "evaluation": spec["evaluation"], "validation_split": spec["data"]["splits"]["validation"],
            "fidelity": spec["fidelity"], "variation": spec["variation"],
            "source_content": sorted(state["sources"].values()), "frozen_world_files": world_files,
        }
        return {"run": str(ctrl.root), "task_id": task_id,
                "validation_acceptance": module["accepted"]["id"],
                "report": str(report_path), "report_sha256": file_hash(report_path),
                "binding": binding_path, "binding_sha256": binding["binding_sha256"],
                "backend": backend_path, "backend_sha256": backend["sha256"],
                "conditions_sha256": digest(conditions),
                "strata": {name: {"trials": item["trials"],
                    "successes": sum(ep["success"] for ep in report["evaluation"]["episodes"] if ep["stratum"] == name)}
                    for name, item in evaluated["strata"].items()}}


def _check_regression(record, candidate, *, forbid_holdout, expected_baseline=None):
    if set(record) != {"task_id", "baseline_run", "candidate_run"}:
        raise ValueError("Regression requires exact task_id, baseline_run and candidate_run")
    task = record["task_id"]
    if Path(record["baseline_run"]).resolve() == Path(record["candidate_run"]).resolve():
        raise ValueError("Baseline and candidate must be separate immutable development runs")
    baseline = _validation(record["baseline_run"], task, forbid_holdout=forbid_holdout)
    proposed = _validation(record["candidate_run"], task, forbid_holdout=forbid_holdout)
    if proposed["backend_sha256"] != candidate["sha256"]:
        raise ValueError("Candidate regression did not execute the proposed backend version")
    if expected_baseline is not None and baseline["backend_sha256"] != expected_baseline:
        raise ValueError("Regression baseline must execute the currently active task backend")
    if baseline["conditions_sha256"] != proposed["conditions_sha256"]:
        raise ValueError("Regression changed task, policy I/O, source world or evaluation conditions")
    if set(baseline["strata"]) != set(proposed["strata"]):
        raise ValueError("Regression strata differ")
    for name, old in baseline["strata"].items():
        new = proposed["strata"][name]
        if old["trials"] != new["trials"] or new["successes"] < old["successes"]:
            raise ValueError("Candidate regressed on development stratum: " + name)
    return {"task_id": task, "baseline": baseline, "candidate": proposed,
            "verdict": "passed_development_nonregression"}


def _verify_version(directory, identity):
    version = _version(directory, identity)
    for task, item in version["candidates"].items():
        package = load_backend(item["path"])
        if package["sha256"] != item["sha256"] or package["manifest"]["task_id"] != task:
            raise ValueError("Released backend identity changed")
    for record in version["regressions"]:
        for side in ("baseline", "candidate"):
            original = record[side]
            current = _validation(original["run"], record["task_id"], forbid_holdout=False)
            if current != original:
                raise ValueError("Released development evidence changed; qualify a new release")
    return version


def promote(root, name, candidates, regressions, reason):
    """Promote a shared tool only after paired, nonregressing multi-task checks."""
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Promotion requires a concrete reason")
    directory = _directory(root, name)
    packages = {}
    for path in candidates:
        package = load_backend(path)
        task = package["manifest"]["task_id"]
        if task in packages:
            raise ValueError("Supply exactly one candidate backend per task")
        packages[task] = {"path": str(Path(path).resolve()), "sha256": package["sha256"]}
    if len(packages) < 2:
        raise ValueError("Shared tool promotion requires at least two distinct tasks")
    if (len(regressions) != len(packages) or {r.get("task_id") for r in regressions} != set(packages)):
        raise ValueError("Every candidate task needs exactly one paired development regression")
    with _lock(directory):
        history = _history(directory)
        parent = history[-1]["active"] if history else None
        previous = _verify_version(directory, parent) if parent else None
        if previous and not set(previous["candidates"]) <= set(packages):
            raise ValueError("A shared upgrade must retain regression coverage for every active task")
        evaluated = []
        for record in sorted(regressions, key=lambda row: row["task_id"]):
            task = record["task_id"]
            baseline = previous["candidates"].get(task, {}).get("sha256") if previous else None
            evaluated.append(_check_regression(record, packages[task], forbid_holdout=True, expected_baseline=baseline))
        if previous and previous["candidates"] == packages:
            raise ValueError("The candidate is already the active backend release")
        payload = {"schema": SCHEMA, "name": name, "parent_release": parent,
                   "candidates": packages, "regressions": evaluated, "reason": reason,
                   "scope": "cross_task_development_nonregression; real validation pending"}
        identity = digest(payload)
        version_path = directory / "versions" / (identity + ".json")
        if not version_path.exists():
            write_new(version_path, {**payload, "sha256": identity})
        _append(directory, history, "promote", identity, reason)
        return {"name": name, "active_version": identity, "previous_version": parent,
                "backends": packages, "status": "active", "real_validation": "pending"}


def rollback(root, name, reason):
    """Restore the previous qualified tool version without modifying any run."""
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Rollback requires a concrete reason")
    directory = _directory(root, name)
    with _lock(directory):
        history = _history(directory)
        if not history:
            raise ValueError("There is no active release to roll back")
        current = _version(directory, history[-1]["active"])
        target = current["parent_release"]
        if target is None:
            raise ValueError("There is no previous qualified release")
        version = _verify_version(directory, target)
        _append(directory, history, "rollback", target, reason)
        return {"name": name, "active_version": target, "rolled_back_version": current["sha256"],
                "backends": version["candidates"], "status": "active", "evidence_reassigned": False}


def status(root, name):
    directory = _directory(root, name)
    if not directory.exists():
        return {"name": name, "status": "unregistered", "active_version": None}
    with _lock(directory):
        history = _history(directory)
        if not history:
            return {"name": name, "status": "unregistered", "active_version": None}
        identity = history[-1]["active"]
        try:
            version = _verify_version(directory, identity)
        except (ValueError, OSError, KeyError) as error:
            return {"name": name, "status": "stale", "active_version": identity,
                    "reason": str(error), "history_events": len(history), "executable": False}
        return {"name": name, "status": "active", "active_version": identity,
                "backends": version["candidates"], "history_events": len(history),
                "executable": True, "scope": version["scope"]}
