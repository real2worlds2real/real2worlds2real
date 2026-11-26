"""Artifact integrity, held-out evaluation accounting, and native twin gates."""
import hashlib
import json
import math
import itertools
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(paths):
    result = {}
    for raw in paths:
        if not Path(raw).is_absolute():
            raise ValueError(f"Evidence/input paths must be absolute: {raw}")
        path = Path(raw).resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"Evidence must name a file, not a directory: {raw}")
        result[str(path)] = file_hash(path)
    return result


def unchanged(files):
    return all(Path(p).is_file() and file_hash(p) == sha for p, sha in files.items())


def compare(check, value):
    if check["kind"] == "review":
        return value is True
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return False
    target = check["target"]
    return {"eq": lambda: value == target, "ge": lambda: value >= target,
            "le": lambda: value <= target}[check["operator"]]()


def evaluate_episodes(spec, module, evaluation):
    if not evaluation or evaluation["split"] != module:
        raise ValueError("Evaluation must identify its declared split")
    episodes = evaluation["episodes"]
    split = spec["data"]["splits"][module]
    ids, cases = set(), set()
    strata = {name: [] for name in spec["evaluation"]["strata"]}
    for ep in episodes:
        case = (ep["group"], ep["seed"], ep["stratum"])
        if ep["id"] in ids or case in cases:
            raise ValueError("Duplicate evaluation episode or scenario")
        ids.add(ep["id"])
        cases.add(case)
        if ep["group"] not in split["groups"] or ep["seed"] not in split["seeds"]:
            raise ValueError("Evaluation episode is outside the contracted split")
        if ep["stratum"] not in strata:
            raise ValueError("Undeclared evaluation stratum")
        strata[ep["stratum"]].append(ep["success"])
    expected = set(itertools.product(split["groups"], split["seeds"], strata))
    if cases != expected:
        raise ValueError("Evaluation must include every declared group/seed/stratum, including failures")
    results = {}
    for name, outcomes in strata.items():
        n = len(outcomes)
        rate = sum(outcomes) / n if n else 0.0
        z = 1.959963984540054
        lower = ((rate + z*z/(2*n) - z*math.sqrt(rate*(1-rate)/n + z*z/(4*n*n))) /
                 (1 + z*z/n)) if n else 0.0
        score = lower if spec["evaluation"]["statistic"] == "wilson_lower_95" else rate
        results[name] = {"trials": n, "success_rate": rate, "wilson_lower_95": lower,
                         "pass": n >= spec["evaluation"]["minimum_trials"] and
                         score >= spec["evaluation"]["success_rate_min"]}
    return {"pass": all(r["pass"] for r in results.values()), "strata": results}


def policy_io(spec, path):
    if not path:
        raise ValueError("Missing policy I/O artifact")
    actual = json.loads(Path(path).read_text())
    fields = ("observations", "actions", "control_hz", "max_latency_ms", "action_alignment",
              "normalization", "deployment_controller")
    expected = {k: spec["policy"][k] for k in fields}
    if actual != expected:
        raise ValueError("Dataset/training/evaluation I/O does not match the deployment contract")


def dataset_manifest(spec, path):
    if not path:
        raise ValueError("Missing dataset manifest")
    manifest = json.loads(Path(path).read_text())
    if manifest.get("version") != 1 or not isinstance(manifest.get("episodes"), list):
        raise ValueError("Expected a version 1 episode manifest")
    ids, paths, sources, count = set(), [], {}, 0
    for ep in manifest["episodes"]:
        required = {"id", "group", "seed", "source", "split", "success", "label", "artifact"}
        if set(ep) != required or type(ep["success"]) is not bool or type(ep["seed"]) is not int:
            raise ValueError("Malformed dataset episode")
        if ep["id"] in ids:
            raise ValueError("Duplicate dataset episode")
        ids.add(ep["id"])
        split = ep["split"]
        if split not in ("train", "validation"):
            raise ValueError("Sealed holdout episodes must not enter the development dataset")
        declared = spec["data"]["splits"][split]
        if ep["group"] not in declared["groups"] or ep["seed"] not in declared["seeds"]:
            raise ValueError("Dataset episode is outside its declared split")
        if spec["data"]["split_unit"] == "source" and ep["source"] in sources and sources[ep["source"]] != split:
            raise ValueError("A source is shared across dataset splits")
        sources[ep["source"]] = split
        if ep["label"] not in ("expert", "recovery_expert"):
            raise ValueError("Policy actions cannot be silently relabelled as expert supervision")
        if spec["data"]["failure_policy"] == "success_only" and not ep["success"]:
            raise ValueError("Failed episodes violate the success-only contract")
        if not ep["success"] and ep["label"] != "recovery_expert":
            raise ValueError("Failed/recovery episodes require explicit recovery labels")
        count += split == "train"
        paths.append(ep["artifact"])
    files = fingerprint(paths)
    return count, files


def native_twin(session_path):
    """Recompute gates and freeze hashes; a self-authored boolean cannot replace S6."""
    from calibrated_scene_reconstruction.scene.session import Session
    from calibrated_scene_reconstruction.scene.twin.gates import status
    from calibrated_scene_reconstruction.scene.twin.freeze import verify

    session = Session(Path(session_path).resolve())
    state = status(session)
    failed = [key for key, value in state["stages"].items() if value["applicable"] and not value["passed"]]
    failed += [f"{kind}.{key}" for kind in ("objects", "supports")
               for key, value in state.get(kind, {}).items() if not value["passed"]]
    if failed:
        raise ValueError("Native twin gates failed: " + ", ".join(failed))
    result = verify(session)
    if not result["unchanged"]:
        raise ValueError("Native twin freeze verification failed")
    return result
