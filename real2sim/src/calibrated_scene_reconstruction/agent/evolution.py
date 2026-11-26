"""Real2sim capability experiments, regression checks and scoped adoption.

The external agent authors code and reviews semantics. This ledger executes
explicit checks and detects stale evidence; it never grants scene approval.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import jsonschema


def _object(properties):
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


TEXT = {"type": "string", "minLength": 1, "pattern": r"\S"}
ID = {"type": "string", "pattern": r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$"}
TEXTS = {"type": "array", "items": TEXT, "minItems": 1, "uniqueItems": True}
CONTEXT = {"type": "object", "additionalProperties": TEXT, "minProperties": 1}
MODULES = ["capture", "support", "objects", "environment", "physics", "assembly", "review", "delivery", "harness"]
CHECK = _object({"id": ID, "kind": {"enum": ["regression", "integration", "native"]},
                 "argv": {"type": "array", "items": TEXT, "minItems": 1},
                 "cwd": TEXT, "inputs": TEXTS,
                 "artifacts": {"type": "array", "items": TEXT, "uniqueItems": True},
                 "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 86400}})
PROPOSAL = _object({"id": ID, "module": {"enum": MODULES}, "context": CONTEXT,
                    "change_kind": {"enum": ["tool", "skill", "setting", "procedure", "framework"]},
                    "gap": TEXT, "hypothesis": TEXT, "acceptance": TEXTS,
                    "baseline": TEXTS, "implementation": TEXTS,
                    "native_required": {"type": "boolean"},
                    "checks": {"type": "array", "items": CHECK, "minItems": 2}})
REVIEW = _object({"summary": TEXT, "applicability": TEXT, "limitations": TEXT,
                  "negative_results": {"type": "array", "items": TEXT},
                  "reuse_procedure": TEXT, "entrypoints": TEXTS})


def schemas():
    return {"proposal": PROPOSAL, "review": REVIEW, "context": CONTEXT}


def _path(raw):
    path = Path(raw)
    if not path.is_absolute():
        raise ValueError(f"Use an absolute path: {raw}")
    return path.resolve()


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint(paths):
    return {str(_path(p)): _hash(_path(p)) for p in paths}


def _changed(files):
    changed = []
    for path, expected in files.items():
        try:
            if _hash(path) == expected:
                continue
        except OSError:
            pass
        changed.append(path)
    return changed


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


@contextlib.contextmanager
def _lock(run):
    with (run / ".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def propose(root, request):
    """Freeze the gap, baseline and validation plan before changing implementation."""
    jsonschema.validate(request, PROPOSAL)
    kinds = {c["kind"] for c in request["checks"]}
    needed = {"regression", "integration"} | ({"native"} if request["native_required"] else set())
    if not needed <= kinds:
        raise ValueError(f"Missing check kinds: {sorted(needed - kinds)}")
    ids = [c["id"] for c in request["checks"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Check ids must be unique")
    for check in request["checks"]:
        if not _path(check["cwd"]).is_dir():
            raise ValueError("Check cwd must exist")
        for path in check["inputs"] + check["artifacts"]:
            _path(path)  # New implementation/test files may not exist yet.
    for path in request["implementation"]:
        _path(path)
    baseline = _fingerprint(request["baseline"])
    run = Path(root).resolve() / request["id"]
    run.mkdir(parents=True, exist_ok=False)
    _write(run / "proposal.json", {"version": 1, "request": request, "baseline": baseline})
    return {"run": str(run), "status": "candidate"}


def _stop(child):
    # Only this command's process group; never touch another session's jobs.
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def check(run, check_id):
    """Execute one planned check, preserving failures and interrupted attempts."""
    run = Path(run).resolve()
    with _lock(run):
        proposal = _read(run / "proposal.json")["request"]
        spec = next((c for c in proposal["checks"] if c["id"] == check_id), None)
        if spec is None:
            raise ValueError(f"Unknown check: {check_id}")
        inputs = _fingerprint(proposal["implementation"] + spec["inputs"] + [str(run / "proposal.json")])
        attempts = run / "checks"
        attempts.mkdir(exist_ok=True)
        directory = attempts / f"{len(list(attempts.iterdir())) + 1:06d}-{check_id}"
        directory.mkdir()
        _write(directory / "request.json", {"check": spec, "inputs": inputs})
        started = time.time()
        code, error, child = None, None, None
        try:
            with (directory / "command.log").open("w") as log:
                child = subprocess.Popen(spec["argv"], cwd=spec["cwd"], stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                code = child.wait(timeout=spec["timeout_seconds"])
        except (OSError, subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            error = type(exc).__name__ + ": " + str(exc)
            if child is not None:
                _stop(child)
            if isinstance(exc, KeyboardInterrupt):
                raise
        finally:
            changed = _changed(inputs)
            artifacts = {}
            for path in spec["artifacts"]:
                try:
                    artifacts.update(_fingerprint([path]))
                except OSError as exc:
                    error = f"Missing/unreadable evidence: {exc}"
            result = {"check": check_id, "kind": spec["kind"], "returncode": code,
                      "error": error, "elapsed_seconds": time.time() - started,
                      "changed_inputs": changed, "passed": code == 0 and error is None and not changed,
                      "files": {**inputs, **artifacts, **_fingerprint([str(directory / "command.log"),
                                                          str(directory / "request.json")])}}
            _write(directory / "result.json", result)
        return {"attempt": str(directory), **result}


def _checks(run):
    latest = {}
    attempts = sorted((run / "checks").glob("*"))
    for directory in attempts:
        request = _read(directory / "request.json")
        result_path = directory / "result.json"
        if result_path.exists():
            result = _read(result_path)
            changed = _changed(result["files"])
            row = {"passed": result["passed"] and not changed, "stale_files": changed,
                   "files": {**result["files"], str(result_path): _hash(result_path)}}
        else:
            row = {"passed": False, "stale_files": [], "files": {}}
        latest[request["check"]["id"]] = {"attempt": directory.name, **row}
    return latest


def promote(run, review):
    """Adopt a scoped procedure after current checks and an agent semantic review."""
    jsonschema.validate(review, REVIEW)
    run = Path(run).resolve()
    with _lock(run):
        proposal = _read(run / "proposal.json")["request"]
        latest = _checks(run)
        missing = [c["id"] for c in proposal["checks"] if not latest.get(c["id"], {}).get("passed")]
        if missing:
            raise ValueError(f"Checks missing, failed, interrupted or stale: {missing}")
        implementation = {str(_path(p)) for p in proposal["implementation"]}
        if not {str(_path(p)) for p in review["entrypoints"]} <= implementation:
            raise ValueError("Reusable entrypoints must belong to the checked implementation")
        # Reviewers must acknowledge failures; the history remains authoritative.
        failed = [p for p in (run / "checks").glob("*/result.json") if not _read(p)["passed"]]
        interrupted = [p for p in (run / "checks").glob("*/request.json")
                       if not p.with_name("result.json").exists()]
        if (failed or interrupted) and not review["negative_results"]:
            raise ValueError("Explain prior failures/interruption and their resolution in negative_results")
        files = _fingerprint([str(run / "proposal.json")] + list(implementation))
        for row in latest.values():
            files.update(row["files"])
        revisions = run / "adoptions"
        revisions.mkdir(exist_ok=True)
        target = revisions / f"{len(list(revisions.glob('*.json'))) + 1:06d}.json"
        _write(target, {"version": 1, "review": review, "files": files,
                        "attempts": {k: v["attempt"] for k, v in latest.items()}})
    return status(run)


def status(run):
    run = Path(run).resolve()
    proposal = _read(run / "proposal.json")["request"]
    latest = _checks(run)
    revisions = sorted((run / "adoptions").glob("*.json"))
    result = {"run": str(run), "id": proposal["id"], "module": proposal["module"],
              "change_kind": proposal["change_kind"],
              "context": proposal["context"], "gap": proposal["gap"], "status": "candidate",
              "checks": {c["id"]: {k: v for k, v in latest.get(c["id"], {}).items() if k != "files"}
                         for c in proposal["checks"]}}
    if revisions:
        adoption = _read(revisions[-1])
        changed = _changed(adoption["files"])
        attempts_changed = adoption["attempts"] != {k: v["attempt"] for k, v in latest.items()}
        result.update(status="stale" if changed or attempts_changed else "adopted",
                      stale_files=changed, checks_changed=attempts_changed,
                      review=adoption["review"], adoption=str(revisions[-1]))
    return result


def recall(root, context):
    """Find scoped procedures; candidates/stale entries are diagnostic leads only."""
    jsonschema.validate(context, CONTEXT)
    rows = []
    for proposal in sorted(Path(root).glob("*/proposal.json")):
        row = status(proposal.parent)
        if all(row["context"].get(k) == v for k, v in context.items()):
            rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(prog="calibrated-scene-reconstruction evolve", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("schemas")
    for name in ("propose", "promote"):
        sub = commands.add_parser(name)
        sub.add_argument("root" if name == "propose" else "run")
        sub.add_argument("request")
    sub = commands.add_parser("check")
    sub.add_argument("run")
    sub.add_argument("check_id")
    sub = commands.add_parser("status")
    sub.add_argument("run")
    sub = commands.add_parser("recall")
    sub.add_argument("root")
    sub.add_argument("context", help="JSON file containing exact context filters")
    args = parser.parse_args(argv)
    try:
        if args.command == "schemas":
            result = schemas()
        elif args.command == "propose":
            result = propose(args.root, _read(args.request))
        elif args.command == "promote":
            result = promote(args.run, _read(args.request))
        elif args.command == "check":
            result = check(args.run, args.check_id)
        elif args.command == "status":
            result = status(args.run)
        else:
            result = recall(args.root, _read(args.context))
    except (OSError, ValueError, jsonschema.ValidationError) as exc:
        parser.exit(2, f"{exc}\n")
    print(json.dumps(result, indent=2))
    return 1 if args.command == "check" and not result["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
