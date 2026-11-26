"""File-only delivery verification for independently installed reconstruction and station hosts."""
import hashlib
import json
from pathlib import Path


LEGACY_SCHEMA = "astra_twin_delivery_v1"
REVISION_SCHEMA = "astra_twin_delivery_v2"


def delivery_pins(record):
    """Yield the accepted assets, native review evidence and revision-specific check files."""
    schema = record.get("schema")
    if schema not in (LEGACY_SCHEMA, REVISION_SCHEMA) or record.get("route") not in ("scene_pipeline", "task_workspace"):
        raise ValueError("Not a twin delivery record")
    acceptance = record["acceptance"]
    yield acceptance
    if record.get("scene"):
        yield record["scene"]
    yield from (record.get("assets") or {}).values()
    if acceptance.get("kind") == "accepted_scene":
        path = Path(acceptance["path"])
        if path.is_file():
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() == acceptance["sha256"]:
                for name, sha in json.loads(raw).get("evidence_files", {}).items():
                    yield {"path": name, "sha256": sha}
    if schema == LEGACY_SCHEMA:
        if "model_revisions" in record or "model_revisions_sha256" in record:
            raise ValueError("Reissue the revision delivery with its current model check evidence")
        return
    revisions = record.get("model_revisions")
    if not revisions or hashlib.sha256(json.dumps(revisions, sort_keys=True).encode()).hexdigest() != record.get("model_revisions_sha256"):
        raise ValueError("Missing or changed model revision evidence")
    for key, candidate in revisions.items():
        module, target = key.split("/", 1)
        identity = {name: candidate[name] for name in ("parent", "entry", "checks", "bindings")}
        identity.update(module=module, target=target, files={name: pin["sha256"] for name, pin in candidate["files"].items()})
        if hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() != candidate["revision"]:
            raise ValueError(f"Model revision no longer matches its declared inputs or checks for {key}")
        evidence = candidate.get("evidence")
        if not isinstance(evidence, dict) or set(evidence) != set(candidate["checks"]):
            raise ValueError(f"Incomplete model check evidence for {key}")
        yield from candidate["files"].values()
        for check, receipt in evidence.items():
            yield receipt
            path = Path(receipt["path"])
            if not path.is_file():
                continue
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != receipt["sha256"]:
                continue  # the receipt itself is already reported as changed
            trial = json.loads(raw)
            event = trial["model_event"]
            if (event["module"] + "/" + event["target"] != key or event["revision"] != candidate["revision"]
                    or event["operation"] + "/" + event["method"] != check or event.get("check_passed") is not True
                    or not trial["files"]):
                raise ValueError(f"Check receipt does not attest {key} {check}")
            for name, sha in trial["files"].items():
                yield {"path": name, "sha256": sha}


def changed_files(record):
    """Rehash each unique file once, including paths shared by multiple review receipts."""
    hashes = {}
    changed = set()
    for pin in delivery_pins(record):
        name = pin["path"]
        if name not in hashes:
            path = Path(name)
            hashes[name] = None
            if path.is_file():
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                hashes[name] = digest.hexdigest()
        if hashes[name] != pin["sha256"]:
            changed.add(name)
    return sorted(changed)
