"""Package a completed simulation policy for independent delivery review.

The small policy/configuration bundle has a content-pinned external dependency
closure. It is not self-contained or robot-deployable. This tool neither opens
an unfinished holdout nor accepts the outer delivery module.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from bound_training import FORMAT, _dimensions, predict, read, write_new
from astra_e2e.oversight.evidence import digest, file_hash, unchanged
from astra_e2e.oversight.policy_bridge import verify_binding


def _completed_report(path, binding, holdout):
    report = read(path)
    state = report.get("state", {})
    if (report.get("schema") != "astra_unified_agent_report_v1"
            or report.get("status") != "simulation_policy_accepted"
            or state.get("done") != report["status"] or state.get("pending") is not None
            or report.get("binding_sha256") != binding["binding_sha256"]
            or report.get("execution_mode") != "production"
            or not (state.get("review_result") or {}).get("accepted")):
        raise ValueError("Delivery requires a complete production final-agent report for this binding")
    directory = path.parent
    if (directory / "runner.lock").exists():
        raise ValueError("Final agent must be stopped before delivery")
    config_path, implementation_path = directory / "config.json", directory / "implementation.json"
    config, implementation = read(config_path), read(implementation_path)
    if (config.get("execution_mode") != report["execution_mode"]
            or read(config["binding"])["binding_sha256"] != binding["binding_sha256"]
            or report.get("config_sha256") != digest({"configuration": config, "implementation": implementation})):
        raise ValueError("Final report does not match its immutable configuration")
    previous, latest, files = None, None, {}
    for item in sorted((directory / "history").glob("*.json")):
        row = read(item)
        body = {k: v for k, v in row.items() if k != "sha256"}
        if (row.get("previous") != previous or digest(body) != row.get("sha256")
                or row.get("config_sha256") != report["config_sha256"]):
            raise ValueError("Final-agent history integrity failure")
        previous, latest = row["sha256"], row["state"]
        files[str(item.resolve())] = file_hash(item)
    if latest != state or report.get("decision_steps") != state.get("step"):
        raise ValueError("Final report must match the completed agent history")
    last = state.get("last") or {}
    receipt_path = last.get("receipt")
    if (not receipt_path or last.get("runner_receipt_sha256") != file_hash(receipt_path)
            or holdout["files"].get(str(Path(receipt_path).resolve())) != file_hash(receipt_path)):
        raise ValueError("Final report does not identify the current accepted holdout receipt")
    receipt = read(receipt_path)
    identity = receipt.get("facts", {}).get("oversight", {})
    if (receipt.get("operation") != "confirm" or receipt.get("status") != "completed"
            or identity.get("module") != "holdout"
            or any(identity.get(k) != binding[k] for k in ("binding_sha256", "task_id", "backend_id"))):
        raise ValueError("Final report receipt belongs to another operation or binding")
    for item in (path, config_path, implementation_path):
        files[str(item)] = file_hash(item)
    seal = directory / "report.integrity.json"
    if seal.exists():
        if read(seal).get("report_sha256") != file_hash(path):
            raise ValueError("Final report integrity seal changed")
        files[str(seal)] = file_hash(seal)
    files.update(implementation)
    return report, files


def _qualified(binding_path, report_path, config_path):
    binding = verify_binding(binding_path)
    from astra_e2e.oversight.controller import Controller
    controller = Controller(binding["oversight_run"])
    plan, state = controller._load()  # A delivery worker may run under its parent's writer lock.
    if (plan["contract"].get("validation_scope") != "simulation_verification"
            or binding.get("simulation_only") is not True or not state.get("holdout_consumed")):
        raise ValueError("This packager requires a completed simulation-verification run")
    accepted = {}
    for name in ("data", "train", "validation", "holdout"):
        row = state["modules"][name]
        if row["status"] != "accepted" or not row["accepted"] or not controller._current(name, plan, state):
            raise ValueError("Delivery requires current accepted " + name)
        accepted[name] = row["accepted"]
    checkpoint_sha = accepted["train"]["checkpoint_sha256"]
    if not checkpoint_sha or any(accepted[name]["checkpoint_sha256"] != checkpoint_sha
                                 for name in ("validation", "holdout")):
        raise ValueError("Training and both evaluations must use the same accepted checkpoint")
    # Only after accepted holdout is established may the finished report be read.
    report, report_files = _completed_report(report_path, binding, accepted["holdout"])
    native = read(config_path)
    if (accepted["holdout"]["files"].get(str(config_path)) != file_hash(config_path)
            or native.get("schema") != "astra_bound_native_v1" or native.get("simulation_only") is not True
            or native.get("backend_id") != binding["backend_id"]
            or native.get("policy_io") != binding["policy_io"]
            or native.get("success_rule") != binding["success_contract"].get("predicates")):
        raise ValueError("Native configuration must be the exact accepted final execution configuration")
    checkpoints = [Path(path) for path, sha in accepted["train"]["files"].items() if sha == checkpoint_sha]
    if len(checkpoints) != 1:
        raise ValueError("Cannot resolve a unique accepted training checkpoint")
    return binding, plan, accepted, report, report_files, native, checkpoints[0]


def _dense_normalizer(checkpoint, binding):
    with np.load(checkpoint, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        arrays = {name: archive[name].copy() for name in archive.files if name != "metadata_json"}
    if (metadata.get("schema") != FORMAT or metadata.get("binding_sha256") != binding["binding_sha256"]
            or metadata.get("world_sha256") != binding["world_sha256"]
            or metadata.get("policy_io") != binding["policy_io"]):
        raise ValueError("Only the bound dense-BC NPZ format is supported by this packager version")
    dim = _dimensions(binding["policy_io"]["observations"])
    action_dim = _dimensions(binding["policy_io"]["actions"])
    if (set(arrays) not in ({"mean", "scale", "w"}, {"mean", "scale", "w1", "w2"})
            or any(not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all()
                   for value in arrays.values())
            or arrays["mean"].shape != (dim,) or arrays["scale"].shape != (dim,)
            or not (arrays["scale"] > 0).all()
            or predict(arrays, arrays["mean"]).shape != (action_dim,)):
        raise ValueError("Checkpoint weights or embedded normalizer violate the bound I/O")
    return metadata, {name: arrays[name] for name in ("mean", "scale")}


def _scene_files(binding):
    from calibrated_scene_reconstruction.scene.session import Session
    session = Session(binding["native_session"])
    roots = {"session": session.dir.resolve(), "config": session.config_dir.resolve()}
    freeze_path = roots["session"] / "qa/independent_freeze.json"
    if file_hash(freeze_path) != binding["native_freeze_sha256"]:
        raise ValueError("Native freeze changed")
    files, exports = {}, []
    for row in read(freeze_path)["files"]:
        root = roots[row["base"]]
        path = (root / row["path"]).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("Scene dependency escapes its frozen root")
        files[str(path)] = row["sha256"]
        if row["base"] == "session" and path.is_relative_to(roots["session"] / "export") and path.suffix in {".usd", ".usda", ".usdc", ".usdz"}:
            exports.append(str(path))
    if not exports:
        raise ValueError("Frozen scene contains no delivery USD or USDZ export")
    return files, exports, roots


def package(binding, agent_report, native_config, output):
    binding_path, report_path, config_path = (Path(p).resolve(strict=True) for p in (binding, agent_report, native_config))
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Delivery output must be a new directory")
    (bound, plan, accepted, report, report_files, native, checkpoint) = _qualified(binding_path, report_path, config_path)
    metadata, normalizer = _dense_normalizer(checkpoint, bound)
    scene_files, exports, roots = _scene_files(bound)
    if any(output.is_relative_to(root) or root.is_relative_to(output) for root in roots.values()):
        raise ValueError("Delivery output must be separate from frozen scene/config roots")
    files = {}
    def merge(values):
        for path, sha in values.items():
            if not Path(path).is_absolute() or (path in files and files[path] != sha):
                raise ValueError("Conflicting or non-absolute dependency identity")
            files[path] = sha
    for group in (bound["evidence_files"], bound["source_files"], scene_files, report_files,
                  *(item["files"] for item in accepted.values())):
        merge(group)
    merge({str(binding_path): file_hash(binding_path), str(config_path): file_hash(config_path),
           str(Path(__file__).resolve()): file_hash(__file__)})
    for path in [native["factory_file"], *native.get("implementation_files", []), *native.get("runtime_asset_files", [])]:
        if accepted["holdout"]["files"].get(path) != file_hash(path):
            raise ValueError("Final execution did not pin a native runtime dependency")
    if not unchanged(files):
        raise ValueError("A required delivery dependency changed")
    output.mkdir(parents=True)
    copies = {"policy.npz": checkpoint, "native_config.json": config_path,
              "world_binding.json": binding_path, "final_agent_report.json": report_path}
    for name, source in copies.items():
        shutil.copyfile(source, output / name)
        if file_hash(output / name) != file_hash(source):
            raise ValueError("Copied delivery artifact changed")
    write_new(output / "policy_io.json", bound["policy_io"])
    with (output / "normalizer.npz").open("xb") as stream:
        np.savez(stream, **normalizer)
    write_new(output / "normalization.json", {
        "format": FORMAT, "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                                     for name, value in normalizer.items()},
        "source_checkpoint_sha256": file_hash(checkpoint),
        "observation_transform": metadata.get("observation_transform"),
        "action_transform": metadata.get("action_transform"),
        "note": "Exact train-derived arrays copied from the accepted checkpoint; no refitting."})
    closure = {"schema": "astra_external_dependency_closure_v1", "portable_self_contained": False,
               "mode": "content_pinned_external_files", "scene_exports": exports,
               "files": [{"path": path, "sha256": sha, "bytes": Path(path).stat().st_size}
                         for path, sha in sorted(files.items())],
               "runtime": {"engine": native["engine"], "factory": native["factory"],
                           "factory_file": native["factory_file"],
                           "execution_packets": [p for p in files if Path(p).name == "packet.json"],
                           "note": "Pinned execution packets contain the actual launcher/cwd/environment. Installed simulator/Python binaries and licensed assets remain external; recreate and preflight the recorded runtime."}}
    write_new(output / "dependencies.json", closure)
    handoff = """# Simulation policy handoff

This package is simulation-only and is not approved for robot deployment.
It may require privileged observations and initialized task fixtures. Read
policy_io.json and native_config.json for the exact observation, action,
controller, initialization, release and success contracts. Real validation is
pending; no real-robot performance or safety claim is made.

policy.npz is the byte-identical accepted training checkpoint evaluated in the
sealed final run. normalizer.npz copies its train-derived mean/scale arrays.
Use the pinned decoder and the exact bound I/O; do not refit normalization.

This is NOT a portable, self-contained package. dependencies.json enumerates
content-pinned external scene exports, raw assets, code, runtime inputs and
evidence. Keep those files at their recorded locations, or perform an explicit
relocation/rebinding and requalification. Recreate the external simulator and
Python runtime described by the execution packets and run exact-launcher
preflight. A checksum inventory is not a runtime installation or a fresh test.

The final split has already been consumed. Do not feed final outcomes back to
the agent or reopen it for tuning. New development requires a new contracted
experiment with new final cases. Before any physical robot use, qualify the
observation source, calibration, controller timing, limits, safety and reset
procedure through a separately authorized human-operated handoff.

Packaging does not approve the outer delivery module. Independently inspect
artifact parity, dependencies and this handoff, then author the ordinary delivery
review with reproducible_package, controller_parity, human_handoff and
no_real_success_claim evidence (plus any task-specific contracted checks).
"""
    (output / "HANDOFF.md").write_text(handoff, encoding="utf-8")
    # Acceptance may change while files are copied; retain partial outputs on failure.
    current = _qualified(binding_path, report_path, config_path)
    if current[0] != bound or current[2] != accepted or not unchanged(files):
        raise ValueError("Bound evidence changed during packaging")
    manifest = {"schema": "astra_simulation_policy_package_v1", "status": "pending_independent_delivery_review",
                "delivery_accepted": False, "simulation_only": True, "robot_deployable": False,
                "portable_self_contained": False, "real_validation": "pending",
                "binding_sha256": bound["binding_sha256"], "world_sha256": bound["world_sha256"],
                "task_id": bound["task_id"], "backend_id": bound["backend_id"],
                "checkpoint_sha256": file_hash(checkpoint), "native_config_sha256": file_hash(config_path),
                "acceptance_ids": {name: item["id"] for name, item in accepted.items()},
                "external_dependency_count": len(files), "format": metadata["schema"],
                "files": {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()},
                "review_obligations": plan["modules"]["delivery"]["checks"]}
    write_new(output / "manifest.json", manifest)
    return {"manifest": str(output / "manifest.json"), "manifest_sha256": file_hash(output / "manifest.json"),
            "checkpoint": str(output / "policy.npz"), "policy_io": str(output / "policy_io.json"),
            "delivery_accepted": False, "robot_deployable": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--agent-report", required=True)
    parser.add_argument("--native-config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(package(args.binding, args.agent_report, args.native_config, args.output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
