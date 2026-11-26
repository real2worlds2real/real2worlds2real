"""Unbound native qualification probe; never accepted as a policy job receipt."""
import argparse
import importlib
from pathlib import Path
import sys
import traceback

import numpy as np

from bound_native import run_episode
from bound_training import read, write_new
from bound_insertion_config import policy_io
from astra_e2e.layout import source_roots
from astra_e2e.oversight.evidence import digest, file_hash, unchanged


def probe(session, config_path, output, *, method="servo", episodes=3, seed_start=91000):
    config = read(config_path)
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Qualification probe output must be a new directory")
    output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[2]
    source_snapshot = {str(p.resolve()): file_hash(p) for folder in (*source_roots(root), root / "end_to_end" / "unified", root / "policy_development")
                       for p in folder.rglob("*.py") if "__pycache__" not in p.parts}
    io = config.get("policy_io", policy_io())
    probe_context = {"schema": "astra_unbound_native_probe_v1", "world_sha256": "unbound_native_qualification_only",
                     "task_id": "source_scene_insertion_qualification", "backend_id": config["backend_id"],
                     "native_session": str(Path(session).resolve()), "policy_io": io}
    name, factory = config["factory"].split(":")
    module = importlib.import_module(name)
    if Path(module.__file__).resolve() != Path(config["factory_file"]).resolve():
        raise ValueError("Probe factory path does not match configured source")
    environment = None
    results = []
    try:
        environment = getattr(module, factory)(probe_context, {"operation": "qualification_probe",
                                                               "diagnostic_directory": str(output)}, config)
        metadata = environment.metadata()
        policy = environment.controller(method, config["collectors"][method])
        for seed in range(episodes):
            case = {"split": "qualification_only", "group": "unbound_probe", "seed": seed_start + seed,
                    "stratum": "nominal", "repeat": 0}
            trace = run_episode(environment, case, policy, obs_dim=3, action_dim=3,
                                low=np.asarray(config["action_low"]), high=np.asarray(config["action_high"]),
                                max_steps=config["max_steps"], success_ticks=config["success_ticks"])
            trace["teacher_used"] = True
            path = output / f"probe_{seed:03d}.json"
            write_new(path, trace)
            results.append({"trace": str(path), "sha256": file_hash(path), "success": trace["success"],
                            "failure": trace["failure"], "initial": trace["initial_measurement"],
                            "final": trace["steps"][-1]["measurement"] if trace["steps"] else None})
        try:
            metadata["qualification_image"] = environment.snapshot(output / "native_probe.png")
        except Exception as error:
            metadata["visualization_failure"] = type(error).__name__ + ": " + str(error)
        metadata["final_diagnostics"] = environment.diagnostics()
        loaded = {str(Path(module.__file__).resolve()) for name, module in list(sys.modules.items())
                  if (name.startswith("calibrated_scene_reconstruction.") or name.startswith("bound_") or name == "__main__")
                  and getattr(module, "__file__", None) and Path(module.__file__).suffix == ".py"}
        loaded.add(str(Path(__file__).resolve()))
        if not loaded <= source_snapshot.keys():
            raise ValueError("Probe loaded an implementation outside the pre-execution code snapshot")
        implementation = {path: source_snapshot[path] for path in sorted(loaded)}
        if not unchanged(implementation):
            raise ValueError("Native probe implementation changed during execution")
    except BaseException as error:
        write_new(output / "probe_failure.json", {
            "schema": "astra_unbound_native_probe_failure_v1", "is_policy_receipt": False,
            "config": str(Path(config_path).resolve()), "config_sha256": file_hash(config_path),
            "factory_sha256": file_hash(config["factory_file"]), "episodes": results,
            "implementation_snapshot": source_snapshot,
            "error": type(error).__name__ + ": " + str(error), "traceback": traceback.format_exc()})
        if environment is not None:
            environment.close()
        raise
    report = {"schema": "astra_unbound_native_probe_v1", "is_policy_receipt": False,
              "scope": "Native physics/IK qualification only; no data, training, validation or holdout claim.",
              "method": method, "qualification_seed_start": seed_start,
              "config": str(Path(config_path).resolve()), "config_sha256": file_hash(config_path),
              "factory_sha256": file_hash(config["factory_file"]), "metadata": metadata, "episodes": results,
              "implementation_sha256": implementation, "implementation_unchanged": unchanged(implementation),
              "successes": sum(item["success"] for item in results),
              "initial_attached_outcome_negative": all(item["initial"] and item["initial"]["attachment_active"]
                                                        and not item["initial"]["success"] for item in results),
              "released_success_only": all(not item["success"] or item["final"]["attachment_active"] is False for item in results),
              "independent_semantic_approval": False}
    write_new(output / "probe_report.json", report)
    print({"probe_report": str(output / "probe_report.json"), "successes": report["successes"], "episodes": len(results)}, flush=True)
    environment.close()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("session", "config", "output"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--method", choices=("servo", "mpc"), default="servo")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=91000)
    args = parser.parse_args()
    print(probe(args.session, args.config, args.output, method=args.method, episodes=args.episodes, seed_start=args.seed_start))


if __name__ == "__main__":
    main()
