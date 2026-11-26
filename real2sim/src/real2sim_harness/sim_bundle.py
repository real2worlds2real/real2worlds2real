"""Prepare a reusable simulator bundle from an existing Blender asset."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

from .artifacts import atomic_json, sha256_file
from .sim_parameters import load_parameters


def prepare_bundle(source, output, blender, *, steps=120, render=False, timeout=900, physics_parameters=None):
    source, output, blender = Path(source).resolve(), Path(output).resolve(), Path(blender).resolve()
    root = Path(__file__).resolve().parents[2]
    local_runtime = os.environ.get("REAL2SIM_SIM_RUNTIME", "docker") == "local"
    if not source.is_file() or not blender.is_file():
        raise FileNotFoundError("A saved .blend and a Blender executable are required")
    if output == root or (not local_runtime and not output.is_relative_to(root)):
        raise ValueError("The current container launcher requires an output directory inside the project")
    if source.is_relative_to(output):
        raise ValueError("Write the simulator bundle separately from the Blender source")
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "conversion_logs"
    logs.mkdir(exist_ok=True)
    source_digest = sha256_file(source)
    progress = []
    active_stage = "initialization"

    def runtime(path):
        if local_runtime:
            return str(Path(path).resolve())
        return str(Path("/workspace/real2sim_harness") / Path(path).resolve().relative_to(root))

    def execute(name, command):
        nonlocal active_stage
        active_stage = name
        print(f"SIM_BUNDLE_STEP {name}", flush=True)
        try:
            with (logs / f"{name}.log").open("w") as stream:
                completed = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                                           timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            progress.append({"name": name, "returncode": None, "timed_out": True,
                             "log": str((logs / f"{name}.log").relative_to(output))})
            raise
        progress.append({"name": name, "returncode": completed.returncode,
                         "log": str((logs / f"{name}.log").relative_to(output))})
        if completed.returncode:
            raise RuntimeError(f"{name} failed with code {completed.returncode}; see {logs / (name + '.log')}")

    def isaac(name, script, *arguments):
        execute(name, [str(root / "scripts/run_isaac_lab3.sh"),
                       runtime(root / "scripts" / script), *map(str, arguments)])

    try:
        execute("blender_export", [str(blender), "--background", str(source), "--python",
                                   str(root / "scripts/export_blend_for_isaac.py"), "--", str(output)])
        manifest = json.loads((output / "manifest.json").read_text())
        active_stage = "parameter_validation"
        parameters = load_parameters(output, create=True)
        if physics_parameters is not None:
            if not isinstance(physics_parameters, dict):
                raise ValueError("Physical parameter overrides must be a JSON object")
            for key, value in physics_parameters.items():
                if isinstance(value, dict) and isinstance(parameters.get(key), dict):
                    parameters[key].update(value)
                else:
                    parameters[key] = value
            atomic_json(output / "physics_parameters.json", parameters)
            load_parameters(output)
        kind = manifest["asset_kind"]
        device = os.environ.get("REAL2SIM_PHYSICS_DEVICE", "cuda:0")
        if device != "cpu" and not re.fullmatch(r"cuda:[0-9]+", device):
            raise ValueError("REAL2SIM_PHYSICS_DEVICE must be cpu or cuda:<device index>")
        render_device = os.environ.get("REAL2SIM_RENDER_DEVICE", device if device.startswith("cuda:") else "cuda:0")
        if not re.fullmatch(r"cuda:[0-9]+", render_device):
            raise ValueError("REAL2SIM_RENDER_DEVICE must be cuda:<device index>")
        common = ["--steps", str(steps), "--viz", "none", "--device", device]
        if kind in {"rigid", "articulated"}:
            isaac("physics_authoring", "author_physx_articulation.py", runtime(output))
            entrypoint = "physx.usda"
            report_name = "physx_validation.json" if kind == "articulated" else "rigid_validation.json"
            arguments = [runtime(output / entrypoint), *common,
                         "--snapshot", runtime(output / "simulation_final.usda")]
            if kind == "articulated":
                arguments.append("--all-joints")
            isaac("conversion_consistency", "audit_sim_conversion.py", runtime(output))
            isaac("simulation_validation", "simulate_physx_asset.py" if kind == "articulated" else "simulate_rigid_asset.py", *arguments)
        elif kind == "deformable":
            isaac("physics_authoring", "author_newton_surface.py", runtime(output))
            isaac("visual_binding", "isaac_surface_binding.py", runtime(output))
            entrypoint, report_name = "newton_surface.usda", "newton_validation.json"
            data = json.loads((output / "sim_mesh.json").read_text())
            protocol = parameters["constraints"]
            extra = ["--pin-mode", "source", "--lift-distance", str(protocol["lift_distance_m"])] if data["pin_indices"] else ["--pin-mode", "none"]
            for key, flag in (("lift_duration_seconds", "--lift-seconds"),
                              ("lift_delay_seconds", "--lift-delay-seconds"),
                              ("quiet_start_seconds", "--quiet-start-seconds"),
                              ("quiet_speed_tolerance_m_s", "--quiet-speed-tolerance")):
                if protocol[key] is not None:
                    extra.extend([flag, str(protocol[key])])
            isaac("conversion_consistency", "audit_sim_conversion.py", runtime(output))
            isaac("simulation_validation", "simulate_newton_surface.py", runtime(output / entrypoint),
                  *common, *extra, "--trajectory", runtime(output / "simulation_motion.npz"),
                  "--snapshot", runtime(output / "simulation_final.usda"))
        else:
            raise ValueError(f"No verified simulator adapter for asset kind {kind!r}")
        validation = json.loads((output / report_name).read_text())
        if not validation.get("ok"):
            raise RuntimeError("The simulator reported a failed asset check")
        active_stage = "source_integrity"
        if sha256_file(source) != source_digest:
            raise RuntimeError("The Blender source changed during conversion")
        if render:
            isaac("isaac_visual_check", "render_usd_asset.py", runtime(output / "simulation_final.usda"),
                  "--output", runtime(output / "isaac_views"), "--resolution", "640",
                  "--viz", "none", "--device", render_device)
        active_stage = "bundle_packaging"
        names = {"manifest.json", "visual.usdc", "physics_parameters.json", entrypoint,
                 report_name, "simulation_final.usda", "physics_manifest.json", "collision_meshes.json",
                 "conversion_validation.json", "visual.materials.json", "visual.normals.json"}
        if kind == "deformable":
            names.update({"sim_mesh.json", "surface_binding.json", "surface_binding.npz", "simulation_motion.npz"})
        for subfolder in ("textures", "shaders", "evidence_dependencies", *( ["isaac_views"] if render else [] )):
            names.update(str(path.relative_to(output)) for path in (output / subfolder).rglob("*") if path.is_file())
        files = {name: sha256_file(output / name) for name in sorted(names) if (output / name).is_file()}
        report = {
            "schema_version": 1, "status": "simulation_verified", "ok": True,
            "asset_kind": kind, "backend": "newton_vbd" if kind == "deformable" else "physx",
            "source_blend": str(source), "source_blend_sha256": source_digest,
            "entrypoint": entrypoint, "parameters": "physics_parameters.json",
            "visual_source": "visual.usdc", "validation": report_name,
            "visual_render_verified": render, "real_physics_calibrated": False,
            "robot_task_implemented": False, "files": files, "steps": progress,
            "physics_device": device,
            "render_device": render_device if render else None,
        }
    except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as exc:
        failed_stage = active_stage
        report = {"schema_version": 1, "ok": False, "status": "conversion_failed",
                  "source_blend": str(source), "source_blend_sha256": source_digest,
                  "failed_stage": failed_stage, "error": str(exc), "steps": progress}
        # Failed motion can still produce an inspectable real simulator state.
        # Return that evidence without promoting a failed asset to success.
        if render and failed_stage == "simulation_validation" and (output / "simulation_final.usda").is_file():
            try:
                isaac("failed_state_visual_check", "render_usd_asset.py", runtime(output / "simulation_final.usda"),
                      "--output", runtime(output / "failed_isaac_views"), "--resolution", "640",
                      "--viz", "none", "--device", render_device)
                report["failure_images"] = [f"failed_isaac_views/{name}.png" for name in ("front", "side", "top")
                                            if (output / "failed_isaac_views" / f"{name}.png").is_file()]
            except (RuntimeError, ValueError, OSError, subprocess.TimeoutExpired) as render_error:
                report["failure_render_error"] = str(render_error)
    atomic_json(output / "asset_bundle.json", report)
    return report


def verify_bundle(folder):
    folder = Path(folder).resolve()
    report = json.loads((folder / "asset_bundle.json").read_text())
    failures = []
    if not report.get("ok") or not report.get("files"):
        failures.append("bundle_has_no_successful_conversion")
    for name, digest in report.get("files", {}).items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder) or not path.is_file() or sha256_file(path) != digest:
            failures.append(f"missing_or_changed:{name}")
    return {"ok": not failures, "failures": failures,
            "checked_files": len(report.get("files", {})), "asset_kind": report.get("asset_kind")}
