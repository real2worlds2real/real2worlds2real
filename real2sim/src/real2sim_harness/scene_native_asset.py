"""Retain source-bound native assets for scene materialization, without flattening physics."""

from __future__ import annotations

import json
import shutil
import subprocess
from copy import deepcopy
from pathlib import Path

from .artifacts import atomic_json, sha256_file
from .sim_parameters import DEFAULTS


def representation(report, kind):
    selected = report.get("selected_simulation_review")
    if kind == "rigid" and selected == "rigid_contact":
        return "rigid_contact"
    if kind == "deformable" and selected == "native_deformable":
        return "surface"
    if kind == "solid" and report.get("solid", {}).get("ok"):
        return "volume"
    return None


def package_native_for_scene(executor, attempt, output):
    attempt, output = Path(attempt).resolve(), Path(output).resolve()
    report = json.loads((attempt / "report.json").read_text())
    kind = report["contract"]["expected"]["kind"]
    mode = representation(report, kind)
    if output.exists() or mode is None:
        raise ValueError("Use a fresh destination and a supported native source representation")
    backend = "newton"
    if mode == "rigid_contact":
        from .contact_asset import verify_evidence
        detail = report["rigid_contact"]
        native = verify_evidence(attempt, detail)
        request = json.loads((native / "request.json").read_text())
        passive = json.loads((native / "runtime.json").read_text())
        followers = []
    else:
        detail = report["native_deformable" if mode == "surface" else "solid"]
        if mode == "surface":
            from .source_deformable import verify_evidence
            native = verify_evidence(attempt, detail)
            if json.loads((native / "surface_asset.json").read_text())["backend"] == "physx_particle_cloth":
                backend = "physx"
        else:
            native = (attempt / detail["directory"]).resolve()
            if not native.is_relative_to(attempt) or native == attempt:
                raise ValueError("Native volume evidence escaped its attempt")
            if detail.get("source_scene_sha256") != sha256_file(attempt / "scene.blend"):
                raise ValueError("Native volume source changed")
            for name, digest in detail.get("cache_manifest", {}).items():
                path = (attempt / name).resolve()
                if not path.is_relative_to(native) or sha256_file(path) != digest:
                    raise ValueError("Native volume evidence changed")
            if not detail.get("ok") or not detail.get("cache_manifest"):
                raise ValueError("Successful native volume evidence is required")
        request = json.loads((native / "behavior_request.json").read_text())
        from .source_deformable import passive_configuration
        passive = passive_configuration(request)
        if backend == "physx":
            # Reuse the actual resolved passive configuration, not the diagnostic
            # request's defaults, grips or test environment.
            passive = json.loads((native / "runtime.json").read_text())
            if any(p.get("source_visual_objects") for p in passive.get("probes", [])):
                raise ValueError("Export fixed source fixtures as separate scene participants")
            passive = dict(passive, probes=[], grasps=[], measurements=[])
        followers = [row["name"] for row in json.loads((native / "visual_binding.json").read_text())["visuals"]]
    output.mkdir(parents=True)
    source = attempt / "scene.blend"
    digest = sha256_file(source)
    root = Path(__file__).resolve().parents[2]
    with (output / "export.log").open("w") as stream:
        process = subprocess.run([str(executor.blender), "--background", str(source), "--python-exit-code", "1", "--python",
                        str(root / "scripts/export_blend_for_isaac.py"), "--", str(output)],
                       stdout=stream, stderr=subprocess.STDOUT, check=False,
                       timeout=executor.timeout_seconds)
    if process.returncode or not (output / "manifest.json").is_file():
        raise RuntimeError("Native scene visual export failed: " + (output/"export.log").read_text(errors="replace")[-3000:])
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest["source_blend_sha256"] != digest or sha256_file(source) != digest:
        raise ValueError("Native source changed during scene export")
    manifest.update(native_representation=mode, native_configuration="native_configuration.json",
                    deformable_visual_objects=followers)
    atomic_json(output / "manifest.json", manifest)
    atomic_json(output / "native_configuration.json", passive)
    atomic_json(output / "native_test_configuration.json", request)
    shutil.copy2(source, output / "source.blend")
    parameters = deepcopy(DEFAULTS)
    env = request.get("environment", {})
    parameters["environment"].update(gravity=env.get("gravity", request.get("gravity", [0, 0, -9.81])),
                                      air_drag_rate_per_second=env.get("air_drag_rate_per_second", 0))
    parameters["material"]["friction"] = env.get("friction", request.get("contact", {}).get("friction", .5))
    settings = request.get("solver", {})
    parameters["solver"].update(substeps=settings.get("substeps", request.get("substeps", 12)),
                                iterations=settings.get("iterations", request.get("iterations", 20)),
                                self_contact=settings.get("self_contact", mode == "surface"))
    if followers:
        from .usd_surface_binding import build
        g = request["geometry"]
        atomic_json(output / "sim_mesh.json", {"vertices": g["vertices"], "faces": g["triangles"], "pin_indices": []})
        if backend == "newton":
            parameters["surface"]["particle_radius"] = request["material"]["contact_radius_m"]
        build(output)
    atomic_json(output / "physics_parameters.json", parameters)
    files = {str(p.relative_to(output)): sha256_file(p) for p in output.rglob("*")
             if p.is_file() and p.suffix != ".log"}
    atomic_json(output / "asset_bundle.json", {
        "schema_version": 1, "ok": True, "status": "verified_source_for_scene_materialization",
        "asset_kind": kind, "backend": backend, "native_representation": mode,
        **({"cloth_configuration": "native_configuration.json"} if backend == "physx" else {}),
        "entrypoint": "visual.usdc", "visual_source": "visual.usdc", "files": files,
        "source_blend": str(source), "source_blend_sha256": digest,
        "source_review_sha256": sha256_file(attempt / "report.json"),
        "scene_simulation_verified": False, "real_physics_calibrated": False,
        "scope": "Retained native construction, passive configuration and source visual mapping; target scene dynamics require separate verification.",
    })
    return output
