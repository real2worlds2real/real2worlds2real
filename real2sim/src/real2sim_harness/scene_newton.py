"""Portable materialization contract for a single-solver Newton scene.

This module needs no Newton import. Actual dynamics belong to the selected
runtime adapter; source bundle validation is not target-backend verification.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .artifacts import atomic_json, sha256_file


def materialization(bundle):
    kind = bundle.get("asset_kind")
    if bundle.get("native_representation"):
        mode = bundle["native_representation"]
        required = {"manifest.json", "physics_parameters.json", "visual.usdc", "native_configuration.json", "native_test_configuration.json", "source.blend"}
        if mode in {"surface", "volume"}:
            required.update({"sim_mesh.json", "surface_binding.json", "surface_binding.npz"})
        missing = sorted(required-set(bundle.get("files", {})))
        return {"ok": not missing and mode in {"surface", "volume", "rigid_contact"},
                "missing_files": missing, "native_representation": mode,
                "source_backend": bundle.get("backend"), "target_backend": "newton",
                "asset_kind": kind, "visual_entrypoint": "visual.usdc",
                "loader": "real2sim_harness.scene_newton_runtime", "target_simulation_verified": False,
                "required_files": sorted(required)}
    if kind == 'fluid':
        required={'visual.usdc','fluid_configuration.json','render_source.blend'}
        missing=sorted(required-set(bundle.get('files',{})))
        return {'ok':not missing,'source_backend':bundle.get('backend'),'target_backend':'newton',
                'asset_kind':kind,'visual_entrypoint':'visual.usdc','loader':'real2sim_harness.scene_fluid',
                'target_simulation_verified':False,'required_files':sorted(required),'missing_files':missing}
    required = {"manifest.json", "physics_parameters.json", "visual.usdc"}
    if kind in {"rigid", "articulated", "hybrid"}:
        required.add("collision_meshes.json")
    if kind in {"deformable", "hybrid"}:
        required.update({"sim_mesh.json", "surface_binding.json", "surface_binding.npz"})
    if kind == "hybrid":
        required.add("hybrid_configuration.json")
    if kind not in {"rigid", "articulated", "deformable", "hybrid"}:
        return {
            "ok": False,
            "reason": "This Newton scene materializer does not yet bind this asset kind",
        }
    missing = sorted(required - set(bundle.get("files", {})))
    if missing:
        return {
            "ok": False,
            "reason": "Retained native construction data is missing",
            "missing_files": missing,
        }
    return {
        "ok": True,
        "source_backend": bundle.get("backend"),
        "target_backend": "newton",
        "asset_kind": kind,
        "visual_entrypoint": "visual.usdc",
        "loader": "real2sim_harness.scene_newton_runtime",
        "target_simulation_verified": False,
        "required_files": sorted(required),
    }


def package_hybrid_for_scene(attempt, output):
    """Retain a source-matched successful hybrid review as a portable asset.

    This preserves native construction and visual files; it does not certify
    that the resulting object has passed the new scene's interaction tests.
    """
    attempt, output = Path(attempt).resolve(), Path(output).resolve()
    detail = json.loads((attempt / "report.json").read_text()).get("hybrid", {})
    source = (attempt / detail.get("directory", "")).resolve()
    if not detail.get("ok") or not source.is_relative_to(attempt) or source == attempt:
        raise ValueError("A successful retained hybrid review is required")
    blend = attempt / "scene.blend"
    if sha256_file(blend) != detail.get("source_scene_sha256"):
        raise ValueError("Hybrid review no longer matches the editable source")
    inventory = detail.get("cache_manifest", {})
    if not inventory or output.exists() or output.is_relative_to(source):
        raise ValueError("Use a fresh destination and a source-matched hybrid inventory")
    inputs = {}
    for name, digest in inventory.items():
        path = (attempt / name).resolve()
        if not path.is_relative_to(source) or not path.is_file() or sha256_file(path) != digest:
            raise ValueError("Hybrid source changed or escaped its inventory: " + name)
        inputs[path.relative_to(source)] = (path, digest)
    required = {
        "manifest.json",
        "physics_parameters.json",
        "visual.usdc",
        "collision_meshes.json",
        "sim_mesh.json",
        "surface_binding.json",
        "surface_binding.npz",
        "hybrid_configuration.json",
    }
    if not required <= {str(path) for path in inputs}:
        raise ValueError("The hybrid review is missing required native construction data")
    output.mkdir(parents=True)
    files = {}
    for name, (path, digest) in inputs.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if sha256_file(target) != digest:
            raise ValueError("Hybrid source changed while packaging")
        files[str(name)] = digest
    report = {
        "schema_version": 1,
        "ok": True,
        "status": "verified_source_for_scene_materialization",
        "asset_kind": "hybrid",
        "backend": "newton",
        "entrypoint": "visual.usdc",
        "visual_source": "visual.usdc",
        "source_blend": str(blend),
        "source_blend_sha256": sha256_file(blend),
        "source_review_sha256": sha256_file(attempt / "report.json"),
        "files": files,
        "scene_simulation_verified": False,
        "real_physics_calibrated": False,
    }
    atomic_json(output / "asset_bundle.json", report)
    return output
