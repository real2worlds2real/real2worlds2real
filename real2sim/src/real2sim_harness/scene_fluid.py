"""Native liquid packages and world-space inputs for the shared scene protocol."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import numpy as np

from .artifacts import atomic_json, sha256_file
from .fluid_environment import static_colliders
from .material_transition import validate_transition


def validate_configuration(configuration):
    if configuration.get("schema_version") != 1:
        raise ValueError("Fluid configuration needs schema_version 1")
    parts = configuration.get("parts")
    if not isinstance(parts, list) or not 1 <= len(parts) <= 100:
        raise ValueError("Fluid configuration needs a bounded nonempty parts list")
    count = 0
    for part in parts:
        points = np.asarray(part.get("positions"), dtype=float)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or len(points) < 50
            or not np.isfinite(points).all()
        ):
            raise ValueError("Fluid parts need at least 50 finite three-dimensional particles")
        if not isinstance(part.get("source_object"), str) or not part["source_object"].strip():
            raise ValueError("Retain the source material object name")
        spacing = part.get("spacing")
        if type(spacing) not in (int, float) or not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("Fluid spacing must be finite and positive")
        material = part.get("material")
        if not isinstance(material, dict) or not material:
            raise ValueError("Retain explicit MPM material parameters")
        if any(
            type(value) not in (int, float) or not np.isfinite(value) or value < 0
            for value in material.values()
        ):
            raise ValueError("MPM material values must be finite and nonnegative")
        if part.get("transition"):
            validate_transition(part["transition"])
        count += len(points)
    if count > 180_000:
        raise ValueError("Combined particle count exceeds the current review budget")
    numerical = configuration.get("solver", {})
    for key in ("voxel_size", "surface_voxel_size"):
        value = numerical.get(key, numerical.get("voxel_size"))
        if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
            raise ValueError("MPM grid and surface spacing must be finite and positive")
    for key, default, maximum in (("substeps", 2, 128), ("max_iterations", 60, 1000)):
        value = numerical.get(key, default)
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError("MPM solver budgets must be bounded positive integers")
    static_colliders(configuration)
    return configuration


def package_fluid_for_scene(executor, attempt, output):
    """Re-export source-matched native construction after successful object review."""
    attempt, output = Path(attempt).resolve(), Path(output).resolve()
    report = json.loads((attempt / "report.json").read_text())
    detail = report.get("fluid", {})
    source = attempt / "scene.blend"
    if not detail.get("ok") or sha256_file(source) != detail.get("source_scene_sha256"):
        raise ValueError("A successful source-matched fluid object review is required")
    folder = (attempt / detail.get("directory", "")).resolve()
    if folder == attempt or not folder.is_relative_to(attempt):
        raise ValueError("Fluid review directory escaped the retained attempt")
    inventory = detail.get("cache_manifest", {})
    if not inventory:
        raise ValueError("Retain the verified fluid cache inventory")
    for name, digest in inventory.items():
        path = (attempt / name).resolve()
        if not path.is_relative_to(folder) or not path.is_file() or sha256_file(path) != digest:
            raise ValueError("Fluid source cache is missing or changed")
    previous = validate_configuration(json.loads((folder / "fluid_configuration.json").read_text()))
    if output.exists() or output.is_relative_to(folder):
        raise ValueError("Choose a fresh fluid package directory")
    output.mkdir(parents=True)
    script = Path(__file__).resolve().parents[2] / "scripts/export_blend_fluid.py"
    with (output / "export.log").open("w") as stream:
        process = subprocess.run(
            [
                str(executor.blender),
                "--background",
                str(source),
                "--python",
                str(script),
                "--",
                str(output / "fluid_configuration.json"),
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=executor.timeout_seconds,
            check=False,
        )
    if process.returncode:
        raise RuntimeError("Fluid scene source export failed; inspect export.log")
    current = validate_configuration(json.loads((output / "fluid_configuration.json").read_text()))
    for key, default in (
        ("parts", None),
        ("solver", None),
        ("floor_z", 0.0),
        ("static_colliders", []),
        ("use_ground_plane", True),
    ):
        if current.get(key, default) != previous.get(key, default):
            raise ValueError("Fluid native construction changed after its accepted review: " + key)
    if current.get("source_scene_sha256") != sha256_file(source):
        raise ValueError("Fluid source changed while packaging")
    files = {
        str(path.relative_to(output)): sha256_file(path)
        for path in output.rglob("*")
        if path.is_file() and path.suffix != ".log"
    }
    if not {"visual.usdc", "fluid_configuration.json", "render_source.blend"} <= set(files):
        raise ValueError("Fluid export omitted required construction or material files")
    atomic_json(
        output / "asset_bundle.json",
        {
            "schema_version": 1,
            "ok": True,
            "asset_kind": "fluid",
            "backend": "newton",
            "status": "verified_source_for_scene_materialization",
            "entrypoint": "visual.usdc",
            "source_blend": str(source),
            "source_blend_sha256": sha256_file(source),
            "source_review_sha256": sha256_file(attempt / "report.json"),
            "files": files,
            "scene_simulation_verified": False,
            "real_physics_calibrated": False,
        },
    )
    return output


def assemble_inputs(scene, output):
    """Place particles in scene metres and make scene supports authoritative."""
    from .scene import verify_scene

    scene, output = Path(scene).resolve(), Path(output).resolve()
    verified = verify_scene(scene)
    if not verified["ok"]:
        raise ValueError("Invalid scene package: " + str(verified["failures"]))
    spec = json.loads((scene / "request.json").read_text())
    if spec["backend"] != "newton":
        raise ValueError("Liquid scenes require the selected Newton backend")
    parts, records, settings = [], {}, []
    for entry in spec["objects"]:
        folder = scene / "assets" / entry["id"]
        bundle = json.loads((folder / "asset_bundle.json").read_text())
        if bundle.get("asset_kind") != "fluid":
            raise ValueError(
                "This MPM scene adapter supports fluids and static supports; dynamic rigid/cloth coupling needs a compatible adapter"
            )
        configuration = validate_configuration(
            json.loads((folder / "fluid_configuration.json").read_text())
        )
        settings.append(configuration["solver"])
        placement = np.asarray(entry["transform"])
        mapping = []
        for index, source in enumerate(configuration["parts"]):
            part = copy.deepcopy(source)
            points = np.asarray(part["positions"]) @ placement[:3, :3].T + placement[:3, 3]
            part["positions"] = points.tolist()
            if part.get("transition"):
                transition = part["transition"]
                if transition.get("height_delay_seconds_per_m", 0) and not np.allclose(
                    placement[2, :3], [0, 0, 1]
                ):
                    raise ValueError(
                        "A rotated spatial transition needs an explicit field-frame adapter"
                    )
                transition["height_origin_m"] = transition.get("height_origin_m", 0) + float(
                    placement[2, 3]
                )
            mapping.append(
                {
                    "global_index": len(parts),
                    "source_part": index,
                    "source_object": source["source_object"],
                }
            )
            parts.append(part)
        records[entry["id"]] = {
            "parts": mapping,
            "render_source": "render_source.blend",
            "transform": entry["transform"],
        }
    boxes = []
    for support in spec["supports"]:
        matrix = np.asarray(support["transform"])
        absolute = np.abs(matrix[:3, :3])
        if (
            not np.allclose(absolute.sum(0), 1)
            or not np.allclose(absolute.sum(1), 1)
            or not np.allclose(absolute, absolute.round())
        ):
            raise ValueError(
                "The current MPM static adapter needs axis-aligned boxes; use an explicit rotated-geometry adapter"
            )
        boxes.append(
            {
                "id": support["id"],
                "size_m": (absolute @ support["size_m"]).tolist(),
                "center_m": matrix[:3, 3].tolist(),
            }
        )
    if not boxes:
        raise ValueError("Declare actual finite scene supports for this liquid scene adapter")
    numerical = {
        "voxel_size": min(s["voxel_size"] for s in settings),
        "surface_voxel_size": min(
            s.get("surface_voxel_size", s["voxel_size"] * 0.5) for s in settings
        ),
        "substeps": max(s.get("substeps", 2) for s in settings),
        "max_iterations": max(s.get("max_iterations", 60) for s in settings),
    }
    configuration = {
        "schema_version": 1,
        "parts": parts,
        "solver": numerical,
        "use_ground_plane": False,
        "static_colliders": boxes,
        "environment_source": "composed_scene_supports",
        "source_scene_sha256": sha256_file(scene / "scene.usda"),
    }
    validate_configuration(configuration)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "fluid_configuration.json", configuration)
    atomic_json(output / "fluid_scene_bindings.json", {"schema_version": 1, "assets": records})
    return configuration
