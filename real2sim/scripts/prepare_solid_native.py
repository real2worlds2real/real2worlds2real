"""Tetrahedralize a Blender control, remap source vertex selections and bind visuals."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from real2sim_harness.artifacts import atomic_json, sha256_file  # noqa: E402
from real2sim_harness.deformable_behavior import validate_behavior  # noqa: E402
from real2sim_harness.deformable_geometry import surface_mesh, tetrahedralize  # noqa: E402
from real2sim_harness.surface_binding import build_binding, deform_surface  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--source-file", default="solid_source.json")
    args = parser.parse_args()
    folder = args.folder
    source_path = (folder / args.source_file).resolve()
    if not source_path.is_relative_to(folder.resolve()):
        raise ValueError("Source geometry record must stay within the native construction folder")
    source = json.loads(source_path.read_text())
    geometry = source.pop("geometry")
    meshing, visuals = source.pop("meshing"), source.pop("visuals")
    if source.get("representation") == "surface":
        vertices, triangles, _ = surface_mesh(geometry["vertices"], geometry["triangles"])
        tet = {"vertices": vertices, "triangles": triangles}
    else:
        tet = tetrahedralize(geometry["vertices"], geometry["triangles"], **meshing)
    distance, mapping = cKDTree(tet["vertices"]).query(np.asarray(geometry["vertices"]))
    if np.max(distance) > 1e-7:
        raise ValueError(
            "Mesher moved an authored control vertex; selections need a new explicit mapping"
        )
    for group in source.get("grasps", []) + source.get("measurements", []):
        ids = np.asarray(group["indices"])
        if (
            ids.ndim != 1
            or not np.issubdtype(ids.dtype, np.integer)
            or not len(ids)
            or ids.min() < 0
            or ids.max() >= len(mapping)
        ):
            raise ValueError("Source selections must name integer source control vertices")
        group["indices"] = mapping[ids].tolist()
    source.update(
        frames=args.frames,
        geometry={
            key: tet[key].tolist() for key in ("vertices", "triangles", "tetrahedra") if key in tet
        },
    )
    validate_behavior(source)
    atomic_json(folder / "behavior_request.json", source)
    arrays, records, maximum = {}, [], 0.0
    for index, visual in enumerate(visuals):
        binding = build_binding(tet["vertices"], tet["triangles"], np.asarray(visual["vertices"]))
        replay = deform_surface(tet["vertices"], tet["triangles"], binding)
        error = float(np.linalg.norm(replay - visual["vertices"], axis=1).max())
        maximum = max(maximum, error)
        for key, value in binding.items():
            arrays[f"visual_{index:03d}_{key}"] = value
        records.append({"name": visual["name"], "index": index, "vertices": len(replay)})
    if maximum > 1e-6:
        raise ValueError("The solid surface binding changed authored geometry")
    np.savez_compressed(folder / "visual_binding.npz", **arrays)
    atomic_json(
        folder / "visual_binding.json",
        {
            "schema_version": 1,
            "visuals": records,
            "static_visual_objects": source.get("source_static_visual_objects", []),
            "rest_binding_max_error_m": maximum,
            "source_scene_sha256": source["source_scene_sha256"],
            "binding_sha256": sha256_file(folder / "visual_binding.npz"),
            "configuration_sha256": sha256_file(folder / "behavior_request.json"),
        },
    )


if __name__ == "__main__":
    main()
