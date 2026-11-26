"""Source-independent numerical feedback for an actual scene MPM execution."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .artifacts import sha256_file
from .fluid_environment import audit_static_boxes


def runtime_report(scene, output, *, expected_frames, fps):
    scene, output = Path(scene), Path(output)
    config = json.loads((output / "fluid_configuration.json").read_text())
    bindings = json.loads((output / "fluid_scene_bindings.json").read_text())
    source = json.loads((output / "mpm/fluid_report.json").read_text())
    if not source.get("runtime", {}).get("version"):
        raise ValueError("MPM runner must report its actual runtime version")
    expected = len(config["parts"])
    with np.load(output / "mpm/particle_motion.npz", allow_pickle=False) as data:
        trajectories = [data[f"part_{index:03d}"].copy() for index in range(expected)]
    finite = all(
        p.ndim == 3
        and p.shape == (expected_frames, len(config["parts"][i]["positions"]), 3)
        and np.isfinite(p).all()
        for i, p in enumerate(trajectories)
    )
    boxes = config["static_colliders"]
    tolerance = config["solver"]["voxel_size"] * 0.5
    static = audit_static_boxes(
        np.concatenate([p.reshape(-1, 3) for p in trajectories]), boxes, tolerance_m=tolerance
    )
    static["native_shape_count"] = source.get("static_environment_check", {}).get(
        "native_shape_count", -1
    )
    static["ok"] = bool(static["ok"] and static["native_shape_count"] >= len(boxes))
    support = []
    for name, asset in bindings["assets"].items():
        points = np.concatenate([trajectories[part["global_index"]][-1] for part in asset["parts"]])
        matches = []
        for box in boxes:
            size, center = np.asarray(box["size_m"]), np.asarray(box["center_m"])
            top = center[2] + size[2] / 2
            inside = (np.abs(points[:, :2] - center[:2]) <= size[:2] / 2).all(axis=1)
            near = np.abs(points[:, 2] - top) <= config["solver"]["voxel_size"] * 2
            count = int((inside & near).sum())
            if count:
                matches.append(
                    {"support": box["id"], "near_surface_particles": count, "top_m": float(top)}
                )
        support.append({"id": name, "ok": bool(matches), "supports": matches})
    frames = sorted((output / "mpm/surfaces").glob("frame_*.npz"))
    surfaces_ok = len(frames) == expected_frames
    for path in frames:
        with np.load(path, allow_pickle=False) as data:
            for index in range(expected):
                key = f"part_{index:03d}"
                vertices, faces = data[key + "_vertices"], data[key + "_faces"]
                surfaces_ok = surfaces_ok and bool(
                    len(vertices)
                    and len(faces)
                    and np.isfinite(vertices).all()
                    and faces.ndim == 2
                    and faces.shape[1] == 3
                    and faces.min() >= 0
                    and faces.max() < len(vertices)
                )
    checks = [
        {"name": "scene_loaded", "ok": True},
        {"name": "finite_state", "ok": bool(finite)},
        {
            "name": "fluid_state",
            "ok": bool(
                source.get("ok")
                and finite
                and source.get("frames") == expected_frames
                and source.get("fps") == fps
            ),
            "method": "Actual MPM particle trajectories and native report",
        },
        {"name": "surface_integrity", "ok": bool(surfaces_ok), "frames": len(frames)},
        {"name": "static_environment_integrity", **static},
        {
            "name": "environment_contacts",
            "ok": all(row["ok"] for row in support),
            "assets": support,
            "method": "Observed particle support with finite native colliders and gravity; no direct contact force measured",
        },
    ]
    return {
        "schema_version": 1,
        "ok": all(check["ok"] for check in checks),
        "scene_sha256": sha256_file(scene / "scene.usda"),
        "runtime": source["runtime"],
        "checked_asset_ids": list(bindings["assets"]),
        "checks": checks,
        "images": [],
        "videos": [],
        "frames": expected_frames,
        "fps": fps,
        "real_physics_calibrated": False,
        "scope": "MPM liquids sharing scene-authored static boxes; dynamic rigid/fluid and cloth/fluid coupling not implemented",
    }
