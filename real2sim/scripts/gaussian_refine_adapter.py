"""Optional rendered-view repair, geometry-aware propagation and actual 3D color refinement."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from real2sim_harness.artifacts import atomic_json, sha256_file
from real2sim_harness.gaussian_background import read_gaussians
from real2sim_harness.gaussian_refinement import propagate_anchors
from real2sim_harness.posed_observations import camera_spec, load_views


def invoke(config_path, replacements):
    config_path = Path(config_path).resolve()
    digest = sha256_file(config_path)
    config = json.loads(config_path.read_text())
    if (
        config.get("schema_version") != 1
        or not isinstance(config.get("argv"), list)
        or not config["argv"]
    ):
        raise ValueError("Use an operator-owned argv runner, schema_version=1")
    argv = []
    for value in config["argv"]:
        if not isinstance(value, str) or not value:
            raise ValueError("Runner argv must contain nonempty strings")
        for key, replacement in replacements.items():
            value = value.replace("{" + key + "}", str(replacement))
        argv.append(value)
    subprocess.run(argv, cwd=config_path.parent, check=True, timeout=1800)
    if sha256_file(config_path) != digest:
        raise ValueError("Nested runner configuration changed during refinement")
    return {"path": str(config_path), "sha256": digest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-runner", type=Path, required=True)
    parser.add_argument("--restoration-runner", type=Path, required=True)
    parser.add_argument("--base-runner", type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if request["operation"] != "refine":
        if args.base_runner is None:
            raise ValueError("This adapter needs --base-runner for train/transcode delegation")
        invoke(args.base_runner, {"request": args.request.resolve(), "output": output})
        return
    observations = Path(request["observations"]).resolve()
    observed_request = json.loads(observations.read_text())
    captures, hashes = load_views(observed_request, observations.parent, required=("image",))
    if len(captures) < 2:
        raise ValueError("Refinement needs at least two registered real capture views")
    source = Path(request["source"]).resolve()
    read_gaussians(source)
    hashes[str(source)] = sha256_file(source)
    hashes[str(observations)] = sha256_file(observations)
    novel = request.get("novel_views", [])
    if not isinstance(novel, list) or not 1 <= len(novel) <= 32:
        raise ValueError("Supply 1..32 explicitly selected novel cameras within capture coverage")
    maximum = float(request.get("max_novel_distance_m", 0.5))
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("max_novel_distance_m must be finite positive in the source frame")
    cameras = []
    for row in novel:
        camera = camera_spec(row)
        nearest = min(np.linalg.norm(camera["center"] - view["center"]) for view in captures)
        if nearest > maximum:
            raise ValueError("Novel camera exceeds the declared capture-neighborhood limit")
        cameras.append(camera)
    raw_views = observed_request["views"] + novel
    render_request = output / "render_request.json"
    atomic_json(
        render_request, {"views": raw_views, "min_opacity": request.get("min_opacity", 0.95)}
    )
    render_root = output / "before"
    dependencies = [
        invoke(
            args.worker_runner,
            {"mode": "render", "request": render_request, "source": source, "output": render_root},
        )
    ]
    rendered = json.loads((render_root / "render_report.json").read_text())["views"]
    if len(rendered) != len(raw_views):
        raise ValueError("Renderer returned an incorrect view count")
    offset = len(captures)
    count = request.get("anchors", min(2, len(novel)))
    if type(count) is not int or not 1 <= count <= min(8, len(novel)):
        raise ValueError("Use 1..8 anchors, bounded by the novel view count")
    # Spread anchors over the agent-selected camera sequence; camera planning remains optional agent work.
    indices = np.linspace(0, len(novel) - 1, count, dtype=int).tolist()
    restore_request = output / "restoration_request.json"
    reference_indices = []
    for i in indices:
        camera = cameras[i]
        # Actual photographs condition compatible restoration backends. This is a
        # reference-selection heuristic, not a visibility certificate.
        scores = [np.linalg.norm(camera["center"] - view["center"])
                  + .25 * (1 - np.clip(camera["pose"][2, :3] @ view["pose"][2, :3], -1, 1))
                  for view in captures]
        reference_indices.append(int(np.argmin(scores)))
    atomic_json(
        restore_request,
        {"images": [str(render_root / rendered[offset + i]["image"]) for i in indices],
         "references": [str(captures[i]["image_path"]) for i in reference_indices]},
    )
    repair_root = output / "restored"
    dependencies.append(
        invoke(args.restoration_runner, {"request": restore_request, "output": repair_root})
    )
    repaired = json.loads((repair_root / "restoration_report.json").read_text())
    if repaired.get("ok") is not True or len(repaired["images"]) != len(indices):
        raise ValueError("Restoration backend failed or omitted anchors")
    anchors = []
    tolerances = {
        "absolute_tolerance_m": request.get("depth_tolerance_m", 0.02),
        "relative_tolerance": request.get("relative_depth_tolerance", 0.02),
    }
    for i, entry in zip(indices, repaired["images"], strict=True):
        depth = np.load(render_root / rendered[offset + i]["depth"], allow_pickle=False)
        rgb = np.asarray(Image.open(repair_root / entry["image"]).convert("RGB")).copy()
        if rgb.shape[:2] != depth.shape:
            raise ValueError("Restoration changed the camera pixel domain")
        previous, valid, _ = propagate_anchors(anchors, depth, cameras[i], **tolerances)
        rgb[valid] = previous[valid]
        anchors.append({"rgb": rgb, "depth": depth, "camera": cameras[i]})
    training_root = output / "training_inputs"
    training_root.mkdir(exist_ok=True)
    training_views = []
    for i, (row, view) in enumerate(zip(observed_request["views"], captures, strict=True)):
        image = training_root / f"capture-{i:03d}.png"
        mask_path = training_root / f"capture-{i:03d}-mask.png"
        Image.fromarray(view["image"]).save(image)
        mask = np.load(render_root / rendered[i]["depth"], allow_pickle=False) > 0
        if "mask" in view:
            mask &= view["mask"] > 0
        if "ignore" in view:
            mask &= view["ignore"] == 0
        Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
        training_views.append(
            {**row, "image": str(image), "mask": str(mask_path), "role": "retained_capture"}
        )
    coverage = []
    for i, (row, camera) in enumerate(zip(novel, cameras, strict=True)):
        depth = np.load(render_root / rendered[offset + i]["depth"], allow_pickle=False)
        rgb, valid, detail = propagate_anchors(anchors, depth, camera, **tolerances)
        image = training_root / f"pseudo-{i:03d}.png"
        mask = training_root / f"pseudo-{i:03d}-mask.png"
        Image.fromarray(rgb).save(image)
        Image.fromarray((valid * 255).astype(np.uint8)).save(mask)
        coverage.append(detail)
        training_views.append(
            {
                **row,
                "image": str(image),
                "mask": str(mask),
                "role": "generated_depth_consistent_pseudo_view",
            }
        )
    if not any(row["covered_pixels"] for row in coverage):
        raise ValueError(
            "No depth-consistent anchor coverage; inspect source geometry/opacity/camera choices"
        )
    training_request = output / "training_request.json"
    atomic_json(
        training_request,
        {
            "views": training_views,
            "iterations": request.get("iterations", 200),
            "learning_rate": request.get("learning_rate", 0.0025),
            "geometry_mode": request.get("geometry_mode", "color_only"),
        },
    )
    trained_root = output / "color_refinement"
    dependencies.append(
        invoke(
            args.worker_runner,
            {
                "mode": "train",
                "request": training_request,
                "source": source,
                "output": trained_root,
            },
        )
    )
    trained = json.loads((trained_root / "training_report.json").read_text())
    if (
        request.get("geometry_mode", "color_only") == "color_only"
        and trained.get("geometry_and_opacity_unchanged") is not True
    ):
        raise ValueError("Worker omitted geometry/opacity preservation check")
    shutil.copy2(trained_root / "refined.ply", output / "refined.ply")
    read_gaussians(output / "refined.ply")
    after_root = output / "after"
    dependencies.append(
        invoke(
            args.worker_runner,
            {
                "mode": "render",
                "request": render_request,
                "source": output / "refined.ply",
                "output": after_root,
            },
        )
    )
    previews = []
    after = json.loads((after_root / "render_report.json").read_text())["views"]
    capture_errors = []
    for i, view in enumerate(captures):
        before = np.asarray(
            Image.open(render_root / rendered[i]["image"]).convert("RGB"), dtype=float
        )
        current = np.asarray(Image.open(after_root / after[i]["image"]).convert("RGB"), dtype=float)
        mask = np.asarray(Image.open(training_views[i]["mask"]).convert("L")) > 128
        if mask.any():
            target = view["image"].astype(float)
            capture_errors.append(
                {
                    "view": i,
                    "pixels": int(mask.sum()),
                    "before_mae_rgb255": float(np.abs(before[mask] - target[mask]).mean()),
                    "after_mae_rgb255": float(np.abs(current[mask] - target[mask]).mean()),
                }
            )
    for i in indices[:4]:
        before = np.asarray(Image.open(render_root / rendered[offset + i]["image"]).convert("RGB"))
        current = np.asarray(Image.open(after_root / after[offset + i]["image"]).convert("RGB"))
        name = f"comparison-{i:03d}.png"
        Image.fromarray(np.concatenate([before, current], axis=1)).save(output / name)
        previews.append(name)
    for name, digest in hashes.items():
        if sha256_file(Path(name)) != digest:
            raise ValueError("Refinement source observations/model changed during execution")
    atomic_json(
        output / "background_report.json",
        {
            "schema_version": 1,
            "ok": True,
            "operation": "refine",
            "backend": trained["backend"],
            "restoration_backend": repaired["backend"],
            "source_ply": "refined.ply",
            "input_hashes": hashes,
            "runtime_dependencies": dependencies,
            "geometry_mode": trained.get("geometry_mode"),
            "geometry_and_opacity_unchanged": trained.get("geometry_and_opacity_unchanged"),
            "preview_images": previews,
            "coverage": coverage,
            "capture_errors": capture_errors,
            "simulation_verified": False,
            "limits": [
                "Restoration and pseudo views are generated proposals, not observations.",
                "Gaussian depth is model-estimated. Reject invalid/disoccluded pixels rather than invent depth.",
                "Actual original captures remain training targets; hidden or low-opacity holes are not certified.",
                "Color-only optimization preserves existing geometry and opacity; poor geometry still needs reconstruction repair.",
                "Completion does not establish improved quality; inspect the before/after views and final scene.",
            ],
        },
    )


if __name__ == "__main__":
    main()
