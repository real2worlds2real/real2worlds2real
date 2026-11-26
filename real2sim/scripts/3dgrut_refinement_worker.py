"""Official 3DGRUT rendering and geometry-frozen SH color refinement in its own environment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["render", "train"], required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.checkout.resolve()))
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from threedgrut.datasets.protocols import Batch
    from threedgrut.model.model import MixtureOfGaussians

    from real2sim_harness.artifacts import atomic_json, sha256_file
    from real2sim_harness.posed_observations import camera_spec

    request = json.loads(args.request.read_text())
    views = request["views"]
    if not 1 <= len(views) <= 100:
        raise ValueError("Use 1..100 bounded views")
    args.output.mkdir(parents=True, exist_ok=True)
    if not OmegaConf.has_resolver("int_list"):
        OmegaConf.register_new_resolver("int_list", lambda items: [int(v) for v in items])
    with initialize_config_dir(
        config_dir=str(args.checkout.resolve() / "configs"), version_base=None
    ):
        conf = compose(
            config_name="apps/colmap_3dgut.yaml",
            overrides=[
                f"path={args.output}",
                f"out_dir={args.output}",
                "num_workers=0",
                "dataset.normalize_world_space=false",
                "model.feature_type=sh",
            ],
        )
    model = MixtureOfGaussians(conf, scene_extent=1.0)
    model.init_from_ply(str(args.source.resolve()))
    model.build_acc()
    frozen = {
        name: getattr(model, name).detach().clone()
        for name in ("positions", "scale", "rotation", "density")
    }
    geometry_mode = request.get("geometry_mode", "color_only")
    if geometry_mode not in ("color_only", "joint_refine"):
        raise ValueError("geometry_mode must be color_only or joint_refine")
    if geometry_mode == "color_only":
        for name in frozen:
            getattr(model, name).requires_grad_(False)

    def batch_for(row):
        camera = camera_spec(row)
        k = camera["K"]
        if not np.allclose(k[[0, 1], [1, 0]], 0):
            raise ValueError("3DGRUT pinhole worker requires zero intrinsic skew")
        y, x = torch.meshgrid(
            torch.arange(camera["height"], device="cuda"),
            torch.arange(camera["width"], device="cuda"),
            indexing="ij",
        )
        rays = torch.stack(
            [
                (x - float(k[0, 2])) / float(k[0, 0]),
                (y - float(k[1, 2])) / float(k[1, 1]),
                torch.ones_like(x),
            ],
            dim=-1,
        ).float()
        rays = torch.nn.functional.normalize(rays, dim=-1)[None]
        params = {
            "resolution": np.array([camera["width"], camera["height"]], np.uint32),
            "shutter_type": "GLOBAL",
            "principal_point": np.array([k[0, 2] + 0.5, k[1, 2] + 0.5], np.float32),
            "focal_length": np.array([k[0, 0], k[1, 1]], np.float32),
            "radial_coeffs": np.zeros(6, np.float32),
            "tangential_coeffs": np.zeros(2, np.float32),
            "thin_prism_coeffs": np.zeros(4, np.float32),
        }
        return Batch(
            rays_ori=torch.zeros_like(rays),
            rays_dir=rays,
            T_to_world=torch.tensor(
                np.linalg.inv(camera["pose"]), device="cuda", dtype=torch.float32
            )[None],
            intrinsics_OpenCVPinholeCameraModelParameters=params,
        )

    records = []
    if args.mode == "render":
        minimum = float(request.get("min_opacity", 0.95))
        if not 0.1 <= minimum <= 1:
            raise ValueError("min_opacity must be .1..1")
        for i, row in enumerate(views):
            batch = batch_for(row)
            with torch.no_grad():
                rendered = model(batch, train=False)
            rgb = rendered["pred_features"][0].detach().float().cpu().numpy()
            opacity = rendered["pred_opacity"][0, ..., 0].detach().float().cpu().numpy()
            distance = rendered["pred_dist"][0, ..., 0].detach().float().cpu().numpy()
            # The renderer integrates ray distance with opacity weights.
            depth = distance / np.maximum(opacity, 1e-8) * batch.rays_dir[0, ..., 2].cpu().numpy()
            depth[(opacity < minimum) | ~np.isfinite(depth) | (depth <= 0)] = 0
            Image.fromarray(np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(
                args.output / f"view-{i:03d}.png"
            )
            np.save(args.output / f"depth-{i:03d}.npy", depth.astype(np.float32))
            np.save(args.output / f"opacity-{i:03d}.npy", opacity)
            visibility_path = None
            if "mog_visibility" in rendered:
                visibility = rendered["mog_visibility"].detach().reshape(-1).float().cpu().numpy()
                if len(visibility) != len(model.positions) or not np.isfinite(visibility).all():
                    raise ValueError("Renderer returned mismatched particle visibility")
                visibility_path = f"particle-visibility-{i:03d}.npy"
                np.save(args.output / visibility_path, visibility)
            elif request.get("require_particle_visibility", False):
                raise ValueError("Installed renderer does not expose actual particle visibility")
            records.append(
                {
                    "image": f"view-{i:03d}.png",
                    "depth": f"depth-{i:03d}.npy",
                    "opacity": f"opacity-{i:03d}.npy",
                    "particle_visibility": visibility_path,
                    "gaussian_source_sha256": sha256_file(args.source),
                    "valid_depth_fraction": float((depth > 0).mean()),
                }
            )
        atomic_json(
            args.output / "render_report.json",
            {
                "ok": True,
                "views": records,
                "depth_source": "opacity-normalized expected Gaussian ray distance converted to camera-Z; model estimate",
            },
        )
        return
    iterations = request.get("iterations", 200)
    if type(iterations) is not int or not 1 <= iterations <= 10000:
        raise ValueError("Use 1..10000 color iterations")
    learning_rate = float(request.get("learning_rate", 0.0025))
    if not np.isfinite(learning_rate) or not 0 < learning_rate <= 0.1:
        raise ValueError("Use a finite color learning rate in (0,.1]")
    groups = [{"params": [model.features_albedo, model.features_specular], "lr": learning_rate}]
    if geometry_mode == "joint_refine":
        for name, rate in (
            ("positions", 1e-5),
            ("scale", 1e-4),
            ("rotation", 1e-4),
            ("density", 0.001),
        ):
            groups.append({"params": [getattr(model, name)], "lr": rate})
    optimizer = torch.optim.Adam(groups)
    losses = []
    for step in range(iterations):
        row = views[step % len(views)]
        batch = batch_for(row)
        target = np.asarray(Image.open(row["image"]).convert("RGB"), dtype=np.float32) / 255
        mask = np.asarray(Image.open(row["mask"]).convert("L")) > 128
        if target.shape[:2] != mask.shape or target.shape[:2] != (
            row["image_size"][1],
            row["image_size"][0],
        ):
            raise ValueError("Training image/mask/camera dimensions disagree")
        if not mask.any():
            continue
        prediction = model(batch, train=True, frame_id=step)["pred_features"][0]
        target_gpu = torch.tensor(target, device="cuda")
        mask_gpu = torch.tensor(mask, device="cuda")
        loss = torch.abs(prediction[mask_gpu] - target_gpu[mask_gpu]).mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite color-refinement loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    if not losses:
        raise ValueError("No observed or propagated pixels available for refinement")
    unchanged = all(
        torch.equal(getattr(model, name).detach(), before) for name, before in frozen.items()
    )
    if geometry_mode == "color_only" and not unchanged:
        raise RuntimeError("Refinement unexpectedly changed Gaussian geometry or opacity")
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise ValueError("Refinement produced nonfinite model parameters")
    model.export_ply(str(args.output / "refined.ply"))
    revision = subprocess.check_output(
        ["git", "-C", str(args.checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    atomic_json(
        args.output / "training_report.json",
        {
            "ok": True,
            "iterations": len(losses),
            "losses": losses,
            "geometry_mode": geometry_mode,
            "geometry_and_opacity_unchanged": unchanged,
            "position_delta_max": float(
                torch.linalg.vector_norm(
                    model.positions.detach() - frozen["positions"], dim=1
                ).max()
            ),
            "source_ply": "refined.ply",
            "backend": {"name": "3DGRUT rendered-view refinement", "version": revision},
        },
    )


if __name__ == "__main__":
    main()
