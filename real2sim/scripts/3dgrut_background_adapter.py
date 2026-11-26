"""Optional operator-owned bridge to an installed official 3DGRUT checkout.

No checkout, model weights, CUDA runtime or authentication is bundled here.
The scene harness invokes this process through a configured argv list.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from real2sim_harness.artifacts import atomic_json
from real2sim_harness.gaussian_background import frame_at


def training_dataset(dataset, output):
    """Bridge our shared COLMAP selected model to upstream's sparse/0 layout."""
    dataset = Path(dataset).resolve()
    if not (dataset / "images").is_dir():
        raise ValueError("Use a COLMAP dataset with images/ and registered sparse poses")
    if not (dataset / "sparse/selected").is_dir():
        if not (dataset / "sparse/0").is_dir():
            raise ValueError("Use a COLMAP dataset with sparse/0 or sparse/selected; recover poses first")
        return dataset
    staged = output / "capture_dataset"
    shutil.copytree(dataset / "images", staged / "images")
    shutil.copytree(dataset / "sparse/selected", staged / "sparse/0")
    return staged


def commands(request, output, python, train_config):
    if request["operation"] == "train":
        dataset = Path(request["dataset"]).resolve()
        if not (dataset / "images").is_dir() or not (dataset / "sparse" / "0").is_dir():
            raise ValueError("Use a COLMAP dataset with images/ and sparse/0; recover poses first")
        iterations = request.get("iterations", 30000)
        downsample = request.get("downsample_factor", 2)
        if not isinstance(iterations, int) or not 1 <= iterations <= 1_000_000 or downsample not in (1, 2, 4, 8):
            raise ValueError("Invalid training iteration or downsample budget")
        return [[python, "train.py", "--config-name", train_config,
                 f"path={dataset}", f"out_dir={output / 'training'}", "experiment_name=background",
                 f"n_iterations={iterations}", f"dataset.downsample_factor={downsample}",
                 "dataset.normalize_world_space=false", "export_ply.enabled=true",
                 f"export_ply.path={output / 'trained.ply'}", "num_workers=4"]]
    source = Path(request["source"]).resolve()
    if not source.is_file() or request.get("format") not in {"particle_field", "nurec_volume"}:
        raise ValueError("Transcode an existing Gaussian asset to particle_field or nurec_volume")
    target = "lightfield" if request["format"] == "particle_field" else "nurec"
    return [[python, "-m", "threedgrut.export.scripts.transcode", str(source),
             "-o", str(output / "raw.usdz"), "--format", target]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--train-config", default="apps/colmap_3dgut.yaml")
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if request["operation"] == "train":
        request["dataset"] = str(training_dataset(request["dataset"], output))
    revision = subprocess.run(["git", "-C", str(args.checkout), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    backend = {"name": "nv-tlabs/3dgrut", "version": revision}
    for argv in commands(request, output, args.python, args.train_config):
        subprocess.run(argv, cwd=args.checkout, check=True)
    result = {"schema_version": 1, "ok": True, "backend": backend,
              "operation": request["operation"], "simulation_verified": False}
    if request["operation"] == "train":
        result.update(source_ply="trained.ply", normalization="disabled; output retains dataset coordinates",
                      next_action="Prepare this source_ply with explicit source-to-scene alignment, crop and optional posed-camera coverage")
    else:
        from pxr import Gf, Usd, UsdGeom
        frame, matrix, _ = frame_at(request)
        raw = Usd.Stage.Open(str(output / "raw.usdz"))
        source_root = raw.GetDefaultPrim()
        if not source_root:
            raise ValueError("Converted Gaussian asset has no default root")
        stage = Usd.Stage.CreateNew(str(output / "background.usda"))
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        UsdGeom.SetStageUpAxis(stage, "Z")
        root = UsdGeom.Xform.Define(stage, "/Background")
        stage.SetDefaultPrim(root.GetPrim())
        root.AddTransformOp().Set(Gf.Matrix4d(matrix.T.tolist()))
        stage.DefinePrim("/Background/Content").GetReferences().AddReference("./raw.usdz", source_root.GetPath())
        stage.GetRootLayer().Save()
        result.update(frame=frame, background={"path": "background.usda", "format": request["format"],
                      "observed": request.get("observed", True), "transform": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]})
    atomic_json(output / "background_report.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
