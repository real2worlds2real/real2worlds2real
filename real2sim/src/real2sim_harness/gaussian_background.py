"""Optional captured backgrounds: CPU editing/export and replaceable training."""
from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

from .artifacts import atomic_json, sha256_file
from .measurement import rigid_transform


def similarity(value):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError("source_to_scene must be a finite affine 4x4 similarity")
    scale = float(np.cbrt(np.linalg.det(matrix[:3, :3])))
    if scale <= 0 or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3) * scale ** 2, rtol=1e-5, atol=1e-10):
        raise ValueError("Gaussian alignment needs positive uniform scale and a proper rotation")
    return matrix, scale


def frame_at(request):
    frame = request.get("frame", {})
    if not isinstance(frame.get("scale_source"), str) or not frame["scale_source"].strip() or not isinstance(frame.get("scale_verified"), bool):
        raise ValueError("Describe estimated or measured scale and its verification status")
    matrix, scale = similarity(frame.get("source_to_scene"))
    return frame, matrix, scale


def _input(base, value):
    if not isinstance(value, str) or not value:
        raise ValueError("Supply an existing input path")
    path = (Path(base) / value).expanduser().resolve()
    if not path.is_file():
        raise ValueError("Missing background input: " + path.name)
    return path


def read_gaussians(path):
    from plyfile import PlyData

    data = PlyData.read(str(path))
    if "vertex" not in data or not 0 < len(data["vertex"]) <= 10_000_000:
        raise ValueError("Expected a bounded nonempty Gaussian vertex table")
    vertices = data["vertex"].data
    required = ["x", "y", "z", "opacity", *[f"scale_{i}" for i in range(3)],
                *[f"rot_{i}" for i in range(4)], *[f"f_dc_{i}" for i in range(3)]]
    if not set(required) <= set(vertices.dtype.names):
        raise ValueError("Expected pre-activation 3DGS PLY properties; a mesh or XYZ point cloud is insufficient")
    rest = sorted((n for n in vertices.dtype.names if n.startswith("f_rest_")), key=lambda n: int(n[7:]))
    if rest != [f"f_rest_{i}" for i in range(len(rest))] or len(rest) not in (0, 9, 24, 45):
        raise ValueError("Expected contiguous 3DGS spherical-harmonic coefficients of degree 0 to 3")
    for name in required + rest:
        if not np.isfinite(vertices[name]).all():
            raise ValueError("Nonfinite Gaussian property: " + name)
    if any(np.max(np.abs(vertices[f"scale_{i}"])) > 80 for i in range(3)):
        raise ValueError("Gaussian log scales exceed the supported numeric range")
    return vertices


def _matrix_columns(vertices, names):
    return np.column_stack([vertices[name] for name in names]).astype(np.float64)


def frustum_coverage(points, cameras):
    """Geometric frustum inclusion; deliberately makes no occlusion/quality claim."""
    counts, directions = np.zeros(len(points), dtype=np.int32), np.zeros_like(points, dtype=float)
    if not isinstance(cameras, list) or not 1 <= len(cameras) <= 500:
        raise ValueError("Supply from 1 to 500 posed cameras")
    for camera in cameras:
        pose = rigid_transform(camera["world_to_camera"])
        intrinsic = np.asarray(camera["intrinsics"], dtype=float)
        width, height = camera["width"], camera["height"]
        if (intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all()
                or not np.allclose(intrinsic[2], [0, 0, 1]) or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
                or not isinstance(width, int) or not isinstance(height, int) or min(width, height) <= 0):
            raise ValueError("Use finite pinhole intrinsics and positive image dimensions")
        xyz = points @ pose[:3, :3].T + pose[:3, 3]
        pixels = xyz @ intrinsic.T
        z = xyz[:, 2]
        uv = pixels[:, :2] / np.maximum(z[:, None], 1e-12)
        visible = (z > 1e-4) & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        center = -pose[:3, :3].T @ pose[:3, 3]
        rays = center - points
        rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-12)
        counts += visible
        directions += rays * visible[:, None]
    mean_diversity = 1 - np.linalg.norm(directions, axis=1) / np.maximum(counts, 1)
    return counts, {"camera_count": len(cameras), "points": len(points),
                    "outside_all_frusta": int((counts == 0).sum()), "one_frustum": int((counts == 1).sum()),
                    "two_or_more_frusta": int((counts >= 2).sum()),
                    "mean_direction_diversity": float(mean_diversity[counts > 0].mean()) if (counts > 0).any() else 0,
                    "method": "pinhole_frustum_inclusion_only; no occlusion or photometric quality test"}


def _crop(vertices, request, matrix, scale, base):
    points = _matrix_columns(vertices, ["x", "y", "z"])
    world = points @ matrix[:3, :3].T + matrix[:3, 3]
    sigma = np.exp(_matrix_columns(vertices, [f"scale_{i}" for i in range(3)])) * scale
    crop = request.get("crop", {})
    extent = float(crop.get("sigma_extent", 3))
    opacity_limit = float(crop.get("min_opacity", 0))
    if not math.isfinite(extent) or not 0 <= extent <= 10 or not 0 <= opacity_limit < 1:
        raise ValueError("Use sigma_extent 0..10 and min_opacity 0..<1")
    radius = sigma.max(axis=1) * extent
    opacity = 1 / (1 + np.exp(-np.clip(vertices["opacity"].astype(float), -80, 80)))
    removed = {"faint": opacity < opacity_limit}
    bounds = crop.get("keep_bounds_m")
    if bounds is not None:
        bounds = np.asarray(bounds, dtype=float)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or not np.all(bounds[1] > bounds[0]):
            raise ValueError("keep_bounds_m needs ordered finite minimum/maximum corners")
        # Conservative whole-particle containment; no partial Gaussian clipping.
        removed["outside_bounds"] = np.any((world - radius[:, None] < bounds[0]) | (world + radius[:, None] > bounds[1]), axis=1)
    boxes = crop.get("exclude_boxes", [])
    if not isinstance(boxes, list) or len(boxes) > 200:
        raise ValueError("Use a bounded list of task-space exclusion boxes")
    for index, box in enumerate(boxes):
        transform = rigid_transform(box["transform"])
        size = np.asarray(box["size_m"], dtype=float)
        if size.shape != (3,) or not np.isfinite(size).all() or not (size > 0).all():
            raise ValueError("Exclusion boxes need three positive sizes")
        local = (world - transform[:3, 3]) @ transform[:3, :3]
        removed[f"exclusion_{index}"] = np.all(np.abs(local) <= size / 2 + radius[:, None], axis=1)
    coverage = {"status": "not_supplied", "method": "no capture cameras supplied"}
    near_camera=crop.get('near_camera_m',0)
    if type(near_camera) not in (int,float) or not math.isfinite(near_camera) or near_camera<0:
        raise ValueError('near_camera_m must be finite and nonnegative')
    camera_path = None
    if request.get("cameras_path"):
        camera_path = _input(base, request["cameras_path"])
        camera_data = json.loads(camera_path.read_text())
        counts, coverage = frustum_coverage(world, camera_data["cameras"])
        minimum = crop.get("min_frustum_views", 0)
        if not isinstance(minimum, int) or not 0 <= minimum <= len(camera_data["cameras"]):
            raise ValueError("Invalid minimum frustum count")
        removed["insufficient_frusta"] = counts < minimum
        if near_camera:
            centers=[]
            for camera in camera_data['cameras']:
                pose=rigid_transform(camera['world_to_camera'])
                centers.append(-pose[:3,:3].T@pose[:3,3])
            near=np.zeros(len(world),dtype=bool)
            # Bound scratch memory; this optional cleanup needs no extra model.
            for start in range(0,len(world),65536):
                block=world[start:start+65536]
                nearest=np.full(len(block),np.inf)
                for center in centers:
                    delta=block-center
                    nearest=np.minimum(nearest,np.einsum('ij,ij->i',delta,delta))
                near[start:start+len(block)]=nearest<=(near_camera+radius[start:start+len(block)])**2
            removed['near_capture_cameras']=near
    elif crop.get("min_frustum_views", 0):
        raise ValueError("Frustum filtering requires actual posed camera evidence")
    elif near_camera:
        raise ValueError('Near-camera cleanup requires actual posed camera evidence in the scene frame')
    max_sigma = crop.get("max_sigma_m")
    if max_sigma is not None:
        if not math.isfinite(float(max_sigma)) or float(max_sigma) <= 0:
            raise ValueError("max_sigma_m must be positive")
        removed["large_particles"] = sigma.max(axis=1) > float(max_sigma)
    remove = np.logical_or.reduce(list(removed.values()))
    if remove.all():
        raise ValueError("Crop removed every Gaussian; review alignment and task zones")
    return vertices[~remove].copy(), world, ~remove, {"input_particles": len(vertices), "kept_particles": int((~remove).sum()),
            "removed_particles": int(remove.sum()), "per_filter_counts_may_overlap": {k: int(v.sum()) for k, v in removed.items()},
            "crop_method": "conservative bounding spheres against whole-particle task boxes", "coverage": coverage}, camera_path


def write_particle_field(vertices, output, matrix):
    from pxr import Gf, Usd, UsdGeom, UsdVol, Vt

    if not hasattr(UsdVol, "ParticleField3DGaussianSplat"):
        raise RuntimeError("This exporter requires OpenUSD with ParticleField3DGaussianSplat; use a configured legacy converter otherwise")
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    root = UsdGeom.Xform.Define(stage, "/Background")
    stage.SetDefaultPrim(root.GetPrim())
    # Keep SH lobes and anisotropic ellipsoids in the same local frame. Baking
    # positions alone would rotate geometry but leave directional appearance wrong.
    root.AddTransformOp().Set(Gf.Matrix4d(matrix.T.tolist()))
    field = UsdVol.ParticleField3DGaussianSplat.Define(stage, "/Background/Gaussians")
    points = _matrix_columns(vertices, ["x", "y", "z"]).astype(np.float32)
    scales = np.exp(_matrix_columns(vertices, [f"scale_{i}" for i in range(3)])).astype(np.float32)
    quats = _matrix_columns(vertices, [f"rot_{i}" for i in range(4)])
    norms = np.linalg.norm(quats, axis=1)
    if (norms < 1e-10).any():
        raise ValueError("Gaussian rotations must be nonzero quaternions")
    quats /= norms[:, None]
    opacity = (1 / (1 + np.exp(-np.clip(vertices["opacity"].astype(float), -80, 80)))).astype(np.float32)
    dc = _matrix_columns(vertices, [f"f_dc_{i}" for i in range(3)])
    rest_names = sorted((n for n in vertices.dtype.names if n.startswith("f_rest_")), key=lambda n: int(n[7:]))
    count = 1 + len(rest_names) // 3
    coefficients = np.zeros((len(vertices), count, 3), dtype=np.float32)
    coefficients[:, 0] = dc
    if rest_names:
        coefficients[:, 1:] = _matrix_columns(vertices, rest_names).reshape(len(vertices), 3, count - 1).transpose(0, 2, 1)
    field.CreatePositionsAttr(Vt.Vec3fArray.FromNumpy(points))
    field.CreateScalesAttr(Vt.Vec3fArray.FromNumpy(scales))
    field.CreateOrientationsAttr([Gf.Quatf(float(q[0]), Gf.Vec3f(*map(float, q[1:]))) for q in quats])
    field.CreateOpacitiesAttr(Vt.FloatArray.FromNumpy(opacity))
    field.CreateRadianceSphericalHarmonicsDegreeAttr(int(round(math.sqrt(count) - 1)))
    sh = field.CreateRadianceSphericalHarmonicsCoefficientsAttr(Vt.Vec3fArray.FromNumpy(coefficients.reshape(-1, 3)))
    sh.SetMetadata("elementSize", count)
    Usd.ColorSpaceAPI.Apply(field.GetPrim()).CreateColorSpaceNameAttr("srgb_rec709_display")
    field.CreateProjectionModeHintAttr("perspective")
    field.CreateSortingModeHintAttr("cameraDistance")
    radius = scales.max(axis=1)[:, None] * 3
    field.CreateExtentAttr([Gf.Vec3f(*map(float, (points - radius).min(0))), Gf.Vec3f(*map(float, (points + radius).max(0)))])
    stage.GetRootLayer().Save()


def prepare_background(request, base, output):
    from plyfile import PlyData, PlyElement

    frame, matrix, scale = frame_at(request)
    observed = request.get("observed", True)
    if not isinstance(observed, bool):
        raise ValueError("observed must explicitly distinguish capture from generated completion")
    source = _input(base, request.get("source_ply"))
    digest = sha256_file(source)
    vertices = read_gaussians(source)
    kept, world, mask, metrics, cameras = _crop(vertices, request, matrix, scale, base)
    PlyData([PlyElement.describe(kept, "vertex")], text=False).write(str(output / "background.ply"))
    write_particle_field(kept, output / "background.usdc", matrix)
    # A top-down geometry diagnostic, not a photorealistic rendering.
    xy = world[:, :2]
    low, high = xy.min(0), xy.max(0)
    pixels = np.clip(((xy - low) / np.maximum(high - low, 1e-6) * 511).astype(int), 0, 511)
    preview = np.full((512, 512, 3), 245, dtype=np.uint8)
    for selected, color in ((~mask, [220, 70, 60]), (mask, [40, 130, 100])):
        p = pixels[selected]
        preview[511 - p[:, 1], p[:, 0]] = color
    Image.fromarray(preview).save(output / "crop_topdown.png")
    if sha256_file(source) != digest:
        raise ValueError("Gaussian source changed during preparation")
    return {"ok": True, "operation": "prepare", "metrics": metrics, "frame": frame,
            "source_sha256": digest, "camera_sha256": sha256_file(cameras) if cameras else None,
            "source_ply": str(output / "background.ply"), "preview_images": ["crop_topdown.png"],
            "background": {"path": str(output / "background.usdc"), "format": "particle_field",
                           "observed": observed, "transform": np.eye(4).tolist()},
            "simulation_verified": False, "visual_only": True,
            "limits": ["Frustum inclusion is not visibility, metric accuracy or rendered coverage certification.",
                       "Exclusion boxes must cover intended target motion; removal may expose uncaptured holes.",
                       "Gaussian backgrounds supply no collisions; scene-level visual review remains necessary."]}


def run_background(store, workspace, *, request_path, timeout_seconds=900):
    from .scene_agent import workspace_file
    from .scene_video import evidence_path

    source = workspace_file(workspace, request_path)
    request = json.loads(source.read_text())
    if request.get("schema_version") != 1 or request.get("operation") not in {"prepare", "train", "transcode", 'refine','cousin'}:
        raise ValueError("Use Gaussian request schema 1 and prepare, train, transcode, refine or cousin")
    root = store.root / "gaussian-backgrounds"
    index = len(list(root.glob("background-[0-9][0-9][0-9]"))) + 1
    output = root / f"background-{index:03d}"
    output.mkdir(parents=True)
    atomic_json(output / "request.json", request)
    try:
        if request["operation"] == "prepare":
            result = prepare_background(request, source.parent, output)
        elif request['operation']=='cousin':
            from .gaussian_cousins import edit_background
            run = json.loads((store.root / 'run.json').read_text())
            if (run.get('generation_mode') == 'scene_cousin'
                    and not run.get('scene_cousin', {}).get('layout_change_request')):
                for edit in request.get('edits', []):
                    transform, _ = similarity(edit.get('transform', np.eye(4).tolist()))
                    if not np.allclose(transform, np.eye(4), rtol=0, atol=1e-6):
                        raise ValueError('Scene cousins preserve layout; use regional appearance edits without moving captured regions')
            result=edit_background(request,source.parent,output)
        else:
            run = json.loads((store.root / "run.json").read_text())
            config_path = run.get("gaussian_runner") or os.environ.get("REAL2SIM_GAUSSIAN_RUNNER")
            if not config_path:
                raise ValueError("Configure --gaussian-runner or REAL2SIM_GAUSSIAN_RUNNER for training/transcoding; CPU prepare needs neither")
            config_path = Path(config_path).resolve()
            config_digest = sha256_file(config_path)
            config = json.loads(config_path.read_text())
            if (config.get("schema_version") != 1 or request["operation"] not in config.get("capabilities", [])
                    or not isinstance(config.get("argv"), list) or not config["argv"]
                    or not all(isinstance(v, str) and v for v in config["argv"])):
                raise ValueError("Configured Gaussian backend does not support this operation")
            # Resolve only declared path fields relative to the agent request.
            normalized = dict(request)
            for key in ("source", "dataset", 'observations'):
                if key in normalized:
                    normalized[key] = str((source.parent / normalized[key]).expanduser().resolve())
            atomic_json(output / "backend_request.json", normalized)
            replacements = {"{request}": str(output / "backend_request.json"), "{output}": str(output)}
            argv = []
            for value in config["argv"]:
                for key, replacement in replacements.items():
                    value = value.replace(key, replacement)
                argv.append(value)
            with (output / "backend.log").open("w") as log:
                process = subprocess.run(argv, cwd=config_path.parent, stdout=log, stderr=subprocess.STDOUT,
                                         timeout=timeout_seconds, check=False)
            result = json.loads((output / "background_report.json").read_text())
            if process.returncode or result.get("ok") is not True or not isinstance(result.get("backend"), dict) or not result["backend"].get("name") or not result["backend"].get("version"):
                raise ValueError("Gaussian backend failed or omitted actual backend identity")
            if request["operation"] in {"train", 'refine'}:
                path = evidence_path(output, result["source_ply"])
                read_gaussians(path)
                result["source_ply"] = str(path)
            else:
                from .scene import _check_stage
                path = evidence_path(output, result["background"]["path"])
                stage, prim, _, _ = _check_stage(path)
                from pxr import Usd, UsdPhysics
                if any(p.HasAPI(UsdPhysics.RigidBodyAPI) or p.HasAPI(UsdPhysics.CollisionAPI) or p.IsA(UsdPhysics.Joint) for p in Usd.PrimRange(prim)):
                    raise ValueError("Gaussian background must remain visual-only")
                kind = result["background"].get("format")
                types = {p.GetTypeName() for p in Usd.PrimRange(prim)}
                if ((kind == "particle_field" and not any(v.startswith("ParticleField") for v in types))
                        or (kind == "nurec_volume" and not types & {"Volume", "OmniNuRecFieldAsset"})
                        or kind not in {"particle_field", "nurec_volume"}):
                    raise ValueError("Converted background does not contain its declared Gaussian representation")
                result["background"]["path"] = str(path)
                del stage
            if sha256_file(config_path) != config_digest:
                raise ValueError("Gaussian backend configuration changed during execution")
            result["runner_config_sha256"] = config_digest
        result.update(background_id=output.name, directory=str(output), simulation_verified=False)
        result["files"] = {str(p.relative_to(output)): sha256_file(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "background_report.json"}
    except (ImportError, OSError, ValueError, RuntimeError, KeyError, subprocess.TimeoutExpired) as exc:
        result = {"ok": False, "background_id": output.name, "directory": str(output), "error": f"{type(exc).__name__}: {exc}"}
        if (output / "backend.log").is_file():
            result["log_tail"] = (output / "backend.log").read_text(errors="replace")[-2000:]
    atomic_json(output / "background_report.json", result)
    store.append_event({"type": "gaussian_background", "operation": request["operation"], "ok": result["ok"], "background_id": output.name})
    return result
