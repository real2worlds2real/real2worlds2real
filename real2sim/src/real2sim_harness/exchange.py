"""Transfer calibrated observations and placements without converting simulation assets."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

from .artifacts import atomic_json, sha256_file
from .measurement import rigid_transform
from .posed_observations import load_views
from .scene import validate_scene
from .sim_bundle import verify_bundle


def _input(session, relative):
    path = (session / relative).resolve(strict=True)
    if not path.is_relative_to(session) or not path.is_file():
        raise ValueError(f"Expected a file in the capture session: {relative}")
    return path


def _retain(source, target, pins):
    before = sha256_file(source)
    shutil.copyfile(source, target)
    if sha256_file(target) != before:
        raise ValueError(f"Input changed during transfer: {source}")
    pins[str(source)] = before


def _legacy_models(session, names, cameras):
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(session / "splat/data/sparse/full"))
    images = {image.name: image for image in reconstruction.images.values()}
    models = {}
    for name, i in names.items():
        camera = reconstruction.cameras[images[name].camera_id]
        if not np.allclose(camera.calibration_matrix(), cameras["K"][i]) or (
            camera.width, camera.height
        ) != tuple(cameras["size"][i]):
            raise ValueError(f"COLMAP calibration differs from the archived pixels for {name}")
        models[name] = camera.model.name
    return models


def scale_evidence(session, verified):
    alignment = _input(session, "splat/align.json")
    source = json.loads(alignment.read_text()).get("scale_from")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Alignment must identify its scale source")
    pins = {str(alignment): sha256_file(alignment)}
    if verified:
        from calibrated_scene_reconstruction.scene.session import Session
        from calibrated_scene_reconstruction.scene.twin.gates import status, review
        native = Session(session)
        native.config_dir = session.parents[1] / "configs" / session.name
        anchor = status(native)["stages"]["S1"]
        if not anchor.get("applicable") or not anchor["passed"]:
            raise ValueError("Verified metric transfer needs a current native S1 scale review")
        row = review(native, "S1")
        proof = _input(session, f"qa/review/S1_r{row['round']}.json")
        pins[str(proof)] = sha256_file(proof)
    return source, pins


def export_views(session, frames, output, *, bundle_to_world=None, verified=False):
    """Write a posed-observation request directly consumable inside a task workspace."""
    session, output = Path(session).resolve(), Path(output).resolve()
    if not isinstance(frames, list) or not 1 <= len(frames) <= 100 or output.exists():
        raise ValueError("Select 1..100 views and a fresh output directory")
    cameras = _input(session, "parsing/posed_frames/cameras.npz")
    scale, pins = scale_evidence(session, verified)
    transform = rigid_transform(np.eye(4) if bundle_to_world is None else bundle_to_world)
    pins[str(cameras)] = sha256_file(cameras)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as staging, np.load(cameras, allow_pickle=False) as data:
        folder = Path(staging) / "observations"
        folder.mkdir()
        indices = {str(name): i for i, name in enumerate(data["names"])}
        selected = {}
        for frame in frames:
            name = frame["name"]
            if name not in indices or name in selected or Path(name).name != name:
                raise ValueError("Select distinct registered image names")
            selected[name] = indices[name]
        if "camera_model" in data:
            models = {name: str(data["camera_model"][i]) for name, i in selected.items()}
        else:
            models = _legacy_models(session, selected, data)
            sparse = session / "splat/data/sparse/full"
            pins.update({str(p): sha256_file(p) for p in sparse.glob("*.bin")})
        rows = []
        for j, frame in enumerate(frames):
            name, i = frame["name"], selected[frame["name"]]
            if models[name] != "PINHOLE":
                raise ValueError(f"{name} needs rectified PINHOLE pixels and matching intrinsics")
            rotation = data["R_world_cam"][i].T
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = rotation
            extrinsic[:3, 3] = -rotation @ data["centre"][i]
            row = {"world_to_camera": (extrinsic @ transform).tolist(), "intrinsics": data["K"][i].tolist(),
                   "image_size": data["size"][i].tolist(), "source": f"{cameras}#{name}"}
            sources = {"image": "splat/data/images/" + name}
            if set(frame) - {"name", "mask", "ignore", "depth"}:
                raise ValueError("Unexpected observation channel")
            sources.update({k: v for k, v in frame.items() if k != "name"})
            for key, relative in sources.items():
                source = _input(session, relative)
                row[key] = f"{j:03d}-{key}{source.suffix}"
                _retain(source, folder / row[key], pins)
            rows.append(row)
        # A registered metric frame alone does not verify absolute scale.
        result = {"schema_version": 1, "frame": {"source": scale, "metric_verified": verified}, "views": rows}
        load_views(result, folder, required=("image",))
        atomic_json(folder / "views.json", result)
        atomic_json(folder / "provenance.json", {"inputs": pins, "bundle_to_world": transform.tolist()})
        folder.rename(output)
    return {"manifest": str(output / "views.json"), "inputs": pins, "views": len(rows)}


def model_to_world(fit):
    from calibrated_scene_reconstruction.assets.object_pose import rotation

    pose = json.loads(fit.read_text())["pose"]
    transform = np.eye(4)
    transform[:3, :3] = rotation(pose)
    transform[:3, 3] = [*pose["xy"], pose["support_z"]]
    return rigid_transform(transform)


def export_geometry(session, round_path, output):
    """Export the exact evaluated surface as a geometry prior alongside editable source."""
    session, output = Path(session).resolve(), Path(output).resolve()
    source = _input(session, Path(round_path) / "mesh.npz")
    fit = _input(session, Path(round_path) / "fit.json")
    if output.exists():
        raise ValueError("Choose a fresh geometry exchange directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    pins = {str(p): sha256_file(p) for p in (source, fit)}
    transform = model_to_world(fit)
    with tempfile.TemporaryDirectory(dir=output.parent) as staging, np.load(source, allow_pickle=False) as archive:
        folder = Path(staging) / "geometry"
        folder.mkdir()
        vertices, faces = archive["V"], archive["F"]
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all()
                or faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.dtype.kind not in "iu"
                or faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("Invalid fitted triangle mesh")
        mesh = {"vertices": vertices, "triangles": faces}
        if "UV" in archive:
            mesh["uv"] = archive["UV"]
        np.savez_compressed(folder / "mesh.npz", **mesh)
        for name in ("object.glb", "scene.blend", "model.py"):
            path = source.parent / name
            if path.is_file():
                _retain(_input(session, path), folder / name, pins)
        atomic_json(folder / "provenance.json", {"inputs": pins, "model_to_world": transform.tolist(),
                    "role": "observed model candidate; runtime physics and task acceptance remain separate"})
        folder.rename(output)
    return {"mesh": str(output / "mesh.npz"), "model_to_world": transform.tolist(),
            "vertices": len(vertices), "triangles": len(faces), "inputs": pins}


def placed_manifest(session, spec, participants):
    """Place runtime bundles in a native calibrated model's frame.

    bundle_to_model explicitly registers the bundle's local origin and axes to
    the fitted geometry. The native mesh may have a different origin.
    """
    session = Path(session).resolve()
    if not participants or "objects" in spec:
        raise ValueError("Supply fitted participants and scene options separately")
    frame = spec["frame"]
    if type(frame.get("scale_verified")) is not bool:
        raise ValueError("Declare whether the native frame requires verified scale")
    scale, pins = scale_evidence(session, frame["scale_verified"])
    spec = {**spec, "frame": {**frame, "scale_source": scale}}
    objects = []
    for item in participants:
        fit = _input(session, item["fit"])
        transform = model_to_world(fit)
        bundle = Path(item["bundle"]).resolve(strict=True)
        checked = verify_bundle(bundle)
        if not checked["ok"]:
            raise ValueError(f"Invalid runtime bundle: {checked['failures']}")
        transform = transform @ rigid_transform(item["bundle_to_model"])
        objects.append({"id": item["id"], "transform": transform.tolist(), "bundle": str(bundle)})
        for path in (fit, bundle / "asset_bundle.json"):
            pins[str(path)] = sha256_file(path)
    result = {**spec, "schema_version": 1, "objects": objects}
    return validate_scene(result, session), pins
