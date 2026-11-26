"""Unaccepted scene inputs for producing missing native intrinsic evidence.

This is deliberately separate from prepare_scene: it validates current source
declarations, fitted mesh/pose and observation hashes, but does not pretend that
the native evidence being measured has already passed semantic review. Only
intrinsic_assay is supported; it cannot issue a bound policy job receipt.
"""
from functools import cached_property
import hashlib
import json
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.core.paths import resolve
from calibrated_scene_reconstruction.scene.objects.object_understanding import validate_build
from calibrated_scene_reconstruction.scene.session import Session
from .runtime_inputs import SceneRuntimeInputs


def _hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _candidate(session, name, section):
    spec = session.scene_graph[section][name]
    fix = spec["fix"]
    directory = session.dir / ("support" if section == "furniture" else "objects") / name / f"round_{fix['round']}"
    files = {}

    def pin(path, expected=None):
        path = Path(path).resolve(strict=True)
        if not path.is_file():
            raise ValueError("Candidate evidence must be a file")
        actual = _hash(path)
        if expected is not None and actual != expected:
            raise ValueError("Candidate source/geometry hash changed: " + str(path))
        files[str(path)] = actual
        return path

    report_path, input_path = pin(directory / "fit.json"), pin(directory / "input.json")
    report, inputs = _read(report_path), _read(input_path)
    pin(input_path, report.get("input_sha256"))
    pin(directory / "model.py", report["model_sha256"])
    source_model = pin(session.config_dir / fix["model"], report["model_sha256"])
    mesh = pin(directory / "mesh.npz", report["mesh_sha256"])
    if (not mesh.is_relative_to(session.dir.resolve()) or resolve(fix["mesh"]).resolve() != mesh
            or fix.get("model_sha256") != report["model_sha256"]):
        raise ValueError("Active candidate must be the current source mesh/model, without replacement")
    pin(session.config_dir / "scene_graph.json")
    pin(session.config_dir / "agent/object_understanding.json")
    actual_ownership = None
    build_for_validation = report["build"]
    if section == "dynamic_objects":
        from calibrated_scene_reconstruction.assets.metric_mesh import asset
        physical = asset(fix)
        actual_ownership = {"parts": [{"name": p["name"], "kind": p["kind"]} for p in physical["parts"]],
                            "links": physical["links"], "joints": physical["joints"],
                            "surface_bindings": physical["surface_bindings"]}
        # Read physical ownership from the actual native mesh loader, not a
        # guessed kind for old Blender visual-construction names. In particular
        # any cloth/deformable/joint metadata remains physical and is validated.
        build_for_validation = {**report["build"], **actual_ownership}
    declaration = validate_build(session, name, build_for_validation)
    # Legacy rounds may predate semantic inventories. The new declaration is
    # bound as a candidate hypothesis, never retroactively marked as old review.
    for group, base in (("inputs", session.dir if section == "dynamic_objects" else None),
                        ("dependencies", None), ("source_assets", None)):
        for raw, sha in inputs.get(group, {}).items():
            if not isinstance(sha, str):
                raise ValueError("Expected exact source asset hashes in candidate input manifest")
            path = Path(raw)
            path = path if path.is_absolute() else (base / path if base is not None else resolve(raw))
            pin(path, sha)
    with np.load(mesh, allow_pickle=False) as archive:
        vertices, faces = archive["V"], archive["F"]
        if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices)
                or not np.isfinite(vertices).all() or faces.ndim != 2 or faces.shape[1:] != (3,)
                or faces.dtype.kind not in "iu" or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("Candidate mesh is not a finite valid triangle surface")
    pose = report["pose"]
    pose_fields = ("xy", "yaw_deg") if section == "furniture" else ("xy", "yaw_deg", "support_z", "local_top", "down_offset")
    if any(fix.get(key) != pose.get(key) for key in pose_fields):
        raise ValueError("Candidate pose differs from its source fit")
    if section == "dynamic_objects":
        from calibrated_scene_reconstruction.assets.metric_mesh import asset, sdf_settings
        asset(fix)  # Physical topology validity; no renderer or pose substitute.
        if fix.get("sdf") != sdf_settings(spec["fit"].get("sdf")):
            raise ValueError("Candidate collider profile differs from current declared source settings")
        if fix.get("opening") != report["build"].get("opening") or fix.get("params") != report["build"]["params"]:
            raise ValueError("Candidate physical opening/parameters differ from its source build")
    else:
        visual = pin(directory / "object.glb", report["visual_sha256"])
        if resolve(fix["visual_glb"]).resolve() != visual:
            raise ValueError("Candidate support visual surface differs from its source geometry")
        for texture in report.get("textures", []):
            pin(resolve(texture["path"]), texture["sha256"])
    return {"object": name, "section": section, "round": fix["round"], "files": files,
            "understanding": declaration, "source_model": str(source_model), "qualified": False,
            "actual_native_mesh_ownership": actual_ownership}


class QualificationSceneInputs(SceneRuntimeInputs):
    """Read-only candidate materialization, not an accepted native twin."""
    qualification_only = True
    qualified = False
    policy_receipts_allowed = False

    def __init__(self, session, record):
        super().__init__(session)
        self.qualification_record = record

    def assert_unchanged(self):
        for candidate in self.qualification_record["candidates"].values():
            for path, sha in candidate["files"].items():
                if not Path(path).is_file() or _hash(path) != sha:
                    raise ValueError("Unaccepted qualification inputs changed: " + path)
        return True

    @cached_property
    def support_models(self):
        from calibrated_scene_reconstruction.scene.support.runtime_inputs import _models
        def check(name):
            self.assert_unchanged()
            item = self.qualification_record["candidates"].get(name)
            if not item or item["section"] != "furniture" or item["qualified"] is not False:
                raise ValueError("Support was not source-bound by this unaccepted qualification record")
        return _models(self.session, check)


def prepare_qualification_scene(session, *, purpose, output):
    """Prepare only intrinsic assay inputs and persist an explicit unaccepted intent.

    No existing review, freeze, mesh or configuration is written. Use ordinary
    prepare_scene after independently reviewing the new native evidence.
    """
    if purpose != "intrinsic_assay":
        raise ValueError("Unaccepted candidate composition is restricted to intrinsic_assay")
    session = session if isinstance(session, Session) else Session(session)
    target = Path(output).resolve()
    if not target.is_relative_to((session.dir / "qa").resolve()) or target.exists():
        raise ValueError("Write a new qualification intent under this session's QA")
    candidates = {}
    for section in ("furniture", "dynamic_objects"):
        for name, spec in session.scene_graph.get(section, {}).items():
            if (spec.get("fix") or {}).get("model"):
                candidates[name] = _candidate(session, name, section)
    if not candidates:
        raise ValueError("Candidate qualification requires current reconstructed model assets")
    record = {"schema": "astra_unaccepted_intrinsic_scene_v1", "purpose": purpose,
              "qualified": False, "policy_receipts_allowed": False, "native_session": str(session.dir.resolve()),
              "candidates": candidates, "producer_sha256": _hash(__file__),
              "scope": "Current source/geometry integrity for measuring missing native intrinsic evidence; no semantic, task, policy or deployment approval."}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return QualificationSceneInputs(session, record)
