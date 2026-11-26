"""Source-bound native contact evidence for rigid assets; ordinary PhysX stays default."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .artifacts import atomic_json, sha256_file

REQUIRED = {
    "input.json",
    "request.json",
    "source.blend",
    "object.glb",
    "runtime.json",
    "motion.npz",
    "validation.json",
    "animation_report.json",
    "native_motion.mp4",
    "encoding_report.json",
    "contact_asset.json",
}


def selected(report, kind):
    return kind == "rigid" and report.get("selected_simulation_review") == "rigid_contact"


def retain_asset_evidence(store, attempt_id, folder, detail):
    """Called only for an explicitly selected, source-derived final review."""
    attempt = store.attempts_dir / attempt_id
    request = json.loads((folder / "request.json").read_text())
    runtime = {
        k: v
        for k, v in request.items()
        if k not in {"controls", "thread_checks", "separation_checks", "source_attempt_id"}
    }
    runtime["controls"] = []
    atomic_json(folder / "runtime.json", runtime)
    shutil.copy2(attempt / "object.glb", folder / "object.glb")
    files = {
        p.name: sha256_file(p)
        for p in folder.iterdir()
        if p.is_file() and p.name in REQUIRED - {"contact_asset.json"}
    }
    bundle = {
        "schema_version": 2,
        "asset_kind": "rigid",
        "backend": "newton_native_sdf",
        "runtime": "runtime.json",
        "source": "source.blend",
        "visual": "object.glb",
        "appearance_source": "source.blend",
        "appearance_review": "native_motion.mp4",
        "preview_material_equivalence_verified": False,
        "loader": "real2sim_harness.rigid_contact_runtime.load_rigid_contact",
        "test_configuration": "request.json",
        "files": files,
        "passive_runtime": True,
        "sim_version_required": None,
        "scope": "Source-bound rigid contact asset under recorded effective parameters. "
                 "Appearance is reviewed by replaying the original Blender source; object.glb is a preview "
                 "without material-equivalence certification. No physical calibration or robot-policy certification.",
    }
    atomic_json(folder / "contact_asset.json", bundle)
    detail.update(
        directory=str(folder.relative_to(attempt)),
        cache_manifest={
            str((folder / n).relative_to(attempt)): sha256_file(folder / n) for n in REQUIRED
        },
        asset_acceptance_updated=False,
        delivery_scope="Native concave rigid contacts and original Blender-material replay with complete movie. "
                       "The GLB preview does not certify transferred appearance. Final acceptance remains with the agent.",
    )
    verify_evidence(attempt, detail)
    report = json.loads((attempt / "report.json").read_text())
    report["rigid_contact"] = detail
    report["selected_simulation_review"] = "rigid_contact"
    report.setdefault("artifacts", {}).update({n: n for n in detail["cache_manifest"]})
    atomic_json(attempt / "report.json", report)


def verify_evidence(attempt, detail):
    """Check source, passive runtime, native trajectory, replay and movie together."""
    attempt = Path(attempt).resolve()
    source_sha = sha256_file(attempt / "scene.blend")
    if not detail.get("ok") or detail.get("source_scene_sha256") != source_sha:
        raise ValueError("Rigid contact requires successful source-matched native evidence")
    folder = (attempt / detail.get("directory", "")).resolve()
    if folder == attempt or not folder.is_relative_to(attempt):
        raise ValueError("Rigid contact evidence must stay inside its saved attempt")
    manifest = detail.get("cache_manifest", {})
    required = {str((folder / n).relative_to(attempt)) for n in REQUIRED}
    if not required <= set(manifest):
        raise ValueError("Rigid contact evidence inventory is incomplete")
    for name, digest in manifest.items():
        path = (attempt / name).resolve()
        if not path.is_relative_to(folder) or not path.is_file() or sha256_file(path) != digest:
            raise ValueError("Rigid contact evidence changed; rerun the selected review")

    def read(n):
        return json.loads((folder / n).read_text())

    req, native, replay, encoding, runtime, bundle = [
        read(n)
        for n in [
            "request.json",
            "validation.json",
            "animation_report.json",
            "encoding_report.json",
            "runtime.json",
            "contact_asset.json",
        ]
    ]
    if (
        req.get("source_scene_sha256") != source_sha
        or sha256_file(folder / "source.blend") != source_sha
        or not req.get("source_bindings")
    ):
        raise ValueError("Rigid contact source and visual bindings are stale or missing")
    if not native.get("ok") or native.get("frames", 0) < 2 or native.get("failures"):
        raise ValueError("Rigid contact native checks did not pass")
    for name in ["request.json", "motion.npz"]:
        if native.get("evidence_sha256", {}).get(name) != sha256_file(folder / name):
            raise ValueError("Rigid contact construction and motion do not match")
    if not all(
        replay.get(k)
        for k in ["ok", "source_material_replay", "fresh_process", "all_motion_framing"]
    ):
        raise ValueError("Rigid contact requires fresh original-material motion replay")
    if replay.get("source_scene_sha256") != source_sha or replay.get(
        "motion_sha256"
    ) != sha256_file(folder / "motion.npz"):
        raise ValueError("Rigid contact replay does not match its source and trajectory")
    if (
        replay.get("frames") != native["frames"]
        or replay.get("fps") != native.get("fps")
        or replay.get("preview_only")
    ):
        raise ValueError("Rigid contact movie must cover the complete native sequence")
    if not encoding.get("ok") or encoding.get("mp4", {}).get("frames") != native["frames"]:
        raise ValueError("Rigid contact movie did not fully decode")
    if detail.get("video", {}).get("sha256") != sha256_file(folder / "native_motion.mp4"):
        raise ValueError("Rigid contact registered movie changed")
    expected_runtime = {
        k: v
        for k, v in req.items()
        if k not in {"controls", "thread_checks", "separation_checks", "source_attempt_id"}
    }
    expected_runtime["controls"] = []
    if runtime != expected_runtime or not bundle.get("passive_runtime"):
        raise ValueError(
            "Rigid contact delivery must retain passive geometry and remove test drives"
        )
    if set(bundle.get("files", {})) != REQUIRED - {"contact_asset.json"}:
        raise ValueError("Rigid contact shareable bundle inventory is incomplete")
    if bundle.get("schema_version", 1) >= 2 and (
        bundle.get("appearance_source") != "source.blend"
        or bundle.get("appearance_review") != "native_motion.mp4"
        or bundle.get("preview_material_equivalence_verified") is not False
    ):
        raise ValueError("Rigid contact appearance must name original-source replay; GLB equivalence is not verified")
    for name, digest in bundle["files"].items():
        if sha256_file(folder / name) != digest:
            raise ValueError("Rigid contact shareable bundle changed")
    return folder


def prepare_motion(store, attempt_id, report):
    from .blender import FinalRenderResult
    from .video_review import register_video

    attempt = store.attempts_dir / attempt_id
    detail = report["rigid_contact"]
    folder = verify_evidence(attempt, detail)
    video = register_video(
        store,
        folder / "native_motion.mp4",
        video_id=f"candidate-{attempt_id}-rigid-contact",
        role="candidate_motion",
    )
    review = {
        "status": "awaiting_agent_review",
        "complete_video_encoded": True,
        "source_scene_sha256": detail["source_scene_sha256"],
        "motion_sha256": sha256_file(folder / "motion.npz"),
        "frames": detail["frames"],
        "directory": detail["directory"],
        "video": video,
        "review_policy": "Agent chooses temporal/view coverage and decides final acceptance.",
    }
    report["motion_review"] = review
    atomic_json(attempt / "report.json", report)
    store.append_event(
        {
            "type": "candidate_motion_returned_for_review",
            "attempt_id": attempt_id,
            "frames": detail["frames"],
        }
    )
    return FinalRenderResult(attempt_id, True, review, [])
