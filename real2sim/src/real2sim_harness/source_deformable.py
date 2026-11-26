"""Source-bound native surface controls, fixed fixtures, replay and asset evidence."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .artifacts import atomic_json, sha256_file
from .blender import FinalRenderResult

REQUIRED = {
    "input.json",
    "source.blend",
    "object.glb",
    "deformable_source.json",
    "behavior_request.json",
    "visual_binding.json",
    "visual_binding.npz",
    "request.json",
    "motion.npz",
    "validation.json",
    "animation_report.json",
    "native_motion.mp4",
    "encoding_report.json",
    "runtime.json",
    "surface_asset.json",
}


def passive_configuration(request, *, resolved_physx=None):
    c = {
        k: v for k, v in request.items() if k not in {"grasps", "measurements", "source_attempt_id"}
    }
    c["grasps"] = []
    c["measurements"] = []
    if resolved_physx is not None:
        c["physx"] = dict(resolved_physx)
    return c


def _runtime_description(validation):
    backend = validation.get("backend_id", "newton_native_vbd")
    if backend == "newton_native_vbd":
        return backend, "real2sim_harness.deformable_runtime.load_deformable", None
    if backend == "physx_particle_cloth":
        settings = validation.get("resolved_physx_settings")
        if not isinstance(settings, dict) or not settings or not validation.get("runtime_version"):
            raise ValueError("PhysX surface evidence needs resolved settings and runtime version")
        return backend, "real2sim_harness.physx_cloth.create_cloth", settings
    raise ValueError("Unsupported native surface runtime")


def verify_evidence(attempt, detail):
    attempt = Path(attempt).resolve()
    digest = sha256_file(attempt / "scene.blend")
    if not detail.get("ok") or detail.get("source_scene_sha256") != digest:
        raise ValueError("Native surface requires successful source-matched evidence")
    folder = (attempt / detail.get("directory", "")).resolve()
    if folder == attempt or not folder.is_relative_to(attempt):
        raise ValueError("Native surface evidence must stay inside its source attempt")
    manifest = detail.get("cache_manifest", {})
    if not {str((folder / n).relative_to(attempt)) for n in REQUIRED} <= set(manifest):
        raise ValueError("Native surface evidence inventory is incomplete")
    for name, sha in manifest.items():
        path = (attempt / name).resolve()
        if not path.is_relative_to(folder) or not path.is_file() or sha256_file(path) != sha:
            raise ValueError("Native surface evidence changed; repeat its review")

    def read(name):
        return json.loads((folder / name).read_text())

    native, replay, meta, c, request, encoded, runtime, bundle = [
        read(n)
        for n in [
            "validation.json",
            "animation_report.json",
            "visual_binding.json",
            "behavior_request.json",
            "request.json",
            "encoding_report.json",
            "runtime.json",
            "surface_asset.json",
        ]
    ]
    if (
        not native.get("ok")
        or native.get("representation") != "surface"
        or not native.get("contact_audit", {}).get("ok")
    ):
        raise ValueError("Native surface numerical or contact checks did not pass")
    if (
        c != request
        or c.get("source_scene_sha256") != digest
        or meta.get("source_scene_sha256") != digest
        or sha256_file(folder / "source.blend") != digest
    ):
        raise ValueError("Native surface source, conversion and simulation differ")
    if (
        meta.get("binding_sha256") != sha256_file(folder / "visual_binding.npz")
        or meta.get("configuration_sha256") != sha256_file(folder / "behavior_request.json")
        or meta.get("rest_binding_max_error_m", 1) > 1e-6
    ):
        raise ValueError("Native surface visual binding changed or fails rest reconstruction")
    for name in ["request.json", "motion.npz"]:
        if native.get("evidence_sha256", {}).get(name) != sha256_file(folder / name):
            raise ValueError("Native surface trajectory or request is stale")
    if not all(
        replay.get(k)
        for k in ["ok", "source_material_replay", "fresh_process", "all_motion_framing"]
    ):
        raise ValueError("Native surface requires fresh source-material replay")
    if (
        replay.get("source_scene_sha256") != digest
        or replay.get("motion_sha256") != sha256_file(folder / "motion.npz")
        or replay.get("rest_binding_max_error_m", 1) > 1e-6
    ):
        raise ValueError("Native surface replay does not match source/trajectory")
    if (
        replay.get("frames") != native["frames"]
        or replay.get("fps") != native["fps"]
        or replay.get("preview_only")
        or not encoded.get("ok")
        or encoded.get("mp4", {}).get("frames") != native["frames"]
    ):
        raise ValueError("Native surface movie does not cover every retained state")
    if detail.get("video", {}).get("sha256") != sha256_file(folder / "native_motion.mp4"):
        raise ValueError("Native surface registered movie changed")
    backend, loader, settings = _runtime_description(native)
    if settings is not None:
        from .physx_cloth import resolve_cloth_settings

        if resolve_cloth_settings({**c, "physx": settings})["settings"] != settings:
            raise ValueError("PhysX surface evidence must retain every resolved setting")
    if bundle.get("backend") != backend or bundle.get("loader") != loader:
        raise ValueError("Native surface bundle names a different backend or loader")
    if runtime != passive_configuration(c, resolved_physx=settings) or not bundle.get("passive_runtime"):
        raise ValueError("Native surface runtime must omit test grasps and measurements")
    if set(bundle.get("files", {})) != REQUIRED - {"surface_asset.json"}:
        raise ValueError("Native surface runtime bundle is incomplete")
    for name, sha in bundle["files"].items():
        if sha256_file(folder / name) != sha:
            raise ValueError("Native surface runtime bundle changed")
    return folder


def _retain(store, attempt_id, folder, detail):
    attempt = store.attempts_dir / attempt_id
    request = json.loads((folder / "behavior_request.json").read_text())
    validation = json.loads((folder / "validation.json").read_text())
    backend, loader, settings = _runtime_description(validation)
    atomic_json(folder / "runtime.json", passive_configuration(request, resolved_physx=settings))
    shutil.copy2(attempt / "object.glb", folder / "object.glb")
    bundle = {
        "schema_version": 1,
        "asset_kind": "deformable",
        "backend": backend,
        "runtime": "runtime.json",
        "source": "source.blend",
        "visual": "object.glb",
        "loader": loader,
        "test_configuration": "behavior_request.json",
        "passive_runtime": True,
        "fixed_source_fixtures_available": bool(request.get("probes")),
        "files": {n: sha256_file(folder / n) for n in REQUIRED - {"surface_asset.json"}},
        "scope": "One source-bound elastic surface with optional fixed analytic fixture context. No plasticity, physical calibration or robot policy.",
    }
    atomic_json(folder / "surface_asset.json", bundle)
    detail.update(
        directory=str(folder.relative_to(attempt)),
        cache_manifest={
            str((folder / n).relative_to(attempt)): sha256_file(folder / n) for n in REQUIRED
        },
    )
    verify_evidence(attempt, detail)
    report = json.loads((attempt / "report.json").read_text())
    report["native_deformable"] = detail
    report["selected_simulation_review"] = "native_deformable"
    report.setdefault("artifacts", {}).update({n: n for n in detail["cache_manifest"]})
    atomic_json(attempt / "report.json", report)


def retain_surface_evidence(store, attempt_id, folder, detail):
    """Register complete externally executed surface evidence using the same gate.

    The folder must be inside the source attempt and satisfy REQUIRED. This does
    not run physics or review images, and never turns a failed trial into success.
    """
    _retain(store, attempt_id, Path(folder), detail)


def run_review(
    store, executor, workspace, request, *, assumptions, render_video, record_asset_evidence
):
    attempt_id = request.get("source_attempt_id")
    if not isinstance(attempt_id, str) or not re.fullmatch(r"attempt-[0-9]{3}", attempt_id):
        raise ValueError("Use a saved source attempt")
    attempt = store.attempts_dir / attempt_id
    source = attempt / "scene.blend"
    source_report = json.loads((attempt / "report.json").read_text())
    if not source_report.get("ok") or not source.is_file():
        raise ValueError("Use a successful source Blender attempt")
    if record_asset_evidence:
        run = json.loads((store.root / "run.json").read_text())
        if not render_video or run["asset_contract"]["kind"] != "deformable":
            raise ValueError(
                "Native surface asset review requires a deformable contract and complete video"
            )
        source_report.update(
            selected_simulation_review="native_deformable",
            native_deformable={
                "ok": False,
                "status": "running",
                "source_scene_sha256": sha256_file(source),
            },
        )
        atomic_json(attempt / "report.json", source_report)
    parent = attempt if record_asset_evidence else store.root / "behavior_tools"
    parent.mkdir(exist_ok=True)
    index = 1
    while (parent / f"native-surface-{index:03d}").exists():
        index += 1
    folder = parent / f"native-surface-{index:03d}"
    folder.mkdir()
    atomic_json(folder / "input.json", request)
    shutil.copy2(source, folder / "source.blend")
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(root / "src"), env.get("PYTHONPATH")]))
    env.setdefault("WARP_CACHE_PATH", str(store.root / ".cache/warp"))
    commands = [
        [
            str(executor.blender),
            "--background",
            str(folder / "source.blend"),
            "--python",
            str(root / "scripts/export_blend_deformable_behavior.py"),
            "--",
            str(folder / "input.json"),
            "--output",
            str(folder / "deformable_source.json"),
        ],
        [
            sys.executable,
            str(root / "scripts/prepare_solid_native.py"),
            str(folder),
            "--source-file",
            "deformable_source.json",
            "--frames",
            str(request.get("frames", 120)),
        ],
        [
            sys.executable,
            str(root / "scripts/simulate_deformable_behavior.py"),
            str(folder / "behavior_request.json"),
            "--output",
            str(folder),
        ],
        [
            str(executor.blender),
            "--background",
            str(folder / "source.blend"),
            "--python",
            str(root / "scripts/render_deformable_behavior.py"),
            "--",
            str(folder),
            "--source-replay",
            "--mode",
            "render" if render_video else "preview",
        ],
    ]
    if render_video:
        commands.append(
            [
                sys.executable,
                str(root / "scripts/encode_object_demo.py"),
                str(folder),
                "--name",
                "native_motion",
            ]
        )
    error = None
    images = []
    video = None
    replay = {}
    try:
        for index, command in enumerate(commands):
            log = folder / f"stage_{index}.log"
            with log.open("w") as stream:
                p = subprocess.run(
                    command,
                    cwd=root,
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=executor.timeout_seconds,
                    check=False,
                )
            if p.returncode and not (
                index == 2 and p.returncode == 2 and (folder / "motion.npz").is_file()
            ):
                raise RuntimeError(
                    f"Native surface stage {index} failed: "
                    + log.read_text(errors="replace")[-3000:]
                )
        replay = json.loads(
            (
                folder / ("animation_report.json" if render_video else "preview_report.json")
            ).read_text()
        )
        frames = replay["images"]
        selected = sorted(set(round(i * (len(frames) - 1) / 6) for i in range(7)))
        images = [folder / frames[i] for i in selected]
        if render_video:
            from .video_review import register_video

            video = register_video(
                store,
                folder / "native_motion.mp4",
                video_id=folder.name,
                role="candidate_motion"
                if record_asset_evidence
                else "deformable_behavior_diagnostic",
            )
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        error = str(exc)
    detail = (
        json.loads((folder / "validation.json").read_text())
        if (folder / "validation.json").is_file()
        else {"ok": False}
    )
    detail.update(
        directory=str(folder),
        source_scene_sha256=sha256_file(source),
        cache_verification=replay,
        assumptions=assumptions,
        video=video,
        asset_acceptance_updated=False,
    )
    if sha256_file(source) != sha256_file(folder / "source.blend"):
        error = "Source changed during native surface review"
    if error:
        detail.update(ok=False, error=error)
    if record_asset_evidence:
        try:
            if not detail.get("ok"):
                raise ValueError("Native surface check failed; delivery remains blocked")
            _retain(store, attempt_id, folder, detail)
        except (OSError, ValueError, KeyError) as exc:
            detail.update(ok=False, error=str(exc))
            report = json.loads((attempt / "report.json").read_text())
            report["native_deformable"] = detail
            atomic_json(attempt / "report.json", report)
    atomic_json(folder / "tool_report.json", detail)
    store.append_event(
        {
            "type": "source_deformable_behavior",
            "ok": detail["ok"],
            "attempt_id": attempt_id,
            "output": str(folder.relative_to(store.root)),
        }
    )
    return FinalRenderResult(attempt_id, detail["ok"], detail, images)


def prepare_motion(store, attempt_id, report):
    from .video_review import register_video

    folder = verify_evidence(store.attempts_dir / attempt_id, report["native_deformable"])
    video = register_video(
        store,
        folder / "native_motion.mp4",
        video_id=f"candidate-{attempt_id}-native-surface",
        role="candidate_motion",
    )
    review = {
        "status": "awaiting_agent_review",
        "complete_video_encoded": True,
        "source_scene_sha256": report["native_deformable"]["source_scene_sha256"],
        "frames": report["native_deformable"]["frames"],
        "video": video,
        "review_policy": "Agent chooses useful temporal/detail coverage and decides delivery.",
    }
    report["motion_review"] = review
    atomic_json(store.attempts_dir / attempt_id / "report.json", report)
    return FinalRenderResult(attempt_id, True, review, [])
