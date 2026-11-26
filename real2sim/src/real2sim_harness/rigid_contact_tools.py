"""Optional agent-facing concave mesh contact diagnostics, separate from acceptance."""

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
from .rigid_contact import validate_request


def schemas():
    return [
        {
            "type": "function",
            "name": "run_rigid_contact_behavior",
            "description": "Optionally test agent-authored closed rigid meshes using native Newton SDF contacts, explicit external wrenches, and measured screw-lead checks. Useful for threads and other concave mating parts. Read simulate-mating-rigid-parts. No helical motion is imposed. Set record_asset_evidence only for a source-bound rigid asset final review; this retains passive runtime data and complete video for the ordinary acceptance gate. Diagnostic calls do not select an acceptance backend.",
            "parameters": {
                "type": "object",
                "properties": {
                    "request_path": {
                        "type": "string",
                        "description": "Workspace-relative JSON with inline meshes or a source_attempt_id and source object bindings, explicit controls and optional measured checks.",
                    },
                    "assumptions": {"type": "string"},
                    "render_video": {"type": "boolean"},
                    "record_asset_evidence": {
                        "type": "boolean",
                        "description": "Explicitly select native SDF rigid asset review. Requires source_attempt_id and render_video=true. Does not itself accept delivery.",
                    },
                },
                "required": [
                    "request_path",
                    "assumptions",
                    "render_video",
                    "record_asset_evidence",
                ],
                "additionalProperties": False,
            },
            "strict": True,
        }
    ]


def run_tool(
    store,
    executor,
    workspace,
    *,
    request_path,
    assumptions,
    render_video,
    record_asset_evidence=False,
):
    if not assumptions.strip() or type(render_video) is not bool:
        raise ValueError("Record physical/control assumptions and select whether to render video")
    workspace = Path(workspace).resolve()
    path = (workspace / request_path).resolve()
    if (
        Path(request_path).is_absolute()
        or not path.is_relative_to(workspace)
        or path.suffix != ".json"
        or not path.is_file()
    ):
        raise ValueError("Use an existing workspace-relative JSON request")
    if path.stat().st_size > 50_000_000:
        raise ValueError("Rigid contact request exceeds the mesh budget")
    request = json.loads(path.read_text())
    source = None
    if request.get("source_attempt_id") is not None:
        attempt_id = request["source_attempt_id"]
        if not isinstance(attempt_id, str) or not re.fullmatch(r"attempt-[0-9]{3}", attempt_id):
            raise ValueError("Invalid source attempt")
        attempt = store.attempts_dir / attempt_id
        if not json.loads((attempt / "report.json").read_text()).get("ok"):
            raise ValueError("Use a successful source Blender attempt")
        source = attempt / "scene.blend"
        if not source.is_file():
            raise ValueError("Source Blender file is missing")
    else:
        request = validate_request(request)
    if type(record_asset_evidence) is not bool:
        raise ValueError("record_asset_evidence must be boolean")
    if record_asset_evidence:
        run = json.loads((store.root / "run.json").read_text())
        if source is None or not render_video or run["asset_contract"]["kind"] != "rigid":
            raise ValueError(
                "Native rigid asset review requires a source attempt, complete video and rigid contract"
            )
        report = json.loads((attempt / "report.json").read_text())
        report["selected_simulation_review"] = "rigid_contact"
        report["rigid_contact"] = {
            "ok": False,
            "source_scene_sha256": sha256_file(source),
            "status": "running",
        }
        atomic_json(attempt / "report.json", report)
    parent = attempt if record_asset_evidence else store.root / "behavior_tools"
    parent.mkdir(exist_ok=True)
    index = 1
    while (parent / f"rigid-contact-{index:03d}").exists():
        index += 1
    output = parent / f"rigid-contact-{index:03d}"
    output.mkdir()
    atomic_json(output / "input.json", request)
    atomic_json(
        output / "provenance.json",
        {"request_sha256": sha256_file(path), "assumptions": assumptions},
    )
    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(root / "src"), env.get("PYTHONPATH")]))
    env.setdefault("WARP_CACHE_PATH", str(store.root / ".cache/warp"))
    commands = [
        [
            sys.executable,
            str(root / "scripts/simulate_rigid_contact.py"),
            str(output / "input.json"),
            "--output",
            str(output),
        ],
        [
            str(executor.blender),
            "--background",
            "--factory-startup",
            "--python",
            str(root / "scripts/render_rigid_contact.py"),
            "--",
            str(output),
            "--mode",
            "render" if render_video else "preview",
        ],
    ]
    if source is not None:
        shutil.copy2(source, output / "source.blend")
        commands[1].insert(2, str(output / "source.blend"))
        commands[1].remove("--factory-startup")
        commands[1].append("--source-replay")
        commands.insert(
            0,
            [
                str(executor.blender),
                "--background",
                str(output / "source.blend"),
                "--python",
                str(root / "scripts/export_blend_rigid_contact.py"),
                "--",
                str(output / "input.json"),
                "--output",
                str(output / "input.json"),
            ],
        )
    simulation_stage = 1 if source is not None else 0
    if render_video:
        commands.append(
            [
                sys.executable,
                str(root / "scripts/encode_object_demo.py"),
                str(output),
                "--name",
                "native_motion",
            ]
        )
    error = None
    images = []
    video = None
    preview = {}
    try:
        for i, command in enumerate(commands):
            log_path = output / f"stage_{i}.log"
            with log_path.open("w") as log:
                p = subprocess.run(
                    command,
                    cwd=root,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=executor.timeout_seconds,
                    check=False,
                )
            if p.returncode and not (
                i == simulation_stage and p.returncode == 2 and (output / "motion.npz").is_file()
            ):
                raise RuntimeError(
                    f"Contact stage {i} failed: " + log_path.read_text(errors="replace")[-2500:]
                )
        preview = json.loads(
            (
                output / ("animation_report.json" if render_video else "preview_report.json")
            ).read_text()
        )
        names = preview["images"]
        selected = sorted(set(round(i * (len(names) - 1) / 6) for i in range(7)))
        images = [output / names[i] for i in selected]
        if render_video:
            from .video_review import register_video

            video = register_video(
                store,
                output / "native_motion.mp4",
                video_id=output.name,
                role="rigid_contact_diagnostic",
            )
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        error = str(exc)
    report = (
        json.loads((output / "validation.json").read_text())
        if (output / "validation.json").is_file()
        else {"ok": False}
    )
    if source is not None and sha256_file(source) != sha256_file(output / "source.blend"):
        report.update(ok=False, error="Source changed during contact review")
    report.update(
        directory=str(output),
        assumptions=assumptions,
        video=video,
        asset_acceptance_updated=False,
        cache_verification=preview,
        source_scene_sha256=sha256_file(source) if source is not None else None,
    )
    if error:
        report.update(ok=False, error=error)
    if record_asset_evidence:
        try:
            if not report.get("ok"):
                raise ValueError("Native contact checks failed; delivery remains blocked")
            from .contact_asset import retain_asset_evidence

            retain_asset_evidence(store, attempt_id, output, report)
        except (OSError, ValueError, KeyError) as exc:
            report.update(ok=False, error=str(exc))
            source_report = json.loads((attempt / "report.json").read_text())
            source_report["rigid_contact"] = report
            atomic_json(attempt / "report.json", source_report)
    atomic_json(output / "tool_report.json", report)
    store.append_event(
        {
            "type": "rigid_contact_behavior",
            "ok": report["ok"],
            "output": str(output.relative_to(store.root)),
        }
    )
    return FinalRenderResult(output.name, report["ok"], report, images)
