"""Optional agent tools for tetrahedral meshing and native behavioral evidence."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .artifacts import atomic_json, sha256_file
from .blender import FinalRenderResult


def schemas():
    result = [
        {
            "type": "function",
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "request_path": {
                        "type": "string",
                        "description": "Workspace-relative JSON; geometry is explicit inline data, in metres.",
                    },
                    "assumptions": {"type": "string"},
                },
                "required": ["request_path", "assumptions"],
                "additionalProperties": False,
            },
            "strict": True,
        }
        for name, description in {
            "tetrahedralize_control_mesh": "Optionally tetrahedralize an agent-authored watertight triangle control mesh using MeshPy/TetGen. Returns a reusable geometry JSON and numerical topology/volume evidence. Read simulate-deformable-behavior. This is mesh conversion, not image reconstruction or physics acceptance.",
            "run_deformable_behavior": "Optionally test explicit thin-shell or tetrahedral-solid geometry, or a saved source surface with fixed source fixtures, using native Newton VBD, physical sphere/box probes, point grasps and release. Returns measured regional response/recovery, validity checks and fresh-process replay images. Read simulate-deformable-behavior. Diagnostic calls do not select delivery. A source-bound surface can explicitly set record_asset_evidence=true with complete video for the normal deformable acceptance gate.",
        }.items()
    ]
    behavior = next(row for row in result if row["name"] == "run_deformable_behavior")
    behavior["parameters"]["properties"]["render_video"] = {
        "type": "boolean",
        "description": "Optionally render every retained frame and register a movie for inspect_video; false returns sampled replay images.",
    }
    behavior["parameters"]["required"].append("render_video")
    behavior["parameters"]["properties"]["record_asset_evidence"] = {
        "type": "boolean",
        "description": "Explicitly select source-bound native surface asset review; requires source_attempt_id and complete video. False keeps diagnostics only.",
    }
    behavior["parameters"]["required"].append("record_asset_evidence")
    return result


def _meshing(request, output):
    from .deformable_geometry import tetrahedralize

    mesh = tetrahedralize(
        request["vertices"],
        request["triangles"],
        max_volume_m3=request["max_volume_m3"],
        quality=request.get("quality", 1.4),
        max_elements=request.get("max_elements", 100000),
        max_surface_edge_m=request.get("max_surface_edge_m"),
    )
    atomic_json(
        output / "geometry.json",
        {key: mesh[key].tolist() for key in ("vertices", "triangles", "tetrahedra")},
    )
    report = {
        "ok": True,
        "vertices": len(mesh["vertices"]),
        "triangles": len(mesh["triangles"]),
        "tetrahedra": len(mesh["tetrahedra"]),
        "volume_m3": float(mesh["rest_volumes"].sum()),
        "minimum_tet_volume_m3": float(mesh["rest_volumes"].min()),
        "geometry": "geometry.json",
        "geometry_sha256": sha256_file(output / "geometry.json"),
        "scope": "Closed-manifold and element validity; no material or physical calibration.",
    }
    atomic_json(output / "validation.json", report)


def run_tool(
    store,
    executor,
    workspace,
    name,
    *,
    request_path,
    assumptions,
    render_video=False,
    record_asset_evidence=False,
):
    if (
        name not in {"tetrahedralize_control_mesh", "run_deformable_behavior"}
        or not assumptions.strip()
    ):
        raise ValueError(
            "Choose a deformable tool and record geometry/material/control assumptions"
        )
    if type(render_video) is not bool or (render_video and name != "run_deformable_behavior"):
        raise ValueError("Video replay is optional for native behavior tests only")
    workspace = Path(workspace).resolve()
    path = (workspace / request_path).resolve()
    if (
        Path(request_path).is_absolute()
        or not path.is_relative_to(workspace)
        or not path.is_file()
        or path.suffix != ".json"
    ):
        raise ValueError("Request must be an existing workspace-relative JSON")
    if path.stat().st_size > 30_000_000:
        raise ValueError("Request exceeds the inline mesh budget")
    request = json.loads(path.read_text())
    if type(record_asset_evidence) is not bool:
        raise ValueError("record_asset_evidence must be boolean")
    if request.get("source_attempt_id") is not None:
        if name != "run_deformable_behavior":
            raise ValueError("Source surface review uses run_deformable_behavior")
        from .source_deformable import run_review

        return run_review(
            store,
            executor,
            workspace,
            request,
            assumptions=assumptions,
            render_video=render_video,
            record_asset_evidence=record_asset_evidence,
        )
    if record_asset_evidence:
        raise ValueError("Native surface delivery evidence requires a source attempt")
    parent = store.root / "behavior_tools"
    parent.mkdir(exist_ok=True)
    number = 1
    while (parent / f"{name}-{number:03d}").exists():
        number += 1
    output = parent / f"{name}-{number:03d}"
    output.mkdir()
    atomic_json(output / "input.json", request)
    atomic_json(
        output / "provenance.json",
        {"tool": name, "request_sha256": sha256_file(path), "assumptions": assumptions},
    )
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(root / "src"), environment.get("PYTHONPATH")))
    )
    environment.setdefault("WARP_CACHE_PATH", str(store.root / ".cache/warp"))
    commands = (
        [
            [
                sys.executable,
                "-m",
                "real2sim_harness.deformable_tools",
                "--request",
                str(output / "input.json"),
                "--output",
                str(output),
            ]
        ]
        if name == "tetrahedralize_control_mesh"
        else [
            [
                sys.executable,
                str(root / "scripts/simulate_deformable_behavior.py"),
                str(output / "input.json"),
                "--output",
                str(output),
            ],
            [
                str(executor.blender),
                "--background",
                "--factory-startup",
                "--python",
                str(root / "scripts/render_deformable_behavior.py"),
                "--",
                str(output),
            ],
        ]
    )
    error = None
    images = []
    video = None
    if render_video:
        commands[-1].extend(["--mode", "render"])
    try:
        for index, command in enumerate(commands):
            with (output / f"stage_{index}.log").open("w") as log:
                process = subprocess.run(
                    command,
                    env=environment,
                    cwd=root,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=executor.timeout_seconds,
                    check=False,
                )
            # A failed physical gate still deserves visual feedback when the cache is finite.
            if process.returncode and not (
                index == 0
                and name == "run_deformable_behavior"
                and process.returncode == 2
                and (output / "motion.npz").is_file()
            ):
                raise RuntimeError(f"Behavior stage {index} failed; inspect its retained log")
        if name == "run_deformable_behavior":
            preview = json.loads(
                (
                    output / ("animation_report.json" if render_video else "preview_report.json")
                ).read_text()
            )
            selected = (
                sorted(set(round(i * (len(preview["images"]) - 1) / 6) for i in range(7)))
                if render_video
                else range(len(preview["images"]))
            )
            images = [output / preview["images"][i] for i in selected]
            if render_video:
                import imageio_ffmpeg

                from .video_review import register_video

                video_path = output / "native_motion.mp4"
                command = [
                    imageio_ffmpeg.get_ffmpeg_exe(),
                    "-y",
                    "-framerate",
                    str(preview["fps"]),
                    "-i",
                    str(output / "frames/frame_%04d.png"),
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-crf",
                    "19",
                    str(video_path),
                ]
                with (output / "encode.log").open("w") as log:
                    subprocess.run(
                        command,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=executor.timeout_seconds,
                        check=True,
                    )
                video = register_video(
                    store, video_path, video_id=output.name, role="deformable_behavior_diagnostic"
                )
    except (OSError, RuntimeError, ImportError, subprocess.SubprocessError) as exc:
        error = str(exc)
    report = (
        json.loads((output / "validation.json").read_text())
        if (output / "validation.json").is_file()
        else {"ok": False}
    )
    report.update(directory=str(output), assumptions=assumptions, tool=name)
    report["video"] = video
    if error:
        report.update(ok=False, error=error)
    report["asset_acceptance_updated"] = False
    atomic_json(output / "tool_report.json", report)
    store.append_event(
        {
            "type": "deformable_behavior_tool",
            "tool": name,
            "ok": report["ok"],
            "output": str(output.relative_to(store.root)),
        }
    )
    return FinalRenderResult(output.name, report["ok"], report, images)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    _meshing(json.loads(args.request.read_text()), args.output)


if __name__ == "__main__":
    main()
