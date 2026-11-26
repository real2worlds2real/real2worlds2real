from __future__ import annotations

import copy
import json
import math
import re
import subprocess
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any

from .artifacts import RunStore, atomic_json, sha256_file
from .metrics import compare_mask_sets


@dataclass(frozen=True)
class AttemptResult:
    attempt_id: str
    ok: bool
    summary: dict[str, Any]
    images: list[Path]


@dataclass(frozen=True)
class FinalRenderResult:
    attempt_id: str
    ok: bool
    summary: dict[str, Any]
    images: list[Path]


def _task_scoped(method):
    @wraps(method)
    def execute(self, store, *args, **kwargs):
        executor = self if getattr(self, "_scope_root", None) == store.root.resolve() else self.for_task(store)
        return method(executor, store, *args, **kwargs)
    return execute


class BlenderExecutor:
    def __init__(self, blender: Path, *, timeout_seconds: int = 180):
        self.blender = blender.resolve()
        self.timeout_seconds = timeout_seconds
        if not self.blender.is_file():
            raise FileNotFoundError(self.blender)

    def for_task(self, store, *, auth_dir=None):
        """Bind generated Blender code to this task without mutating other workers."""
        from .task_execution import blender_proxy

        clone = copy.copy(self)
        clone._original_blender = getattr(self, "_original_blender", self.blender)
        clone._auth_dir = auth_dir or getattr(self, "_auth_dir", None)
        clone._scope_root = store.root.resolve()
        clone.blender = blender_proxy(clone._original_blender, store.root, auth_dir=clone._auth_dir)
        return clone

    @_task_scoped
    def run_fluid(self, store, *, attempt_id: str, frames: int, assumptions: str):
        from .fluid import run_fluid
        return run_fluid(self, store, attempt_id=attempt_id, frames=frames, assumptions=assumptions)

    @_task_scoped
    def run_simulation(self, store, **arguments):
        from .sim_review import run_simulation
        return run_simulation(self, store, **arguments)

    @_task_scoped
    def run_hybrid(self, store, **arguments):
        from .hybrid import run_hybrid
        return run_hybrid(self, store, **arguments)

    @_task_scoped
    def run_strands(self, store, **arguments):
        from .strands import run_strands
        return run_strands(self, store, **arguments)

    @_task_scoped
    def run_solids(self, store, **arguments):
        from .solids import run_solids
        return run_solids(self, store, **arguments)

    @_task_scoped
    def run_fracture(self, store, **arguments):
        from .fracture import run_fracture
        return run_fracture(self, store, **arguments)

    @_task_scoped
    def execute(
        self,
        store: RunStore,
        *,
        script: str,
        plan: str,
        change_summary: str,
    ) -> AttemptResult:
        if "def build_scene" not in script:
            raise ValueError("Blender script must define build_scene()")
        if len(script.encode()) > 250_000:
            raise ValueError("Blender script exceeds the 250 KB phase-1 limit")
        attempt_id, attempt = store.new_attempt()
        script_path = attempt / "agent_scene.py"
        script_path.write_text(script, encoding="utf-8")
        atomic_json(
            attempt / "request.json",
            {"attempt_id": attempt_id, "plan": plan, "change_summary": change_summary},
        )
        harness = Path(__file__).with_name("blender_harness.py")
        command = [
            str(self.blender),
            "--background",
            "--factory-startup",
            "--python",
            str(harness),
            "--",
            str(script_path),
            str(attempt),
            str(store.root / "run.json"),
        ]
        store.append_event({"type": "attempt_started", "attempt_id": attempt_id, "plan": plan})
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            stdout, stderr = completed.stdout[-20_000:], completed.stderr[-20_000:]
            (attempt / "stdout.log").write_text(stdout, encoding="utf-8")
            (attempt / "stderr.log").write_text(stderr, encoding="utf-8")
            report_path = attempt / "report.json"
            report = json.loads(report_path.read_text()) if report_path.is_file() else {}
            run = json.loads((store.root / "run.json").read_text())
            if report.get("ok") and run.get("reference_masks"):
                reference_dir = store.root / "references" / "masks"
                report["silhouette"] = compare_mask_sets(reference_dir, attempt)
                atomic_json(report_path, report)
            ok = completed.returncode == 0 and bool(report.get("ok"))
            summary = {
                "attempt_id": attempt_id,
                "ok": ok,
                "returncode": completed.returncode,
                "geometry": report.get("geometry"),
                "export": report.get("export"),
                "contract": report.get("contract"),
                "articulation": report.get("articulation"),
                "cousin_validation": report.get("cousin_validation"),
                "functional_structure": report.get("functional_structure"),
                "camera_alignment_previews": report.get("camera_alignment_previews", []),
                "silhouette": report.get("silhouette"),
                "artifacts": report.get("artifacts", {}),
                "stdout_tail": stdout[-4000:],
                "stderr_tail": stderr[-4000:],
                "error": report.get("error"),
            }
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or "")[-20_000:]
            stderr = (exc.stderr or "")[-20_000:]
            (attempt / "stdout.log").write_text(stdout, encoding="utf-8")
            (attempt / "stderr.log").write_text(stderr, encoding="utf-8")
            summary = {
                "attempt_id": attempt_id,
                "ok": False,
                "error": f"Blender exceeded {self.timeout_seconds} seconds",
                "stdout_tail": stdout[-4000:],
                "stderr_tail": stderr[-4000:],
            }
            atomic_json(attempt / "report.json", summary)
            ok = False
        review_names = report.get("review_images", []) if "report" in locals() else []
        if not review_names:
            review_names = ["preview_front.png", "preview_side.png", "preview_back.png"]
        images = [attempt / name for name in review_names if (attempt / name).is_file()]
        store.append_event(
            {"type": "attempt_finished", "attempt_id": attempt_id, "ok": ok, "summary": summary}
        )
        return AttemptResult(attempt_id, ok, summary, images)

    @_task_scoped
    def run_deformation(
        self,
        store: RunStore,
        *,
        attempt_id: str,
        frames: int,
        assumptions: str,
    ) -> FinalRenderResult:
        if not re.fullmatch(r"attempt-[0-9]{3}", attempt_id):
            raise ValueError("Invalid attempt id")
        if frames not in {48, 72, 96, 120, 180} or not assumptions.strip():
            raise ValueError("Choose 48/72/96/120/180 frames and record physical assumptions")
        attempt = store.attempts_dir / attempt_id
        scene = attempt / "scene.blend"
        report_path = attempt / "report.json"
        report = json.loads(report_path.read_text())
        if not report.get("ok") or not scene.is_file():
            raise ValueError("Run deformation on a successful saved Blender attempt")
        index = len(list(attempt.glob("deformation-[0-9][0-9][0-9]"))) + 1
        output = attempt / f"deformation-{index:03d}"
        output.mkdir()
        harness = Path(__file__).with_name("blender_deformation_harness.py")
        command = [str(self.blender), "--background", str(scene), "--python", str(harness), "--"]
        store.append_event({"type": "deformation_started", "attempt_id": attempt_id, "frames": frames})
        atomic_json(output / "request.json", {"frames": frames, "assumptions": assumptions})
        detail = {}
        try:
            baked = subprocess.run(
                [*command, "simulate", str(output), str(frames)],
                check=False, capture_output=True, text=True, timeout=self.timeout_seconds,
            )
            (output / "stdout.log").write_text(baked.stdout[-20_000:])
            (output / "stderr.log").write_text(baked.stderr[-20_000:])
            path = output / "deformation.json"
            detail = json.loads(path.read_text()) if path.is_file() else {}
            detail["ok"] = baked.returncode == 0 and bool(detail.get("ok"))
            if detail["ok"]:
                # A new Blender process must reproduce the baked control vertices.
                verified = subprocess.run(
                    [str(self.blender), "--background", str(output / "simulation.blend"),
                     "--python", str(harness), "--", "verify", str(output)],
                    check=False, capture_output=True, text=True, timeout=self.timeout_seconds,
                )
                (output / "verification.log").write_text((verified.stdout + verified.stderr)[-20_000:])
                verification_path = output / "cache_verification.json"
                verification = json.loads(verification_path.read_text()) if verification_path.is_file() else {}
                detail["cache_verification"] = verification
                detail["ok"] = verified.returncode == 0 and bool(verification.get("ok"))
                if not detail["ok"]:
                    detail["error"] = "Native cache did not replay correctly in a fresh Blender process"
            elif not detail.get("error"):
                detail["error"] = (baked.stdout + baked.stderr)[-4000:]
        except subprocess.TimeoutExpired:
            detail.update(ok=False, error=f"Deformation exceeded {self.timeout_seconds} seconds")
        detail.update(attempt_id=attempt_id, assumptions=assumptions, directory=output.name)
        detail["cache_manifest"] = {
            str(path.relative_to(attempt)): sha256_file(path)
            for path in [output / "simulation.blend", *sorted(output.rglob("*.bphys"))]
            if path.is_file()
        }
        atomic_json(output / "deformation.json", detail)
        report["deformation"] = detail
        for path in output.rglob("*"):
            if path.is_file() and path.suffix in {".json", ".blend", ".bphys", ".npz", ".png"}:
                name = str(path.relative_to(attempt))
                report.setdefault("artifacts", {})[name] = name
        atomic_json(report_path, report)
        images = [output / name for name in detail.get("images", []) if (output / name).is_file()]
        store.append_event({"type": "deformation_finished", "attempt_id": attempt_id, "ok": detail.get("ok", False), "directory": output.name})
        # Cache filenames/hashes are retained on disk, not copied into every model response.
        summary = {key: value for key, value in detail.items() if key not in {"cache_manifest", "cache_files"}}
        summary["cache_file_count"] = len(detail.get("cache_files", []))
        return FinalRenderResult(attempt_id, bool(detail.get("ok")), summary, images)

    @_task_scoped
    def render_final(
        self,
        store: RunStore,
        *,
        attempt_id: str,
        resolution: int,
        samples: int,
    ) -> FinalRenderResult:
        if not re.fullmatch(r"attempt-[0-9]{3}", attempt_id):
            raise ValueError("Invalid attempt id")
        if resolution not in {768, 1024, 1536} or samples not in {64, 128, 256}:
            raise ValueError("Unsupported final-render quality")
        attempt = store.attempts_dir / attempt_id
        scene = attempt / "scene.blend"
        report_path = attempt / "report.json"
        if not scene.is_file() or not report_path.is_file():
            raise ValueError("Attempt has no saved Blender scene")
        report = json.loads(report_path.read_text())
        if not report.get("ok"):
            raise ValueError("Only a successful Blender attempt can receive a final render")
        harness = Path(__file__).with_name("blender_final_harness.py")
        command = [
            str(self.blender),
            "--background",
            str(scene),
            "--python",
            str(harness),
            "--",
            str(attempt),
            str(resolution),
            str(samples),
        ]
        store.append_event(
            {
                "type": "final_render_started",
                "attempt_id": attempt_id,
                "resolution": resolution,
                "samples": samples,
            }
        )
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            stdout, stderr = completed.stdout[-20_000:], completed.stderr[-20_000:]
            (attempt / "final_render_stdout.log").write_text(stdout, encoding="utf-8")
            (attempt / "final_render_stderr.log").write_text(stderr, encoding="utf-8")
            final_path = attempt / "final_render.json"
            final = json.loads(final_path.read_text()) if final_path.is_file() else {}
            ok = completed.returncode == 0 and bool(final.get("ok"))
            final.update(returncode=completed.returncode)
            if not ok:
                final.update(stdout_tail=stdout[-4000:], stderr_tail=stderr[-4000:])
        except subprocess.TimeoutExpired as exc:
            final = {
                "ok": False,
                "error": f"Final render exceeded {self.timeout_seconds} seconds",
                "stdout_tail": (exc.stdout or "")[-4000:],
                "stderr_tail": (exc.stderr or "")[-4000:],
            }
            ok = False
        images = [attempt / name for name in final.get("images", []) if (attempt / name).is_file()]
        report["final_render"] = final
        artifacts = report.setdefault("artifacts", {})
        for path in [*images, attempt / "final_render.json"]:
            if path.is_file():
                artifacts[path.name] = path.name
        atomic_json(report_path, report)
        store.append_event(
            {
                "type": "final_render_finished",
                "attempt_id": attempt_id,
                "ok": ok,
                "images": [path.name for path in images],
            }
        )
        return FinalRenderResult(attempt_id, ok, final, images)

    @_task_scoped
    def inspect_surface(
        self, store: RunStore, *, attempt_id: str, center: list[float], size_m: list[float],
        view_direction: list[float], frame: int | None,
    ) -> FinalRenderResult:
        if not re.fullmatch(r"attempt-[0-9]{3}", attempt_id):
            raise ValueError("Invalid attempt id")
        if any(len(values) != 3 or any(not math.isfinite(value) for value in values) for values in (center, size_m, view_direction)):
            raise ValueError("Center, size and direction must be finite three-vectors")
        if min(size_m) <= 0 or sum(value**2 for value in view_direction) <= 1e-12:
            raise ValueError("Region size and view direction must be nonzero")
        attempt = store.attempts_dir / attempt_id
        report_path = attempt / "report.json"
        report = json.loads(report_path.read_text())
        if not report.get("ok"):
            raise ValueError("Surface review requires a successful saved attempt")
        scene = attempt / "scene.blend"
        native_replay = None
        if frame is not None:
            native_kind = next((key for key in ("strand", "hybrid", "solid", "rigid_contact", "native_deformable") if report.get(key, {}).get("ok") and (key not in {"rigid_contact", "native_deformable"} or report.get("selected_simulation_review") == key)), None)
            if native_kind:
                native = report[native_kind]
                folder = (attempt / native["directory"]).resolve()
                if (not folder.is_relative_to(attempt.resolve()) or not 0 <= frame < native["frames"]
                        or native.get("source_scene_sha256") != sha256_file(scene)):
                    raise ValueError("Native frame inspection requires current source and a valid cache index")
                if not native.get("cache_manifest"):
                    raise ValueError("Native frame inspection requires a verified cache manifest")
                for name, digest in native["cache_manifest"].items():
                    path = (attempt / name).resolve()
                    if not path.is_relative_to(attempt.resolve()) or not path.is_file() or sha256_file(path) != digest:
                        raise ValueError("Native surface cache changed; rerun simulation before inspection")
                native_replay = {"kind": native_kind, "directory": str(folder), "cache_frame": frame}
            else:
                deformation = report.get("deformation", {})
                if not deformation.get("ok") or not deformation.get("frame_start", 1) <= frame <= deformation.get("frame_end", 0):
                    raise ValueError("Frame inspection requires a passed deformation cache covering that frame")
                scene = attempt / deformation["directory"] / "simulation.blend"
        output = attempt / f"surface-review-{len(list(attempt.glob('surface-review-*'))) + 1:03d}"
        output.mkdir()
        request = {"center": center, "size_m": size_m, "view_direction": view_direction, "frame": frame}
        if native_replay:
            request["native_replay"] = native_replay
        atomic_json(output / "request.json", request)
        store.append_event({"type": "surface_review_started", "attempt_id": attempt_id, **request})
        try:
            completed = subprocess.run(
                [str(self.blender), "--background", str(scene), "--python",
                 str(Path(__file__).with_name("blender_surface_harness.py")), "--", str(output / "request.json")],
                check=False, capture_output=True, text=True, timeout=self.timeout_seconds,
            )
            (output / "stdout.log").write_text(completed.stdout[-20_000:])
            (output / "stderr.log").write_text(completed.stderr[-20_000:])
            path = output / "surface_review.json"
            review = json.loads(path.read_text()) if path.is_file() else {}
            review["ok"] = completed.returncode == 0 and bool(review.get("ok"))
            if not review["ok"]:
                review.setdefault("error", (completed.stdout + completed.stderr)[-4000:])
        except subprocess.TimeoutExpired:
            review = {"ok": False, "error": f"Surface review exceeded {self.timeout_seconds} seconds"}
        review.update(attempt_id=attempt_id, directory=output.name)
        atomic_json(output / "surface_review.json", review)
        report.setdefault("surface_reviews", []).append({"directory": output.name, "ok": review["ok"], **request})
        images = [output / name for name in review.get("images", []) if (output / name).is_file()]
        for path in [*images, output / "surface_review.json", output / "request.json"]:
            name = str(path.relative_to(attempt))
            report.setdefault("artifacts", {})[name] = name
        atomic_json(report_path, report)
        store.append_event({"type": "surface_review_finished", "attempt_id": attempt_id, "ok": review["ok"], "directory": output.name})
        return FinalRenderResult(attempt_id, review["ok"], review, images)
