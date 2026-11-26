from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class RunStore:
    root: Path

    @classmethod
    def create(
        cls,
        runs_root: Path,
        *,
        references: list[Path],
        objective: str,
        model: str,
        max_turns: int,
        tool_schemas: list[dict[str, Any]],
        asset_kind: str = "rigid",
        front_axis: str = "+Y",
        reference_mask_dir: Path | None = None,
        required_parts: list[str] | None = None,
    ) -> RunStore:
        if asset_kind not in {"auto", "rigid", "articulated", "deformable", "hybrid", "fluid", "strand", "fracture", "solid"}:
            raise ValueError(f"Unsupported asset kind: {asset_kind}")
        if front_axis not in {"+X", "-X", "+Y", "-Y"}:
            raise ValueError(f"Unsupported front axis: {front_axis}")
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        root = runs_root.resolve() / f"run-{stamp}-{uuid.uuid4().hex[:8]}"
        root.mkdir(parents=True)
        refs = root / "references"
        refs.mkdir()
        copied = []
        for index, source in enumerate(references):
            source = source.resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            target = refs / f"{index:02d}{source.suffix.lower()}"
            shutil.copy2(source, target)
            copied.append(
                {
                    "source_name": source.name,
                    "path": str(target.relative_to(root)),
                    "sha256": sha256_file(target),
                }
            )
        reference_masks = {}
        if reference_mask_dir is not None:
            mask_root = refs / "masks"
            mask_root.mkdir()
            for source in sorted(reference_mask_dir.resolve().glob("mask_*.png")):
                target = mask_root / source.name
                shutil.copy2(source, target)
                reference_masks[source.name] = str(target.relative_to(root))
        atomic_json(
            root / "run.json",
            {
                "schema_version": 2,
                "created_at": datetime.now(UTC).isoformat(),
                "status": "running",
                "objective": objective,
                "model": model,
                "max_turns": max_turns,
                "contract_inference_required": asset_kind == "auto",
                "references": copied,
                "tools": tool_schemas,
                "asset_contract": {
                    "kind": None if asset_kind == "auto" else asset_kind,
                    "front_axis": front_axis,
                    "up_axis": "+Z",
                    "units": "m",
                    "required_parts": sorted(set(required_parts or [])),
                },
                "reference_masks": reference_masks,
            },
        )
        return cls(root)

    @property
    def workspace_dir(self) -> Path:
        """Locate this task's active workspace independently of the model transport."""
        context = self.root / "external/context.json"
        if context.is_file():
            path = Path(json.loads(context.read_text())["workspace"]).resolve()
        elif (self.root / "codex/workspace").is_dir():
            path = (self.root / "codex/workspace").resolve()
        else:
            path = (self.root / "workspace").resolve()
        if not path.is_relative_to(self.root.resolve()) or path == self.root.resolve():
            raise ValueError("Workspace must be a separate directory inside this task")
        return path

    @property
    def attempts_dir(self) -> Path:
        path = self.root / "attempts"
        path.mkdir(exist_ok=True)
        return path

    def new_attempt(self) -> tuple[str, Path]:
        run = json.loads((self.root / "run.json").read_text())
        if run.get("contract_inference_required"):
            raise ValueError("Inspect the references and call declare_asset_contract before building")
        existing = sorted(self.attempts_dir.glob("attempt-*"))
        attempt_id = f"attempt-{len(existing) + 1:03d}"
        path = self.attempts_dir / attempt_id
        path.mkdir()
        return attempt_id, path

    def declare_asset_contract(
        self,
        *,
        kind: str,
        required_parts: list[str],
        observations: str,
        uncertainty: str,
    ) -> dict[str, Any]:
        if kind not in {"rigid", "articulated", "deformable", "hybrid", "fluid", "strand", "fracture", "solid"}:
            raise ValueError("Unsupported inferred asset kind")
        if not isinstance(required_parts, list) or not 1 <= len(required_parts) <= 80:
            raise ValueError("Declare 1 to 80 semantic part names")
        if any(not isinstance(part, str) or not part.strip() for part in required_parts):
            raise ValueError("Part names must be nonempty strings")
        if not observations.strip() or not uncertainty.strip():
            raise ValueError("Reference observations and uncertainty must be recorded")
        run_path = self.root / "run.json"
        run = json.loads(run_path.read_text())
        if not run.get("contract_inference_required"):
            raise ValueError("The asset contract is already fixed")
        if any(self.attempts_dir.glob("attempt-*")):
            raise ValueError("Cannot change the asset contract after building starts")
        contract = run["asset_contract"]
        if run.get("cousin") and kind != run["cousin"]["parent"]["asset_contract"]["kind"]:
            raise ValueError("A cousin must retain the parent's simulation asset kind")
        contract["kind"] = kind
        contract["required_parts"] = sorted(
            set(contract.get("required_parts", [])) | {part.strip() for part in required_parts}
        )
        analysis = {
            "asset_contract": contract,
            "observations": observations,
            "uncertainty": uncertainty,
            "source": "agent_inferred_from_references",
        }
        atomic_json(self.root / "asset_analysis.json", analysis)
        run["contract_inference_required"] = False
        run["asset_analysis"] = "asset_analysis.json"
        atomic_json(run_path, run)
        self.append_event({"type": "asset_contract_declared", **analysis})
        return analysis

    def append_event(self, event: dict[str, Any]) -> None:
        row = {"at": datetime.now(UTC).isoformat(), **event}
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def reopen(self, reason: str, evidence: dict[str, Any]) -> dict[str, Any]:
        """Withdraw an accepted delivery without discarding its source or history."""
        if not reason.strip():
            raise ValueError("A concrete rejection reason is required")
        run_path = self.root / "run.json"
        run = json.loads(run_path.read_text())
        final_path = self.root / "final.json"
        if run.get("status") != "accepted" or not final_path.is_file():
            raise ValueError("Only an accepted run can be reopened")
        final = json.loads(final_path.read_text())
        revision = len(list((self.root / "reviews").glob("rejection-*"))) + 1
        review_dir = self.root / "reviews" / f"rejection-{revision:03d}"
        review_dir.mkdir(parents=True)
        review = {"status": "needs_revision", "reason": reason, "evidence": evidence,
                  "rejected_attempt": final["attempt_id"],
                  "rejected_source_sha256": sha256_file(self.attempts_dir / final["attempt_id"] / "scene.blend"),
                  "rejected_report_sha256": sha256_file(self.attempts_dir / final["attempt_id"] / "report.json"),
                  "at": datetime.now(UTC).isoformat()}
        atomic_json(review_dir / "previous_final.json", final)
        atomic_json(review_dir / "review.json", review)
        codex_path = self.root / "codex" / "report.json"
        if codex_path.is_file():
            codex = json.loads(codex_path.read_text())
            atomic_json(review_dir / "previous_codex_report.json", codex)
            codex.pop("final", None)
            codex.update(status="needs_revision", human_review=str((review_dir / "review.json").relative_to(self.root)))
            atomic_json(codex_path, codex)
        run.pop("accepted_attempt", None)
        run.update(status="needs_revision", latest_rejection=str((review_dir / "review.json").relative_to(self.root)))
        atomic_json(run_path, run)
        # Historical acceptance stays in reviews; final.json denotes active delivery.
        final_path.unlink()
        self.append_event({"type": "acceptance_withdrawn", **review})
        return review

    def accept(self, attempt_id: str, reason: str, *, commit: bool = True) -> dict[str, Any]:
        attempt = self.attempts_dir / attempt_id
        report_path = attempt / "report.json"
        if attempt.parent != self.attempts_dir or not report_path.is_file():
            raise ValueError("Unknown attempt")
        report = json.loads(report_path.read_text())
        if not report.get("ok"):
            raise ValueError("Only a successful Blender attempt can be accepted")
        run = json.loads((self.root / "run.json").read_text())
        tool_names = {tool.get("name") for tool in run.get("tools", [])}
        if run.get('measurement_input'):
            from .measurement import require_current_review
            require_current_review(self, attempt, report)
        if run.get("cousin"):
            from .cousin_contract import validate_cousin
            cousin = run["cousin"]
            check = validate_cousin(cousin["parent"]["functional_structure"], report.get("functional_structure", {}),
                                   allow_surface_only=cousin["allow_surface_only"],
                                   sibling_geometry=[row["geometry_fingerprint"] for row in cousin["accepted_siblings"]])
            if not check["ok"]:
                raise ValueError("Cousin structure/geometry validation failed: " + ", ".join(check["failures"]))
        if run.get("latest_rejection"):
            rejection = json.loads((self.root / run["latest_rejection"]).read_text())
            if (sha256_file(attempt / "scene.blend") == rejection["rejected_source_sha256"]
                    and sha256_file(report_path) == rejection["rejected_report_sha256"]):
                raise ValueError("The rejected source and evidence are unchanged; correct and recheck the reported defect")
        if not report.get("contract", {}).get("ok"):
            raise ValueError("Asset contract must pass before acceptance")
        if run.get("asset_contract", {}).get("kind") == "articulated":
            if not report.get("articulation", {}).get("ok"):
                raise ValueError("Articulated assets must pass the motion and hierarchy audit")
        native_surface = (run.get("asset_contract", {}).get("kind") == "deformable"
                          and report.get("selected_simulation_review") == "native_deformable")
        if native_surface:
            from .source_deformable import verify_evidence as verify_surface
            verify_surface(attempt, report.get("native_deformable", {}))
        if run.get("asset_contract", {}).get("kind") in {"deformable", "hybrid"} and not native_surface and not (
            run["asset_contract"]["kind"] == "hybrid" and "run_hybrid" in tool_names
        ):
            deformation = report.get("deformation", {})
            if not deformation.get("ok") or not deformation.get("cache_verification", {}).get("ok"):
                raise ValueError("Deformable assets must pass native simulation and fresh-process cache replay")
            for obj in deformation.get("objects", []):
                if obj.get("type") == "CLOTH":
                    check = obj.get("stretch_check", {})
                    ratio = obj.get("metrics", {}).get("edge_length_ratio_p95")
                    if check.get("ok") is False or (not check and ratio is not None and ratio > 2):
                        raise ValueError("Cloth has widespread excessive edge stretch; repair physics before acceptance")
            if deformation.get("source_scene_sha256") != sha256_file(attempt / "scene.blend"):
                raise ValueError("Deformation evidence does not match the saved source scene")
            manifest = deformation.get("cache_manifest", {})
            if not manifest:
                raise ValueError("Deformation cache manifest is missing")
            for name, digest in manifest.items():
                path = (attempt / name).resolve()
                if not path.is_relative_to(attempt.resolve()) or not path.is_file() or sha256_file(path) != digest:
                    raise ValueError("Deformation cache is missing or changed; rerun the simulation")
        tool_names = {tool.get("name") for tool in run.get("tools", [])}
        from .contact_asset import selected as contact_selected
        from .contact_asset import verify_evidence as verify_contact
        native_contact = contact_selected(report, run.get("asset_contract", {}).get("kind"))
        if native_contact:
            verify_contact(attempt, report.get("rigid_contact", {}))
        special_backend = native_contact or native_surface or run.get("asset_contract", {}).get("kind") in {"fluid", "strand", "fracture", "solid"} or (
            run.get("asset_contract", {}).get("kind") == "hybrid" and "run_hybrid" in tool_names
        )
        if "run_simulation" in tool_names and not special_backend:
            if report.get("simulation") is None:
                raise ValueError(
                    "Final simulator acceptance is required: call run_simulation with render_views=True, "
                    "inspect its results, repair any failure, then submit again"
                )
        if report.get("simulation") is not None and not (native_contact or native_surface):
            from .sim_bundle import verify_bundle
            simulation = report["simulation"]
            if not simulation.get("ok"):
                raise ValueError("The selected simulator check failed; resolve it before acceptance")
            if simulation.get("source_scene_sha256") != sha256_file(attempt / "scene.blend"):
                raise ValueError("Simulator evidence does not match the saved Blender source")
            manifest_path = (attempt / simulation["bundle_manifest"]).resolve()
            if (not manifest_path.is_relative_to(attempt.resolve())
                    or not manifest_path.is_file()
                    or sha256_file(manifest_path) != simulation.get("bundle_sha256")
                    or not verify_bundle(manifest_path.parent)["ok"]):
                raise ValueError("Simulator bundle is missing or changed; rerun the selected check")
            bundle = json.loads(manifest_path.read_text())
            if "run_simulation" in tool_names and (
                not bundle.get("visual_render_verified")
                or not all(f"isaac_views/{name}.png" in bundle.get("files", {}) for name in ("front", "side", "top"))
            ):
                raise ValueError(
                    "Final simulator acceptance needs target-rendered images: call run_simulation with "
                    "render_views=True, inspect the returned views and correct defects before accepting"
                )
        if run.get("asset_contract", {}).get("kind") == "fracture":
            fracture = report.get("fracture", {})
            if not fracture.get("ok") or not fracture.get("cache_verification", {}).get("ok"):
                raise ValueError("Fracture requires native PhysX impact and fresh-process fragment replay")
            if fracture.get("source_scene_sha256") != sha256_file(attempt / "scene.blend"):
                raise ValueError("Fracture evidence does not match the saved source")
            if not fracture.get("cache_manifest"):
                raise ValueError("Fracture cache evidence is missing")
            for name, digest in fracture["cache_manifest"].items():
                path = (attempt / name).resolve()
                if not path.is_relative_to(attempt.resolve()) or not path.is_file() or sha256_file(path) != digest:
                    raise ValueError("Fracture evidence changed; rerun the simulation")
        if run.get("asset_contract", {}).get("kind") == "strand":
            strand = report.get("strand", {})
            if not strand.get("ok") or not strand.get("cache_verification", {}).get("ok"):
                raise ValueError("Strands require actual Newton rod dynamics and fresh-process visual replay")
            if strand.get("source_scene_sha256") != sha256_file(attempt / "scene.blend"):
                raise ValueError("Strand evidence does not match the saved source")
            if not strand.get("cache_manifest"):
                raise ValueError("Strand cache evidence is missing")
            for name, digest in strand["cache_manifest"].items():
                path = (attempt / name).resolve()
                if not path.is_relative_to(attempt.resolve()) or not path.is_file() or sha256_file(path) != digest:
                    raise ValueError("Strand evidence changed; rerun the simulation")
        if run.get("asset_contract", {}).get("kind") == "solid":
            solid = report.get("solid", {})
            replay = solid.get("cache_verification", {})
            if not solid.get("ok") or not replay.get("ok") or not replay.get("source_material_replay") or not replay.get("fresh_process"):
                raise ValueError("Solids require actual Newton FEM and fresh-process source-material replay")
            if solid.get("source_scene_sha256") != sha256_file(attempt / "scene.blend"):
                raise ValueError("Solid evidence does not match the saved source")
            folder = (attempt / solid.get("directory", "")).resolve()
            if folder == attempt.resolve() or not folder.is_relative_to(attempt.resolve()):
                raise ValueError("Solid evidence needs its retained native construction directory")
            required = {str((folder / name).relative_to(attempt)) for name in (
                "solid_source.json", "behavior_request.json", "visual_binding.json", "visual_binding.npz",
                "request.json", "motion.npz", "validation.json", "preview_report.json")}
            manifest = solid.get("cache_manifest", {})
            if not required <= set(manifest):
                raise ValueError("Solid native construction and replay inventory is incomplete")
            for name, digest in manifest.items():
                path = (attempt / name).resolve()
                if not path.is_relative_to(folder) or not path.is_file() or sha256_file(path) != digest:
                    raise ValueError("Solid evidence changed; rerun the simulation")
        if run.get("asset_contract", {}).get("kind") == "fluid":
            fluid = report.get("fluid", {})
            if not fluid.get("ok") or not fluid.get("cache_verification", {}).get("ok"):
                raise ValueError("Fluid assets must pass Newton material simulation and fresh-process surface replay")
            if fluid.get("source_scene_sha256") != sha256_file(attempt / "scene.blend"):
                raise ValueError("Fluid evidence does not match the source scene")
            if not fluid.get("cache_manifest"):
                raise ValueError("Fluid cache manifest is missing")
            for name, digest in fluid["cache_manifest"].items():
                path = (attempt / name).resolve()
                if not path.is_relative_to(attempt.resolve()) or not path.is_file() or sha256_file(path) != digest:
                    raise ValueError("Fluid cache is missing or changed; rerun the material simulation")
        if run.get("asset_contract", {}).get("kind") == "hybrid" and "run_hybrid" in tool_names:
            hybrid = report.get("hybrid", {})
            if not hybrid.get("ok") or not hybrid.get("cache_verification", {}).get("ok") or not hybrid.get("target_render", {}).get("ok"):
                raise ValueError("Hybrid assets must pass actual Newton rigid/cloth simulation, fresh replay and target-rendered review")
            if hybrid.get("source_scene_sha256") != sha256_file(attempt / "scene.blend"):
                raise ValueError("Hybrid evidence does not match the source scene")
            if not hybrid.get("cache_manifest"):
                raise ValueError("Hybrid cache manifest is missing")
            if run.get("latest_rejection") and not hybrid.get("surface_intersections", {}).get("ok"):
                raise ValueError("Revised hybrid must pass the complete cached-motion surface crossing check")
            for name, digest in hybrid["cache_manifest"].items():
                path = (attempt / name).resolve()
                if not path.is_relative_to(attempt.resolve()) or not path.is_file() or sha256_file(path) != digest:
                    raise ValueError("Hybrid cache is missing or changed; rerun the simulation")
        if run.get("asset_contract", {}).get("kind") in {"deformable", "hybrid"} and "inspect_surface" in tool_names:
            if not any(review.get("ok") for review in report.get("surface_reviews", [])):
                raise ValueError("Deformable assets need material/clay/grazing local surface evidence before acceptance")
        if "render_final" in tool_names and not report.get("final_render", {}).get("ok"):
            raise ValueError("Attempt must pass a final Cycles render before acceptance")
        for required in ("scene.blend", "object.glb"):
            if not (attempt / required).is_file():
                raise ValueError(f"Attempt is missing {required}")
        final = {
            "status": "accepted",
            "delivery_status": "simulation_verified" if native_contact or native_surface or any(report.get(key, {}).get("ok") for key in ("simulation", "fluid", "hybrid", "strand", "fracture", "solid")) else "blender_only",
            "attempt_id": attempt_id,
            "reason": reason,
            "accepted_at": datetime.now(UTC).isoformat(),
            "artifacts": {
                name: str((attempt / name).relative_to(self.root))
                for name in report.get(
                    "artifacts", {"scene.blend": "scene.blend", "object.glb": "object.glb"}
                )
                if (attempt / name).is_file()
            },
            "report": report,
        }
        if not commit:
            return final
        atomic_json(self.root / "final.json", final)
        run_path = self.root / "run.json"
        run["status"] = "accepted"
        run["accepted_attempt"] = attempt_id
        atomic_json(run_path, run)
        self.append_event({"type": "accepted", "attempt_id": attempt_id, "reason": reason})
        return final

    def finish_without_acceptance(self, status: str) -> None:
        run_path = self.root / "run.json"
        run = json.loads(run_path.read_text())
        if run.get("status") == "accepted":
            return
        run["status"] = status
        atomic_json(run_path, run)
