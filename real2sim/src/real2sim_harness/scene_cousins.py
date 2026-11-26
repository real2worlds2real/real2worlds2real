"""Scene-cousin families over accepted scenes and existing functional object workers."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import uuid
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .artifacts import RunStore, atomic_json, sha256_file

SCENE_COUSIN_INSTRUCTIONS = """
This is one requested cousin of an accepted scene, not another pixel-matched twin.
Read context/scene_cousin.json, context/parent_scene.json and generate-scene-cousins.
The parent images/videos describe the accepted parent. Interpret the current raw user request and variant
prompt yourself. By default preserve the parent's recognizable layout, room structure and support relationships.
Keep operation-object placement anchors, support proxies and background placement anchors unchanged by default.
Change appearance and object morphology within those anchors, preserving usable clearances and function.
If layout_change_request is non-null, the user requested layout changes: follow that scope and revalidate
supports, collisions, clearances and operation. Otherwise do not use rearrangement, a global scene transform
or moved internal background geometry as cousin diversity. The original user prompt remains authoritative.
Reuse unchanged verified participants from the parent library. For a redesigned participant, use
build_scene_cousin_object with the parent's object_id so its functional structure survives.
Articulated joint identities/types/connectivity, mechanical loops and cloth attachment ownership remain;
metric dimensions, joint positions, shapes and materials may change. Revalidate actual target motion.
The participant ID set and simulation kinds remain fixed in this initial scene-cousin interface; decorative
environment details can be authored as mesh backgrounds. If requested participant topology is unsupported,
report that limitation rather than silently dropping an operation target.
Choose optional background tools: code-generated mesh environments, Gaussian cousin edits, or an installed
image-conditioned background backend. Keep background visuals, actual supports and participant placement
consistent. New unobserved regions are inferred. Background appearance supplies no collisions.
Do not recursively request cousins or start extra variants. The family coordinator owns quantity and progress.
Finish only with accept_scene after current whole-scene simulator and visual review; object or background
success alone is insufficient. Failed final feedback is an opportunity to repair in this same session.
"""

SCENE_PLANNING_INSTRUCTIONS = """Plan requested cousins of an accepted scene.
Read task.json, context/parent_scene.json and generate-scene-cousins. Interpret the user's natural-language
quantity and scope; preserve the raw request. Use submit_cousin_plan with one prompt per requested scene.
Prompts may coordinate environment appearance and individual object redesign; no rigid prompt grammar is required.
No fixed variant count is implied. An explicit count overrides inferred quantity. Default to meaningful
object-shape redesign unless the user specifically requests appearance-only variants.
Preserve operation-target identities and function. Joint topology and attachment ownership remain, while
dimensions and appearance can adapt within the parent's placement anchors. Preserve relative positions,
room structure, support relationships and clearances. Keep object/support/background placement anchors
and support proxy dimensions unchanged by default; do not invent furniture rearrangement for diversity.
If the user requests layout changes in natural language, summarize only their requested scope in the
variant's layout_change_request; otherwise use null. This is an internal interpretation, not a required
user-facing switch. Requested layout variation is valid scaling and needs fresh physical/visual checks.
All-mesh and optional Gaussian backgrounds are both valid.
Do not model or simulate during planning. Tool choice in the subsequent stage belongs to the scene agent.
Set execute=false for planning-only requests; preserve unresolved decisions with a concise question.
Finish using submit_cousin_plan.
"""


def _inventory(folder):
    return {
        str(p.relative_to(folder)): sha256_file(p) for p in sorted(folder.rglob("*")) if p.is_file()
    }


def _check_inventory(folder, files):
    folder = Path(folder).resolve()
    for name, digest in files.items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder) or not path.is_file() or sha256_file(path) != digest:
            raise ValueError("Retained scene-cousin parent/library changed: " + name)


def _extract_structure(executor, source, output):
    from .task_execution import policy_for, sandbox_command

    extractor = Path(__file__).resolve().parents[2] / "scripts/extract_asset_structure.py"
    output.parent.mkdir(parents=True, exist_ok=True)
    before = sha256_file(source)
    # A caller may place a family outside the parent's task directory. Inspect
    # its retained copy within the family scope, with only structure output
    # writable, instead of inheriting the unrelated parent's Blender proxy.
    scope = output.parent.parent.resolve()
    if not source.resolve().is_relative_to(scope):
        raise ValueError("Extract structure from the retained family library")
    binary = getattr(executor, "_original_blender", executor.blender)
    command = sandbox_command(
        [str(binary), "--background", str(source), "--python-exit-code", "1",
         "--python", str(extractor), "--", str(output)],
        policy_for(scope, executable=binary, writable=[output.parent]),
    )
    with output.with_suffix(".log").open("w") as log:
        result = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=executor.timeout_seconds,
            check=False,
        )
    if result.returncode or not output.is_file():
        raise RuntimeError(
            "Parent participant structure extraction failed; inspect its retained log"
        )
    if sha256_file(source) != before:
        raise ValueError("Parent editable source changed during structure extraction")
    return json.loads(output.read_text())


def create_planning_run(
    parent,
    executor,
    *,
    prompt,
    count=None,
    model=None,
    max_turns=12,
    families_root=None,
    scene_runner=None,
    gaussian_runner=None,
):
    from .capabilities import TASK_TOOLS as CODEX_TOOLS
    from .scene_review import _current_runtime, _media_path, scene_folder
    from .sim_bundle import verify_bundle

    parent = RunStore(Path(parent).resolve())
    final = json.loads((parent.root / "final_scene.json").read_text())
    if final.get("status") != "accepted":
        raise ValueError("Scene cousins require an accepted parent scene")
    if count is not None and (type(count) is not int or count < 1):
        raise ValueError("An explicit cousin count must be a positive integer")
    if not isinstance(prompt, str) or (not prompt.strip() and count is None):
        raise ValueError("Supply a natural-language cousin request or an explicit count")
    folder = scene_folder(parent, final["scene_id"])
    runtime = _current_runtime(folder)
    if sha256_file(folder / "scene_bundle.json") != final["scene_bundle_sha256"]:
        raise ValueError("Parent scene no longer matches its final receipt")
    parent_run = json.loads((parent.root / "run.json").read_text())
    runners = {}
    for key, override in (("scene_runner", scene_runner), ("gaussian_runner", gaussian_runner)):
        value = override or parent_run.get(key)
        if value is not None:
            path = Path(value).resolve()
            if not path.is_file():
                raise ValueError("Scene cousin operator configuration does not exist: " + key)
            runners[key] = str(path)
    spec = json.loads((folder / "request.json").read_text())
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    family = Path(families_root or parent.root / "cousins") / ("family-" + stamp)
    family.mkdir(parents=True)
    # Preserve the exact accepted package separately from the usable source library.
    bundle = json.loads((folder / "scene_bundle.json").read_text())
    snapshot = family / "parent_scene"
    for name, digest in bundle["files"].items():
        source, target = folder / name, snapshot / name
        if not source.resolve().is_relative_to(folder.resolve()):
            raise ValueError("Parent package inventory escapes its directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha256_file(target) != digest:
            raise ValueError("Parent changed during snapshot")
    shutil.copy2(folder / "scene_bundle.json", snapshot / "scene_bundle.json")
    catalog = {}
    layout = dict(spec)
    layout["objects"] = []
    for row in spec["objects"]:
        destination = family / "library" / row["id"]
        shutil.copytree(snapshot / "assets" / row["id"], destination)
        metadata_path = destination / "asset_bundle.json"
        metadata = json.loads(metadata_path.read_text())
        editable = destination / "editable/scene.blend"
        if editable.is_file():
            metadata.update(
                source_blend="editable/scene.blend", source_blend_sha256=sha256_file(editable)
            )
            atomic_json(metadata_path, metadata)
        if not verify_bundle(destination)["ok"]:
            raise ValueError("Parent participant cannot be retained as a verified source bundle")
        catalog[row["id"]] = {
            "bundle": str(destination),
            "asset_kind": metadata["asset_kind"],
            "manifest_sha256": sha256_file(metadata_path),
            "editable_blend": str(editable) if editable.is_file() else None,
            "transform": row["transform"],
        }
        if editable.is_file():
            structure = _extract_structure(
                executor, editable, family / "structures" / (row["id"] + ".json")
            )
            contract = structure["asset_contract"]
            if contract["kind"] != metadata["asset_kind"]:
                raise ValueError(
                    "Parent source contract differs from its verified scene representation"
                )
            catalog[row["id"]].update(
                functional_structure=structure,
                asset_contract=contract,
                geometry_fingerprint=structure["geometry_fingerprint"],
            )
        layout["objects"].append({**row, "bundle": str(destination)})
    layout["backgrounds"] = [
        {**row, "path": str(snapshot / "backgrounds" / (row["id"] + ".usdz"))}
        for row in spec["backgrounds"]
    ]
    # Retain accepted scene views and continuous evidence for optional review.
    evidence = family / "parent_evidence"
    evidence.mkdir()
    visual = []
    for i, name in enumerate(runtime["report"].get("images", [])[:8]):
        source = _media_path(folder / runtime["output"], name)
        target = evidence / f"view-{i:03d}{source.suffix}"
        shutil.copy2(source, target)
        visual.append(target)
    movies = []
    for i, name in enumerate(runtime["report"].get("videos", [])[:4]):
        source = _media_path(folder / runtime["output"], name)
        target = evidence / f"motion-{i:03d}{source.suffix}"
        shutil.copy2(source, target)
        movies.append(str(target))
    if not visual:
        visual = [parent.root / row["path"] for row in parent_run["references"]]
    objective = prompt or f"Generate {count} functional cousins of this accepted scene."
    planner = RunStore.create(
        family / "planning",
        references=visual,
        objective=objective,
        model=model or parent_run["model"],
        max_turns=max_turns,
        tool_schemas=CODEX_TOOLS,
        asset_kind="auto",
    )
    context = planner.root / "context"
    context.mkdir()
    info = {
        "parent_run": str(parent.root),
        "parent_scene_id": final["scene_id"],
        "accepted_bundle_sha256": final["scene_bundle_sha256"],
        "acceptance_reason": final["reason"],
        "original_twin_objective": parent_run["objective"],
        "catalog": catalog,
        "layout": layout,
        "parent_movies": movies,
        "scene_runtime": final["runtime"],
    }
    atomic_json(context / "parent_scene.json", info)
    atomic_json(context / "parent_manifest.json", layout)
    from .video_review import register_video

    for i, name in enumerate(movies):
        register_video(planner, name, video_id=f"parent-scene-{i:03d}", role="parent_scene")
    batch = {
        "schema_version": 1,
        "family_kind": "scene",
        "status": "awaiting_plan",
        "request": prompt,
        "explicit_count": count,
        "parent": info,
        "planner": str(planner.root.relative_to(family)),
        "snapshot_files": _inventory(snapshot),
        "library_files": _inventory(family / "library"),
        "runners": runners,
        "parent_context_sha256": sha256_file(context / "parent_scene.json"),
        "variants": [],
    }
    atomic_json(family / "family.json", batch)
    record = json.loads((planner.root / "run.json").read_text())
    record.update(
        generation_mode="scene_cousin_planning",
        cousin_family=str(family),
        cousin_explicit_count=count,
        cousin_request=prompt,
        contract_inference_required=False,
    )
    atomic_json(planner.root / "run.json", record)
    return family, planner


def verify_parent(family, batch):
    _check_inventory(family / "parent_scene", batch["snapshot_files"])
    _check_inventory(family / "library", batch["library_files"])
    context = family / batch["planner"] / "context/parent_scene.json"
    if (
        sha256_file(context) != batch["parent_context_sha256"]
        or json.loads(context.read_text()) != batch["parent"]
    ):
        raise ValueError("Parent scene-cousin context changed")


def create_child(family, index, *, max_turns):
    from .capabilities import TASK_TOOLS as CODEX_TOOLS
    from .video_review import register_video

    family = Path(family).resolve()
    batch = json.loads((family / "family.json").read_text())
    verify_parent(family, batch)
    variant = batch["variants"][index]
    planner = RunStore(family / batch["planner"])
    planner_run = json.loads((planner.root / "run.json").read_text())
    references = [planner.root / row["path"] for row in planner_run["references"]]
    child = RunStore.create(
        family / f"variant-{index + 1:03d}",
        references=references,
        objective=variant["prompt"],
        model=planner_run["model"],
        max_turns=max_turns,
        tool_schemas=CODEX_TOOLS,
        asset_kind="auto",
    )
    shutil.copytree(planner.root / "context", child.root / "context")
    record = json.loads((child.root / "run.json").read_text())
    siblings = []
    for row in batch["variants"]:
        if row["status"] == "accepted":
            final = json.loads((family / row["run"] / "final_scene.json").read_text())
            siblings.append(
                {
                    "index": row["index"],
                    "prompt": row["prompt"],
                    "geometry_fingerprint": final["cousin"]["geometry_fingerprint"],
                }
            )
    cousin = {
        "family": str(family),
        "variant_index": index + 1,
        "variant_count": batch["count"],
        "raw_user_request": batch["request"],
        "shared_notes": batch.get("shared_notes", ""),
        "variant_prompt": variant["prompt"],
        "allow_surface_only": not variant["geometry_change_expected"],
        "layout_change_request": variant.get("layout_change_request"),
        "accepted_siblings": siblings,
        "execution_request": batch.get("execution_request"),
    }
    record.update(
        generation_mode="scene_cousin", contract_inference_required=False, scene_cousin=cousin
    )
    record.update(batch.get("runners", {}))
    atomic_json(child.root / "run.json", record)
    atomic_json(child.root / "context/scene_cousin.json", cousin)
    for i, path in enumerate(batch["parent"].get("parent_movies", [])):
        register_video(child, path, video_id=f"parent-scene-{i:03d}", role="parent_scene")
    variant.update(status="running", run=str(child.root.relative_to(family)))
    atomic_json(family / "family.json", batch)
    return child


def run_family(family, executor, *, agent_options, execute_plan=False):
    from .codex_backend import run_codex_agent

    family = Path(family).resolve()
    batch = json.loads((family / "family.json").read_text())
    if batch["status"] == "needs_input" or (batch.get("execute") is False and not execute_plan):
        return batch
    if execute_plan:
        batch.update(
            execute=True,
            execution_request="An explicit later request authorizes execution of this retained plan; original design requirements remain.",
        )
        atomic_json(family / "family.json", batch)
    if not batch.get("count"):
        raise ValueError("Resume scene cousin planning before executing")
    verify_parent(family, batch)
    for index in range(batch["count"]):
        batch = json.loads((family / "family.json").read_text())
        row = batch["variants"][index]
        if row["status"] == "accepted":
            child = RunStore(family / row["run"])
            final = json.loads((child.root / "final_scene.json").read_text())
            from .scene_review import _current_runtime, scene_folder

            _current_runtime(scene_folder(child, final["scene_id"]))
            validate_variant(child, final["scene_id"])
            continue
        existing = bool(row.get("run"))
        child = (
            RunStore(family / row["run"])
            if existing
            else create_child(family, index, max_turns=agent_options["max_turns"])
        )
        record = json.loads((child.root / "run.json").read_text())
        if agent_options["max_turns"] > record["max_turns"]:
            if not 1 <= agent_options["max_turns"] <= 30:
                raise ValueError("Use at most 30 turns per invocation; cumulative counters are retained")
            record["max_turns"] = agent_options["max_turns"]
            atomic_json(child.root / "run.json", record)
        options = {k: v for k, v in agent_options.items() if k != "max_turns"}
        try:
            if (child.root / "final_scene.json").is_file():
                final = json.loads((child.root / "final_scene.json").read_text())
                from .scene_review import _current_runtime, scene_folder

                _current_runtime(scene_folder(child, final["scene_id"]))
                validate_variant(child, final["scene_id"])
                result = {"status": "accepted"}
            else:
                result = run_codex_agent(
                    executor,
                    child,
                    model=record["model"],
                    objective=record["objective"],
                    max_turns=record["max_turns"],
                    resume_existing=existing and (child.root / "codex/thread.json").is_file(),
                    **options,
                )
        except (OSError, ValueError, RuntimeError, TimeoutError) as error:
            result = {"status": "error", "error_type": type(error).__name__}
        batch = json.loads((family / "family.json").read_text())
        row = batch["variants"][index]
        row["status"] = result["status"]
        if result.get("error_type"):
            row["error_type"] = result["error_type"]
        if result["status"] != "accepted":
            batch.update(status="incomplete", unfinished_index=index + 1)
            atomic_json(family / "family.json", batch)
            return batch
        final = json.loads((child.root / "final_scene.json").read_text())
        validate_variant(child, final["scene_id"])
        batch["status"] = "running"
        atomic_json(family / "family.json", batch)
    batch["status"] = "complete"
    atomic_json(family / "family.json", batch)
    return batch


def _background_geometry(path):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    values = []
    gaussians = []
    for prim in stage.Traverse():
        if prim.GetTypeName() == "ParticleField3DGaussianSplat":
            from pxr import UsdVol

            field = UsdVol.ParticleField3DGaussianSplat(prim)
            points = np.asarray(field.GetPositionsAttr().Get(), dtype=float)
            scales = np.asarray(field.GetScalesAttr().Get(), dtype=float)
            quats = field.GetOrientationsAttr().Get()
            if not len(points):
                continue
            q = np.array([[v.GetReal(), *v.GetImaginary()] for v in quats], dtype=float)
            q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
            w, x, y, z = q.T
            rotations = np.stack(
                [
                    1 - 2 * (y * y + z * z),
                    2 * (x * y - z * w),
                    2 * (x * z + y * w),
                    2 * (x * y + z * w),
                    1 - 2 * (x * x + z * z),
                    2 * (y * z - x * w),
                    2 * (x * z - y * w),
                    2 * (y * z + x * w),
                    1 - 2 * (x * x + y * y),
                ],
                axis=1,
            ).reshape(-1, 3, 3)
            matrix = np.asarray(
                UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            ).T
            axes = np.einsum("ij,njk->nik", matrix[:3, :3], rotations * scales[:, None, :])
            covariance = axes @ np.swapaxes(axes, 1, 2)
            world = points @ matrix[:3, :3].T + matrix[:3, 3]
            gaussians.append(
                np.column_stack(
                    [
                        world,
                        covariance[:, 0, 0],
                        covariance[:, 0, 1],
                        covariance[:, 0, 2],
                        covariance[:, 1, 1],
                        covariance[:, 1, 2],
                        covariance[:, 2, 2],
                    ]
                )
            )
            continue
        if prim.IsA(UsdGeom.Mesh) or prim.GetTypeName().startswith("ParticleField"):
            attrs = {}
            for name in (
                "points",
                "faceVertexCounts",
                "faceVertexIndices",
                "positions",
                "scales",
                "orientations",
            ):
                attr = prim.GetAttribute(name)
                if attr and attr.HasAuthoredValueOpinion():
                    data = np.asarray(attr.Get())
                    payload = (
                        json.dumps([str(v) for v in data.tolist()]).encode()
                        if data.dtype.kind == "O"
                        else data.tobytes()
                    )
                    attrs[name] = [
                        str(data.dtype),
                        list(data.shape),
                        hashlib.sha256(payload).hexdigest(),
                    ]
            if attrs:
                values.append(
                    [
                        str(
                            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                                Usd.TimeCode.Default()
                            )
                        ),
                        attrs,
                    ]
                )
    if gaussians:
        data = np.round(np.concatenate(gaussians), 8)
        order = np.lexsort(tuple(data[:, i] for i in reversed(range(data.shape[1]))))
        values.append(
            [
                "gaussian_world_geometry",
                len(data),
                hashlib.sha256(data[order].tobytes()).hexdigest(),
            ]
        )
    return sorted(values, key=lambda value: json.dumps(value, sort_keys=True))


def build_object(
    executor,
    store,
    workspace,
    worker,
    *,
    object_id,
    objective,
    max_calls,
    geometry_change_expected=True,
    asset_id=None,
    reference_paths=None,
):
    from .capabilities import TASK_TOOLS as CODEX_TOOLS
    from .scene_agent import package_object_result, workspace_file

    record = json.loads((store.root / "run.json").read_text())
    if record.get("generation_mode") != "scene_cousin":
        raise ValueError("Functional scene-cousin workers require an accepted parent scene")
    if (
        not isinstance(objective, str)
        or not objective.strip()
        or type(geometry_change_expected) is not bool
    ):
        raise ValueError("Supply a participant redesign objective and geometry-change intent")
    if type(max_calls) is not int or not 1 <= max_calls <= 100:
        raise ValueError("Choose a worker budget from 1 to 100")
    family = Path(record["scene_cousin"]["family"]).resolve()
    batch = json.loads((family / "family.json").read_text())
    verify_parent(family, batch)
    parent = batch["parent"]["catalog"].get(object_id)
    if parent is None:
        raise ValueError("Use a parent operation object_id from context/parent_scene.json")
    if not parent.get("editable_blend") or not parent.get("functional_structure"):
        return {
            "ok": False,
            "status": "parent_editable_source_missing",
            "next_action": "The participant can be reused, but redesign requires its editable Blender source and structural contract.",
        }
    index_path = store.root / "scene_cousin_objects.json"
    index = json.loads(index_path.read_text()) if index_path.is_file() else {}
    if asset_id is not None:
        if asset_id not in index or index[asset_id]["object_id"] != object_id:
            raise ValueError("Resume using the returned asset_id for this object_id")
        row = index[asset_id]
        child = RunStore(store.root / row["run"])
        if (
            row["objective"] != objective
            or row["geometry_change_expected"] != geometry_change_expected
        ):
            raise ValueError(
                "Resume retains the original assignment; start a new asset for changed intent"
            )
    else:
        asset_id = f"asset-{len(index) + 1:03d}"
        refs = (
            [workspace_file(workspace, p) for p in reference_paths]
            if reference_paths
            else [store.root / r["path"] for r in record["references"]]
        )
        if not 1 <= len(refs) <= 30:
            raise ValueError(
                "Provide 1..30 image references or use the retained parent scene views"
            )
        contract = parent["asset_contract"]
        child = RunStore.create(
            store.root / "scene-cousin-object-runs",
            references=refs,
            objective=objective,
            model=record["model"],
            max_turns=12,
            tool_schemas=CODEX_TOOLS,
            asset_kind="auto",
            front_axis=contract["front_axis"],
        )
        context = child.root / "context"
        context.mkdir()
        shutil.copy2(parent["editable_blend"], context / "parent_scene.blend")
        atomic_json(context / "parent_structure.json", parent["functional_structure"])
        parent_info = {
            "asset_contract": contract,
            "functional_structure": parent["functional_structure"],
            "scene_parent": batch["parent"]["parent_run"],
            "object_id": object_id,
            "source_scene_sha256": sha256_file(Path(parent["editable_blend"])),
        }
        siblings = []
        for variant in batch["variants"]:
            if variant["status"] != "accepted":
                continue
            sibling_path = family / variant["run"] / "scene_cousin_objects.json"
            if not sibling_path.is_file():
                continue
            for other in json.loads(sibling_path.read_text()).values():
                if other["object_id"] == object_id and other.get("geometry_fingerprint"):
                    siblings.append({"geometry_fingerprint": other["geometry_fingerprint"]})
        cousin = {
            "parent": parent_info,
            "raw_user_request": record["scene_cousin"]["raw_user_request"],
            "variant_prompt": objective,
            "shared_notes": record["scene_cousin"]["shared_notes"],
            "allow_surface_only": not geometry_change_expected,
            "accepted_siblings": siblings,
        }
        cousin["execution_authorized"] = batch.get("execute", True)
        cousin["execution_request"] = record["scene_cousin"].get("execution_request")
        child_record = json.loads((child.root / "run.json").read_text())
        child_record.update(
            generation_mode="cousin", cousin=cousin, parent_context="context/parent.json"
        )
        atomic_json(child.root / "run.json", child_record)
        atomic_json(context / "parent.json", parent_info)
        atomic_json(context / "cousin_task.json", cousin)
        atomic_json(
            context / "scene_parent.json",
            {
                "original_user_request": record["scene_cousin"]["raw_user_request"],
                "participant_assignment": objective,
                "object_id": object_id,
                "scope": "Current scene-cousin intent governs this participant. Parent scene images are context, not a pixel-matching target.",
            },
        )
        index[asset_id] = {
            "object_id": object_id,
            "objective": objective,
            "geometry_change_expected": geometry_change_expected,
            "run": str(child.root.relative_to(store.root)),
            "status": "running",
        }
        atomic_json(index_path, index)
    if not (child.root / "final.json").is_file():
        try:
            result = worker(
                executor,
                child,
                objective=objective,
                max_calls=max_calls,
                resume_existing=(child.root / "codex/thread.json").is_file(),
            )
        except (OSError, ValueError, RuntimeError, TimeoutError) as error:
            return {
                "ok": False,
                "asset_id": asset_id,
                "object_id": object_id,
                "run": str(child.root),
                "error": str(error),
                "next_action": "Resume this retained worker using asset_id.",
            }
        if result["status"] != "accepted":
            return {
                "ok": False,
                "asset_id": asset_id,
                "object_id": object_id,
                "run": str(child.root),
                "worker_status": result["status"],
            }
    packaged = package_object_result(executor, child, asset_id)
    if packaged.get("ok"):
        final = json.loads((child.root / "final.json").read_text())
        index = json.loads(index_path.read_text())
        index[asset_id].update(
            status="accepted",
            bundle=packaged["bundle"],
            bundle_sha256=sha256_file(Path(packaged["bundle"]) / "asset_bundle.json"),
            final_sha256=sha256_file(child.root / "final.json"),
            geometry_fingerprint=final["report"]["functional_structure"]["geometry_fingerprint"],
        )
        atomic_json(index_path, index)
    return {**packaged, "object_id": object_id}


def scene_geometry(manifest, object_fingerprints):
    values = {"objects": [], "supports": [], "backgrounds": []}
    for row in sorted(manifest["objects"], key=lambda r: r["id"]):
        values["objects"].append([row["id"], object_fingerprints[row["id"]], row["transform"]])
    for row in sorted(manifest["supports"], key=lambda r: r["id"]):
        values["supports"].append([row["id"], row["size_m"], row["transform"]])
    for row in sorted(manifest["backgrounds"], key=lambda r: r["id"]):
        values["backgrounds"].append(
            [row["id"], row["transform"], _background_geometry(row["path"])]
        )
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def validate_layout(parent, current, *, layout_change_request=None):
    """Preserve measurable placement anchors; semantic room structure needs visual review."""
    changed = []
    for group in ("objects", "supports", "backgrounds"):
        before = {row["id"]: row for row in parent[group]}
        after = {row["id"]: row for row in current[group]}
        if before.keys() != after.keys():
            raise ValueError("Scene cousin must preserve layout participant IDs: " + group)
        for name, row in after.items():
            if not np.allclose(row["transform"], before[name]["transform"], rtol=0, atol=1e-6):
                if layout_change_request is None:
                    raise ValueError(
                        "Scene cousin must preserve layout placement anchor unless the user requests layout changes: "
                        + name
                    )
                changed.append({"group": group, "id": name, "change": "placement"})
            if group == "supports" and not np.allclose(
                row["size_m"], before[name]["size_m"], rtol=0, atol=1e-6
            ):
                if layout_change_request is None:
                    raise ValueError(
                        "Scene cousin must preserve support proxy dimensions unless the user requests layout changes: "
                        + name
                    )
                changed.append({"group": group, "id": name, "change": "support_dimensions"})
    return changed


def validate_variant(store, scene_id):
    from .scene_review import scene_folder

    record = json.loads((store.root / "run.json").read_text())
    cousin = record.get("scene_cousin")
    if cousin is None:
        return None
    family = Path(cousin["family"]).resolve()
    batch = json.loads((family / "family.json").read_text())
    verify_parent(family, batch)
    parent = batch["parent"]
    current = json.loads((scene_folder(store, scene_id) / "request.json").read_text())
    old = parent["catalog"]
    if {row["id"] for row in current["objects"]} != set(old):
        raise ValueError("Scene cousin must preserve the operation participant ID set")
    planned = batch["variants"][cousin["variant_index"] - 1]
    layout_request = planned.get("layout_change_request")
    if cousin.get("layout_change_request") != layout_request:
        raise ValueError("Scene cousin layout intent differs from its retained user-request plan")
    layout_changes = validate_layout(
        parent["layout"], current, layout_change_request=layout_request
    )
    index_path = store.root / "scene_cousin_objects.json"
    receipts = json.loads(index_path.read_text()) if index_path.is_file() else {}
    fingerprints, parent_fingerprints = {}, {}
    for row in current["objects"]:
        object_id = row["id"]
        metadata = json.loads((Path(row["bundle"]) / "asset_bundle.json").read_text())
        if metadata["asset_kind"] != old[object_id]["asset_kind"]:
            raise ValueError("Scene cousin changed participant simulation kind: " + object_id)
        digest = sha256_file(Path(row["bundle"]) / "asset_bundle.json")
        baseline = old[object_id]["manifest_sha256"]
        parent_fingerprints[object_id] = old[object_id].get("geometry_fingerprint", baseline)
        if digest == baseline:
            fingerprints[object_id] = parent_fingerprints[object_id]
            continue
        match = next(
            (
                r
                for r in receipts.values()
                if r["object_id"] == object_id and r.get("bundle_sha256") == digest
            ),
            None,
        )
        if match is None:
            raise ValueError(
                "Redesigned participant needs its functional cousin worker receipt: " + object_id
            )
        child = RunStore(store.root / match["run"])
        if sha256_file(child.root / "final.json") != match["final_sha256"]:
            raise ValueError(
                "Participant cousin final receipt changed; regenerate its scene binding"
            )
        final = json.loads((child.root / "final.json").read_text())
        child.accept(final["attempt_id"], final["reason"], commit=False)
        fingerprints[object_id] = final["report"]["functional_structure"]["geometry_fingerprint"]
    geometry = scene_geometry(current, fingerprints)
    prior = scene_geometry(parent["layout"], parent_fingerprints)
    if not cousin["allow_surface_only"]:
        if geometry == prior:
            raise ValueError(
                "Scene cousin requires shape redesign or user-requested layout variation, not an unchanged clone"
            )
        if geometry in {row["geometry_fingerprint"] for row in cousin["accepted_siblings"]}:
            raise ValueError("Scene cousin duplicates an accepted sibling geometry")
    return {
        "ok": True,
        "geometry_fingerprint": geometry,
        "parent_geometry_fingerprint": prior,
        "geometry_changed": geometry != prior,
        "layout_anchors_preserved": not layout_changes,
        "layout_change_request": layout_request,
        "layout_anchor_changes": layout_changes,
        "scope": "Participant kind, functional-worker lineage, default fixed placement/support anchors unless the user requests layout variation, and exact geometry reuse. Astra reviews conformity to the requested layout scope, internal background layout, recognizable room structure, support relationships, clearances and semantic design using current simulator evidence.",
    }
