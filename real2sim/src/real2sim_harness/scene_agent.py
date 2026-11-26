"""Optional scene orchestration over the existing persistent Astra transport."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from .artifacts import RunStore, atomic_json, sha256_file


def _tool(name, description, properties, required):
    return {"type": "function", "name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required,
                           "additionalProperties": False}, "strict": True}


SCENE_TOOLS = [
    _tool('build_scene_cousin_object', 'Scene-cousin only: redesign one operation participant from its accepted parent, retaining functional structure and using the existing full object-cousin simulation/acceptance loop. Reuse unchanged parent bundles directly instead. asset_id resumes a retained worker. Scope and geometry-change intent are inferred by Astra from the user prompt.', {
        'object_id':{'type':'string'},'objective':{'type':'string'},
        'max_calls':{'type':'integer','minimum':1,'maximum':100},
        'geometry_change_expected':{'type':'boolean'}, 'asset_id':{'type':['string','null']},
        'reference_paths':{'type':['array','null'],'items':{'type':'string'},'minItems':1,'maxItems':30}},
        ['object_id','objective','max_calls','geometry_change_expected','asset_id','reference_paths']),
    _tool("plan_scene_physics", "Select one physics backend for the interacting task, rather than separate engines per object. Rigid/articulated, cloth and soft-body tasks default to PhysX. Fluid/strand tasks retain a Newton default. Explicit backend requests take priority; an adapter must verify the complete interaction set. The result is a capability request, not a simulation success.", {
        "objects": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "string"}, "kind": {"type": "string", "enum": ["rigid", "articulated", "deformable", "hybrid", "fluid", "strand", "fracture", "solid"]}},
            "required": ["id", "kind"], "additionalProperties": False}},
        "interactions": {"type": ["array", "null"], "items": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 2}},
        "preferred_backend": {"type": "string", "enum": ["auto", "physx", "newton"]}}, ["objects", "interactions", "preferred_backend"]),
    _tool("build_scene_object", "Use our complete object-generation and simulator-review pipeline for one scene participant. The object worker sees your selected images/video plus the task objective and must pass its normal acceptance. Reuse asset_id to resume an unfinished worker. Retain scene context and scale evidence in the objective. Final scene-level backend conversion may still be needed.", {
        "reference_paths": {"type": "array", "items": {"type": "string"}, "minItems": 0, "maxItems": 30},
        "objective": {"type": "string"}, "max_calls": {"type": "integer", "minimum": 1, "maximum": 100},
        "asset_id": {"type": ["string", "null"]}}, ["reference_paths", "objective", "max_calls", "asset_id"]),
    _tool("build_scene_background", "Execute a workspace Blender Python file defining build_scene() to author a visual mesh environment and export editable Blender plus USD materials. Keep interactive objects out; supports/colliders are specified separately. Optional alternative to a captured Gaussian background; generated hidden regions must not be called observed.", {
        "script_path": {"type": "string"}}, ["script_path"]),
    _tool("assemble_scene", "Compose a workspace scene manifest into a self-contained USD package with complete verified object bundles, visual backgrounds and static supports. Preserves joint/material relationships, relocates world joint anchors and returns structure checks. Composition alone is not simulator acceptance. Read compose-interactive-scene for the manifest contract.", {
        "manifest_path": {"type": "string"}}, ["manifest_path"]),
    _tool("run_scene_simulation", "Run the operator-configured simulator adapter on a composed scene. Set record_video when continuous operation evidence would help; the runner must support scene_video. Returned movies are available to inspect_video. Recording is optional, especially useful near final review of coupled motion. Review actual feedback and repair; no simulator version is forced.", {
        "scene_id": {"type": "string", "pattern": "^scene-[0-9]{3}$"},
        "record_video": {"type": "boolean"},
        "physics_hz": {"type": ["integer", "null"], "minimum": 1,
                       "description": "Optional physics frequency override; null keeps the runner default. Requires configurable_timestep capability and preserves the configured simulation duration."},
        "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 3600}}, ["scene_id", "timeout_seconds", "record_video", "physics_hz"]),
    _tool("run_gaussian_background", "Optionally prepare a captured Gaussian background: CPU crop/deduplicate and coverage inspection plus standard ParticleField export, CPU cousin appearance/region edits, or train/transcode/refine through an explicitly configured backend. Cousins preserve layout unless the user's request calls for layout changes. Optional refinement repairs novel-view proposals, propagates them with depth/occlusion checks and optimizes the actual 3D model while retaining original captures. Read prepare-gaussian-background for request contracts, coverage and limits. Gaussian appearance provides no collisions and does not replace editable operation targets. Full mesh is also valid.", {
        "request_path": {"type": "string"}, "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 3600}}, ["request_path", "timeout_seconds"]),
    _tool("accept_scene", "Finish a scene task after current scene/runtime checks pass and you judge the returned visual and functional evidence satisfactory. Failed or stale evidence rejects acceptance and permits repair. This is the required finish for scene generation.", {
        "scene_id": {"type": "string", "pattern": "^scene-[0-9]{3}$"}, "reason": {"type": "string"}}, ["scene_id", "reason"]),
]

SCENE_INSTRUCTIONS = """You are Astra, controlling an interactive Real2Sim scene task.
Read task.json and the startup task workflow. Read compose-interactive-scene when composing the scene. Understand the supplied images/video and user objective.
The user's explicitly named operation targets, parts, actions and exclusions take priority over inferred targets
or reconstruction-module suggestions. Preserve the original request when assigning participant workers.
You choose how to recover the scene, which tools to use and where additional evidence is useful.
For optional internal investigation, project_access.json lists readable framework source and this task's records.
Use native file/search tools as needed; only the agent workspace is writable.
Separate visual environment, static collision/support surfaces and editable interactive participants.
Use our build_scene_object pipeline for generated participants; reuse existing verified bundles when suitable.
Plan the backend for the whole interaction task. A per-object PhysX bundle is not a permanent engine assignment.
Cross-engine coupling is not the default. Selected-backend capabilities and any required conversion must be supplied by
a compatible adapter; never flatten an articulation or discard deformation data to make assembly pass.
Backgrounds may be captured meshes/Gaussians or code-generated environments. Mark unseen completion as inferred.
Keep original scene context when inspecting crops. Report estimated scale when metric evidence is unavailable.
Reference geometry, video tools and web search are optional aids, subject to the configured availability.
The scene manifest, checks and simulator protocol are in scene_contract.md. Write source/manifest files inside
your workspace. Compose, run the configured simulator, inspect evidence and repair until the task passes.
The same scene must be judged, not only isolated object renders. Check duplication/ghosts, support, scale,
placement, interpenetration, exposed background and the intended interactions. Request record_video when
continuous motion evidence is useful, then inspect the returned video ID with the shared video tools.
It is optional and especially useful for coupled motion; do not render a movie on every intermediate edit.
The optional run_gaussian_background tool handles captured backgrounds independently of the mesh branch.
Tool success alone is not your visual acceptance.
For an original twin scene that explicitly requests later cousins, retain that natural-language request with
request_cousins; the optional scene-cousin stage begins after this scene is accepted. Never invent extra variants.
Finish with accept_scene. If a required runtime capability is unavailable, preserve the work and clearly
report the missing adapter rather than claiming a delivered operable scene. Budgets remain enforced.
"""


def workspace_file(workspace, name):
    path = (Path(workspace) / name).resolve()
    if not path.is_relative_to(Path(workspace).resolve()) or not path.is_file():
        raise ValueError("Use a retained file inside the scene workspace")
    return path


def build_background(executor, store, workspace, *, script_path):
    source = workspace_file(workspace, script_path)
    if source.suffix != ".py":
        raise ValueError("Background source must be Python")
    index = len(list((store.root / "environments").glob("environment-*"))) + 1
    output = store.root / "environments" / f"environment-{index:03d}"
    output.mkdir(parents=True)
    shutil.copy2(source, output / "build.py")
    harness = Path(__file__).with_name("blender_scene_background.py")
    try:
        with (output / "build.log").open("w") as stream:
            process = subprocess.run([str(executor.blender), "--background", "--python", str(harness), "--",
                                      str(output / "build.py"), str(output)], stdout=stream, stderr=subprocess.STDOUT,
                                     timeout=executor.timeout_seconds, check=False)
        if process.returncode or not (output / "environment.usdc").is_file():
            raise RuntimeError("Visual environment export failed")
        # Ensure the export can be referenced without dropping root transforms or dependencies.
        from .scene import _check_stage
        _check_stage(output / "environment.usdc")
        export = json.loads((output / "export.json").read_text())
        return {"ok": True, "path": str(output / "environment.usdc"), "editable_source": str(output / "environment.blend"),
                "directory": str(output), "preview_images": export.get("preview_images", []),
                "format": "mesh", "visual_only": True, "source_sha256": sha256_file(output / "build.py")}
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": str(exc), "log_tail": (output / "build.log").read_text(errors="replace")[-2500:]}


def build_object(executor, store, workspace, worker, *, reference_paths, objective, max_calls, asset_id):
    from .video import prepare_reference_inputs
    from .video_review import retain_input_videos
    if not objective.strip() or not 1 <= max_calls <= 100 or not 0 <= len(reference_paths) <= 30:
        raise ValueError("Describe the participant and supply optional references within the worker budget")
    paths = [workspace_file(workspace, name) for name in reference_paths]
    index_path = store.root / "scene_objects.json"
    index = json.loads(index_path.read_text()) if index_path.is_file() else {}
    if asset_id is not None:
        if not re.fullmatch(r"asset-[0-9]{3}", asset_id) or asset_id not in index:
            raise ValueError("Use a returned asset-NNN ID to resume")
        child = RunStore(store.root / index[asset_id]["run"])
        record = json.loads((child.root / "run.json").read_text())
        if record["objective"] != objective:
            raise ValueError("Resume retains the original objective; request a new asset for a new objective")
    else:
        asset_id = f"asset-{len(index) + 1:03d}"
        input_dir = store.root / "scene-object-inputs" / asset_id
        references, inputs = prepare_reference_inputs(paths, input_dir, video_frames=12)
        from .capabilities import TASK_TOOLS as CODEX_TOOLS
        child = RunStore.create(store.root / "scene-object-runs", references=references, objective=objective,
                                model=json.loads((store.root / "run.json").read_text())["model"], max_turns=12,
                                tool_schemas=CODEX_TOOLS, asset_kind="auto")
        retain_input_videos(child.root, inputs)
        atomic_json(child.root / "input_manifest.json", {"inputs": inputs})
        parent = json.loads((store.root / "run.json").read_text())
        context = child.root / "context"
        context.mkdir()
        scene_references = []
        for reference_index, record in enumerate(parent.get("references", [])):
            source = store.root / record["path"]
            target = context / f"scene_reference_{reference_index:03d}{source.suffix.lower()}"
            shutil.copy2(source, target)
            scene_references.append(str(target.relative_to(child.root)))
        atomic_json(context / "scene_parent.json", {"original_user_request": parent["objective"],
            "participant_assignment": objective, "scene_references": scene_references,
            "reference_mode": "reference_guided" if references else "text_only_concept",
            "scope": "Apply the original user's constraints relevant to your assigned participant. The parent scene worker handles the broader environment and other participants. Do not replace an explicitly requested operation target with an inferred alternative."})
        index[asset_id] = {"run": str(child.root.relative_to(store.root)), "objective": objective}
        atomic_json(index_path, index)
    final_path = child.root / "final.json"
    if not final_path.is_file():
        retained = (child.root / "codex/thread.json").is_file()
        try:
            result = worker(executor, child, objective=objective, max_calls=max_calls, resume_existing=retained)
        except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
            return {"ok": False, "asset_id": asset_id, "run": str(child.root), "error": str(exc),
                    "next_action": "The retained object worker can be resumed with this asset_id."}
        if result["status"] != "accepted":
            return {"ok": False, "asset_id": asset_id, "run": str(child.root), "worker_status": result["status"]}
    return package_object_result(executor, child, asset_id)


def package_object_result(executor, child, asset_id):
    if hasattr(executor, "for_task"):
        executor = executor.for_task(child)
    final = json.loads((child.root / 'final.json').read_text())
    attempt = child.attempts_dir / final["attempt_id"]
    current = child.accept(final["attempt_id"], final["reason"], commit=False)
    simulation = current["report"].get("simulation", {})
    bundle = attempt / simulation["directory"] if simulation.get("directory") else None
    from .scene_native_asset import package_native_for_scene, representation
    kind = json.loads((child.root / "run.json").read_text())["asset_contract"]["kind"]
    if representation(current["report"], kind):
        root = child.root / "scene-assets"
        destination = root / f"native-{len(list(root.glob('native-*'))) + 1:03d}"
        bundle = package_native_for_scene(executor, attempt, destination)
    if current["report"].get("hybrid", {}).get("ok"):
        from .scene_newton import package_hybrid_for_scene
        root = child.root / "scene-assets"
        destination = root / f"hybrid-{len(list(root.glob('hybrid-*'))) + 1:03d}"
        bundle = package_hybrid_for_scene(attempt, destination)
    if current['report'].get('fluid',{}).get('ok'):
        from .scene_fluid import package_fluid_for_scene
        root=child.root/'scene-assets'
        destination=root/f"fluid-{len(list(root.glob('fluid-*')))+1:03d}"
        bundle=package_fluid_for_scene(executor,attempt,destination)
    if bundle is None or not (bundle / "asset_bundle.json").is_file():
        return {"ok": False, "asset_id": asset_id, "asset_kind": json.loads((child.root / "run.json").read_text())["asset_contract"]["kind"],
                "run": str(child.root), "status": "dedicated_runtime_binding_required",
                "next_action": "The accepted participant uses a dedicated fluid/rod/hybrid/fracture package. Preserve it and supply a scene adapter for that package; do not flatten it into a rigid mesh."}
    return {"ok": True, "asset_id": asset_id, "bundle": str(bundle), "run": str(child.root),
            "asset_kind": json.loads((bundle / "asset_bundle.json").read_text())["asset_kind"],
            "scene_backend_still_to_be_validated": True}
