"""Drive native reconstruction methods from a single current revision per target."""
from pathlib import Path
import shutil

from astra_station.common import digest, file_hash, read, write_new
from real2sim_harness.model_revision import apply_fit, check_tree, from_template, inside, snapshot, workspace_blender_source

METHODS = {
    "object.build": {"blender": "workspace.run_blender", "metric_blender": "session.object_blender_run",
                     "support_blender": "session.support_blender_run"},
    "object.fit": {"posed": "workspace.fit_multiview_geometry", "metric": "session.object_fit",
                   "support": "session.support_fit"},
    "object.test": {"metric_task": "session.object_task_check", "simulation": "workspace.run_simulation", "contact": "workspace.run_rigid_contact_behavior",
                    "cloth": "workspace.run_deformation", "deformable": "workspace.run_deformable_behavior",
                    "hybrid": "workspace.run_hybrid", "fluid": "workspace.run_fluid", "strand": "workspace.run_strands",
                    "solid": "workspace.run_solids", "fracture": "workspace.run_fracture"},
    "environment.build": {"blender": "workspace.build_scene_background", "gaussian": "workspace.run_gaussian_background",
                          "metric": "session.stage_s3"},
    "environment.assemble": {"scene": "workspace.assemble_scene"},
    "environment.test": {"simulation": "workspace.run_scene_simulation"},
}
DOMAINS = ("object", "environment")
REVISIONS = {f"{domain}.revise" for domain in DOMAINS}


def schemas():
    text = {"type": "string", "minLength": 1}
    target = {"type": "string", "pattern": "[A-Za-z0-9_-]{1,128}"}
    sha = {"type": "string", "pattern": "[0-9a-f]{64}"}
    rows = {}
    for op in (*sorted(REVISIONS), "object.from_template", "object.adopt_fit", *METHODS):
        fields = {"target": target}
        if op == "object.from_template":
            fields.update(parent={"anyOf": [sha, {"type": "null"}]}, recipe=text)
        elif op in REVISIONS:
            fields.update(parent={"anyOf": [sha, {"type": "null"}]}, entry=text,
                          files={"type": "array", "minItems": 1, "items": text})
        elif op == "object.adopt_fit":
            fields.update(revision=sha, fit_step=text)
        else:
            fields.update(revision=sha, method={"enum": list(METHODS[op])}, options={"type": "object"})
        required = list(fields)
        if op in {"object.build", "environment.build"}:
            fields["method"]["default"] = "blender"
            required.remove("method")
        if op in REVISIONS or op == "object.from_template":
            domain = op.split(".")[0]
            checks = [cap + "/" + method for cap in METHODS if cap in {domain + ".fit", domain + ".test"}
                      for method in METHODS[cap]]
            if domain == "environment":
                fields["bindings"] = {"type": "object", "additionalProperties": False, "properties": {
                    key: text for key in ("build/blender", "build/metric", "build/gaussian", "assemble/scene", "settings")}}
            fields["checks"] = {"type": "array", "items": {"enum": checks}}
        rows[op] = {"type": "object", "required": required, "properties": fields, "additionalProperties": False}
    return rows


def models(receipts):
    """Project the current heads from the workflow's retained, current receipt prefix."""
    heads = {}
    for pin in receipts:
        event = read(pin["path"]).get("model_event")
        if not event:
            continue
        key = event["module"] + "/" + event["target"]
        if "candidate" in event:
            heads[key] = {"candidate": event["candidate"], "uses": []}
        else:
            heads[key]["uses"].append({**event, "receipt": pin})
    return heads


def _local_model(candidate, source, native):
    from calibrated_scene_reconstruction.scene.session import Session
    from calibrated_scene_reconstruction.scene.objects.blender.fit import model_path
    from calibrated_scene_reconstruction.scene.support.blender import model_path as support_path
    from calibrated_scene_reconstruction.scene.background.blender_env import specification
    from calibrated_scene_reconstruction.scene.twin.gates import options

    session = Session(source)
    if native == "session.stage_s3":
        if not options(session):
            raise ValueError("A revision-bound room stage needs a native twin.json")
        if options(session)["background"] == "blender":
            entry = specification(session)["model_path"]
        else:
            entry = session.config_dir / "twin.json"
    elif native.startswith("session.support_"):
        if candidate["target"] != session.scene_graph["task"]["support"]:
            raise ValueError("Support target differs from the native scene")
        entry = support_path(session)
    else:
        entry = model_path(session, candidate["target"])
    entry = Path(entry).resolve(strict=True)
    root = entry
    for _ in Path(candidate["entry"]).parts:
        root = root.parent
    if inside(root, candidate["entry"]) != entry:
        raise ValueError("Native model entry differs from the candidate")
    check_tree(candidate, root)
    return root


def _active_metric_object(candidate, source, member):
    from calibrated_scene_reconstruction.scene.session import Session
    session = Session(source)
    graph = session.scene_graph
    name = candidate["target"]
    block = graph.get("dynamic_objects", {}).get(name, graph.get("furniture", {}).get(name, {}))
    fix = block.get("fix", {})
    if fix.get("model_sha256") != candidate["files"][member] or not fix.get("mesh"):
        raise ValueError("The exported metric object differs from the current model revision")
    mesh = (source.parents[1] / fix["mesh"]).resolve(strict=True)
    if not mesh.is_relative_to(source):
        raise ValueError("Exported model geometry must belong to the checked native session")
    expected = fix.get("mesh_sha256") or read(mesh.parent / "fit.json")["mesh_sha256"]
    if file_hash(mesh) != expected:
        raise ValueError("The exported metric mesh is stale")
    return fix


def _build_use(head, source, field):
    for use in reversed(head["uses"]):
        if use["source"] == str(source) and field in use:
            return use[field]
    raise ValueError("Build this revision in the selected native workspace before testing it")


def run(chain, step, args, folder, receipts, execute_native, workspace):
    op = step["operation"]
    module, action = op.split(".")
    target = args["target"]
    source = Path(step["source"]).resolve(strict=True)
    head = models(receipts).get(module + "/" + target)
    candidate = head["candidate"] if head else None
    revision = candidate["revision"] if candidate else None
    event = {"module": module, "target": target, "source": str(source)}
    if action in {"revise", "from_template"}:
        if args["parent"] != revision:
            raise ValueError("Revision parent must match the current candidate")
        checks = set(args.get("checks", []))
        if candidate:
            checks.update(candidate["checks"])
        if action == "from_template":
            candidate = from_template(source, folder, target=target, recipe=args["recipe"], parent=revision, checks=checks)
        else:
            candidate = snapshot(source, folder, module=module, target=target, entry=args["entry"],
                                 files=args["files"], parent=revision, checks=checks, bindings=args.get("bindings"))
        pins = {str(Path(candidate["root"]) / name): sha for name, sha in candidate["files"].items()}
        return candidate, pins, {**event, "candidate": candidate}
    if not candidate or args["revision"] != revision:
        raise ValueError("Select the current model revision")
    if action == "adopt_fit":
        fits = [use for use in head["uses"] if use["operation"] == "object.fit"
                and use["step_id"] == args["fit_step"] and use["source"] == str(source)]
        if not fits or not fits[-1].get("parameters"):
            raise ValueError("Select a bounded parameter fit of the current revision")
        candidate = apply_fit(candidate, fits[-1]["parameters"], folder)
        pins = {str(Path(candidate["root"]) / name): sha for name, sha in candidate["files"].items()}
        return candidate, pins, {**event, "candidate": candidate, "fit_step": args["fit_step"]}
    native = METHODS[op][args["method"]]
    member = candidate["bindings"].get(action + "/" + args["method"], candidate["entry"])
    candidate = {**candidate, "entry": member}
    opts = dict(args["options"])
    event.update(revision=revision, member=member, step_id=step["id"], operation=op, method=args["method"], native=native)
    check_tree(candidate, candidate["root"])
    input_pins = {}
    if native.startswith("session."):
        if (source / "qa/independent_freeze.json").exists():
            raise ValueError("Create a mutable derivative before running model tools on a frozen scene")
        root = _local_model(candidate, source, native)
        if native == "session.object_task_check":
            event["round"] = _active_metric_object(candidate, source, member)["round"]
        if native.startswith("session.object_") and native != "session.object_task_check":
            _bind(opts, "object", target)
        # Model parameters belong to the revision. A search produces a proposal for a new revision.
        if native.endswith("blender_run") and opts.get("params"):
            raise ValueError("Set explicit build parameters in the model revision")
        if native == "session.stage_s3" and opts.get("object_route", "agent_blender") != "agent_blender":
            raise ValueError("A background step cannot change the object route")
        event["native_inputs"] = {str(root / name): sha for name, sha in candidate["files"].items()}
    elif action == "test":
        field = "attempt_id" if module == "object" else "scene_id"
        identity = _build_use(head, source, field)
        event["tested_build"] = identity
        if args["method"] in {"contact", "deformable"}:
            request = inside(workspace(source), opts["request_path"])
            if read(request).get("source_attempt_id") != identity:
                raise ValueError("The native physics request must bind the active source_attempt_id")
        else:
            _bind(opts, field, identity)
    else:
        work = workspace(source)
        entry, input_pins = _workspace_inputs(work, source, folder, candidate, head, action, args["method"], opts)
    result, pins = execute_native(chain, {**step, "operation": native}, opts, folder)
    if result.get("ok") is False:
        return result, pins, event
    if not all(Path(p).is_file() and file_hash(p) == sha for p, sha in input_pins.items()):
        raise ValueError("Native inputs changed during model execution")
    if native.startswith("session."):
        check_tree(candidate, root)
    if native == "workspace.run_blender":
        attempt = result["attempt_id"]
        built = source / "attempts" / attempt
        if file_hash(built / "agent_scene.py") != input_pins[str(entry)]:
            raise ValueError("Native attempt used a different Blender program")
        event.update(attempt_id=attempt, artifact=str(built / "scene.blend"))
    elif native == "workspace.build_scene_background":
        if result["source_sha256"] != file_hash(entry):
            raise ValueError("Background used a different Blender program")
        event["artifact"] = result["path"]
    elif native == "workspace.assemble_scene":
        event.update(scene_id=result["scene_id"], artifact=str(Path(result["directory"]) / "scene_bundle.json"))
    elif native == "workspace.run_gaussian_background":
        # Native modes can yield a diagnostic only. Only a USD background can bind a delivery.
        path = (result.get("background") or {}).get("path")
        if path and Path(path).suffix.lower() in {".usd", ".usda", ".usdc", ".usdz"}:
            event["artifact"] = str((Path(result["directory"]) / path).resolve(strict=True))
    if "artifact" in event:
        pins[event["artifact"]] = file_hash(event["artifact"])
    if native in {"session.object_fit", "session.support_fit"}:
        event["round"] = result["round"]
    if native == "session.object_fit":
        fit = inside(source, result["round_dir"] + "/fit.json")
        pins[str(fit)] = file_hash(fit)
        event["parameters"] = read(fit)["params"]
    if native == "workspace.fit_multiview_geometry":
        if result["geometry_optimized"]:
            event["parameters"] = result["parameters"]
        report_dir = Path(result["directory"])
        for name in ("report.json", "parameters.json", result["mesh"]):
            p = report_dir / name
            pins[str(p)] = file_hash(p)
        event["check_passed"] = result.get("quality_targets_passed") is True
    elif action in {"test", "fit"}:
        event["check_passed"] = result.get("ok") is True and result.get("pass", True) is True
    if action == "test":
        evidence_dir = result.get("directory")
        if native == "workspace.run_scene_simulation":
            evidence_dir = source / "scenes" / identity / result["output"]
            runtime = read(evidence_dir / "runtime_report.json")["runtime"]
            event["runtime"] = {"name": runtime["name"], "version": runtime["version"]}
        elif native == "session.object_task_check":
            # A later native task run can replace its live report; keep this exact trial.
            native_report = source / "qa/task_check.json"
            copy = folder / "task_check.json"
            shutil.copyfile(native_report, copy)
            pins[str(copy)] = file_hash(copy)
        if evidence_dir:
            directory = Path(evidence_dir)
            if not directory.is_absolute():
                directory = source / "attempts" / identity / directory
            directory = directory.resolve(strict=True)
            if not directory.is_relative_to(source):
                raise ValueError("Test evidence is outside the native source")
            for path in directory.rglob("*"):
                if path.is_file() and path.resolve().is_relative_to(directory):
                    pins[str(path)] = file_hash(path)
    event["outputs"] = {p: sha for p, sha in pins.items() if Path(p).name != "request.json"}
    return result, {**input_pins, **pins}, event


def _workspace_inputs(work, source, folder, candidate, head, action, method, opts):
    input_pins = {}
    stage = work / ("model-" + folder.name)
    shutil.copytree(candidate["root"], stage)
    check_tree(candidate, stage)
    input_pins.update({str(stage / name): sha for name, sha in candidate["files"].items()})
    entry = stage / candidate["entry"]
    if method == "blender":
        code = workspace_blender_source(candidate)
        if code != entry.read_text(encoding="utf-8"):
            entry = entry.with_name("adapter-" + folder.name + ".py")
            entry.write_text(code, encoding="utf-8")
        if len(candidate["files"]) > 1:
            entry = _resource_loader(entry, stage, folder, blender=True)
        _bind(opts, "script_path", str(entry.relative_to(work)))
    elif method == "posed":
        if len(candidate["files"]) > 1 and entry.suffix == ".py":
            entry = _resource_loader(entry, stage, folder, blender=False)
        input_pins.update(_fit_request(work, source, stage, entry, opts))
    else:
        if action == "assemble":
            spec = read(entry)
            for layer in spec.get("backgrounds", []):
                if "from_model_step" in layer:
                    step_id = layer.pop("from_model_step")
                    hits = [u for u in head["uses"] if u["step_id"] == step_id
                            and u["operation"] == "environment.build" and u.get("artifact")]
                    if "path" in layer or not hits:
                        raise ValueError("Select a built background layer from the current revision")
                    layer["path"] = hits[-1]["artifact"]
            # Preserve the manifest directory for its other relative paths.
            derived = entry.with_name("composed-" + folder.name + ".json")
            write_new(derived, spec)
            entry = derived
        field = "manifest_path" if action == "assemble" else "request_path"
        _bind(opts, field, str(entry.relative_to(work)))
    input_pins[str(entry)] = file_hash(entry)
    return entry, input_pins


def _bind(opts, key, value):
    if key in opts and opts[key] != value:
        raise ValueError("The current revision owns " + key)
    opts[key] = value


def _resource_loader(entry, stage, folder, *, blender):
    # Native workers copy a single script to an isolated attempt before execution.
    # Load the immutable staged tree in that same worker so relative asset paths survive.
    path = stage.with_name("loader-" + folder.name + ".py")
    code = ("import runpy, sys\n" + f"sys.path.insert(0, {str(entry.parent)!r})\n"
            + f"_model = runpy.run_path({str(entry)!r})\n")
    if blender:
        code += "def build_scene():\n    return _model['build_scene']()\n"
    else:
        code += "globals().update({k: v for k, v in _model.items() if not k.startswith('__')})\n"
    path.write_text(code, encoding="utf-8")
    return path


def _fit_request(work, source, stage, entry, opts):
    """Keep observation coordinates relative to the original native request."""
    recipe = inside(work, opts["request_path"])
    request = read(recipe)
    pins = {str(recipe): file_hash(recipe)}
    if "model_script" in request or "mesh" in request:
        raise ValueError("The active revision supplies the fit geometry")
    for row in request.get("views", []):
        for key in ("image", "mask", "depth", "ignore"):
            if key in row:
                path = (recipe.parent / row[key]).resolve(strict=True)
                # Native observation tools allow retained native evidence as well as the workspace.
                if not path.is_relative_to(source):
                    raise ValueError("Observation input is outside the native task")
                row[key] = str(path)
                # The native worker verifies file scope and preserves its own snapshot.
                pins[str(path)] = file_hash(path)
    if entry.suffix == ".npz":
        request.update(mesh=str(entry), evaluate_only=True)
    else:
        request["model_script"] = str(entry)
        request.setdefault("model_format", "blender")
    path = stage.with_suffix(".json")
    write_new(path, request)
    opts["request_path"] = str(path.relative_to(work))
    return pins


def bind_delivery(heads, delivery, source):
    """Each current candidate must reach the native acceptance owner's actual artifact."""
    if not heads:
        return delivery
    workspace = delivery["route"] == "task_workspace"
    accepted = delivery["acceptance"]
    if workspace and delivery["level"] == "scene":
        scene = Path(delivery["scene"]["path"]).parent
        bundle = read(scene / "scene_bundle.json")
        final = read(accepted["path"])
        if not final.get("runtime_review") or not final.get("evidence_files"):
            raise ValueError("Run accept_scene again to bind this model delivery to the reviewed runtime trial")
        trial = str(inside(scene, final["runtime_review"]))
        if final["evidence_files"].get(trial) != file_hash(trial):
            raise ValueError("The accepted runtime review is missing or changed; run accept_scene again")
    if not workspace:
        frozen = {str((source if row["base"] == "session" else Path(delivery["source"]["config"])) / row["path"]):
                  row["sha256"] for row in read(accepted["path"])["files"] if row["base"] in {"session", "config"}}
    revisions = {}
    for key, head in heads.items():
        candidate = head["candidate"]
        evidence = {}
        if workspace and delivery["level"] == "scene" and candidate["module"] == "environment":
            layers = {u["step_id"]: u for u in head["uses"] if u["operation"] == "environment.build" and u.get("artifact")}
            for use in layers.values():
                path = use["artifact"]
                if bundle["source_dependencies"].get(path) != use["outputs"][path]:
                    raise ValueError("Accepted scene omitted a built background layer for " + key)
        for check in candidate["checks"]:
            op, method = check.split("/")
            matches = [u for u in head["uses"] if (u["operation"], u["method"]) == (op, method)]
            valid = bool(matches and matches[-1].get("check_passed"))
            if valid and "tested_build" in matches[-1]:
                field = "attempt_id" if candidate["module"] == "object" else "scene_id"
                tested = matches[-1]
                identity = tested["tested_build"]
                valid = identity == _build_use(head, tested["source"], field) and workspace
                if valid and (delivery["level"] == "object" or field == "scene_id"):
                    accepted_id = accepted["attempt"] if delivery["level"] == "object" else accepted["scene_id"]
                    valid = tested["source"] == str(source) and identity == accepted_id
                    if valid and field == "scene_id":
                        valid = tested["outputs"].get(trial) == final["evidence_files"][trial]
                elif valid:
                    build = next(u for u in reversed(head["uses"]) if u.get(field) == identity
                                 and u["source"] == tested["source"])
                    name = "assets/" + candidate["target"] + "/editable/scene.blend"
                    valid = bundle["files"].get(name) == build["outputs"][build["artifact"]]
            if not valid:
                raise ValueError("Current revision still needs " + key + " " + check)
            checked = matches[-1]
            evidence[check] = checked["receipt"]
        latest = {(u["source"], u["native"]): u for u in head["uses"]
                  if u.get("artifact") or u.get("native_inputs")}
        materialized = list(latest.values())
        if not workspace and candidate["module"] == "object":
            metric = [u for u in materialized if u["source"] == str(source) and u.get("native_inputs")]
            if metric:
                fix = _active_metric_object(candidate, source, metric[-1]["member"])
                if any(u["round"] != fix["round"] for u in metric if "round" in u):
                    raise ValueError("The delivered metric round differs from the selected round")
        for use in reversed(materialized):
            artifact = use.get("artifact")
            if not workspace:
                if (Path(use["source"]) == source and use.get("native_inputs")
                        and all(frozen.get(p) == h for p, h in use["native_inputs"].items())):
                    _local_model({**candidate, "entry": use["member"]}, source, use["native"])
                    break
            elif delivery["level"] == "object":
                if (artifact and Path(use["source"]) == source and use.get("attempt_id") == accepted["attempt"]
                        and any(p["path"] == artifact for p in delivery["assets"].values())):
                    break
            elif artifact:
                sha = use["outputs"][artifact]
                if use.get("scene_id") == accepted["scene_id"] and Path(artifact) == scene / "scene_bundle.json":
                    break
                if candidate["module"] == "object":
                    name = "assets/" + candidate["target"] + "/editable/scene.blend"
                    if bundle["files"].get(name) == sha:
                        break
                elif bundle["source_dependencies"].get(artifact) == sha:
                    break
        else:
            raise ValueError("Native delivery lacks the active model revision for " + key)
        revisions[key] = {k: candidate[k] for k in ("revision", "parent", "entry", "checks", "bindings")}
        revisions[key]["evidence"] = evidence
        revisions[key]["files"] = {name: {"path": str(Path(candidate["root"]) / name), "sha256": sha}
                                    for name, sha in candidate["files"].items()}
    return {**delivery, "schema": "astra_twin_delivery_v2",
            "model_revisions": revisions, "model_revisions_sha256": digest(revisions)}
