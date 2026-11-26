"""Append-only source revalidation of an unchanged, historically frozen rigid mesh.

This is a migration of evidence contracts, not a fitting or geometry-repair
route. It copies exact candidate bytes, evaluates current source observations
at the exact inherited pose, and leaves native/semantic/structural acceptance
to their ordinary gates. Historical fit and input records are never edited.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path, PurePosixPath
import shutil

from . import store
from .contract import contract, write_json
from calibrated_scene_reconstruction.scene.objects.object_understanding import declaration, validate_build


RECORD = "fixed_candidate_revalidation.json"
REQUIRED = ("fit.json", "model.py", "mesh.npz")


def model_path(session, obj):
    from .fit import model_path as current_model_path
    return current_model_path(session, obj)


def _evaluate_source(session, obj, out, pose, opening, number):
    from . import evidence
    metrics = evidence.run(session, obj, out / "mesh.npz", deepcopy(pose), out, number)
    metrics["opening_sheet"] = evidence.opening_sheet(session, obj, pose, opening, out)
    return metrics


def _relative_file(root, relative):
    relative = PurePosixPath(str(relative).replace("\\", "/"))
    path = (Path(root) / str(relative)).resolve()
    if relative.is_absolute() or not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise ValueError("Expected an existing file inside its evidence root: " + str(relative))
    return path


def _frozen_files(manifest):
    record = json.loads(Path(manifest).read_text())
    state = record.get("status")
    current = record.get("version") == 1 and isinstance(state, dict) and state.get("passed") is True
    legacy = ("version" not in record and state == "accepted_image_and_object_model_gates"
              and isinstance(record.get("code"), str) and len(record["code"]) == 40
              and isinstance(record.get("frozen_at"), str) and bool(record["frozen_at"]))
    if not (current or legacy):
        raise ValueError("A historically accepted freeze is required; this is not an approval bypass")
    files = {}
    for row in record["files"]:
        relative = PurePosixPath(row["path"].replace("\\", "/"))
        if row["base"] not in {"config", "session"} or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid inherited freeze entry")
        files[row["base"], str(relative)] = row["sha256"]
    return files


def _bound(path, base, relative, frozen):
    expected = frozen.get((base, str(PurePosixPath(str(relative).replace("\\", "/")))))
    if not expected or store.digest(path) != expected:
        raise ValueError("Required inherited file is missing from the freeze or changed: " + str(relative))
    return expected


def _graph_signature(session, graph):
    """Allow metadata/round relocation only; preserve actual geometry and physics."""
    graph = deepcopy(graph)
    if "structure" in graph:
        from calibrated_scene_reconstruction.scene.graph import close_room
        graph["structure"] = close_room(graph["structure"])
    repository = session.dir.resolve().parents[1]
    for group in ("dynamic_objects", "furniture"):
        for spec in graph.get(group, {}).values():
            fit = spec.get("fit") or {}
            for key in ("geometry_review", "geometry_features"):
                fit.pop(key, None)
            fix = spec.get("fix") or {}
            if fix.get("mesh"):
                fix["mesh"] = store.digest(_relative_file(repository, fix["mesh"]))
                fix.pop("round", None)
    return graph


def _physical_build(session, fix, old_build):
    """Derive ownership from the actual NPZ loader, retaining old visual metadata."""
    from calibrated_scene_reconstruction.assets import metric_mesh
    mesh = _relative_file(session.dir.resolve().parents[1], fix["mesh"])
    actual = metric_mesh.asset({**fix, "mesh": str(mesh)})
    if (not actual["parts"] or any(p["kind"] != "rigid" for p in actual["parts"])
            or actual["joints"] or actual["surface_bindings"]
            or old_build.get("joints") or old_build.get("surface_bindings")):
        raise ValueError("Fixed-candidate migration currently supports only actual rigid assemblies without joints")
    result = deepcopy(old_build)
    result["legacy_visual_parts"] = deepcopy(old_build.get("parts"))
    result["parts"] = [{"name": p["name"], "kind": p["kind"], "vertex_count": len(p["V"]),
                        "triangle_count": len(p["F"])} for p in actual["parts"]]
    result["links"] = deepcopy(actual["links"])
    result["joints"] = deepcopy(actual["joints"])
    result["surface_bindings"] = deepcopy(actual["surface_bindings"])
    result["physical_ownership_source"] = {
        "method": "Actual metric_mesh.asset(fix) NPZ interpretation; no inferred dynamic-owner fallback",
        "mesh_sha256": store.digest(mesh), "loader_sha256": store.digest(metric_mesh.__file__),
        "implicit_single_rigid_owner": not bool(actual["links"]),
        "historical_visual_metadata_preserved": True}
    return result


def _parent(session, obj, freeze, graph_path):
    frozen = _frozen_files(freeze)
    _bound(graph_path, "config", "scene_graph.json", frozen)
    graph = json.loads(Path(graph_path).read_text())
    if _graph_signature(session, graph) != _graph_signature(session, session.scene_graph):
        raise ValueError("Source geometry, pose, physics or observation policy changed; use a genuine new fit")
    # Confirm every active parent's mesh against the historical manifest. Hash
    # comparison of two current paths alone cannot establish historical identity.
    for group in ("dynamic_objects", "furniture"):
        for spec in graph.get(group, {}).values():
            fix = spec.get("fix") or {}
            if not fix.get("mesh"):
                continue
            path = _relative_file(session.dir.resolve().parents[1], fix["mesh"])
            if not path.is_relative_to(session.dir.resolve()):
                raise ValueError("Inherited active geometry must belong to this session")
            _bound(path, "session", path.relative_to(session.dir.resolve()), frozen)
    fix = graph["dynamic_objects"][obj]["fix"]
    parent = session.dir / "objects" / obj / f"round_{fix['round']}"
    hashes = {str((parent / name).relative_to(session.dir)): _bound(
        parent / name, "session", (parent / name).relative_to(session.dir), frozen) for name in REQUIRED}
    old = json.loads((parent / "fit.json").read_text())
    # Some historical freezes recorded fit.json but not input.json. The frozen
    # fit's explicit input hash is a genuine transitive binding, not a new hash
    # invented by the migration. Missing hashes still block this route.
    input_relative = str((parent / "input.json").relative_to(session.dir))
    expected_input = old.get("input_sha256")
    if not expected_input or store.digest(parent / "input.json") != expected_input:
        raise ValueError("Frozen fit does not bind the inherited input manifest")
    hashes[input_relative] = expected_input
    binding_basis = {path: {"kind": "direct_inherited_freeze"} for path in hashes}
    binding_basis[input_relative] = {"kind": "transitive_frozen_fit", "via": str((parent / "fit.json").relative_to(session.dir)), "field": "input_sha256"}
    if (old.get("input_sha256") != store.digest(parent / "input.json")
            or old.get("model_sha256") != store.digest(parent / "model.py")
            or old.get("mesh_sha256") != store.digest(parent / "mesh.npz")
            or contract(model_path(session, obj))["sha256"] != old["model_sha256"]):
        raise ValueError("Historical candidate/input/model binding is inconsistent")
    build = _physical_build(session, fix, old["build"])
    validate_build(session, obj, build)
    if fix.get("params") != build.get("params") or fix.get("opening") != build.get("opening"):
        raise ValueError("Inherited export differs from its build report")
    for key in ("xy", "yaw_deg", "support_z", "local_top", "down_offset", "support_rotation"):
        if fix.get(key) != old["pose"].get(key):
            raise ValueError("Inherited exported pose differs from its measured source fit")
    return old, parent, fix, hashes, frozen, binding_basis, build


def _look_files(session, look, frozen):
    files = {}
    for key in ("mesh", "albedo", "photometric", "material_finishes"):
        value = (look or {}).get(key)
        if value:
            path = _relative_file(session.dir, value)
            files[str(value)] = _bound(path, "session", value, frozen)
    if look and not files:
        raise ValueError("A look needs frozen file dependencies before it can be retained")
    return files


def run(session, obj, *, inherited_freeze, inherited_scene_graph):
    if (session.dir / "qa/independent_freeze.json").exists():
        raise ValueError("Create an unfrozen derivative before revalidating a fixed candidate")
    inherited_freeze, inherited_scene_graph = Path(inherited_freeze).resolve(), Path(inherited_scene_graph).resolve()
    if not inherited_freeze.is_relative_to(session.dir.resolve()):
        raise ValueError("Archive the inherited freeze inside the derivative's provenance first")
    understanding = declaration(session, obj)
    old, parent, fix, parent_hashes, frozen, binding_basis, build = _parent(session, obj, inherited_freeze, inherited_scene_graph)
    look_files = _look_files(session, fix.get("look"), frozen)
    graph_signature = _graph_signature(session, session.scene_graph)
    out, number = store.start(session, obj, model_path(session, obj), budget=1)
    try:
        for name in ("mesh.npz", "object.glb", "scene.blend", "preview_0.png", "preview_1.png", "source_solids.npz"):
            if (parent / name).is_file():
                shutil.copy2(parent / name, out / name)
        if any(store.digest(out / name) != old[key] for name, key in (
                ("model.py", "model_sha256"), ("mesh.npz", "mesh_sha256"))):
            raise ValueError("Fixed candidate copy changed model or mesh bytes")
        shutil.copy2(inherited_scene_graph, out / "inherited_scene_graph.json")
        # Actual current observation evaluation; never reuse historical scores.
        metrics = _evaluate_source(session, obj, out, old["pose"], old["build"].get("opening"), number)
        if (graph_signature != _graph_signature(session, session.scene_graph)
                or understanding != declaration(session, obj)
                or any(store.digest(session.dir / path) != sha for path, sha in parent_hashes.items())):
            raise ValueError("Inputs changed during fixed-candidate source evaluation")
        record = {"version": 1, "object": obj, "parent_round": old["round"],
                  "inherited_freeze": str(inherited_freeze.relative_to(session.dir.resolve())),
                  "inherited_freeze_sha256": store.digest(inherited_freeze), "parent_files": parent_hashes,
                  "inherited_acceptance_scope": json.loads(inherited_freeze.read_text())["status"],
                  "parent_binding_basis": binding_basis,
                  "inherited_scene_graph_sha256": store.digest(out / "inherited_scene_graph.json"),
                  "parent_export": fix, "look_files": look_files,
                  "current_input_sha256": store.digest(out / "input.json"),
                  "current_evidence_sha256": store.digest(out / "evidence.json"),
                  "candidate_sha256": {"model.py": old["model_sha256"], "mesh.npz": old["mesh_sha256"]},
                  "producer_sha256": store.digest(__file__), "geometry_changed": False,
                  "pose_changed": False, "native_approval": False,
                  "scope": "Exact inherited rigid candidate, newly evaluated against current source observations; "
                           "new native, semantic and structural reviews remain mandatory."}
        write_json(out / RECORD, record)
        result = {"round": number, "model_sha256": old["model_sha256"], "mesh_sha256": old["mesh_sha256"],
                  "input_sha256": store.digest(out / "input.json"), "mesh": str((out / "mesh.npz").relative_to(session.dir)),
                  "params": deepcopy(old["params"]), "build": build, "pose": deepcopy(old["pose"]),
                  "metrics": metrics, "ok": bool(metrics["ok"]), "converged": True, "evaluations": 1,
                  "optimized_parameters": [], "pose_parameters": [], "trace": [],
                  "optimizer_message": "No optimization; exact historical candidate source re-evaluation",
                  "fixed_candidate_revalidation": {"path": RECORD, "sha256": store.digest(out / RECORD)}}
        if metrics.get("source_pose_controls"):
            from .source_acceptance import decision
            result["source_acceptance"] = decision(session, obj, out / "mesh.npz", result)
            result["effective_ok"] = result["source_acceptance"]["pass"]
        write_json(out / "fit.json", result)
        write_json(out / "params.json", result["params"])
        verify(session, obj, out, result)
        store.event(session, obj, "fixed_candidate_revalidated", round=number, parent_round=old["round"], ok=result["ok"])
        return result, out
    except Exception as exc:
        write_json(out / "error.json", {"ok": False, "error": str(exc), "scope": "Failed new source revalidation; historical round retained"})
        store.event(session, obj, "failed", round=number, error=str(exc))
        raise


def verify(session, obj, out, result):
    pointer = result.get("fixed_candidate_revalidation")
    if pointer is None:
        return None
    if pointer.get("path") != RECORD or store.digest(out / RECORD) != pointer.get("sha256"):
        raise ValueError("Fixed-candidate lineage changed")
    record = json.loads((out / RECORD).read_text())
    freeze = _relative_file(session.dir, record["inherited_freeze"])
    if store.digest(freeze) != record["inherited_freeze_sha256"] or store.digest(__file__) != record["producer_sha256"]:
        raise ValueError("Inherited freeze or revalidation producer changed")
    old, parent, fix, parent_hashes, frozen, binding_basis, build = _parent(session, obj, freeze, out / "inherited_scene_graph.json")
    if record.get("object") != obj or record["parent_files"] != parent_hashes or record["parent_export"] != fix:
        raise ValueError("Fixed candidate parent changed")
    if record.get("parent_binding_basis") != binding_basis:
        raise ValueError("Fixed candidate provenance binding changed")
    if record["look_files"] != _look_files(session, fix.get("look"), frozen):
        raise ValueError("Inherited appearance files changed")
    if (store.digest(out / "input.json") != record["current_input_sha256"]
            or store.digest(out / "evidence.json") != record["current_evidence_sha256"]
            or store.digest(out / "inherited_scene_graph.json") != record["inherited_scene_graph_sha256"]):
        raise ValueError("Revalidated input/source observations changed")
    measured = json.loads((out / "evidence.json").read_text())
    if any(result.get("metrics", {}).get(key) != value for key, value in measured.items()):
        raise ValueError("Revalidation fit scores differ from the fresh observation report")
    if (result["pose"] != old["pose"] or result["build"] != build or result["params"] != old["params"]
            or result["ok"] is not bool(measured["ok"])):
        raise ValueError("Fixed candidate pose/build/score changed")
    for name, expected in record["candidate_sha256"].items():
        if store.digest(out / name) != expected:
            raise ValueError("Fixed candidate bytes changed")
    return record


def inherited_look(session, obj, out, result, previous, replacement):
    """Retain appearance only after verifying exact frozen candidate lineage."""
    record = verify(session, obj, out, result)
    if not record or previous.get("look") != record["parent_export"].get("look"):
        return None
    for key in ("model_sha256", "params", "xy", "yaw_deg", "support_z", "support_rotation", "opening", "sdf"):
        if previous.get(key) != replacement.get(key):
            return None
    return deepcopy(previous.get("look"))


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session"); parser.add_argument("object")
    parser.add_argument("--inherited-freeze", required=True)
    parser.add_argument("--inherited-scene-graph", required=True)
    args = parser.parse_args()
    result, out = run(Session(args.session), args.object, inherited_freeze=args.inherited_freeze,
                      inherited_scene_graph=args.inherited_scene_graph)
    print(json.dumps({"round": str(out), "source_pass": result["ok"], "native_approval": False,
                      "mesh_sha256": result["mesh_sha256"]}, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
