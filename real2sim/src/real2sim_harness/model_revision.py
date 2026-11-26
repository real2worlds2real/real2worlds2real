"""Immutable editable inputs shared by object and environment reconstruction tools."""
from __future__ import annotations

import ast
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from .artifacts import sha256_file


def inside(root, name):
    root = Path(root).resolve(strict=True)
    path = Path(name)
    if path.is_absolute() or path.as_posix() != name or ".." in path.parts:
        raise ValueError("Candidate members must use normalized relative file names")
    source = (root / path).resolve(strict=True)
    if not source.is_relative_to(root) or not source.is_file():
        raise ValueError("Candidate member outside model root or missing: " + name)
    return source


def snapshot(source, folder, *, module, target, entry, files, parent, checks, bindings=None):
    """Keep the entire declared model tree, including relative resource paths."""
    if len(files) != len(set(files)) or entry not in files:
        raise ValueError("List each candidate file once, including its entry point")
    if any(name not in files for name in (bindings or {}).values()):
        raise ValueError("Each environment binding must identify a member of this revision")
    root = Path(folder) / "model"
    root.mkdir()
    hashes = {}
    for name in sorted(files):
        path = inside(source, name)
        dest = root / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        hashes[name] = sha256_file(path)
        shutil.copyfile(path, dest)
        if sha256_file(dest) != hashes[name]:
            raise ValueError("Model changed during snapshot: " + name)
    body = dict(module=module, target=target, entry=entry, files=hashes, parent=parent, checks=sorted(checks), bindings=bindings or {})
    identity = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return {**body, "revision": identity, "root": str(root)}


def check_tree(candidate, root):
    for name, expected in candidate["files"].items():
        if sha256_file(inside(root, name)) != expected:
            raise ValueError("Native input differs from the active model revision: " + name)


def workspace_blender_source(candidate):
    """Use one authored program for both Blender workers. No host execution of asset code."""
    code = inside(candidate["root"], candidate["entry"]).read_text(encoding="utf-8")
    defs = {node.name for node in ast.parse(code).body if isinstance(node, ast.FunctionDef)}
    if "build_scene" in defs:
        return code
    if candidate["module"] == "object" and "build" in defs:
        from calibrated_scene_reconstruction.scene.objects.blender.contract import contract, parameters
        spec = contract(inside(candidate["root"], candidate["entry"]))
        links = spec.get("links", {})
        if (len(links) > 1 or any(link["kind"] != "rigid" for link in links.values())
                or spec.get("joints") or spec.get("joints_dynamic") or spec.get("surface_bindings_dynamic")):
            raise ValueError("Supply build_scene() binding the declared mechanisms and material to native metadata")
        if candidate["entry"] == "model.py" and "template.json" in candidate["files"]:
            from calibrated_scene_reconstruction.assets.blender_kit import is_rigid_template
            from calibrated_scene_reconstruction.assets.kits import FIT
            # Older immutable snapshots contained only this exact metric program.
            # Edited and external models must declare their own Blender contract.
            for kit in FIT:
                if is_rigid_template(kit) and code == _kit_source(kit, spec["params"], spec["fixed"]):
                    return code + _KIT_WORKSPACE_ENTRY
        return code + "\n\ndef build_scene():\n    return build(" + repr(parameters(spec)) + ")\n"
    if candidate["module"] == "environment" and "build" in defs:
        config = json.loads(inside(candidate["root"], candidate["bindings"].get("settings", "environment.json")).read_text())
        if not isinstance(config.get("context"), dict) or not isinstance(config.get("parameters"), dict):
            raise ValueError("Supply the room builder's parameters and observed context in the settings member")
        return code + "\n\ndef build_scene():\n    return build(" + repr(config["parameters"]) + ", " + repr(config["context"]) + ")\n"
    raise ValueError("Supply build_scene() for the workspace worker or an object build(params) contract")


def apply_fit(candidate, values, folder):
    """Replace only the declared initial parameter values. Keep bounds, frames and material data."""
    from calibrated_scene_reconstruction.scene.objects.blender.contract import contract, parameters

    original = inside(candidate["root"], candidate["entry"])
    spec = contract(original)
    parameters(spec, values)  # validates names and bounds without importing asset code
    new = {k: {**row, "init": float(values.get(k, row["init"]))} for k, row in spec["params"].items()}
    if new == spec["params"]:
        raise ValueError("The fit did not change any model parameter")
    text = original.read_bytes()
    nodes = [n.value for n in ast.parse(text).body if isinstance(n, (ast.Assign, ast.AnnAssign))
             and any(isinstance(t, ast.Name) and t.id == "PARAMS"
                     for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))]
    if len(nodes) != 1:
        raise ValueError("One literal PARAMS declaration is required")
    node = nodes[0]
    lines = text.splitlines(keepends=True)
    start = sum(map(len, lines[:node.lineno - 1])) + node.col_offset
    end = sum(map(len, lines[:node.end_lineno - 1])) + node.end_col_offset
    with tempfile.TemporaryDirectory(dir=folder) as temp:
        tree = Path(temp) / "source"
        shutil.copytree(candidate["root"], tree)
        entry = tree / candidate["entry"]
        entry.write_bytes(text[:start] + repr(new).encode() + text[end:])
        return snapshot(tree, folder, module="object", target=candidate["target"],
                        entry=candidate["entry"], files=list(candidate["files"]), parent=candidate["revision"],
                        checks=candidate["checks"])


def _kit_source(kind, params, fixed):
    # Keep this native source stable so previously generated revisions can be recognized.
    return (
        "from calibrated_scene_reconstruction.assets.blender_kit import build as build_kit\n"
        "from calibrated_scene_reconstruction.assets.kits import opening\n\n"
        f"KIT = {kind!r}\nPARAMS = {params!r}\nFIXED = {fixed!r}\n\n"
        "def build(p):\n    build_kit(KIT, p)\n\n"
        "def OPENING(p):\n    return opening(KIT, p)\n"
    )


_KIT_WORKSPACE_ENTRY = (
    "\n\ndef build_scene():\n"
    "    from calibrated_scene_reconstruction.assets.blender_kit import build_workspace\n"
    "    p = {name: row['init'] for name, row in PARAMS.items()}\n"
    "    build_workspace(KIT, build, {**FIXED, **p})\n"
)


def from_template(source, folder, *, target, recipe, parent, checks):
    """Translate an explicitly selected native kit to an editable metric Blender source."""
    from calibrated_scene_reconstruction.assets.blender_kit import is_rigid_template
    from calibrated_scene_reconstruction.assets.kits import FIT
    from calibrated_scene_reconstruction.scene.objects.blender.contract import contract

    text = inside(source, recipe).read_bytes()
    spec = json.loads(text)
    if set(spec) - {"kit", "parameters", "fixed"} or spec.get("kit") not in FIT:
        raise ValueError("Select an installed fitted kit and its source-estimated parameters")
    kind = spec["kit"]
    if not is_rigid_template(kind):
        raise ValueError("Supply custom kits through object.revise with an explicit build_scene() asset contract")
    initial = spec["parameters"]
    if set(initial) != set(FIT[kind]):
        raise ValueError("Provide all fitted kit parameters: " + ", ".join(FIT[kind]))
    params = {name: {"init": initial[name], "lo": lo, "hi": hi} for name, (lo, hi) in FIT[kind].items()}
    with tempfile.TemporaryDirectory(dir=folder) as temp:
        root = Path(temp)
        entry = root / "model.py"
        entry.write_text(_kit_source(kind, params, spec.get("fixed", {})) + _KIT_WORKSPACE_ENTRY, encoding="utf-8")
        contract(entry)
        (root / "template.json").write_bytes(text)
        return snapshot(root, folder, module="object", target=target, entry="model.py",
                        files=["model.py", "template.json"], parent=parent, checks=checks)
