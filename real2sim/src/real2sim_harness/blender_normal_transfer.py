"""Detect source normal detail silently dropped by Blender's USD exporter.

This checks connected normal-detail transfer, not shader or pixel equivalence.
It does not bake textures or certify that their scale matches a photograph.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def export_normal_transfer(usd_path, objects):
    from pxr import Usd, UsdShade

    path = Path(usd_path)
    stage = Usd.Stage.Open(str(path))
    by_name = {}
    for prim in stage.Traverse():
        if prim.IsA(UsdShade.Material):
            name = prim.GetAttribute("userProperties:blender:data_name").Get()
            if name:
                by_name.setdefault(name, []).append(prim)
    materials = {m.name: m for obj in objects if obj.type == "MESH" for m in obj.data.materials if m}
    rows = []
    failures = []
    for name, material in sorted(materials.items()):
        if not material.use_nodes:
            continue
        outputs = [n for n in material.node_tree.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output]
        pending = [link.from_node for n in outputs for link in n.inputs["Surface"].links]
        seen, normals = set(), []
        while pending:
            node = pending.pop()
            if node in seen:
                continue
            seen.add(node)
            if node.type == "BSDF_PRINCIPLED" and node.inputs["Normal"].is_linked:
                normals.append(node.name)
            # Group interfaces require a graph adapter rather than a guessed check.
            if node.type == "GROUP" and node.node_tree:
                pending.extend(n for n in node.node_tree.nodes if n.type == "GROUP_OUTPUT" and n.is_active_output)
            pending.extend(link.from_node for socket in node.inputs for link in socket.links)
        if not normals:
            continue
        matches = by_name.get(name, [])
        row = {"source_material": name, "source_normal_nodes": normals,
               "usd_material": str(matches[0].GetPath()) if len(matches) == 1 else ""}
        rows.append(row)
        if len(matches) != 1:
            failures.append("normal_material_identity_missing_or_ambiguous:" + name)
    report = {"schema_version": 1, "materials": rows, "failures": failures,
              "scope": "Connected Principled normal-detail paths and referenced texture files only; "
                       "not shader, UV, physical-scale or appearance equivalence.", "render_verified": False}
    receipt = path.with_suffix(".normals.json")
    receipt.write_text(json.dumps(report, indent=2) + "\n")
    check = audit_normal_transfer(path, receipt)
    report.update(ok=check["ok"], failures=check["failures"], texture_files=check["texture_files"])
    receipt.write_text(json.dumps(report, indent=2) + "\n")
    if not report["ok"]:
        raise ValueError("Source normal detail was not preserved in USD: " + ", ".join(report["failures"]) +
                         ". Bake the source bump/normal to a UV image, connect it through a Normal Map node, "
                         "and re-export; inspect the saved receipt and target-render the repaired material.")
    return report


def audit_normal_transfer(usd_path, receipt_path):
    from pxr import Usd, UsdShade

    path = Path(usd_path)
    report = json.loads(Path(receipt_path).read_text())
    stage = Usd.Stage.Open(str(path))
    failures = list(report.get("failures", []))
    texture_files = {}
    for row in report["materials"]:
        name = row["source_material"]
        material = UsdShade.Material.Get(stage, row["usd_material"]) if row["usd_material"] else None
        shader = material.ComputeSurfaceSource()[0] if material else None
        normal = shader.GetInput("normal") if shader else None
        if not normal or not normal.HasConnectedSource():
            failures.append("normal_connection_missing:" + name)
            continue
        pending = [normal.GetConnectedSource()[0].GetPrim()]
        seen, textures = set(), []
        while pending:
            prim = pending.pop()
            if prim.GetPath() in seen:
                continue
            seen.add(prim.GetPath())
            node = UsdShade.Shader(prim)
            if node.GetIdAttr().Get() == "UsdUVTexture":
                asset = node.GetInput("file").Get()
                file = Path(asset.resolvedPath) if asset and asset.resolvedPath else None
                if file is None or not file.is_file() or not file.resolve().is_relative_to(path.parent.resolve()):
                    failures.append("normal_texture_missing_or_external:" + name)
                else:
                    relative = str(file.resolve().relative_to(path.parent.resolve()))
                    digest = hashlib.sha256(file.read_bytes()).hexdigest()
                    expected = report.get("texture_files", {}).get(relative)
                    if expected is not None and digest != expected:
                        failures.append("normal_texture_changed:" + relative)
                    texture_files[relative] = digest
                    textures.append(relative)
            for socket in node.GetInputs():
                source = socket.GetConnectedSource()
                if source:
                    pending.append(source[0].GetPrim())
        if not textures:
            failures.append("normal_texture_path_missing:" + name)
    if "texture_files" in report and set(texture_files) != set(report["texture_files"]):
        failures.append("normal_texture_inventory_changed")
    return {"ok": not failures, "failures": sorted(set(failures)),
            "checked_materials": len(report["materials"]), "texture_files": texture_files,
            "render_verified": False}
