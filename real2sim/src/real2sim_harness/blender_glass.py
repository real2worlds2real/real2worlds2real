"""Preserve a supported Blender dielectric as native RTX glass, with an audit.

The universal Preview Surface remains available to other USD consumers. Its
opacity is never used as a substitute for transmission in the MDL context.
Only constant, directly connected, fully transmissive Principled materials are
supported. Unsupported glass graphs fail explicitly instead of becoming opaque.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from pathlib import Path
import sys

if not __package__:  # Blender also executes this file directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real2sim_harness.asset_metadata import asset_value


def _group_has_transmission(tree, seen=None):
    seen = set() if seen is None else seen
    if tree in seen:
        return False
    seen.add(tree)
    for node in tree.nodes:
        if node.type in {"BSDF_GLASS", "BSDF_REFRACTION"}:
            return True
        if node.type == "BSDF_PRINCIPLED" and (
                node.inputs["Transmission Weight"].is_linked
                or node.inputs["Transmission Weight"].default_value > 0):
            return True
        if node.type == "GROUP" and node.node_tree and _group_has_transmission(node.node_tree, seen):
            return True
    return False


def glass_contract(objects):
    materials = {m.name: m for obj in objects for m in obj.data.materials if m}
    rows = []
    for name, material in sorted(materials.items()):
        if not material.use_nodes:
            continue
        nodes = material.node_tree.nodes
        outputs = [n for n in nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output]
        if not outputs or not outputs[0].inputs["Surface"].is_linked:
            continue
        output = outputs[0]
        surface = output.inputs["Surface"].links[0].from_node
        reachable, pending = set(), [surface]
        while pending:
            node = pending.pop()
            if node in reachable:
                continue
            reachable.add(node)
            pending.extend(link.from_node for socket in node.inputs for link in socket.links)
        candidates = [n for n in reachable if n.type == "BSDF_PRINCIPLED"
                      and (n.inputs["Transmission Weight"].is_linked
                           or n.inputs["Transmission Weight"].default_value > 0)]
        if not candidates:
            if any(n.type in {"BSDF_GLASS", "BSDF_REFRACTION"} for n in reachable):
                raise ValueError(f"{name}: glass/group graph needs an explicit target material adapter")
            for node in reachable:
                if node.type == "GROUP" and node.node_tree and _group_has_transmission(node.node_tree):
                    raise ValueError(f"{name}: grouped glass needs an explicit target material adapter")
            continue
        if surface.type != "BSDF_PRINCIPLED" or len(candidates) != 1:
            raise ValueError(f"{name}: mixed glass shader needs an explicit target material adapter")
        def constant(socket, surface=surface, name=name):
            value = surface.inputs[socket]
            if value.is_linked:
                raise ValueError(f"{name}: linked glass input {socket!r} is not supported")
            return value.default_value
        transmission = float(constant("Transmission Weight"))
        alpha = float(constant("Alpha"))
        roughness = float(constant("Roughness"))
        ior = float(constant("IOR"))
        color = list(constant("Base Color"))[:3]
        if not all(math.isfinite(x) for x in [transmission, alpha, roughness, ior, *color]):
            raise ValueError(f"{name}: nonfinite glass parameters")
        if abs(transmission - 1) > 1e-6 or abs(alpha - 1) > 1e-6:
            raise ValueError(f"{name}: native glass requires Transmission Weight=1 and Alpha=1; "
                             "partial transmission and alpha-fading require a separate adapter")
        if (not 0 <= roughness <= 1 or not 1 <= ior <= 4
                or any(not 0 <= c <= 1 for c in color)):
            raise ValueError(f"{name}: glass parameters outside the supported range")
        if (float(constant("Metallic")) != 0 or float(constant("Coat Weight")) != 0
                or float(constant("Anisotropic")) != 0
                or float(constant("Subsurface Weight")) != 0
                or float(constant("Sheen Weight")) != 0
                or float(constant("Thin Film Thickness")) != 0
                or (surface.inputs.get("Weight") is not None and float(constant("Weight")) != 1)
                or float(constant("Specular IOR Level")) != .5
                or any(abs(v-1) > 1e-6 for v in list(constant("Specular Tint"))[:3])
                or surface.inputs["Normal"].is_linked or output.inputs["Volume"].is_linked
                or output.inputs["Displacement"].is_linked):
            raise ValueError(f"{name}: modified lobes, coated, metallic, anisotropic, normal-mapped or volume glass needs its own adapter")
        emission = list(constant("Emission Color"))[:3]
        if any(emission) and float(constant("Emission Strength")):
            raise ValueError(f"{name}: emissive glass needs its own adapter")
        thin = asset_value(material, "real2sim_thin_walled", False)
        if type(thin) is not bool:
            raise ValueError(f"{name}: real2sim_thin_walled must be a boolean")
        rows.append({"source_material": name, "glass_color": color,
                     "frosting_roughness": roughness, "glass_ior": ior, "thin_walled": thin})
    return rows


def export_native_glass(usd_path, objects):
    """Add a packaged MDL render context without changing source or bindings."""
    rows = glass_contract([o for o in objects if o.type == "MESH"])
    path = Path(usd_path)
    report = {"schema_version": 1, "materials": rows, "render_verified": False,
              "scope": "Constant full-transmission Principled dielectric mapped to OmniGlass. "
                       "Tint/roughness appearance requires target-render review; no shader equivalence claim.",
              "universal_fallback": "USD Preview Surface retained; it does not certify glass optics"}
    if rows:
        from pxr import Gf, Sdf, Usd, UsdShade

        source = Path(os.environ.get("REAL2SIM_MDL_ROOT", ""))
        modules = ("OmniGlass.mdl", "OmniGlass_Opacity.mdl", "OmniPBRBase.mdl")
        if not os.environ.get("REAL2SIM_MDL_ROOT") or any(not (source / n).is_file() for n in modules):
            raise ValueError("Transmissive export requires REAL2SIM_MDL_ROOT pointing to the "
                             "installed renderer's core/Base MDL directory")
        destination = path.parent / "shaders"
        destination.mkdir(exist_ok=True)
        report["shader_files"] = {}
        for name in modules:
            target = destination / name
            shutil.copy2(source / name, target)
            report["shader_files"][f"shaders/{name}"] = hashlib.sha256(target.read_bytes()).hexdigest()
        stage = Usd.Stage.Open(str(path))
        by_name = {}
        for prim in stage.Traverse():
            if prim.IsA(UsdShade.Material):
                name = prim.GetAttribute("userProperties:blender:data_name").Get()
                if name:
                    by_name.setdefault(name, []).append(prim)
        for row in rows:
            matches = by_name.get(row["source_material"], [])
            if len(matches) != 1:
                raise ValueError(f"Glass material identity missing or ambiguous: {row['source_material']}")
            material = UsdShade.Material(matches[0])
            shader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild("NativeGlass"))
            shader.SetSourceAsset(Sdf.AssetPath("./shaders/OmniGlass.mdl"), "mdl")
            shader.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
            # USD dependency scanners cannot see imports inside MDL source.
            # Retain them as asset-valued dependencies so scene/USDZ packaging
            # cannot silently omit the native material's implementation.
            shader.GetPrim().CreateAttribute("remo:mdlDependencies", Sdf.ValueTypeNames.AssetArray).Set(
                [Sdf.AssetPath("./shaders/" + n) for n in modules[1:]])
            shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*row["glass_color"]))
            for name in ("frosting_roughness", "glass_ior"):
                shader.CreateInput(name, Sdf.ValueTypeNames.Float).Set(row[name])
            shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(row["thin_walled"])
            out = shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
            out.GetAttr().SetMetadata("renderType", "material")
            for kind in ("surface", "volume", "displacement"):
                material.CreateOutput("mdl:" + kind, Sdf.ValueTypeNames.Token).ConnectToSource(out)
            row["usd_material"] = str(material.GetPath())
        stage.GetRootLayer().Save()
    receipt = path.with_suffix(".materials.json")
    receipt.write_text(json.dumps(report, indent=2) + "\n")
    return report


def audit_native_glass(usd_path, receipt_path):
    """Check authored shader semantics and packaged library hashes, not pixels."""
    from pxr import Usd, UsdShade

    path, receipt = Path(usd_path), Path(receipt_path)
    report = json.loads(receipt.read_text())
    stage = Usd.Stage.Open(str(path))
    failures = []
    for name, digest in report.get("shader_files", {}).items():
        file = (path.parent / name).resolve()
        if (not file.is_relative_to(path.parent.resolve()) or not file.is_file()
                or hashlib.sha256(file.read_bytes()).hexdigest() != digest):
            failures.append("glass_shader_missing_or_changed:" + name)
    for row in report["materials"]:
        material = UsdShade.Material.Get(stage, row["usd_material"])
        connection = material.GetSurfaceOutput("mdl").GetConnectedSource() if material else None
        if not connection:
            failures.append("native_glass_surface_missing:" + row["source_material"])
            continue
        shader = UsdShade.Shader(connection[0].GetPrim())
        for kind in ("volume", "displacement"):
            other = material.GetOutput("mdl:" + kind).GetConnectedSource()
            if not other or other[0].GetPrim().GetPath() != shader.GetPath():
                failures.append(f"native_glass_context_changed:{row['source_material']}:{kind}")
        asset = shader.GetSourceAsset("mdl")
        if (not asset or asset.path != "./shaders/OmniGlass.mdl"
                or shader.GetSourceAssetSubIdentifier("mdl") != "OmniGlass"):
            failures.append("native_glass_shader_changed:" + row["source_material"])
        modules = shader.GetPrim().GetAttribute("remo:mdlDependencies").Get()
        if not modules or {p.path for p in modules} != {
                "./shaders/OmniGlass_Opacity.mdl", "./shaders/OmniPBRBase.mdl"}:
            failures.append("native_glass_dependencies_changed:" + row["source_material"])
        for key in ("glass_color", "frosting_roughness", "glass_ior", "thin_walled"):
            value = shader.GetInput(key).Get()
            expected = row[key]
            if key == "glass_color":
                matches = value is not None and len(value) == len(expected) and all(
                    abs(a-b) < 1e-6 for a, b in zip(value, expected, strict=True))
            elif key == "thin_walled":
                matches = value == expected
            else:
                matches = value is not None and abs(value-expected) < 1e-6
            if not matches:
                failures.append(f"native_glass_input_changed:{row['source_material']}:{key}")
    return {"ok": not failures, "failures": failures,
            "checked_materials": len(report["materials"]), "render_verified": False}
