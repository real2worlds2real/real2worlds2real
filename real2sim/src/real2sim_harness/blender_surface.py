"""Shared Blender mesh-role and coarse-to-fine deformation evidence helpers."""

from __future__ import annotations

import math
from contextlib import contextmanager

import bpy
import numpy as np


import sys
from pathlib import Path

if not __package__:  # Blender also executes this file directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real2sim_harness.asset_metadata import asset_value


def is_asset_object(obj):
    return obj.type in {"MESH", "CURVE", "FONT"} and not any(
        asset_value(obj, tag) for tag in ("real2sim_ground", "real2sim_fixture", "real2sim_sim_proxy", "real2sim_collision_proxy")
    )


@contextmanager
def portable_image_uv_scales(objects):
    """Encode constant Vector Math UV multiplication as glTF-supported Mapping.

    The two node forms are equivalent for UV input. Changes exist only during
    export and are restored even on failure; the editable source is preserved.
    """
    edits, conversions = [], []
    materials = {slot.material for obj in objects for slot in getattr(obj, "material_slots", []) if slot.material and slot.material.use_nodes}
    try:
        for material in materials:
            tree = material.node_tree
            for texture in list(tree.nodes):
                if texture.bl_idname != "ShaderNodeTexImage" or not texture.inputs["Vector"].is_linked:
                    continue
                original = texture.inputs["Vector"].links[0].from_socket
                node = original.node
                if node.bl_idname != "ShaderNodeVectorMath" or node.operation != "MULTIPLY":
                    continue
                first, second = node.inputs[:2]
                if first.is_linked == second.is_linked:
                    continue
                variable, constant = (first, second) if first.is_linked else (second, first)
                source = variable.links[0].from_socket
                if not (source.node.bl_idname == "ShaderNodeUVMap" or (source.node.bl_idname == "ShaderNodeTexCoord" and source.name == "UV")):
                    continue
                scale = tuple(constant.default_value)
                if not all(math.isfinite(value) for value in scale):
                    continue
                mapping = tree.nodes.new("ShaderNodeMapping")
                mapping.vector_type = "POINT"
                mapping.inputs["Scale"].default_value = scale
                tree.links.new(source, mapping.inputs["Vector"])
                edits.append((tree, texture.inputs["Vector"], original, mapping))
                tree.links.new(mapping.outputs["Vector"], texture.inputs["Vector"])
                conversions.append({"material": material.name, "image": texture.image.name if texture.image else None, "scale": list(scale)})
        yield conversions
    finally:
        for tree, target, original, mapping in reversed(edits):
            tree.links.new(original, target)
            tree.nodes.remove(mapping)


def sampled_world_points(obj, indices=None, limit=4096):
    bpy.context.view_layer.update()
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        if indices is None:
            indices = np.unique(np.linspace(0, len(mesh.vertices) - 1, min(len(mesh.vertices), limit)).astype(np.int32))
        if not len(indices) or int(max(indices)) >= len(mesh.vertices):
            raise ValueError(f"Invalid or changing visual topology: {obj.name}")
        coordinates = np.empty((len(mesh.vertices), 3), dtype=np.float64)
        mesh.vertices.foreach_get("co", coordinates.ravel())
        matrix = np.asarray(evaluated.matrix_world, dtype=np.float64)
        return coordinates[indices] @ matrix[:3, :3].T + matrix[:3, 3], indices, len(mesh.vertices)
    finally:
        evaluated.to_mesh_clear()


def proxy_bindings(scene, physical):
    physical_names = {obj.name for obj, _ in physical}
    proxies = [obj for obj in scene.objects if asset_value(obj, "real2sim_sim_proxy")]
    records = []
    for proxy in proxies:
        drivers = []
        for obj in scene.objects:
            if not is_asset_object(obj) or obj.hide_render:
                continue
            for modifier in obj.modifiers:
                if modifier.type not in {"SURFACE_DEFORM", "MESH_DEFORM"}:
                    continue
                target = modifier.target if modifier.type == "SURFACE_DEFORM" else modifier.object
                if target != proxy:
                    continue
                drivers.append({
                    "object": obj.name, "modifier": modifier.name, "type": modifier.type,
                    "bound": bool(modifier.is_bound),
                    "enabled": bool(modifier.show_viewport and modifier.show_render),
                })
        records.append({
            "proxy": proxy.name,
            "drivers": drivers,
            "ok": proxy.name in physical_names and bool(drivers) and all(row["bound"] and row["enabled"] for row in drivers),
        })
    return records
