"""Blender bridge for posed-reference tools; run with Blender --background SOURCE --python ... -- ARGS."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy
import numpy as np


def arrays(obj):
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        matrix = np.asarray(obj.matrix_world, dtype=float)
        vertices = np.array([vertex.co[:] for vertex in mesh.vertices], dtype=float)
        vertices = (
            vertices @ matrix[:3, :3].T + matrix[:3, 3]
        ) * bpy.context.scene.unit_settings.scale_length
        triangles = np.array([triangle.vertices[:] for triangle in mesh.loop_triangles], np.int32)
        result = {"vertices": vertices, "triangles": triangles}
        if mesh.uv_layers.active:
            layer = mesh.uv_layers.active.data
            result["uv"] = np.array(
                [
                    [layer[loop].uv[:] for loop in triangle.loops]
                    for triangle in mesh.loop_triangles
                ],
                dtype=float,
            )
        if np.linalg.det(matrix[:3, :3]) < 0:
            result["triangles"] = result["triangles"][:, [0, 2, 1]]
            if "uv" in result:
                result["uv"] = result["uv"][:, [0, 2, 1]]
        return result
    finally:
        evaluated.to_mesh_clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["export", "apply"])
    parser.add_argument("--object", required=True, help="Exact Blender mesh object name")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mesh",
        type=Path,
        help="Previously exported/baked mesh .npz for exact correspondence checking",
    )
    parser.add_argument("--texture", type=Path, help="Baked PNG atlas")
    parser.add_argument(
        "--unwrap", action="store_true", help="Use Blender Smart UV Project before export"
    )
    parser.add_argument(
        "--prepared-blend", type=Path, help="Save the matching UV-prepared source copy"
    )
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    obj = bpy.data.objects.get(args.object)
    if obj is None or obj.type != "MESH":
        raise ValueError(
            "Choose one existing mesh object; combine meshes/UVs explicitly when appropriate"
        )
    if args.unwrap:
        if (
            args.operation != "export"
            or args.prepared_blend is None
            or args.prepared_blend.suffix != ".blend"
        ):
            raise ValueError(
                "--unwrap requires export and --prepared-blend for the matching editable source"
            )
        if args.prepared_blend.resolve() == Path(bpy.data.filepath).resolve():
            raise ValueError("Save UV preparation to a different Blender file")
        if bpy.context.object and bpy.context.object.mode != "OBJECT":
            bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(island_margin=0.02)
        bpy.ops.object.mode_set(mode="OBJECT")
        args.prepared_blend.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(args.prepared_blend.resolve()), copy=True)
    current = arrays(obj)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.operation == "export":
        if args.output.suffix != ".npz":
            raise ValueError("Export output must use .npz")
        np.savez_compressed(args.output, **current)
        print(
            "REFERENCE_MESH_EXPORTED",
            len(current["vertices"]),
            len(current["triangles"]),
            "uv" in current,
        )
        return
    if args.mesh is None or args.texture is None or args.output.suffix != ".blend":
        raise ValueError("Apply requires --mesh, --texture and a new .blend output")
    if args.output.resolve() == Path(bpy.data.filepath).resolve():
        raise ValueError(
            "Keep the source Blender file; save the textured variant to a different file"
        )
    with np.load(args.mesh, allow_pickle=False) as retained:
        for key in ("vertices", "triangles"):
            if (
                key not in current
                or key not in retained
                or current[key].shape != retained[key].shape
                or not np.allclose(current[key], retained[key], atol=1e-7, rtol=1e-7)
            ):
                raise ValueError(
                    "Geometry/UVs changed since export; re-export and rebake rather than misapply the atlas"
                )
        if "uv" not in retained:
            raise ValueError("Baked mesh has no UV atlas")
        retained_uv = retained["uv"].copy()
    if retained_uv.shape != (len(current["triangles"]), 3, 2) or not np.isfinite(retained_uv).all():
        raise ValueError("Invalid baked UV coordinates")
    if "uv" not in current or not np.allclose(current["uv"], retained_uv, atol=1e-7, rtol=1e-7):
        # xatlas can add UV seams without changing geometry. Transfer those
        # corners to the original editable mesh when its topology corresponds.
        obj.data.calc_loop_triangles()
        base_faces = np.array(
            [triangle.vertices[:] for triangle in obj.data.loop_triangles], np.int32
        )
        negative = obj.matrix_world.to_3x3().determinant() < 0
        if negative:
            base_faces = base_faces[:, [0, 2, 1]]
        if len(obj.data.vertices) != len(current["vertices"]) or not np.array_equal(
            base_faces, current["triangles"]
        ):
            raise ValueError(
                "Generated UV topology differs from the editable mesh; prepare UVs before modifiers or update the construction source"
            )
        assignments = {}
        for triangle, values in zip(obj.data.loop_triangles, retained_uv, strict=True):
            loops = list(triangle.loops)
            if negative:
                loops = [loops[0], loops[2], loops[1]]
            for loop, coordinate in zip(loops, values, strict=True):
                if loop in assignments and not np.allclose(
                    assignments[loop], coordinate, atol=1e-6
                ):
                    raise ValueError(
                        "The UV atlas splits a polygon internally; triangulate or prepare Blender UVs before export and rebake"
                    )
                assignments[loop] = coordinate
        layer = obj.data.uv_layers.new(name="ObservedUV")
        obj.data.uv_layers.active = layer
        for loop, coordinate in assignments.items():
            layer.data[loop].uv = coordinate
    if obj.data.uv_layers.active is None:
        raise ValueError("The editable source needs the matching UV layer")
    image = bpy.data.images.load(str(args.texture.resolve()), check_existing=False)
    image.colorspace_settings.name = "sRGB"
    image.pack()
    material = bpy.data.materials.new(obj.name + "_observed_texture")
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    node = material.node_tree.nodes.new("ShaderNodeTexImage")
    node.image = image
    uv = material.node_tree.nodes.new("ShaderNodeUVMap")
    uv.uv_map = obj.data.uv_layers.active.name
    material.node_tree.links.new(uv.outputs["UV"], node.inputs["Vector"])
    # Transparent atlas pixels fall back to the user's editable base color.
    mix = material.node_tree.nodes.new("ShaderNodeMixRGB")
    mix.blend_type = "MIX"
    mix.inputs[1].default_value = (0.5, 0.5, 0.5, 1)
    material.node_tree.links.new(node.outputs["Alpha"], mix.inputs[0])
    material.node_tree.links.new(node.outputs["Color"], mix.inputs[2])
    material.node_tree.links.new(mix.outputs[0], shader.inputs["Base Color"])
    obj.data.materials.clear()
    obj.data.materials.append(material)
    for polygon in obj.data.polygons:
        polygon.material_index = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output.resolve()), copy=True)
    print("REFERENCE_TEXTURE_APPLIED", args.output.name)


if __name__ == "__main__":
    main()
