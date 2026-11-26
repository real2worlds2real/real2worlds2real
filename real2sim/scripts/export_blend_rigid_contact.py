"""Export source-derived collision bodies and visual bindings for contact diagnostics."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import bpy


if not __package__:  # Blender also executes this file directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from real2sim_harness.asset_metadata import asset_value, is_harness_datablock


def main():
    p = argparse.ArgumentParser()
    p.add_argument("configuration", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    c = json.loads(a.configuration.read_text())
    deps = bpy.context.evaluated_depsgraph_get()
    bound = set()
    bindings = []
    for spec in c["bodies"]:
        frame = bpy.data.objects.get(spec["frame_object"])
        if frame is None:
            raise ValueError("Missing rigid source frame")
        matrix = frame.evaluated_get(deps).matrix_world.copy()
        location, rotation, scale = matrix.decompose()
        if any(abs(v - 1) > 1e-5 for v in scale):
            raise ValueError("Rigid source frames must have applied unit scale")
        if abs(matrix.to_3x3().determinant() - 1) > 1e-5:
            raise ValueError("Rigid source frames must be proper rotations without shear")
        spec["pose"] = [*location, rotation.x, rotation.y, rotation.z, rotation.w]
        inverse = matrix.inverted()
        vertices, triangles = [], []
        for name in spec["collision_objects"]:
            obj = bpy.data.objects.get(name)
            if obj is None or obj.type != "MESH":
                raise ValueError("Collision objects must name source meshes")
            evaluated = obj.evaluated_get(deps)
            mesh = evaluated.to_mesh()
            try:
                mesh.calc_loop_triangles()
                local = inverse @ evaluated.matrix_world
                offset = len(vertices)
                vertices.extend(list(local @ v.co) for v in mesh.vertices)
                triangles.extend([offset + i for i in t.vertices] for t in mesh.loop_triangles)
            finally:
                evaluated.to_mesh_clear()
        spec["vertices"], spec["triangles"] = vertices, triangles
        visuals = []
        for name in spec["visual_objects"]:
            obj = bpy.data.objects.get(name)
            if obj is None or obj.type != "MESH" or obj.hide_render or name in bound:
                raise ValueError("Visual objects must be unique visible source meshes")
            if any(
                m.type in {"ARMATURE", "CLOTH", "SOFT_BODY", "MESH_CACHE", "MESH_SEQUENCE_CACHE"}
                for m in obj.modifiers
            ):
                raise ValueError("Rigid replay cannot replace a deforming source solver")
            bound.add(name)
            visuals.append(
                {
                    "object": name,
                    "body_local_matrix": [
                        list(row) for row in inverse @ obj.evaluated_get(deps).matrix_world
                    ],
                }
            )
        if not visuals:
            raise ValueError("Each source body requires visible source geometry")
        bindings.append({"body": spec["id"], "visuals": visuals})
    expected = {
        o.name
        for o in bpy.context.scene.objects
        if o.type == "MESH"
        and not o.hide_render
        and not asset_value(o, "real2sim_sim_proxy")
        and not asset_value(o, "real2sim_ground")
        and not is_harness_datablock(o.name)
    }
    if expected != bound:
        raise ValueError(
            "Visible source meshes are unbound or incorrectly excluded: "
            + repr(sorted(expected ^ bound))
        )
    c["source_scene_sha256"] = hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest()
    c["source_bindings"] = bindings
    a.output.write_text(json.dumps(c))


if __name__ == "__main__":
    main()
