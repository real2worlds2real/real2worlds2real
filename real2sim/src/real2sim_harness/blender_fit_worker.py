"""Blender-side parameter evaluation; no simulation or asset-acceptance claims."""
import importlib.util
import json
import sys
from pathlib import Path

import bpy

# Blender embeds Python; use the same checkout as the native authoring worker.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def evaluate(source, request):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if Path(source).suffix.lower() == '.blend':
        if request['parameters']:
            raise ValueError('Saved Blender inspection cannot change parameters')
        bpy.ops.wm.open_mainfile(filepath=str(source), load_ui=False, use_scripts=False)
        result = {}
    else:
        spec = importlib.util.spec_from_file_location("parameter_model", source)
        model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model)
        params = request["parameters"]
        declared = getattr(model, "PARAMS", None)
        if isinstance(declared, dict) and declared and all(
            isinstance(v, dict) and set(v) == {"init", "lo", "hi"} for v in declared.values()
        ):
            if any(k not in declared or not declared[k]["lo"] <= v <= declared[k]["hi"] for k, v in params.items()):
                raise ValueError("Fit exceeds the model's declared parameter contract")
            params = {**getattr(model, "FIXED", {}), **{k: v["init"] for k, v in declared.items()}, **params}
        result = model.build(params)
    selected = request.get("selected_objects")
    roots = request.get('selected_roots')
    if selected is not None and roots is not None:
        raise ValueError('Select exact mesh objects or identity roots, not both')
    if roots is not None:
        if not isinstance(roots, list) or not roots or any(not isinstance(n, str) or not n for n in roots):
            raise ValueError('Use nonempty exact identity-root names')
        if any(bpy.context.scene.objects.get(n) is None for n in roots):
            raise ValueError('A requested identity root is missing')
        selected = {obj.name for name in roots
                    for obj in [bpy.context.scene.objects[name], *bpy.context.scene.objects[name].children_recursive]
                    if obj.type == 'MESH' and not obj.hide_render}
    objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"
               and (obj.name in selected if selected is not None else not obj.hide_render)]
    if not objects or (selected is not None and {obj.name for obj in objects} != set(selected)):
        raise ValueError("Model produced no selected mesh or requested object names are missing")
    vertices, faces, offset = [], [], 0
    graph = bpy.context.evaluated_depsgraph_get()
    for obj in sorted(objects, key=lambda obj: obj.name):
        evaluated = obj.evaluated_get(graph)
        mesh = evaluated.to_mesh()
        mesh.calc_loop_triangles()
        vertices += [list(evaluated.matrix_world @ vertex.co) for vertex in mesh.vertices]
        reverse = evaluated.matrix_world.to_3x3().determinant() < 0
        faces += [[offset + index for index in (tuple(reversed(t.vertices)) if reverse else t.vertices)]
                  for t in mesh.loop_triangles]
        offset += len(mesh.vertices)
        evaluated.to_mesh_clear()
    features = result.get("features", []) if isinstance(result, dict) else []
    return {"vertices": vertices, "triangles": faces, "features": features,
            "selected_objects": sorted(obj.name for obj in objects)}


def main():
    source = sys.argv[sys.argv.index("--") + 1]
    for line in sys.stdin:
        try:
            result = {"ok": True, "mesh": evaluate(source, json.loads(line))}
        except Exception as error:
            result = {"ok": False, "error": str(error)}
        sys.stdout.write("REAL2SIM_FIT_RESULT=" + json.dumps(result) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
