"""Summarize simulator-relevant Blender objects in the open scene."""

import json
import sys
from pathlib import Path

import bpy


def main() -> None:
    args = sys.argv[sys.argv.index("--") + 1 :]
    output = Path(args[0])
    objects = []
    for obj in bpy.data.objects:
        if obj.type not in {"MESH", "EMPTY"}:
            continue
        props = {}
        for key in obj.keys():
            if key == "_RNA_UI":
                continue
            value = obj[key]
            if isinstance(value, (str, int, float, bool)):
                props[key] = value
            elif hasattr(value, "to_list"):
                props[key] = value.to_list()
        item = {
            "name": obj.name,
            "type": obj.type,
            "parent": obj.parent.name if obj.parent else None,
            "props": props,
            "matrix_world": [list(row) for row in obj.matrix_world],
            "modifiers": [m.type for m in obj.modifiers],
            "hide_render": obj.hide_render,
        }
        if obj.type == "MESH":
            item["vertices"] = len(obj.data.vertices)
            item["polygons"] = len(obj.data.polygons)
        objects.append(item)
    result = {
        "source": bpy.data.filepath,
        "scene_props": {key: bpy.context.scene[key] for key in bpy.context.scene.keys() if key != "_RNA_UI"},
        "objects": objects,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, default=str))
    print(f"INSPECT_BLEND_SIM_OK objects={len(objects)} output={output}")


if __name__ == "__main__":
    main()
