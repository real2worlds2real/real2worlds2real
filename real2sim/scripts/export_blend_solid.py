"""Export an explicit closed solid control and editable source visual vertices."""

import hashlib
import json
import sys
from pathlib import Path

import bpy

if not __package__:  # Blender also executes this file directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from real2sim_harness.asset_metadata import asset_key, asset_value

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src/real2sim_harness"))
from blender_surface import is_asset_object  # noqa: E402

output = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
output.mkdir(parents=True, exist_ok=True)
controls = [obj for obj in bpy.context.scene.objects if asset_value(obj, "real2sim_solid")]
if len(controls) != 1:
    raise ValueError("This solid adapter needs one explicit closed control mesh per asset")
control = controls[0]
if control.type != "MESH" or control.modifiers or control.data.shape_keys:
    raise ValueError("Solid control must be a baked editable mesh without physics or modifiers")
spec = json.loads(control[asset_key(control, "real2sim_solid")])
if set(spec) - {
    "material",
    "max_volume_m3",
    "quality",
    "max_elements",
    "max_surface_edge_m",
    "visual_objects",
}:
    raise ValueError("Unknown solid-control metadata")
visuals = []
names = spec.get("visual_objects", [])
if not names or len(set(names)) != len(names):
    raise ValueError("Declare unique visible mesh names driven by this solid")
all_visuals = {
    obj.name for obj in bpy.context.scene.objects if is_asset_object(obj) and not obj.hide_render
}
if set(names) != all_visuals:
    raise ValueError("Every visible asset surface must be included in the solid binding")
for name in names:
    obj = bpy.data.objects[name]
    if (
        obj.type != "MESH"
        or obj.data.shape_keys
        or any(
            mod.type in {"CLOTH", "SOFT_BODY", "ARMATURE", "SURFACE_DEFORM", "MESH_DEFORM"}
            for mod in obj.modifiers
        )
    ):
        raise ValueError("Use source mesh visuals without another deformation solver")
    for modifier in obj.modifiers:
        if modifier.show_viewport != modifier.show_render or (
            modifier.type == "SUBSURF" and modifier.levels != modifier.render_levels
        ):
            raise ValueError("Match viewport/render modifier settings before native visual binding")
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    try:
        visuals.append(
            {"name": name, "vertices": [list(evaluated.matrix_world @ v.co) for v in mesh.vertices]}
        )
    finally:
        evaluated.to_mesh_clear()
control.data.calc_loop_triangles()
behavior = json.loads(asset_value(bpy.context.scene, "real2sim_solid_behavior", "{}"))
if set(behavior) - {"environment", "solver", "probes", "grasps", "measurements", "review", "fps"}:
    raise ValueError("Unknown solid behavior fields")
record = {
    "schema_version": 1,
    "representation": "volume",
    "source_scene_sha256": hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest(),
    "geometry": {
        "vertices": [list(control.matrix_world @ v.co) for v in control.data.vertices],
        "triangles": [list(tri.vertices) for tri in control.data.loop_triangles],
    },
    "material": spec["material"],
    "meshing": {
        key: spec[key]
        for key in ("max_volume_m3", "quality", "max_elements", "max_surface_edge_m")
        if key in spec
    },
    "visuals": visuals,
    **behavior,
}
(output / "solid_source.json").write_text(json.dumps(record, indent=2))
