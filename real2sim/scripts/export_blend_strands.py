"""Export generic line-mesh rods, bound surface meshes and endpoint rigid bodies."""

import hashlib
import json
import sys
from pathlib import Path

import bpy

if not __package__:  # Blender also executes this file directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from real2sim_harness.asset_metadata import asset_key, asset_value

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from real2sim_harness.strand_contract import validate_configuration  # noqa: E402

output = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
output.mkdir(parents=True, exist_ok=True)
strands = []
for obj in bpy.context.scene.objects:
    if not asset_value(obj, "real2sim_strand"):
        continue
    if obj.type != "MESH" or obj.data.polygons:
        raise ValueError("real2sim_strand belongs on an ordered line MESH, without faces")
    spec = json.loads(obj[asset_key(obj, "real2sim_strand")])
    n = len(obj.data.vertices)
    edges = {tuple(sorted(e.vertices)) for e in obj.data.edges}
    if edges != {(i, i + 1) for i in range(n - 1)}:
        raise ValueError("Centerline vertex order must follow a single open chain")
    points = [list(obj.matrix_world @ v.co) for v in obj.data.vertices]
    defaults = {
        "radius": 0.003,
        "density": 1100.0,
        "stretch_rigidity": 300.0,
        "shear_rigidity": 100.0,
        "bend_rigidity": 0.00005,
        "twist_rigidity": 0.00003,
        "stretch_damping": 0.05,
        "bend_damping": 0.00005,
        "twist_damping": 0.00005,
        "attachments": [],
    }
    allowed = set(defaults) | {"visual_objects"}
    if set(spec) - allowed:
        raise ValueError("Unknown strand fields: " + str(set(spec) - allowed))
    defaults.update(spec)
    for name in defaults.get("visual_objects", []):
        visual = bpy.data.objects.get(name)
        if visual is None or visual.type != "MESH" or visual.hide_render:
            raise ValueError("Strand visuals must be visible MESH objects: " + name)
    strands.append({"name": obj.name, "points": points, **defaults})
bodies = []
for obj in bpy.context.scene.objects:
    if not asset_value(obj, "real2sim_strand_body"):
        continue
    spec = json.loads(obj[asset_key(obj, "real2sim_strand_body")])
    if set(spec) - {"radius", "half_height", "mass_kg"}:
        raise ValueError("Rigid endpoint body uses radius, half_height, mass_kg")
    if not all(k in spec for k in ("radius", "half_height", "mass_kg")):
        raise ValueError("Declare capsule proxy and mass explicitly")
    descendants = list(obj.children_recursive)
    visuals = [v.name for v in ([obj] + descendants) if v.type == "MESH" and not v.hide_render]
    if not visuals:
        raise ValueError("A rigid endpoint body needs visible geometry")
    bodies.append(
        {
            "name": obj.name,
            "matrix_world": [list(row) for row in obj.matrix_world],
            "visual_objects": visuals,
            **spec,
        }
    )
env = {
    "gravity": [0, 0, -9.81],
    "floor_height": 0.0,
    "friction": 0.5,
    "air_drag_rate_per_second": 0.5,
}
env.update(json.loads(asset_value(bpy.context.scene, "real2sim_strand_environment", "{}")))
solver = {"substeps": 16, "iterations": 12}
solver.update(json.loads(asset_value(bpy.context.scene, "real2sim_strand_solver", "{}")))
c = {
    "schema_version": 1,
    "source_scene_sha256": hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest(),
    "strands": strands,
    "bodies": bodies,
    "environment": env,
    "solver": solver,
    "motion": json.loads(asset_value(bpy.context.scene, "real2sim_strand_motion", "[]")),
    "quality": json.loads(asset_value(bpy.context.scene, "real2sim_strand_quality", "{}")),
}
validate_configuration(c)
(output / "strand_configuration.json").write_text(json.dumps(c, indent=2))
print("STRAND_EXPORT", len(strands), len(bodies), flush=True)
