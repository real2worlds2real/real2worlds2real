"""Export an authored surface, detailed visuals and fixed analytic source fixtures."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/real2sim_harness"))
from blender_surface import is_asset_object  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("configuration", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    c = json.loads(a.configuration.read_text())
    if c.get("representation", "surface") != "surface":
        raise ValueError(
            "This source-surface adapter supports surface controls; use run_solids for volume assets"
        )
    control = bpy.data.objects.get(c["control_object"])
    if control is None or control.type != "MESH" or control.modifiers or control.data.shape_keys:
        raise ValueError("Use one baked editable surface control without another physics solver")
    names = c["visual_objects"]
    static = c.get("static_visual_objects", [])
    if not names or len(set(names + static)) != len(names + static):
        raise ValueError("Source visual owners must be nonempty, unique and disjoint")
    expected = {
        obj.name
        for obj in bpy.context.scene.objects
        if is_asset_object(obj) and not obj.hide_render
    }
    if set(names + static) != expected:
        raise ValueError("Every visible source asset mesh must be deformable or explicitly fixed")
    deps = bpy.context.evaluated_depsgraph_get()
    visuals = []
    for name in names:
        obj = bpy.data.objects[name]
        if (
            obj.type != "MESH"
            or obj.data.shape_keys
            or any(
                m.type in {"CLOTH", "SOFT_BODY", "ARMATURE", "SURFACE_DEFORM", "MESH_DEFORM"}
                for m in obj.modifiers
            )
        ):
            raise ValueError("Source followers cannot contain another deformation solver")
        for m in obj.modifiers:
            if m.show_viewport != m.show_render or (
                m.type == "SUBSURF" and m.levels != m.render_levels
            ):
                raise ValueError("Match viewport and render geometry before binding")
        evaluated = obj.evaluated_get(deps)
        mesh = evaluated.to_mesh()
        try:
            visuals.append(
                {
                    "name": name,
                    "vertices": [list(evaluated.matrix_world @ v.co) for v in mesh.vertices],
                }
            )
        finally:
            evaluated.to_mesh_clear()
    frames, fps = c.get("frames", 120), c.get("fps", 24)
    span = (frames - 1) / fps
    probes = c.get("probes", [])
    represented = set()
    for probe in probes:
        name = probe.get("source_object")
        if name is None:
            continue
        if name not in static:
            raise ValueError("A source-bound analytic fixture must be an explicitly fixed visual")
        obj = bpy.data.objects[name]
        if obj.type != "MESH":
            raise ValueError("Fixed source fixtures must be meshes")
        keys = np.asarray(probe.get("keyframes", [[0, 0, 0, 0], [span, 0, 0, 0]]), dtype=float)
        if keys.ndim != 2 or keys.shape[1] != 4 or np.any(keys[:, 1:] != 0):
            raise ValueError("Source-bound fixtures currently require fixed poses")
        evaluated = obj.evaluated_get(deps)
        mesh = evaluated.to_mesh()
        try:
            points = np.asarray([evaluated.matrix_world @ v.co for v in mesh.vertices])
        finally:
            evaluated.to_mesh_clear()
        lo, hi = points.min(axis=0), points.max(axis=0)
        probe["position_m"] = ((lo + hi) / 2).tolist()
        size = hi - lo
        if probe["shape"] == "box":
            rotation = np.asarray(evaluated.matrix_world)[:3, :3]
            if ((np.abs(rotation) > 1e-7).sum(axis=0) != 1).any() or (
                (np.abs(rotation) > 1e-7).sum(axis=1) != 1
            ).any():
                raise ValueError("Analytic box source fixtures must be world-axis aligned")
            probe["size_m"] = size.tolist()
        elif probe["shape"] == "sphere":
            if np.ptp(size) > 1e-5:
                raise ValueError("A sphere fixture cannot be an ellipsoid")
            probe["radius_m"] = float(size.max() / 2)
        else:
            raise ValueError("Supported source fixtures are box/sphere analytic bounds")
        probe["keyframes"] = keys.tolist()
        probe["source_visual_objects"] = [name]
        probe["conversion_scope"] = (
            "Fixed analytic bounds derived from the evaluated source mesh; inspect proxy conservatism."
        )
        represented.add(name)
    if represented != set(static):
        raise ValueError(
            "Every fixed source visual needs an explicit analytic probe, including declared visual-only fixtures"
        )
    control.data.calc_loop_triangles()
    record = {
        k: v
        for k, v in c.items()
        if k
        not in {
            "source_attempt_id",
            "control_object",
            "visual_objects",
            "static_visual_objects",
            "geometry",
            "record_asset_evidence",
        }
    }
    record.update(
        schema_version=1,
        representation="surface",
        frames=frames,
        fps=fps,
        source_scene_sha256=hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest(),
        geometry={
            "vertices": [list(control.matrix_world @ v.co) for v in control.data.vertices],
            "triangles": [list(t.vertices) for t in control.data.loop_triangles],
        },
        visuals=visuals,
        meshing={},
        source_static_visual_objects=static,
    )
    a.output.write_text(json.dumps(record))


if __name__ == "__main__":
    main()
