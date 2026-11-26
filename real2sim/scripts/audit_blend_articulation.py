"""Read-only articulation audit for the .blend currently opened by Blender."""

import json
import math

import bpy


def descendants(root):
    pending = list(root.children)
    found = []
    while pending:
        obj = pending.pop()
        found.append(obj)
        pending.extend(obj.children)
    return found


def serializable(value):
    if hasattr(value, "to_list"):
        return value.to_list()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return list(value) if hasattr(value, "__iter__") else str(value)


def audit_joint(joint):
    moving = [obj for obj in descendants(joint) if obj.type == "MESH"]
    original = joint.rotation_euler.copy()
    lower = float(joint.get("lower_limit_degrees", 0.0))
    upper = float(joint.get("upper_limit_degrees", 0.0))

    def positions(angle_degrees):
        joint.rotation_euler[2] = math.radians(angle_degrees)
        bpy.context.view_layer.update()
        return {obj.name: obj.matrix_world.translation.copy() for obj in moving}

    at_lower = positions(lower)
    at_upper = positions(upper)
    joint.rotation_euler = original
    bpy.context.view_layer.update()
    spans = {
        name: (at_upper[name] - at_lower[name]).length
        for name in at_lower
    }
    constraints = [
        {
            "name": constraint.name,
            "type": constraint.type,
            "use_limit_z": getattr(constraint, "use_limit_z", None),
            "min_z": getattr(constraint, "min_z", None),
            "max_z": getattr(constraint, "max_z", None),
        }
        for constraint in joint.constraints
    ]
    return {
        "name": joint.name,
        "properties": {key: serializable(joint[key]) for key in joint.keys()},
        "current_rotation_degrees": [round(math.degrees(v), 4) for v in original],
        "moving_mesh_count": len(moving),
        "moving_meshes": sorted(obj.name for obj in moving),
        "max_origin_motion_m": round(max(spans.values(), default=0.0), 6),
        "constraints": constraints,
    }


joints = [obj for obj in bpy.data.objects if obj.get("joint_type")]
report = {
    "ok": bool(joints) and all(descendants(joint) for joint in joints),
    "joint_count": len(joints),
    "joints": [audit_joint(joint) for joint in joints],
}
print("REMO_ARTICULATION_AUDIT=" + json.dumps(report, sort_keys=True))
