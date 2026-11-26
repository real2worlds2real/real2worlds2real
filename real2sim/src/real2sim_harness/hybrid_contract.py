"""Category-independent metadata for articulated rigid bodies joined to cloth."""

from __future__ import annotations

import math


def finite_number(value, label, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return float(value)


def validate_hybrid_contract(configuration, joints, control):
    """Reject ambiguous ownership, invalid loops or controls before simulation."""
    by_name = {joint["name"]: joint for joint in joints}
    bodies = {"__base__", *by_name}
    count = len(control["vertices"])
    assigned = set()
    attachments = configuration.get("attachments", [])
    if not attachments:
        raise ValueError("Hybrid cloth needs explicit particle-to-body attachments")
    for attachment in attachments:
        if attachment["body"] not in bodies:
            raise ValueError(f"Attachment body does not exist: {attachment['body']}")
        indices = attachment.get("particle_indices", [])
        if not indices:
            raise ValueError("Attachment groups must contain particles")
        for index in indices:
            if type(index) is not int or index < 0 or index >= count:
                raise ValueError(f"Attachment particle index outside the control mesh: {index}")
            if index in assigned:
                raise ValueError(f"Particle {index} has conflicting or duplicate attachments")
            assigned.add(index)
        finite_number(attachment.get("stiffness", 1e4), "Attachment stiffness", minimum=0.001)
        finite_number(attachment.get("damping", 1.0), "Attachment damping", minimum=0)
    if set(control.get("pin_indices", [])) - assigned:
        raise ValueError("Every native cloth pin needs an explicit hybrid body attachment; do not silently pin it in world space")
    for constraint in configuration.get("loop_constraints", []):
        if constraint["type"] not in {"ball", "fixed"}:
            raise ValueError("Loop constraints currently support ball or fixed joints")
        if constraint["body_a"] not in bodies or constraint["body_b"] not in bodies:
            raise ValueError("Loop constraint refers to a missing rigid body")
        if constraint["body_a"] == constraint["body_b"]:
            raise ValueError("A loop constraint needs distinct bodies")
        anchor = constraint.get("anchor_world", [])
        if len(anchor) != 3:
            raise ValueError("Loop anchors must have three metre coordinates")
        for value in anchor:
            finite_number(value, "Loop anchor")
    motion = configuration.get("motion", [])
    if not motion:
        raise ValueError("Hybrid operability review needs at least one joint control trajectory")
    controlled = set()
    for record in motion:
        name = record["joint"]
        if name not in by_name or name in controlled:
            raise ValueError(f"Missing or duplicate controlled joint: {name}")
        controlled.add(name)
        spec = by_name[name]
        if spec["joint_type"] in {"revolute", "continuous"}:
            lower, upper = [math.radians(spec[key] - spec["current_degrees"])
                            for key in ("lower_degrees", "upper_degrees")]
        elif spec["joint_type"] == "prismatic":
            lower, upper = spec["lower_m"], spec["upper_m"]
        else:
            raise ValueError("Hybrid controls support revolute or prismatic joints")
        keys = record.get("keyframes", [])
        if len(keys) < 2 or any(not isinstance(key, list) or len(key) != 2 for key in keys):
            raise ValueError("Motion keyframes must contain at least two [seconds, relative_SI_value] pairs")
        previous = -1.0
        for time, value in keys:
            time = finite_number(time, "Keyframe time", minimum=0)
            value = finite_number(value, "Keyframe value")
            if time <= previous:
                raise ValueError("Motion keyframe times must strictly increase")
            if not lower - 1e-6 <= value <= upper + 1e-6:
                raise ValueError(f"Motion target exceeds authored limits for {name}")
            previous = time
        if keys[0] != [0, 0]:
            raise ValueError("Start a hybrid review at the authored rest pose with keyframe [0, 0]")
        if max(abs(key[1]) for key in keys) < 1e-4:
            raise ValueError("A hybrid trajectory must exercise a joint")
    return {"ok": True, "attachment_particles": len(assigned), "controlled_joints": sorted(controlled),
            "loop_constraints": len(configuration.get("loop_constraints", [])),
            "motion_duration_seconds": max(record["keyframes"][-1][0] for record in motion)}
