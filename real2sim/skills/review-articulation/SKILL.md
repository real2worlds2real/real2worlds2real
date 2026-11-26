---
name: review-articulation
description: Verify that an articulated Blender asset has real joint hierarchy, plausible limits, and correct motion rather than a static posed part.
---

# Review articulation evidence

Use this skill for doors, drawers, wheels, handles and other moving assemblies after a successful Blender execution.

The executor reads these exact custom properties on each joint Empty:

- `joint_type`: `revolute` or `prismatic`.
- `joint_axis`: a signed local axis such as `+X`, `-Y`, or `+Z`.
- For `revolute`: `lower_limit_degrees` and `upper_limit_degrees`, expressed in degrees.
- For `prismatic`: `lower_limit_m` and `upper_limit_m`, expressed in metres.

Generic aliases such as `joint_limit_lower` or `limit_lower` are not read, and radians are not inferred. For example, a full-turn revolute range can use `lower_limit_degrees = -180` and `upper_limit_degrees = 180`. A `LIMIT_ROTATION` constraint does not replace these metadata fields.

For an oblique axis, a fixed oriented parent Empty can hold the mount transform while the child joint holds the variable local-axis rotation. Keep this distinction explicit so pose sampling rotates the joint without changing its mounting direction.

Read `attempts/<attempt_id>/report.json`: its `articulation` section records
the hierarchy and sampled motion, and `articulation_previews` lists the pose
images. Inspect those images with your native image reader. Check:

- each intended joint has the correct type and local axis;
- lower and upper limits are finite and physically plausible;
- every moving mesh is a descendant of the joint, while static meshes are not;
- sampled mesh-point motion is nonzero and follows the declared joint type and signed local axis; do not use object-origin motion alone, because centered rotating parts such as turntables can rotate correctly while their origins remain fixed;
- lower, middle and upper renders preserve attachment at the pivot and avoid obvious interpenetration.

Reject a visually open part that is merely positioned beside the chassis. Fix the editable source and rerun Blender; do not patch exported GLB artifacts.

For a free base, verify housing visuals follow the base as well as moving-part
visuals following joints. A correct rest image or moving collision body does not
establish visual rigid ownership. Conversion audits check ownership and rest
geometry; inspect actual target motion too. Keep joint descendants separate when
repairing housing hierarchy. Open poses must show plausible wall thickness and
return surfaces; check overlapping coplanar faces before blaming roughness or
lighting for unstable highlights. For final-window motion metrics, use
[review-numerical-evidence](../review-numerical-evidence/SKILL.md), selecting the
intended body by name rather than assuming index zero is the housing.

A free-flight joint exercise establishes the tested joint response, not stable
operation on a countertop. When supported operation matters, test the unchanged
passive asset in an appropriate support fixture and inspect transition poses,
housing clearance and support response. Keep temporary test drives separate from
delivery; quiet rigid-window statistics alone cannot establish support.
