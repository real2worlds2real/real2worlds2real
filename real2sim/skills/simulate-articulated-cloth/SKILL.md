---
name: simulate-articulated-cloth
description: Build and review functional mechanisms whose articulated rigid links carry a deforming cloth surface, using native two-way attachments in Newton.
---

# Simulate articulated cloth

Use `run_hybrid(attempt_id, frames, parameters_json, assumptions)` for a `hybrid` asset. It exports the saved source, runs Newton 1.7 unified VBD with native two-way rigid/particle attachments, reconstructs the retained appearance, and returns actual motion samples and Isaac RTX views. It is optional during modeling and mandatory for final hybrid acceptance. A static canopy, disconnected parts or independent appearance animation cannot establish the requested mechanical function.

Keep all appearance and geometry in editable Blender source. Use the usual positive-axis Euler joint Empty hierarchy for rigid links, with `joint_type`, `joint_axis`, and finite `lower_limit_degrees`/`upper_limit_degrees` or `lower_limit_m`/`upper_limit_m`. Put fixed mounting orientation on a parent Empty. Every moving rigid mesh belongs below the corresponding joint. The base consists of other rigid meshes. `real2sim_mass_kg`, `real2sim_drive_stiffness` and `real2sim_drive_damping` on a joint are optional assumed defaults. Cloth visuals must not be parented to individual links; use attachments to move them.

Provide one cloth control mesh with a native CLOTH modifier. It may be visible, or tagged `real2sim_sim_proxy=True` with visible Surface Deform/Mesh Deform followers. Coarse-to-fine detail is optional. The source frame 1 is the unstretched reference configuration, and simulation starts there. A trajectory may first gather a surface and then deploy it again; do not compress the cloth's material rest lengths to create a folded appearance. Source modifiers are retained, while replay replaces native cloth motion with actual Newton nodes.

On the control mesh, define `real2sim_attachments` as a JSON list of vertex groups and rigid-body names:

```python
control['real2sim_attachments'] = json.dumps([
    {'group': 'attachment_A', 'body': 'moving_link_A',
     'stiffness': 10000.0, 'damping': 1.0},
    {'group': 'attachment_base', 'body': '__base__'},
])
```

`body` names a joint Empty or `__base__`. Group weights above 0.5 attach those control vertices. Each vertex has one owner. All source CLOTH pin vertices must be explicitly assigned; they are not silently pinned in world space. The body-local anchors are derived from the authored reference pose. Matching HOOK modifiers can supply group/body annotations when explicit metadata is absent. Keep cloth particles outside collider surfaces by their collision radius, rather than initializing interpenetrating pins.

For mechanisms with a closed linkage, record extra constraints in scene `real2sim_loop_constraints` as JSON:

```python
scene['real2sim_loop_constraints'] = json.dumps([
    {'type': 'ball', 'body_a': 'moving_link_A',
     'body_b': 'moving_link_B', 'anchor_world': [x, y, z]},
])
```

Supported extra constraints are `ball` and `fixed`. Anchors are metre coordinates in the authored reference pose. Choose real meeting points inferred from the mechanism. A hierarchy alone cannot close a linkage. Prefer a small set of user-operated controls whose motion propagates through passive joints and constraints.

Record a motion test in scene `real2sim_hybrid_motion` as JSON:

```python
scene['real2sim_hybrid_motion'] = json.dumps([
    {'joint': 'user_control',
     'keyframes': [[0, 0], [2, displacement], [4, 0]]},
])
```

Each value is relative to the authored frame-1 pose: radians for a revolute joint, metres for a prismatic joint. Revolute limits are the source absolute Euler-degree limits; prismatic limits are offsets from the authored position. Times strictly increase and the first pair is `[0, 0]`. Targets must remain inside limits. The source geometry and constraints must be consistent with that rest pose. Use native Blender animation only for intermediate inspection; Newton drives the final trajectory.

Choose 96, 120, 180 or 240 frames at 24 FPS, enough to cover the trajectory and settling. `parameters_json=None` uses defaults; a partial JSON override can edit `surface`, `solver`, `rigid`, or `material` sections of `physics_parameters.json`. Thin cloth often needs a smaller particle radius than the generic 0.008 m default. Numerical assumptions remain editable and are not measured physical calibration.

`real2sim_fixed_base=True` holds the base for the mechanism test. Set it deliberately and record the assumption; downstream users can change this. The current adapter filters the frame's own rigid-rigid collisions at intersecting hinge proxies. Cloth self-contact is selectable and rigid/cloth contact remains active. This is not a completed robot manipulation task.

Inspect the whole sampled motion and final target-rendered views, not just finite-state success. Check control motion, material continuity, attachment drift, excessive stretch, and whether the requested function actually occurs. If a check fails, inspect the failing stage before changing geometry. Integration/runtime failures are distinct from demonstrated source defects. Repair and resubmit within the same session. Final acceptance also needs local surface evidence and a final Blender appearance render; native Blender `run_deformation` is optional for this hybrid path.

For transient penetrations or exposed internal links, read `review-dynamic-contact`. `run_hybrid` checks every cached frame for control/visual surface crossings and returns named pairs and witness locations in `surface_intersections.json`; crossings reject the check even if attachments and joint targets pass. Full-surface rigid/soft contact is enabled by default. `solver.hybrid_full_surface_contact` and `solver.hybrid_rigid_soft_dat` are editable switches, not substitutes for sound initial clearances. For complex motion, consider the video-reading tool to inspect relevant sides and suspect intervals; choose the coverage and detail needed rather than following a fixed viewing schedule.
