---
name: simulate-slender-bodies
description: Generate and review ropes, cords or cables using native Newton rods or an optional PhysX capsule chain, selecting the target scene's backend and inspecting actual continuity, bending and contact.
---

# Simulate slender bodies

For a requested shared PhysX scene, the optional
[PhysX capsule-chain API](../../docs/PHYSX_MATERIAL_CAPABILITIES.md#capsule-chain-rope)
authors an arbitrary centerline as passive capsules and native D6 joints. It
supports local rigid-tool contact in Isaac Sim 5.1 and complements the Newton
route below. Select it from the user's backend/mechanical needs. Preserve
centerline length and total mass, audit joint gaps and actual capsule support,
and inspect contact motion. The default capsule appearance needs a separate
binding for detailed authored visual surfaces. Its angular drives/limits are
not Newton rod material coefficients. Existing `run_strands` receipts remain
specific to their Newton route; a native PhysX check needs its own retained
current-source evidence and the task's configured completion path.

Use asset kind `strand` and `run_strands(attempt_id, frames, assumptions)` for open, unbranched ropes/cables. This adapter uses native Newton 1.7 Rod/VBD stretch, shear, bend and twist, not cloth ribbons or prescribed surface animation. Available frames: 96, 120, 144, 192 or 240 at 24 FPS. Intermediate calls are optional; a successful source-matched native run and fresh-process replay are required for delivery. Native Blender Cloth is not required on this route.

## Editable source representation

Create an ordered line MESH with no faces, 4–513 vertices and exactly edges `(i,i+1)`. Coordinates follow one open centerline, without repeated points. Tag it `real2sim_sim_proxy=True` so it remains in `.blend` but not the visible export. Set its `real2sim_strand` property to JSON:

```python
control['real2sim_strand'] = json.dumps({
    'radius': 0.003, 'density': 1100.0,
    'stretch_rigidity': 300.0, 'shear_rigidity': 100.0,
    'bend_rigidity': 0.00005, 'twist_rigidity': 0.00003,
    'stretch_damping': 0.05,
    'bend_damping': 0.00005, 'twist_damping': 0.00005,
    'visual_objects': ['surface_mesh'],
    'attachments': [{'node': 0, 'body': 'endpoint_body'}],
})
```

These numbers are illustrative assumptions, not calibrated values. Lengths are metres; density is kg/m³; stretch/shear rigidity is N; bend/twist rigidity is N·m². The native builder derives segment-length-aware joint coefficients. Damping fields are per-joint Newton coefficients, not Blender damping. Actual rest curvature/material frames follow the authored centerline. Avoid extreme segment-length variation and tight initial overlaps. Resolving an existing coil is distinct from proving knot manipulation.

Each named visual must be a visible editable MESH without shape keys. Multiple visible meshes, UVs and materials are allowed. The adapter binds source vertices to the centerline and smoothly interpolated native material frames; it preserves reference geometry numerically and replays the source meshes/materials. Build actual cylindrical thickness and useful surface detail as warranted by the reference; choose a suitable centerline resolution rather than simulating every fibre. No hardcoded object recipe is needed.

## Optional rigid endpoints

Create a rigid root object, commonly an Empty, with visible child meshes. All child geometry follows its physical frame. Apply scale into geometry and keep the root matrix a rigid rotation/translation. Tag its `real2sim_strand_body` with JSON:

```python
body['real2sim_strand_body'] = json.dumps({
    'radius': 0.012, 'half_height': 0.06, 'mass_kg': 0.04,
})
```

This explicitly declares a capsule collision proxy centered on the root, with its cylinder along local Z; half_height excludes the hemispherical caps. Choose a proxy consistent with the visual body. The visual mesh may have detailed shape/material independent of that proxy. Attach node 0 or the final node to a declared root by name. Native ball joints attach endpoints at their authored positions. Keep the cord exit and proxy geometry consistent. Collision filtering covers only the explicitly attached endpoint neighborhood; other rod/body and nonneighbor rod contacts stay active.

## Chosen operation and environment

For an optional controlled grasp, store scene `real2sim_strand_motion` as JSON:

```python
scene['real2sim_strand_motion'] = json.dumps([
    {'body': 'endpoint_body',
     'keyframes': [[0,0,0,0], [1,0,0,0], [3,0,0,0.25], [5,0,0,0.25]]},
])
```

Keyframes are `[seconds, world_dx, world_dy, world_dz]`, relative to the authored pose. Start at `[0,0,0,0]`, increase time strictly, finish before the final retained timestamp `(frames-1)/24`. Targets use smooth interpolation. Driven roots are kinematic grasp constraints, while the rest of the strand responds dynamically. A root with no trajectory remains a dynamic rigid body. Add a stationary trajectory when a second root should be held. Current controls prescribe translation; rotations and release schedules are not yet exposed. Choose an informative operation, with sufficient slack and time, rather than forcing endpoint separation beyond material length.

Optional scene JSON `real2sim_strand_environment` overrides `gravity` (default `[0,0,-9.81]`), `floor_height` (0), `friction` (0.5), and `air_drag_rate_per_second` (0.5). Drag is a native linear still-air force, not image smoothing. Keep the visual support consistent with the actual plane. Optional `real2sim_strand_solver` exposes `substeps` (16) and `iterations` (12). More resolution or more substeps are questions to test, not automatic quality guarantees.

For optional obstacle interactions, `real2sim_strand_environment` also accepts `capsule_obstacles`, each with unique `id`, `center_m`, unit `axis`, positive `radius_m` and `half_height_m`. These are static world colliders, rendered as diagnostic fixtures rather than semantic asset parts. Independent auditing checks their clearance against every retained rod/endpoint capsule. `contact_stiffness` (default 1e5) and `contact_damping` (1) are explicit numerical contact coefficients; optional solver `contact_history=true` retains native friction/contact history using matching contacts, following Newton's cable examples. Existing defaults are preserved.

Before interpreting a failed winding test as a bad rope, check whether a feasible slack path exists and whether the motion traps the rope between a fixture base and the ground. A rounded capsule balanced on its bottom cap can create an unintended pinch point. Use an appropriate support/collider shape and lift height; test and retain the failure rather than loosening penetration or connection-gap checks. A successful open-curve winding test is still distinct from a knot certificate. The replay CLI optionally accepts `--view-direction x y z` to expose contact and both endpoints; fixtures need not dominate the asset's camera envelope.

Motion reports include temporal speed/acceleration diagnostics. For an explicitly unforced hold after the grasp stops, optional scene JSON `real2sim_strand_quality` accepts `quiet_start_seconds` and `quiet_speed_tolerance_m_s`; choose a meaningful time window and tolerance. A moving target is not an unforced hold. Independent capsule-proxy auditing checks every retained frame for nonlocal crossings and ground penetration, while excluding adjacent segments and declared attachment neighborhoods. Proxy checks do not replace visual inspection of the actual mesh.

## Review and delivery

Review the seven returned actual-motion samples and diagnostics: connection gaps, endpoint drift, stretch, ground contact and the intended action. Checks do not certify knots, contact-free geometry or real-world material accuracy. A continuous rendered tube alone cannot prove physical continuity: native connection metrics must also pass. Inspect dynamic surfaces for visible pinching, angular seams, detachment and unnatural oscillation. Use material/clay/grazing inspection when local appearance is uncertain.

For useful dynamic review, `render_motion_video` replays the verified cache; `inspect_video` can examine chronological intervals, exact frames and crops. The model chooses viewing coverage. Rendering or encoding is not acceptance. Keep editable source, `strand_configuration.json`, `strand_motion.npz`, validation and source hashes together. `real2sim_harness.strand_runtime.load_strand_asset(configuration)` is the native Newton loader; this is a dedicated rod runtime, not the existing rigid/cloth USD adapter or a completed robot task. Final `accept_result` rejects missing, failed or changed native evidence.

The loader accepts an optional prepopulated Newton `builder`, allowing other bodies or a compatible robot model to share the same finalized VBD world. Existing scene objects retain their IDs. Endpoint rigid bodies load as passive dynamic bodies by default, with their authored masses and native ball connections. The validation script explicitly passes `prescribed_motion=True` to install demonstration grasps; downstream users may omit that flag and use contact or their own constraints. To retain the configured still-air environment, call `real2sim_harness.strand_runtime.apply_strand_environment(runtime, state)` after clearing forces in each substep. It is a shared-world linear force; the caller owns world configuration and controller integration. No robot policy is supplied.

Implementation follows the installed official Newton `Rod` / `ModelBuilder.add_rod` API and VBD cable examples: https://newton-physics.github.io/newton/latest/api/_generated/newton.ModelBuilder.html . Read local installed signatures when extending the adapter; cable API names changed between releases.
