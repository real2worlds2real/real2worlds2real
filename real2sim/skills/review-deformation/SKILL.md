---
name: review-deformation
description: Build and review actual Blender cloth or soft-body deformation with retained simulation caches and rendered motion evidence.
---

# Review native deformation

For garment target simulation, default to PhysX particle cloth and read
[the garment guidance](../simulate-deformable-behavior/physx-garments.md).
The Blender workflow below remains a construction and deformation diagnostic;
it does not replace the requested PhysX garment review.

The host uses Blender 4.5's existing CLOTH/SOFT_BODY solvers. `run_deformation(attempt_id, frames, assumptions)` loads a successful saved attempt, bakes its enabled native physics, returns seven sampled renders and checks cache replay in another Blender process. Supported lengths are 48, 72, 96, 120 or 180 frames, beginning at `scene.frame_start`; choose `scene.render.fps` in source. A short initial test is useful before expensive final rendering.

The source chooses the physical mesh, material parameters, collision objects, constraints and any animated excitation. Preserve the observed reference shape at the initial frame, while making later deformation observable. If the task lacks measured scale, stiffness, thickness, density or motion, label those choices as assumptions. Static photos do not uniquely determine physical parameters.

For cloth-like surface construction and detailed appearance, consult `refine-deformable-surface` before choosing the simulation/detail representation. Proxy use is optional; real visible motion is required. New runs exposing `inspect_surface` require at least one successful material/clay/grazing comparison before deformable acceptance. Review the images, not just the execution flag.

Local conventions and common pitfalls:

- Put CLOTH or SOFT_BODY on actual asset mesh geometry. Keep it enabled in viewport and render. Use at most one such solver per object. Rigid-body motion, shape-key animation alone, and static wrinkle sculpting cannot satisfy the deformation gate.
- Apply object scale before physics. Keep a reasonably uniform control mesh, with at most 250k control vertices. Place high-resolution subdivision and visual thickness after physics; match subdivision viewport/render levels to avoid cache surprises.
- Native cloth collision needs COLLISION modifiers on the other objects, plus cloth collision enabled. Enable self-collision when folds can contact. The host's automatic review ground is visual only: explicitly add a collider if your physical scenario needs one.
- Mark external test supports `obj['real2sim_fixture'] = True`; they remain visible during the motion test but are excluded from object geometry statistics and GLB exports. They are not semantic asset parts.
- A parent transform does not make decorative geometry follow cloth vertices. Use a surface-following modifier or integrate details into the deforming mesh/material; inspect seams and attachments throughout the sampled motion.
- Do not bake inside `build_scene()` or freeze a physically simulated mesh into a static replacement. The host handles baking and retains the original editable source. Initial pose matching and subsequent physical motion are both reviewable.

The diagnostics remove the best-fitting rigid rotation/translation before measuring shape change, and report edge-length ratios and bounded/finite motion. These do not certify self-intersection-free contact or real-world dynamics. Inspect the returned frames for penetration, stretching, detached details, unstable folds and an appropriate physical test. Repair source and rerun rather than patching retained caches.

For large edge-stretch outliers, the live report includes the worst edge's control-vertex indices, initial/current lengths, sample index and world-space midpoints. Use those locations to diagnose nearly collapsed rest edges, inconsistent thickness or contact offsets, and attachment constraints. A passing finite-motion check alone is insufficient evidence to dismiss a stretch warning; fix it or document a defensible physical interpretation.

Interpret local extrema alongside absolute displacement, spatial extent, persistence and garment-wide distributions. Review the affected region in motion, including release and settling when relevant. Persistent jitter, visible penetration and widespread distortion matter even if scalar checks pass; an isolated short-edge ratio needs context before it becomes a task-level failure. Keep diagnostic results distinct from the visual review and the task's stated acceptance requirements.

Native CLOTH also has a deliberately loose sanity gate: its aggregate 95th-percentile edge-length ratio must stay at or below 2. This rejects widespread severe stretch, not isolated short-edge outliers. It is not a measured fabric tolerance. An unusually extensible modeled material can explicitly set `obj['real2sim_cloth_stretch_p95_limit']` above 1 with a documented physical assumption; ordinary woven fabric should aim far tighter than the default, rather than increasing the threshold to hide solver failures.

Deformable/hybrid acceptance requires successful native physics plus fresh-process replay. Keep `simulation.blend` together with its `blendcache_*` directory. `object.glb` is a static visual snapshot; it does not carry this native cloth simulation. The generic simulation playback renderer can make a movie from the retained cache.

Implementation references: [point cache API](https://docs.blender.org/api/4.5/bpy.types.PointCache.html), [baking operators](https://docs.blender.org/api/4.5/bpy.ops.ptcache.html). Backend versions and dependencies are recorded in [the environment reference](../../docs/ENVIRONMENTS.md).
