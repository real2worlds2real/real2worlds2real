---
name: compose-interactive-scene
description: Assemble and review a scene from complete operable object bundles, a visual background and environment collisions, using task-level physics selection and simulator feedback.
---

Read `scene_contract.md` in a scene workspace, or `docs/SCENES.md` in the repository, for manifest and adapter schemas.

For observed twins, carry the five
[evidence obligations](../plan-scene-generation/SKILL.md#evidence-obligations-for-observed-scene-twins)
through composition and final review. Flexible tool choice and iteration order
do not permit skipping the necessary evidence or closing an unresolved finding.

Choose the scene decomposition from the input and task. Keep original views available when cropping objects. Use our object generation for interactive participants; visual backgrounds can be captured meshes/Gaussians or code-generated Blender environments. Mark generated hidden regions as inferred, and estimated scale as unverified. Background reconstruction methods remain optional.

Reuse the shared reference/video, COLMAP/VGGT, PnP and camera/board tools when applicable; there is no separate scene copy of these tools. Camera/board tools require actual corresponding evidence and an explicitly justified frame. Per-object dimension/surface reviews belong to the object worker rather than an unscoped scene-wide dimension target.

For reference twins, review corresponding scene renders for each important
participant's proportions, placement, occlusion and support, using complementary
supplied views where they constrain the issue. Crop or isolate a participant
when the overview hides a defect, then return to the composed view. Successful
camera fitting on another object and successful isolated physics checks do not
close that participant's visual findings.

Retain the planning workflow's current-source boundary baseline for every
important participant after scene placement is applied. A successful self-authored
landmark/extents fit does not replace this review. Check each participant's local
and per-view differences before spending the remaining repair budget on isolated
physics or presentation that cannot change the reconstruction accuracy.

When a clear discrepancy persists, follow the diagnostic routes in
[plan-scene-generation](../plan-scene-generation/SKILL.md) and retain evidence of
the correction. Visual inspection identifies the conflict and selects a repair;
an unexplained parameter compromise does not resolve it. A manual correction is
a candidate until fresh corresponding-view evidence verifies its effect; when
available observations constrain the discrepancy, include a quantitative check
of that property. Prefer bounded fitting over repeated hand tuning. Select masks
by actual object identity and visible structure; a high model confidence or a
union of several detections cannot establish the intended boundary.

After geometry, placement or camera changes, recompute the affected projections,
residuals and overlays from the actual current mesh and camera inputs. Retain
their source hashes or exact parameter/camera snapshots. Old numbers may remain
as historical diagnostics, but cannot describe a changed candidate. Keep major
visual findings open until their own current evidence resolves them; a deadline
may leave an honest partial scene, not a verified twin.

Plan the backend for the whole interaction task with `plan_scene_physics`. Rigid/articulated, cloth and soft-body tasks default to PhysX. Fluid/strand tasks retain a Newton default for the shared task. Honor explicit user backend choices; missing capabilities require a compatible adapter, not a silent backend change. Request the relevant contact capabilities. Neither a Newton label nor adjacent PhysX/Newton assets establish coupling.

`build_scene_object` retains complete object generation/review runs; an unfinished asset ID can be resumed. `build_scene_background` exports editable visual Blender environments. Keep supports/collision proxies separate. Assemble complete verified bundles with rigid placement transforms. Never flatten joints, cloth bindings, fluid state or rods to satisfy a rigid import path. Backend conversion requests are real missing work.

Text-only concept scenes may use `reference_paths: []` for a participant; record estimated dimensions and avoid real-capture reconstruction claims. For environment appearance/layout review, optionally author up to four cameras and set `scene['real2sim_preview_background']=True`. The background tool returns their source-material previews to the agent. These are visual authoring feedback, not proof that operation participants or collisions exist. The final movie can use moving cameras and operation closeups; camera presentation never substitutes for actual physics or whole-scene acceptance.

The native Newton scene route can materialize retained rigid/articulated geometry, cloth topology/detailed bindings and hybrid attachments into one shared solver. It requires the corresponding native source files and a configured adapter. Inspect reported nonzero contact forces and pair coverage, not only successful import. Cloth/rigid checks, joint motion and attachment checks apply to their actual participants. Rope scene binding and unsupported renderer representations still require their own adapters; standalone success is not scene compatibility.

For a source-bound PhysX thin surface, use the [Sim 5.1 particle-cloth scene route](../../docs/SCENES.md#physx-particle-cloth-scenes-on-sim-51). It preserves the surface's passive settings and visual binding while sharing PhysX with rigid participants. Inspect retained deformation, support penetration, placement and actual motion; its collider-proximity witnesses are not force measurements. Honor its current single-surface and control limitations, or supply an extended adapter for the requested task.

Accepted liquid workers can return native fluid bundles for the same assembly/runtime tools. The MPM scene route transforms particles into scene metres and uses scene support boxes as its collision environment, replacing the standalone floor. It preserves source materials and returns actual liquid state, surface, support and video evidence. Its initial capability is fluids plus finite fixed axis-aligned support boxes; moving rigid/fluid and cloth/fluid interactions require additional verified capabilities. Do not put a table that should move into a static support just to satisfy this adapter.

An operator may configure multiple runtime routes once. The harness selects a route from the scene's actual capability requirements, including an optional video request. Use `plan_scene_physics` to reason about the whole task; do not supply runtime shell commands. Missing contact capabilities mean the requested interaction is not supported by the configured adapters. Preserve the assets and return the concrete gap so it can be repaired.

After `assemble_scene`, use `run_scene_simulation` with the operator-configured adapter. Review returned images and, when useful, videos through `inspect_video`. Check scene-level size/pose consistency, support/contact, duplicate background objects, holes exposed by motion, frame coverage, joint/cloth intersections and task functionality. Composition is not simulation verification; a passing program report is not your visual verdict.

When automatic framing misses relevant scene context, supply the manifest's `review_camera` (`eye_m`, `target_m`, optional `up` and `horizontal_fov_deg`). Retain a view that lets you assess the requested environment and interactions; a tight view of one object is insufficient to judge the whole room.

If rigid participants start at rest, distinguish missing contact evidence from
an actual contact failure. When direct reports are unavailable, the example
adapter optionally supports an explicit runtime-only free-release probe via
`--rigid-review-drop-m`; see `docs/SCENES.md`. Check that the displacement is
appropriate for surrounding obstacles and inspect the resulting motion. Keep
passive placement unchanged, retain the probe in the report, and preserve the
existing support-response criteria rather than accepting stationary objects as
proof of a successful drop test.

When the task requires rigid participants to rest on named supports, declare
`support_requirements`; read the
[per-relation contract](../../docs/SCENES.md#required-support-responses).
The harness requires a compatible runner and checks each declared relation against
explicit bounds. Contact elsewhere is insufficient. Choose release height for
the actual obstacles. Do not require support for participants that should be
moving, held or unsupported.

For useful continuous motion evidence, request `record_video:true` in `run_scene_simulation`. This requires a `scene_video` runner; a missing capability returns an error, not a substitute slideshow. Use returned video IDs with the shared `inspect_video` tool, selecting intervals, exact frames and crops as needed. Prefer this near final review of coupled motion or suspected transient defects; do not render a movie after every edit. Gaussian backgrounds can use the optional `prepare-gaussian-background` skill, while full mesh remains a valid branch.

For moving-camera movies, inspect stationary background surfaces over time as well as operation targets. Floors, tiled layers, rugs, walls and decals can shimmer from overlapping coplanar faces, unresolved detail, sampling or temporal denoising even when individual stills look clean. Use short adjacent-frame intervals or a registered background crop when useful. If layers share the same visible plane, repair the visual construction rather than filtering the movie to conceal the defect. A structural backing and decorative surface need distinct visible depths. Geometry/contact checks alone do not certify temporal rendering stability.

A visual-only repair can reuse physical states only after the adapter proves all physical participants/files, placements, supports, controls, solver settings and frame/timing unchanged and verifies the saved cache hashes. Reassemble and render the revised scene and obtain a new visual verdict. Changes to physical inputs require new simulation.

Repair the responsible layer and reassemble/retest when source or placement changes. `accept_scene` finishes only with current runtime evidence and your reasoned visual/functional verdict. Missing solver capabilities or version adapters block operable delivery; preserve the editable work and report the actual gap.
