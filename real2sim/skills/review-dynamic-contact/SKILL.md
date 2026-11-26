---
name: review-dynamic-contact
description: Diagnose transient interpenetration, exposed internal parts, or separation between physics and detailed appearance during an asset's simulated motion.
---

# Review dynamic contact

A finite trajectory, successful joint tracking and small attachment gaps do not establish collision correctness. Sparse motion thumbnails can miss short-lived crossings. When contact is uncertain, consider both sides and relevant stages of the operation; a good endpoint cannot override a demonstrated defective intermediate pose. Choose the viewing intervals and detail that resolve the uncertainty.

For rigid thin-wall or cavity operations, read
[rigid-contact.md](../review-simulation/rigid-contact.md). A final body can be
clear of material after tunneling through a barrier; inspect intermediate
contact and applicable containment or floor-ordering evidence too.

Temporal quality matters independently of collision checks. Persistent wind-like flutter during a smooth lift or stationary hold, abrupt local buzzing and repeated shape pops require explanation or correction even if the mesh stays finite and connected. Check what forces and controls were actually configured; do not invent wind or deliberate shaking to excuse unexplained motion. `review-simulation` describes optional hold diagnostics and contact-scale/damping checks. Real-world parameter fitting is separate from rejecting visibly unnatural numerical behavior.

For recurring lift/hold instability, read [lift-motion.md](lift-motion.md): investigate task records and source directly, compare free-region trajectories with the actual controller, and distinguish contact scale, constraints, solver assumptions and replay. These are general investigation criteria, not a dedicated object-specific tool or a mandatory fixed checklist.

For hybrid assets, `run_hybrid` now checks every cached frame for transverse triangle crossings between rigid colliders and both the control surface and evaluated visual followers. Read `surface_intersections.json` in its returned directory. It identifies the frame/time, named surface/rigid pair, and world-space witness points. These are surface crossings, not signed penetration depths. The check does not certify continuous-time contact, containment, cloth self-intersection or unseen visual details. Use local images to decide what a reported intersection means.

Locate the cause before revising:

- Crossings already in the authored reference pose: inspect rest geometry, collision thickness, sewing anchors and rod/cloth clearance. Stronger constraints cannot repair an initially contradictory contact layout.
- Control surface crossings during motion: inspect collider coverage, time step, contact sampling and features thinner than control-mesh cells. Particle-only contacts can miss a rigid edge through a triangle interior. The hybrid backend supports full-surface rigid/soft contact using rigid SDFs; its voxel size must resolve thin features. DAT truncation is separately selectable and is not a global nonpenetration guarantee.
- Control surface clear but detailed surface crosses: inspect subdivision shrinkage, thickness direction, binding/rest pose and transform replay. Raising the physics resolution alone may not fix the visual follower.
- Cloth and rigid poses disagree between target renderer and Blender: diagnose coordinate conversion/replay before changing the source model.

Keep material clearances, collision radii and solver settings editable. Do not hide structural rods, remove essential colliders, recolor artifacts into the cloth, patch rendered frames or disable the meaningful motion test to obtain a pass. A legitimate opening or sewn interface should be modeled explicitly; blanket exclusions between the entire cloth and mechanism conceal failures.

For final movie review, time-indexed sequences and full-resolution frames/crops are available. Choose the temporal coverage needed for the asset; short transient defects may warrant dense intervals or complete coverage. There is no fixed mandatory viewing recipe. Contact sheets are an image representation of a movie, not native video playback. Do not claim to have reviewed a final movie when only a few pre-render samples were returned. Geometry checks and visual review complement each other; neither alone certifies all contact behavior.

Return detected defects to the same generation session with source/cache provenance, timestamps, part names, images and the failed check. Preserve previous evidence, correct the source or demonstrated integration defect, rerun the simulation, then review the new rendered operation before accepting it.

For complex dynamic candidates, the optional `render_motion_video` tool renders the selected retained trajectory and returns it to the common `inspect_video` reader. Read `inspect-video` for overview, dense interval, exact-frame and crop requests. Use full movies when they materially help final review; intermediate numeric/geometry checks remain cheaper. Review newly rendered motion again when the source or trajectory changes.
