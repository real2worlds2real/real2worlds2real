---
name: simulate-mating-rigid-parts
description: Test interacting concave rigid parts with native SDF mesh contacts, explicit external wrenches, and measured motion evidence. Useful for threaded closures, screw pairs and contact-dependent insertion or release.
---

# Contact-dependent rigid mechanisms

Use this optional capability when the actual mating surface matters. A revolute joint alone cannot demonstrate a screw advancing through threads or disengaging. Keep the original editable Blender geometry and material assumptions; derive collision meshes from evaluated source geometry rather than drawing a second unrelated asset for the test.

`run_rigid_contact_behavior(request_path, assumptions, render_video, record_asset_evidence)` accepts explicit closed rigid meshes and external test controls. Read [the request contract](request.md) when preparing a test. It returns native Newton XPBD/SDF states, measured checks, fresh-process collision-mesh replay and, optionally, a registered movie for `inspect_video`. Inline-only replay uses diagnostic materials. The optional source-attempt binding replays the saved source materials; read the contract before using that branch. With record_asset_evidence=false, both branches remain diagnostics. For a rigid asset, record_asset_evidence=true explicitly selects the source-bound native SDF review for final acceptance; it requires source binding and a complete video. The ordinary PhysX route remains the default.

A selected native asset review retains source geometry, the actual trajectory, original-material video and a passive runtime bundle inside the saved attempt. Source/cache/video changes invalidate that evidence. Complete render_final, inspect the returned evidence and use accept_result for final delivery; recording evidence does not itself accept the asset. render_motion_video exposes the retained native movie, and inspect_surface(frame=...) can inspect the actual native state from another view, including a removed cap interior.

Source-bound appearance review loads the retained Blender source and replays its
materials. The accompanying GLB is a preview; it does not certify preservation of
procedural bump, coatings or other Blender shading in another renderer. A task
that requires target-renderer delivery must also establish that renderer's
appearance. Custom exporters must retain and inspect the relevant material
graphs/textures rather than exporting only color/metallic/roughness and claiming
the original finish. A successful contact test does not resolve shape or finish
differences from the reference.

The loader is passive by default. Drives are explicitly external test fixtures, not hidden asset motors. Explain which degrees of freedom a fixture holds and which the contact solver determines. In an engaged screw test, an angular drive and lateral/tilt stabilization may be useful; imposing axial position as a function of angle would defeat that test. A later independent bounded lift can test removal. If a fixed-duration spin continues after disengagement, the demo can imply too many required turns; distinguish engaged rotation from free rotation. An optional contact-loss stop can end angular drive after observed contact disappears for a dwell, while preserving the independent lift. See the request contract.

For a stalled engagement or failed disengagement, read [thread release](thread-release.md)
for geometry and control diagnostics.

For threads, consider checking:

- Real male/female surfaces, matching handedness and lead, finite ends, clearance, accessible cap interior, and mesh resolution relative to flank size. Lead means axial travel per revolution; for a multi-start thread it is not the spacing between adjacent ridges.
- While engaged, measured axial travel versus unwrapped relative rotation, in a declared interval that excludes initial seating and disengagement. A good residual alone does not prove physical contact.
- Use [review-numerical-evidence](../review-numerical-evidence/SKILL.md) for the relative-pose audit, naming moving/reference bodies and quaternion order. Inspect the full trace for later escape even when fitting only the engaged interval. Inspect evaluated triangulation of concave end faces: a concave n-gon can fill a required opening unexpectedly.
- Useful negative controls: pull while rotation is held; spin with the helical ridge removed; reverse rotation if the end stops permit it. These are optional experiments chosen for the mechanism, not a mandatory fixed sequence for every object.
- Release after the actual thread ends clear; continuous motion and no visible interpenetration. Merely disabling a joint on a timer is not evidence of contact-driven disengagement.

For difficult mating contacts, inspect the actual video, including the approach, engagement and release, plus close crops or another view where needed. Numerical success does not override visible defects. When reducing video FPS, retain sufficient physics substeps: output sampling rate and the contact integration timestep have different purposes. Fine flanks and stiff grasp gains may need a smaller timestep. Refine geometric/SDF resolution or the grasp fixture when warranted; do not hide collision surfaces or relax checks merely to pass.

This capability supplies generic rigid contact experiments and an optional source-bound rigid delivery route. It does not supply an automatic bottle template, measured torque calibration, a universal fastener standard or a robot policy. Native Newton runtime support is separate from a collaborator's Isaac Sim version. A PhysX implementation needs its own compatible SDF/contact adapter and checks; successful Newton motion is not PhysX validation.
