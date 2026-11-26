---
name: twin-deliver
description: Review calibrated_scene_reconstruction task-zone appearance, backgrounds, physics task checks and delivery artifacts through S2–S6, then freeze a reconstruction before separate development evaluation.
---

For shared scene reconstruction use [the environment workflow](../reconstruct-environment/SKILL.md). This skill supplies its native procedures.

# Scene review and delivery

If review exposes a missing environment/physics/checking capability, use
[real2sim-evolve](../real2sim-evolve/SKILL.md) to validate and retain an extension.
An adopted procedure never replaces this session's native delivery evidence.

Read `qa/report.md` and inspect every applicable object and surface, including
those without automatic flags. Use `status` to identify missing, stale, rejected
or blocked stage/OBJ records. A successful tool invocation is not a passing review.

Review the assembled task zone in `qa/object_closeups`, support overlays, markings
and robot views. Compare shape, position, orientation and appearance with the
current capture. Confirm that scene assembly or clearance has not displaced an
otherwise correct fit. Resolve causes at the appropriate layer, regenerate the
affected outputs and append a new review. Defaults remain unchanged unless the
task input explicitly specifies alternative acceptance and scoped relaxations.
Use `scene.acceptance` to retain raw default failures and record separate
effective verdicts. A relaxation activates only after every required input
criterion passes with current native evidence; it cannot excuse invalid
material/contact states, stale inputs, native errors or changes to frozen assets.

S3 reviews the mesh/completed/splat background and is inapplicable in `zone` mode.
For splats, inspect held-out capture comparisons, coverage and free-view renders;
look for floaters, holes, doubled objects and seams with the task meshes. Optimize
background realism using these images, without changing frozen task geometry.
For `environment_quality_version: 1`, record the whole scene through
`python -m calibrated_scene_reconstruction.scene.review.environment SESSION REQUEST.json` before passing S3.
The request names this session's `render_review`, current `candidate_inputs`,
`kind` (`twin`/`cousin`), `backend` (`gs`/`blender`/`hybrid`), `checks` and
`findings`. Every criterion in the module's `CHECKS` requires a verdict,
explanation and images from the actual render review. Use multiple captured and
novel cameras. Cousins changed in shape/layout are judged for visual coherence,
not pixel equality with the original. Passing artifacts are hash-bound; changed
assets, renders or protected task/support/robot state invalidate the review.
For a spatial cousin or task-area clutter, use the [room depth and obstacle checks](../reconstruct-environment/references/experience/spatial-cousin-review.md) when reviewing the composed environment.
S4 reviews surface materials, clipped colours, baked objects/shadows and texture
repetition. Unseen construction/material choices must remain explicit inferences.

Write `configs/<name>/agent/task_check.py` for the declared task. It sets the end
state and checks it under physics using the task-check context. Run the task
check and inspect `qa/task_check.json` and its image. S5 needs a fresh passing task
result and applicable object/clearance evidence. The scene gate reads the result
file; it must not import task code. A failed task is investigated from capture,
geometry or collision evidence; do not resize a cavity simply to make it pass.

For mechanism/physical-prior captures inspect per-mechanism results and nominal,
low and high physical trials. All required checks and all three trials must pass
with the current source, mesh and physics fingerprints. Physical samples retain
their own motion/image evidence. Distinguish prior-only material estimates from
properties constrained by real observed motion. A successful task does not
identify real mass, friction or modulus. Export passive task assets: no animation,
kinematic pose control, active test drives or temporary task fixtures.

Every reconstruction pipeline validates the task objects themselves: insertion,
stacking, seating, stable contact and passive meshing as applicable. Grasp
planning, IK, robot reach and autonomous robot execution are not reconstruction
acceptance criteria by default. An explicit task input can require a particular
hand-level interaction. Then test that actual hand/contact mechanism, including
causal controls such as zero friction for a friction grasp; a generic object
force fixture cannot substitute for the requested fingers. Full autonomous robot
planning is required only if the input requests it. Use object-level fixtures or bounded guides when needed,
disclose them and inspect the released physical state. Rendering standpoints use
the captured/calibrated robot pose without operational placement checks. Robot
geometry and robot/table placement still need image/metric review.

New mechanism deliveries require `simulation.json` with `physics_hz`, native
`friction_type` and an explanatory `basis`. Checking and delivery use the same
profile; GPU flexible simulations require patch friction. Inspect the reopened
package's actual clock, contact model and SDF settings. An S6 review includes
`package_report: "qa/package_<asset-stem>.json"`; the gate binds that report and
asset bytes to the current task evidence and simulation profile.

Use `python -m calibrated_scene_reconstruction.agent.tool record --request review_request.json --output <dir>`
to append review rounds. The request contains `session` and `review` with `stage`,
`checks`, `review`, `findings` and boolean `pass`. All findings in a passing review
must be resolved. The tool chooses the next filename number and rejects overwrites.
The existing `calibrated_scene_reconstruction.twin` CLI remains compatible.

S6 requires all applicable S0–S5 gates and current OBJ rounds. Generate the tour,
USD and self-contained USDZ, inspect capture/free-view/closeup frames, and open
the USDZ alone to check dependencies and physics. Report options, scale/check,
observed sizes/placements, image metrics, task result, inferences and limitations.

Freeze the accepted source, geometry, poses and reviews before any independent
evaluation. Evaluate the exported geometry and keep the results separate from
model fitting, initialization, bounds and gates. Report image matching and
physical-size accuracy separately, including missing measurements.

Use `python -m calibrated_scene_reconstruction.scene.twin.freeze sessions/<name>` after S6, then
`--verify` to check immutable delivery files. The common freeze includes current
support and object rounds, materials, native physical-trial evidence, reviews,
USD/Z and visualizations. Never overwrite a freeze to accept later changes.
Recorded-contact visualizations use `apps/scene/render_task_trace.py` with a
current passing trace and its delivered USD/Z. Clearly label them as replay of
checked native physics; they are not an additional physical experiment.

For a workspace-only tour, `scene_tour.py --background zone --path auto` keeps a
continuous free camera aimed at the support and task objects. Review beginning,
middle, close-up and end frames; a successful encoding with an empty-background
pan is not useful delivery evidence. `--path free` requests that path explicitly.
moving part occlusion when it overlaps a transmitting body: real-time refraction can
incorrectly show the inner bottle surface through the moving part. Compare a path-traced
view before attributing that display error to geometry or physics.
