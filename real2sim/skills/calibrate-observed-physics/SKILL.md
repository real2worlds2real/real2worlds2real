---
name: calibrate-observed-physics
description: Task-level workflow for physical alignment to observed motion, from observation and experiment design through target-backend trials, validation and parameter writeback. Use for standalone calibration or a requested follow-on to object/scene generation.
---

# Calibrate observed physics

You choose the hypotheses, parameters, next experiments and stopping criterion.
Use this task workflow when the user asks for physical alignment, whether the
asset already exists or is being generated. Read the shared
[working methods](../work-with-task-evidence/SKILL.md) at startup or continuation.
Ordinary asset generation does not require measured material parameters.

For an existing usable asset, begin with the observation and target runtime;
there is no required regeneration step. If geometry or functional structure
must first be built or repaired, use [generate-editable-object](../generate-editable-object/SKILL.md)
or [plan-scene-generation](../plan-scene-generation/SKILL.md) for that work, then
resume calibration. Keep observations, trials, validation and the selected asset
configuration in the task workspace. The native experiment interface below is
the standalone execution entry; a generation-task CLI mode is not required.

When selecting observation tools, [segment-reference-views](../segment-reference-views/SKILL.md) provides optional masks with explicit tracking limits, [measure-reference-geometry](../measure-reference-geometry/SKILL.md) handles supplied scale/calibration evidence, and [recover-multiview-geometry](../recover-multiview-geometry/SKILL.md) can register a consistent static scene. Use [inspect-video](../inspect-video/SKILL.md) if its frame/interval interface helps review motion. For suspected interpenetration or unstable contact, consult [review-dynamic-contact](../review-dynamic-contact/SKILL.md) before interpreting the residual as a material-parameter error. Read only the guidance applicable to the experiment and available evidence.

The framework keeps observation, experiment evidence, validation and delivery
contracts; you decide the iterations within them. Write case-specific motion
equations, estimators and experiment code in the task workspace when useful.
Reuse an existing numerical implementation when it fits the hypothesis. You do
not need to match a predefined motion category. Keep case algorithms in the task
workspace.

Choose the experiment backend from the target asset's intended runtime. A PhysX
asset should be calibrated on PhysX; a Newton asset on its Newton solver. Another
demo's material types or current compute availability do not determine this choice.
Set `experiment.target_backend` explicitly. Shared observations and comparison
tools do not imply that fitted coefficients transfer unchanged between solvers.

Read [the experiment interface](../../docs/PHYSICS_INFERENCE.md) when preparing
tracks, trial callbacks or optional searches. Existing camera/metric tools,
segmentation, native video inspection and simulator interfaces remain available.
Inspect raw records or write analysis code whenever that gives better evidence.
Use native file/code access for queries. Choose tools for useful computation or
execution, and write task-local analysis code when the existing interfaces do
not fit the experiment.
For existing tracker archives, masks or measured video landmarks, optionally use
`prepare_motion_observations`; read [motion evidence](../../docs/MOTION_OBSERVATIONS.md)
for exact frame/time/pixel conventions. This consumes perception results without
mandating a tracker. Match the same observable in reality and simulation: visible
mask centroid, projected body origin and persistent surface landmark differ.

For registered pixel tracks with a justified motion plane, optionally use
`project_motion_to_plane`; read [motion geometry and fitting](../../docs/MOTION_GEOMETRY_FITTING.md).
It reuses camera/scale evidence and preserves missing samples. Do not project a
raised object's image centroid onto its support plane as if it were a contact
point. An optional caller-supplied slide fitter can suggest initial friction and
speed, retaining competing stop explanations and uncertainty. It does not select
your experiment or replace target-simulator validation.

For following or moving-camera recordings, read [camera-motion separation](moving-camera.md)
when registration or target-pose recovery needs attention.

## Establish what the observation constrains

- Preserve original timestamps and playback-rate evidence. Distinguish camera
  motion from object motion. Image tracks need a common registered view; metric
  tracks need justified scale and coordinates. Static-scene camera recovery does
  not establish the trajectory of moving objects.
- Separate measured, missing, interpolated and predicted samples. Occlusion is
  not zero displacement. Mask overlap with a hand is a contact hypothesis.
- Identify external controls versus the target's response. Replay a hand/tool or
  gripped boundary when supported; simulate the target's free response. A copied
  target trajectory is not parameter validation. Free-release intervals can be
  useful without reconstructing every preceding manipulation.
- Select only parameters supported by the experiment. Free sliding usually
  constrains effective friction, not mass. Keep unidentifiable parameters as
  priors/ranges; do not infer absolute stiffness from unmeasured loading alone.
  Force/torque measurements are optional future evidence, not prerequisites.

## Run and diagnose

Choose a next experiment for the ambiguity it can resolve. A useful diagnostic
changes one plausible cause while holding the other conditions fixed, and checks
the predicted direction of the effect rather than just whether a score improves.
If several parameter settings explain the same motion, seek another initial state
or excitation that separates their predictions. Retain priors or ranges for
quantities the available recordings cannot distinguish.

Select observables that expose the proposed mechanism: an endpoint can miss
incorrect intermediate motion; contact timing, both participants' trajectories
or rotation can reveal different failures. Compare residuals before and after
events, and test initialization, geometry and measurement hypotheses alongside
material parameters. These are experiment-design options, not a required sequence
or a list of supported motion categories.

Propose a candidate yourself and call `run_trial`, or run your own simulator
experiment and call `compare_physics_observations`. Search is optional: bounded
scans or an ask/tell optimizer can help after selecting useful variables and
bounds. No fixed optimizer, motion taxonomy or iteration count is required.

Keep the initial state, external controls, geometry, camera, clock and numerical
solver settings fixed for a comparable trial series. If these need correction,
retain a new experiment rather than attributing their effect to material physics.
Check effective parameter bindings, contact-pair combination rules and runtime
version. A declared requested parameter is not proof that the backend used it.

Existing native runner callbacks provide initialization, per-step controls and
observation; you can author the experiment directly. For timed changes such as
release or drive switching, optional `interventions` plus `apply_event` execute
once on integration boundaries and retain `actions.jsonl` with timing/readback.
The callback defines the physical action; its successful return alone does not
prove the intended contact change occurred. Read the experiment interface for
callback signatures and timing. Inspect records directly; no query tool is needed.
CPU execution is appropriate only when it preserves the intended model and
numerical settings. Leave GPU-dependent perception/rendering/runtime checks for
the target hardware; simplified tests do not replace final-backend validation.

Inspect residual curves and the corresponding video intervals. Distinguish wrong
geometry, controls or contact conditions from wrong physical parameters. Do not
warp time to hide wrong dynamics. Consider alternate hypotheses when several
parameter sets fit equally well. Use a different observed motion/interval for
held-out validation when available; label same-sequence fitting honestly.
Signed residuals are simulation minus observation in the declared coordinate
order; per-axis bias and RMSE can help localize disagreement without selecting
its cause. For several experiments, optional native `evaluate_candidate` executes
one fixed candidate and retains fit/validation results separately. You choose
the experiments and their independence evidence. If validation results guide
retuning, they are now development evidence, not an untouched held-out check.

## Align complex interactions

For particularly complex interactions, prioritize the main objects' behaviour
and the important visual outcomes identified from the reference and the user's
goal. Assess the overall motion and interaction sequence, rather than requiring
every secondary object's trajectory or final pose to match exactly. Review the
rendered result alongside motion measurements so that a good trajectory score
does not hide a major visual mismatch.

When many uncertain parameters could affect the result, use reasonable priors to
fix a subset initially, recording them as assumptions rather than measurements.
Identify the parameters most likely to affect the main behaviour, then adjust
one at a time while holding the others fixed to understand each change's effect.
Revisit the fixed assumptions when the evidence calls for it. This is a practical
experiment-design option for the agent, not a prescribed parameter order or a
mandatory optimizer.

Keep physical validity distinct from visual alignment. Check for significant
interpenetration, missed collisions and unstable contact; a matching main
trajectory does not establish that the interactions are valid. Sudden jumps or
rapid jitter warrant investigation, but are not automatically numerical errors:
inspect actual collider geometry, contact records and state changes to distinguish
physical impacts or applied controls from solver artifacts. Assess penetration
relative to the geometry and solver tolerances rather than assuming every small
overlap is a failure. Resolve numerical defects before interpreting parameter
changes as better physical calibration.

Record the integration step and solver settings with the selected candidate.
The PhysX scene adapter supports an optional `physics_hz` request when its route
declares `configurable_timestep`; see [scene execution](../../docs/SCENES.md).
Read back the actual step. A smaller step passing contact checks does not by
itself establish trajectory convergence. Test the composed delivery assets:
an analytic experiment primitive and an exported mesh approximation can respond
differently despite matching nominal dimensions and material coefficients.

## Conclude with evidence

Report estimated values, units, assumptions, remaining ambiguity and supporting
trials. A numerical score is not final acceptance. Apply selected parameters
through the asset's editable source/configuration, rerun current target simulation
and review the result. Earlier acceptance does not certify changed physics.
For repository-generated assets, return to the corresponding object workflow
or [compose-interactive-scene](../compose-interactive-scene/SKILL.md) for current
source-bound acceptance. For an externally supplied asset, retain an equivalent
target-runtime check tied to the actual written configuration; do not claim a
repository acceptance receipt that was never produced.

Current implementation supplies track/event comparison, recorded native trial
callbacks, Newton and PhysX parameter/stepping adapters and an optional optimizer
bridge. Newton models, controls and observations can be authored by you; read the
experiment interface for the native adapter. The Newton and PhysX adapters provide optional runtime checks for parameter sensitivity and reset. Retain actual checks on the selected runtime and asset. Coupled PhysX mass properties can be authored into assets; live scalar
mass updates and arbitrary tracking are not supplied by that interface. Per-case
tracking or estimation may be agent-authored and checked against independent
observations. Do not claim arbitrary-video calibration is validated merely because
these interfaces are available.
