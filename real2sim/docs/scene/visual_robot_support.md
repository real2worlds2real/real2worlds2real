# Visual robot scale and Blender task supports

This route is under validation on independently initialized cup, peg and gear
captures. A completed optimizer run is not a calibrated reconstruction. Current
real-capture attempts can abstain; inspect `splat/robot_visual_fit.json` and S1.

Select `scale: "robot_visual"` in `configs/<name>/twin.json`, and write
`robot_visual.json` with `{"robot":"vega_1p_sharpa_arm","state_input":"none"}`.
Use the actual robot asset identity for other robots. This explicit route never
loads `robot_state.json`, board scale or an earlier reconstruction's fit. CAD
joint frames, limits and home values are legal model inputs; home is only one
optimization seed. The depth model supplies a provisional drafting coordinate
scale, not an approved metre.

S0 registers this capture and labels its actual floor and task support. At least
30 distinct reliable tracks are required per plane. When the primary image has
too few tracks, the pipeline requests masks on other already selected, registered
views and retries placement using the same session's COLMAP model. Review the
additional masks in `parsing/placement_plane_masks`. The support carving stage
uses those actual labelled tabletop tracks for its height, so nearby task-object
height clusters cannot masquerade as a second tabletop. Carving inherits this
choice from the completed capture alignment when no explicit plane setting is
provided; omitting `splat.json` must not revert to an unlabelled height histogram.
Occlusion labels include
visible fixtures/cables even when these belong to the background instead of task
meshes. This does not modify video
keyframe selection.

Both primary-photo and supplementary floor proposals are checked against
independently labelled support surfaces. A tabletop labelled as floor cannot
vote for the floor plane merely because it has many reliable feature tracks.
`qa/placement_plane_semantics.json` retains primary candidate overlap decisions;
if fewer than 30 true floor tracks remain, acquire other registered source
views. Original proposals stay intact. No desk-height target or CAD residual
selects which floor pixels survive.

S1 segments the robot, hands and independent occluders in existing registered
images. Contradictory robot/occluder pixels are unknown; views dominated by
ambiguous hand or occluder labels are rejected. Only actual tracked pixel
observations vote for a robot point: track length at least 3, reprojection error
below 1.5 px, at least 2 positive sightings and at least 60% positive votes. Actual positive-observation frame names are retained per track.
Per-image actual track counts are retained for review. Textureless images can
still constrain silhouettes; missing depth texture alone does not reject an
image. A scale-invariant sparse-component association first requires a dominant
body component containing at least 60% of labelled tracks. Separate components,
including a disconnected wheeled base, remain associated when they share at
least two actual observation cameras with that body. Repeated labels on a remote
device cannot establish the robot's identity by themselves. Ambiguous captures
without a dominant identity abstain. Before fitting, labels must overlap at least three independent robot-track
projections from this capture. These may originate in other cameras, so a
textureless image remains usable. This identifies the full segmented instance;
it neither clips the mask to CAD nor creates depth observations. Rejected false
labels and their identity support are recorded. Video/COLMAP frame selection
remains unchanged.

The shared backend transfers legal CAD triangles, kinematics and current-capture
observations between the USD and CUDA environments. A coarse 3D registration
provides a hypothesis, then differentiable rendering jointly estimates root XY,
yaw, scale and non-finger articulation within CAD limits. The fixed CAD root's
lowest surface rests on the observed floor. Home and intermediate articulation
provide additional starting hypotheses; no joint state is treated as measured.
Fingers remain explicitly unestimated; both image loss and sparse-track voting
ignore their labelled regions. Calibration occluders come from the current
support, task-object descriptions and visible clutter, so foreground task
objects are unknown evidence rather than holes in the robot. Broad instance
labels are replaced by the task's more specific semantic description when available.

`visual_render_pipeline SESSION prepare` writes immutable round inputs;
`visual_render_fit INPUT OUTPUT` fits five starting hypotheses in the recon CUDA
environment; `visual_render_pipeline SESSION assess --apply` verifies and applies
only a passing decision. One quarter of the labelled camera views are held out
from image gradients. The default image optimizer reserves sparse 3D observations
for independent validation. Rendered masks and all source inputs are fingerprinted.
Failed attempts and prior decisions remain available for review.
Robot and occluder observation revisions can be selected with `--robot-suffix`
and `--occluder-suffix`; write new masks rather than replacing earlier evidence.
For reproducible deployed runs, `robot_visual.json` (or `robot_calibration.json`
for uncertain commands) can set `observations.robot_suffix`, `robot_prompts`,
`occluder_suffix` and `occluder_prompts`. Prompt overrides are nonempty lists of
individual strings. `pipeline/run_session.sh` uses
`scene.capture.calibration_masks` to acquire these exact revisions, preserving
the old role defaults when no overrides are set. An optional `background_suffix`
also acquires independent background labels; `background_prompts` can override
the default scene classes. Change suffixes when revising the labelling problem;
cache reuse rejects changed prompts. These settings alter semantic evidence on
already registered frames, not video/COLMAP frame selection or acceptance gates.
Named identity/occlusion mistakes can be rejected through
`observations.rejected_frames: {"frame.jpg": "raw-image reason"}` or repeatable
`prepare --reject-frame 'frame.jpg=raw-image reason'`. The same review excludes
actual track votes and whole silhouette views, preserves the masks and records
every reason in the immutable input manifest. Unknown frame names and empty
reasons fail. Do not select rejections from CAD fit errors or dimension error.
An experimental `--background-suffix` uses independently segmented scene
background as negative evidence inside the observed robot's image hull. Unlabelled
interior pixels remain unknown, never invented foreground. Views with over 25%
unlabelled interior relative to robot area abstain. The exterior silhouette and
independently labelled openings remain constrained. This observation policy uses
no projected CAD and requires separate raw-image review; it is not enabled by
default or approval of earlier failed fits. External robot power adapters are
treated as independent occluders rather than CAD geometry.
A full-circle root search initializes scale/XY/yaw in the current images. Rejected coarse3D
articulation falls back to legal neutral/home hypotheses. `--resume-from` retains
separate rendered lineages only within this capture and with identical
observations/CAD. Image losses weight each view by its valid observed robot pixels, preventing a
tiny ambiguous fragment from dominating full-body observations. All image gates
and held-out camera identities remain retained. Learning rates decay during refinement. Exact engine sources
are archived with the prepared inputs; completed decisions remain auditable after
a software update.
An optional `--contour-weight 1` adds an observed-mask distance-field objective,
normalized by observed image extent. It supplies a pose direction even when
silhouettes do not overlap. It changes fitting only, not image/depth acceptance;
it remains experimental and disabled by default pending cross-capture results.
`prepare --validate-approved` and `assess` test a new
backend without replacing the approved anchor; validation cannot apply a scale.
`bind-validation --validation <proof.json>` can certify that unchanged anchor
only when a passing, fingerprinted hypothesis matches its complete pose and scale
exactly. A nearby optimized hypothesis cannot certify the currently rendered
pose. The resulting certificate binds all new evidence without changing scene
coordinates; subsequent anchor checks reject stale validation inputs.

The shared criteria require median actual-CAD silhouette IoU at least 0.85,
local difference at most 0.02, at least 8 labelled views, and held-out IoU at least
0.85. For `depth_observed_v2`, CAD samples need actual validation-depth track
observations in at least two cameras, in addition to facing the camera and being
labelled visible. The projected depth-support window is the unchanged 20 mm
criterion expressed in each camera, not a candidate-dependent 3D distance test.
Their trimmed residual must be at most 20 mm, with at least 30% support and at least
two independently supported links. At least 100 depth-observed CAD samples are
required. Untextured white surfaces have no sparse depth observation; the complete
image silhouettes still constrain them. All-visible/sparse coverage is retained
separately as a diagnostic. Packages without actual track sightings retain the
older `visible_v1` metric scope; never compare these percentages as one metric. Hidden CAD undersides are
not sparse-map errors; the full observable silhouette is checked separately.
At least two plausible starting hypotheses must pass every gate, and their scale
spread must be at most 5%. This spread is an ambiguity diagnostic, not a bound on
true dimensional error. S1 also requires explicit agent review of camera overlays
and robot/table placement. Failed fits keep alignment unchanged. An applied anchor
cannot be reapplied; start a new session for independent calibration. Never use true
task/table sizes as a fitting objective or a gate.

`robot_visual` defaults to `support_route: "agent_blender"`. Other scale routes
can opt in explicitly; existing sessions keep their carved support route. After
`support`, the new `support_model` stage requires agent source at
`configs/<name>/agent/support/<support>/model.py`, referenced by
`furniture.<support>.fit.model`. Missing source writes
`qa/support_model_request.json`; it does not silently select a template.

The support uses the same literal `PARAMS`, `FIXED`, and `build(params)` contract
as task objects: metres, z up, bottom bounding-box centre at the origin. Model the
visible tabletop, edge, thickness, legs and braces; disclose hidden construction.
The highest horizontal surface is the task surface. Initialize from this capture's
calibrated images and observed plane. Shared material slots preserve the distinct
finishes of Boolean-unioned parts. A generic 10 m extent limit permits furniture;
task objects retain their 2 m sanity limit. Neither limit is a size-accuracy check.

`splat/support_carve.json` also records `oriented_footprint` (`xy`, `size`,
`yaw_deg`): a minimum-area rectangle of the observed carved region in metres.
It is an initialization hypothesis. Occluded boundaries or stray connected
regions can distort it; inspect the masks and fit the full support instance
before approving its dimensions or pose.

Use `support_blender_run`, `support_fit`, `support_export`, or
`support_refine_run` through `python -m calibrated_scene_reconstruction.agent.tool`. The fit uses existing
support and occluder masks in at least 8 metric cameras, optimizing parameters and
XY/yaw.
An optional `furniture.<support>.fit.initial_pose` can continue from this capture's
own observed pose; it cannot expand the existing XY/yaw search bounds. The exact
seed is retained in the immutable round input manifest.
Declare texture files in `fit.assets` using config-relative paths under the
model's directory. Each round copies those files beside its archived source,
preserving relative paths. Both originals and copies are fingerprinted; source
code can load them relative to `__file__`, and GLB image textures remain exported.
`scene.support.observations` associates the full support instance using its
observed plane position, rather than unioning neighbouring same-class desks.
Optional `furniture.<support>.fit.observation_parts` contains observed semantic
part descriptions, such as a table leg. These are associated by attachment and
floor/footprint evidence independently of the candidate mesh. Every accepted
and ambiguous observation is recorded. Background-mask dilation is undone
before geometric fitting when legacy masks are used. Optimization weights each
view by its valid observed support pixels; all selected views and the unchanged
median acceptance criteria remain in the final evidence. Every round keeps source, input hashes, meshes, GLB, previews and exact
silhouette evidence under `support/<support>/round_<n>/`. Numerical criteria are
median IoU at least 0.85, local difference at most 0.02, and model top within 10 mm of
this capture's observed support plane. This last criterion compares reconstruction
evidence, not measured furniture height.

Selected part masks are retained separately and fingerprinted. Cropped or
near-boundary leg endpoints do not establish floor contacts. Optional
`fit.part_materials` maps an observed part description to a Blender material
name, using transferred Boolean face-material identities. New rounds also check
each declared part separately: median positive recall must reach0.85 and median
outside-support pixels, normalized by observed part pixels, must stay below0.02.
Absent part evidence fails. Actual whole-support ray visibility keeps hidden
legs from explaining visible labels. These are image checks, not floor-contact
or stability certificates. Set `fit.part_acceptance_version: 1` when continuing
an older support so its previous aggregate-only round cannot satisfy the new
request. Existing frozen rounds are not retroactively presented as having these checks.

`fit.part_loss_weight` (default1) changes only search weighting, never part gates.
Search weights part views by observed positive pixels so a nearly occluded
fragment cannot dominate through a tiny denominator; final per-view evidence
and unweighted medians remain. `fit.optimize_pose`
(default true) can hold the capture's own pose during a component pass. These
policies are bound in the round manifest and checked during export verification.
`fit.optimize_parameters` can select a subset of declared shape parameters.
Optional `fit.search_starts` (1..12) uses deterministic initializations within
the declared bounds; `fit.search_width` (240..960) changes proposal resolution
only. Final CPU checks always use960. Interrupted searches retain an explicitly
unapproved `fit_progress.json`; optimizer convergence never establishes approval.

Before fitting, review source instances. An optional `fit.part_identity_review`
names a config JSON with `version: 1`, `support`, `observations_sha256` and
`exclusions`. Each exclusion names an archived `frame`, `mask`, `mask_sha256`,
`source_sha256` and source-RGB `basis`. A wrong-instance mask becomes unknown
in both whole and part evidence, never background. Candidate errors must not
select excluded pixels. Changed RGB, masks or observation manifests invalidate
the review. This correction is not a whole-object or support approval.
`fit.optimize_shape: false` additionally holds the literal source initializer
dimensions. With both shape and pose held, one actual Blender evaluation records
a material-only round and recomputes the unchanged image/plane gates. Initialize
only from this capture's own reviewed round and verify geometry equality before
accepting an appearance-only revision.

For incomplete support labels, an experimental opt-in
`fit.known_background_suffix` names an independently segmented floor/wall mask
revision. Generate it with `scene.background.splat.masks --prompts floor,wall
--inside 0 --suffix <revision>` on this capture. The fitter undoes the recorded
mask dilation; conflicting positive/negative labels and unlabelled pixels are
unknown. A view needs at least200 actual negative pixels and100 support pixels.
Views with contradictory labels over25% of the visible support are rejected.
At least8 views remain necessary. IoU/local thresholds remain0.85/0.02, evaluated
on the explicitly labelled domain. Every round reports domain pixel counts and
binds the policy and background masks. These metrics have a different observation
scope from the legacy full-frame complement and must be labelled accordingly.
Inspect the masks and rendered geometry: inferred structure in unknown pixels
has not been measured or validated by a high IoU.

Review the evidence and record `SUP.<support>` separately. S2 and S6 require its
current passing round. Changing source, observations, export pose, task-contact
box, mesh or appearance invalidates that review. Runtime replaces the old support
parts and box collider with the exported Blender visual parts and an identical
static triangle collision surface. Tables are not authored as dynamic bodies.
Review object/table contact and robot/table pose in the rendered scene before
physical task checks and delivery. Frozen sessions reject support iteration.
