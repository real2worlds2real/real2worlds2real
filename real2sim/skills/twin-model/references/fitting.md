# Source-grounded model fitting and evidence

Maintained reference for [the modeling skill](../SKILL.md). Apply the sections relevant to the current representation and task; preserve all applicable local gates.

The default reconstruction route is agent-authored Blender geometry. If Blender
is unsuitable for a particular part, choose a more appropriate geometry/CAD
tool or native representation and record why. Generated geometry can be
assembled/imported through the model's declared, snapshotted assets; the
iteration harness does not require every surface to be constructed with
Blender operators. Do not treat an unavailable backend as implemented. Final
simulation and task feasibility always use Isaac Sim5.1 with the actual
delivered geometry, contacts, joints and deformable representation. A result
from a different simulator or a visual animation cannot replace those checks.
Missing
source creates `qa/model_requests.json` and stops. Templates require explicit
`--object-route template` or per-object `fit.route: "template"`; a legacy `fit.kit`
alone does not select fallback. Never silently replace a failed model with a kit.

For `twin.support_route: "agent_blender"`, include the task support in this loop.
Write `agent/support/<support>/model.py` and set `furniture.<support>.fit.model`.
Use `support_blender_run`, `support_fit`, `support_export` or `support_refine_run`.
Review visible top/edge/thickness/legs, the observed plane, object contact and
robot/table placement. Record `SUP.<support>` for the current exported round;
S2 and delivery require it. See [the support contract](../../../docs/scene/visual_robot_support.md).
Trace visible structural connections on full-resolution source images before
assigning a foot, post or brace to the support: a robot base or neighbouring
chair can be directly behind it. A frame resting on a tabletop can also cause
semantic labels to omit the tabletop outside the frame. Preserve source labels
and review the complete support; use independently acquired background labels
for an explicit observation domain when appropriate. A high score in unknown
regions does not validate hidden construction.

Write `configs/<name>/agent/objects/<object>/model.py`. Its literal `PARAMS`
dictionary contains finite `init`, `lo`, `hi` for each fitted parameter; literal
`FIXED` contains disjoint structural/material priors. `build(params)` creates the
solid in Blender. `OPENING(params)` optionally declares local `xy`, `rim_z`,
`depth`, `radius`. Coordinates are metres, z up, origin at the bottom bounding-box
centre. The source runs only inside Blender. Inspect the actual contract at
`src/calibrated_scene_reconstruction/scene/objects/blender/contract.py` before using a new feature.

For staged iteration, optional `dynamic_objects.<object>.fit.optimize_parameters`
selects distinct `PARAMS` names for that numerical round. Omit it to optimize all
parameters; `[]` fits pose while keeping the declared initial shape. Other
parameters retain their source `init` values. The round stores the selection and
complete resulting parameters. This can separate costly flexible initialization
from pose fitting, but every round still evaluates the complete image evidence
with the same thresholds. Fixing parameters is not geometry approval: revise
the source or parameter selection when rigid or flexible accuracy fails.

Optional `fit.assets` declares config-relative data files under the model's own
`assets/` directory. Load them relative to `__file__`; each fit copies them beside
the source snapshot and fingerprints originals/copies. Changes invalidate that
round. A native relaxed soft-body candidate can be stored this way as initial
geometry; it remains subject to image/material/contact/task checks after rebuild.
`tasks.mechanisms.initialization.relaxation_trace` records solver states without
prescribing cloth positions or applying cloth forces. It never approves a task
and must not be used inside a task check to hide an invalid captured initial state.

For an observed contact-connected assembly, optional
`fit.mask_component_gap_pixels` (0..8) associates later prompt parts with the full
first-part segmentation. Order `fit.prompt` with the semantic body first. This
retains body fragments separated by occlusion and rejects disconnected same-class
reference items. Inspect raw/component-selected masks; the small gap represents
segmentation uncertainty, not a fitted geometry tolerance. Do not use this policy
for observed detached parts. Centroid-ray disagreements are review flags on the
agent route; they cannot automatically remove Blender accuracy observations.

Author geometry from this capture: holes, cavities, handles, corner bays, thin
plates and contact surfaces. Bounds and initial pose come from calibrated image
evidence or explicitly stated priors. Neither true dimensions nor task-test
success may set a hole diameter or adjust an unseen floor to force a pass.

Acquire observations with:

```bash
python -m calibrated_scene_reconstruction.scene.objects.object_fit sessions/<name> --from-config --observations-only
```

Fit through `object_fit` in `calibrated_scene_reconstruction.agent.tool` (or legacy `calibrated_scene_reconstruction.objects.tool fit`),
using a JSON request containing `session`, `object` and optional
`max_evaluations`. Read `python -m calibrated_scene_reconstruction.agent.tool schemas` for exact contracts.
Each round stores a source snapshot, inputs, parameters, mesh, previews and
evidence in `objects/<object>/round_<k>/`.
Before local shape refinement the current model is compared at24 full-circle
yaw hypotheses against all cached observations. The selected pose and every
orientation score are snapshotted in `initial_yaw_hypotheses.json`. This prevents
an elongated body from shrinking to explain a bad initial direction; it changes
initialization only, retaining every observation and the same acceptance gates.

Inspect the full per-frame evidence, worst views, Blender previews and opening
overlays. A whole-object silhouette does not establish cavity size or unseen
depth. Distinguish an occluding hand from missing object geometry; retain the
observation set when comparing model revisions. Numeric acceptance stays median
IoU ≥ 0.85, median local difference ≤ 0.02 and at least eight frames. Optimizer
convergence is reported separately. A visible wrong feature cannot be explained
away as a limitation merely because the numeric gate passes.

For new and revised models, list source-visible structural features in
`agent/modeling_plan.json` under `physical_instances.<object>.geometry_features`.
Declare matching `fit.geometry_features` as `{id, description}` entries (or
inherit the list from the plan), and name a session QA
`fit.geometry_review` file. Inspect neutral, untextured geometry from captured,
novel and task-operation poses, in addition to textured source comparisons.
Texture cannot substitute for layered petals, slats, handles, cavities or other
visible structure. A silhouette envelope is an initialization, not a finished
semantic model. Separate observed detail from inferred hidden construction.
`scene.objects.structural_review` blocks export/OBJ acceptance until every
declared feature has current source/neutral-view evidence and an explicit agent
verdict bound to the model and mesh. It verifies the review's provenance, not
automatic semantic recognition. Missing structure must fail even when insertion
or silhouette metrics pass. Rebuilding it requires fresh physical validation.
Record a separate `view_checks` verdict and reason for every neutral/novel/task
view. A convincing front view cannot override a wrong profile or operation pose.
When a source-shaped envelope deforms detailed geometry, inspect its interior
features as well as the outline; matching the envelope can still elongate a
central feature or flatten layered surfaces. Keep such a candidate unaccepted.
Check local feature directions against enlarged raw source crops: a part may
exist yet be reversed, folded to the wrong side, or attached at the wrong place.
Record observed directions separately from hidden construction priors. Do not
assume a familiar class template's orientation from the object name alone.

For visually distinct regions within one rigid object, declare
`fit.source_regions`: each entry needs `id`, source `prompt`, exact Blender
`materials`, source-only `part_review` and hash-bound `evidence_files`.
Use independent source labels to check each region's extent and boundary during
fitting and acceptance, without inventing separate physical links. A passing
whole-object silhouette must not conceal a wrong head/shaft or handle/body split.
Review the raw labels before fitting; never adjust them to make the model pass.

The shape optimizer weights views by their native observed object pixel counts;
crop resizing must not make a small distant observation outweigh a sharp close
view. Acceptance medians still include every reviewed cached view without this
weighting. Preserve the per-round engine snapshot when comparing optimizer changes.

When partial labels or occluding hands require an explicit observation domain,
set `fit.observation_domain` with independent `background_suffix` and
`occluder_suffix` mask revisions. Unknown pixels never come from model mismatch.
Keep all cached views and at least eight informative foreground/background
observations; bind native labels and metadata to the round. Contradictory labels
fail. Reacquisition preserves old masks and starts in `splat/object_fit/history`.
GPU rasterization may propose parameters, but final numeric acceptance uses the
same CPU triangle coverage and unchanged IoU/local/frame thresholds.
If a GPU proposal fails that final CPU check, retain the failed round and
continue its same-capture shape/pose on the CPU path. Disabling CUDA selects
the existing CPU fallback. Preserve observations, physical priors and thresholds;
neither a passing GPU score nor optimizer convergence substitutes for CPU proof.

Revise the source and start another round after rejection. Export only a numeric
passing round. Record `OBJ.<object>` with its evidence, verdict and resolved
findings, matched to the current source/export hashes. Review textures and
`qa/object_closeups` afterward. Set `fit.accepted: true` only for the reviewed
current round; do not edit its source initializer afterward. For an improvement,
open a new round instead. The rendered and SDF collision surfaces must match;
mass uses solid volume and material density unless a justified mass is supplied.

Before modeling with supplemental media, read `object_inputs` and inspect its
role/state-specific boards. Confirm identity and distinguish original-scene
placement from closeup part geometry or motion. A moving/open part is not a new
static view of the closed model. Extra image constraints require an estimated
camera and the corresponding part state; unregistered images remain high-level
agent evidence. Source/plan/selected-image hashes bind object and task checks.
Inspect HDR color metadata and source-derived exposure alternates before
reasoning about part connectivity or cloth folds. A bright merged region does
not establish a connection. Resolve exposure reviews for the intended evidence
use; inspect another source view if details remain clipped or occluded. Corrected
color evidence requires affected primary RGB, segmentation and fitting to be
refreshed; supplemental corrections cannot certify old source masks. Material
tuning starts after reviewing the geometry and contact interpretation again.
Do not identify force-dependent material constants from an unmeasured hand
demonstration. Preserve prior uncertainty where the evidence is insufficient.
For failed source-hold tests inspect the saved transient states and intrinsic
material-region motion relative to the measured primary rigid frame. Check
which regions touch the clamp/support before interpreting motion as bending or
sliding. These diagnostics neither identify a modulus nor replace source or
task gates. Revise unsupported shape bounds, folds and initialization hypotheses
before assigning a visual mismatch to a material value. A disabled-gravity or
disabled-contact control is diagnostic only and cannot certify a task.
Inspect `strain_hotspots` in immutable material coordinates and actual state
coordinates. An assumed tight transition or compressed layer layout can create
large membrane strain and transient motion; first compare source-supported
geometric alternatives. Hotspots do not identify stress, yield or a modulus,
and they never justify relaxing the declared material validity limits.
When static shape retention is uncertain, use `scene.objects.source_motion` on
the existing observation cache to compare direct material tracks with an
independently labelled rigid reference. Review its source boards; epipolar
consistency alone does not prove a static part, and insufficient matches do
not pass. `scene.objects.source_landmarks` can recover direct-material sparse
tracks for geometry proposals, with measured parallax/reprojection quality
and pixel-noise covariance. Camera/scale bias and incorrect correspondence
remain extra uncertainty. Candidate errors must not choose either tool's
source pixels, and sparse coverage must not invent hidden material. Preserve
all silhouette views and the independent material-area constraint.
For expensive models, optional `fit.proposal_fixed` may lower declared numeric
`FIXED` sampling settings during search. Keep constitutive/contact hypotheses
unchanged. This is a proposal only: the harness restores the complete model,
preserves its rigid placement, reruns full CPU source/admissibility checks and
requires all native/task gates. Coarse builders cannot export delivery assets.
Validate native impulse and rigid-contact controls for the actual backend and
solver before adopting a more stable passive profile. For the installed
Isaac5.1 surface-FEM adapter, PGS has those controls; the tested TGS profile
failed them. Formal nodal forces reject unsupported profiles. A backend's
unused bending description is not implemented fold memory; do not treat
surface natural curvature and particle distance-based bend memory as aliases.
Distinguish material identity from exterior depth when a shell is translucent.
Review the actual source evidence and bind `transmitting_links` plus
`transmission_basis` before using transmitted part pixels. The actual model
must have transmitting PBR faces. The straight-ray visibility proxy does not
measure refraction or haze and never removes solid collision geometry. Static
and dynamic source checks must use the same optical interpretation. Preserve
opaque occlusions and every material/contact/task gate.
If new evidence reveals an unresolved connection or mechanism that changes the
task, record it in `mechanisms.json/open_questions` with its role, impact and
source. Continue independent fitting and clearly conditional experiments, but
do not certify task feasibility until the task-relevant questions are resolved
with an answer and evidence. For example, extraction of a detached sheet does
not prove separation of a connected roll. See the additional-inputs reference.
For new cloth models use `material_state_contract: 2`: validate intrinsic
material lengths/area, principal strains, solid penetration and cloth
self-intersection at initialization and throughout task motion. Keep native
self-collision enabled. The exported stage also checks initial surface
crossings. A relaxed geometry candidate or a diagnostic with contacts disabled
is not a passing task. The triangle self-intersection measurement alone does
not certify contact thickness or continuous collision handling.

`fit.part_anchors` optionally maps a later query to `top`, `bottom`, `left`,
`right` or `center` of the observed first semantic-part mask. It never uses the
model render to choose query pixels; raw per-part proposals remain archived.

Appearance baking preserves numbered `look` revisions within the geometry round.
Default `fit.texture_mode: "multiview"` retains observed patterns. For visibly
unpatterned objects the agent may select `"uniform"`, deriving one robust colour
from observed texels so illumination/chart seams are not asserted as texture.
`fit.texture_gain` is an optional bounded display gain, not measured reflectance.
Every material change requires fresh S4 review; frozen sessions reject it.
The multiview baker aligns relative exposure/white balance using overlapping
surface samples, retains printed detail and reports disconnected view groups.
This is not absolute reflectance identification. Agent PBR regions such as
paint versus metal stay bound to original physical faces after UV baking.

For physics recordings, see [video physics](../../../docs/video_physics.md) or
[replay calibration](../../../docs/real2sim_physics.md). Their default route also
requires a reviewed Blender artifact. Their motion initialization does not
approve a primitive as the final model, and estimated scale remains uncertain.
