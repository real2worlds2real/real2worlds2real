---
name: twin-model
description: Author and iterate task-object Blender models against calibrated_scene_reconstruction calibrated silhouettes, visible openings and closeups, recording current OBJ reviews before S2 acceptance.
---

# Task-object model iteration

For a shared reconstruction plan use [the object workflow](../reconstruct-object/SKILL.md). This skill supplies its native procedures.

Before choosing an object representation, read the matching
[experience modules](references/experience/index.md): open vessels, rigid
insertion, gears, bonded flexible surfaces and observed surface depth. Compose the relevant methods for the
task; use the distilled procedures rather than historical fitted values.
For unfamiliar settings or a capability gap, use
[real2sim-evolve](../real2sim-evolve/SKILL.md). Retain validated lessons in the
matching module using its shared template and record the tested applicability.

Read explicit `task.json.acceptance` conditions before choosing model detail or
experiments. A user-specified functional condition can authorize a scoped
relaxation of a default appearance/response gate. Record the original metrics
and failed verdict separately, and activate the relaxation only after the
specified native test passes (`scene.acceptance`). Preserve rigid precision,
full material quantity, valid geometry/contacts and input provenance. Do not
replace a requested friction grasp with node forces or attachments.

All reconstructed objects need semantic review, including furniture. Check
identity, component inventory, repeated structure/layout, connections, support,
materials and rigid/flexible behavior. Symmetry is a source-reviewed construction
prior, not a universal rule; do not approve unsupported asymmetric legs because
their silhouettes fit. SUP gates require current semantic and structural reviews.

When the input permits reusing local reconstructions, inspect matching existing
assets before authoring a replacement. Reuse the editable construction, complete
material topology, texture and supported material priors where compatible. For
a folded sheet, retain the whole sheet rather than copying only its visible
surface. Record the exact source asset and hashes, what was reused, its failed
checks and the new source-dependent changes. Fit the current capture's scale,
pose, folds and contact route; an old rim-draped sheet is not already a valid
sheet threaded through an aperture. Old QA is evidence about that old candidate,
not approval of the new scene. Do not consult prior assets when the user has
required an independent reconstruction without them.

Before repeating parameter or material optimization, compare the model family
against source closeups. If its structure cannot express the observed shape
(for example a broad locally gathered sheet modeled as uniform pleats along a
prescribed arc), stop that search and revise the construction. Record
`fit.representation_review` with source and candidate evidence. Its
`replace_representation` decision blocks ordinary fit/export of that model.
Inspect the new representation before revising the decision. Do not bypass a
rejection with low-level optimizer calls. Explicit hypothesis-falsification
diagnostics are allowed; they confer no physical or source approval.

Reconcile the declared input inventory with the reconstruction graph, registered
non-mesh components and explicitly requested task-reset tools before reviewing only existing
objects. Use `task.object_inventory_contract: 1` for new named inventories;
[coverage rules](../../docs/scene/object_inventory.md) block S2/S6 on omissions.

For a hollow vessel, watertightness and positive total volume are insufficient.
Check that the inner and outer material boundaries do not cross, retain positive
wall thickness, and leave the intended cavity empty. Rotational profiles can use
`assets.vessel_meridian`; arbitrary shapes need an equivalent actual-material
check. Preserve source geometry checks after a repair. Place contents using
signed distance to the real solid, not a vertical upper envelope that would
mistake overhanging fruit or a container's empty cavity for penetration. Native
settling must preserve the source-consistent arrangement as well as reach rest.

When a container hides source-visible contents, inspect the raw RGB and actual
occluding triangles before changing child geometry or labels. Joint parent/child
source feedback and opt-in actual-triangle support placement are documented in
[containers and loose contents](../../docs/agent/container_contents.md). Rebuild
the editable producers and repeat source, solid and native checks before adoption.

Keep a thin formed part's panel thickness separate from the height of its rim,
lip or recess. A silhouette-correct moving part can have the wrong mass and gravity
torque if its recessed interior is filled solid. Inspect the actual material
volume, wall sections and COM before blaming hinge friction or cloth stiffness.
`assets.articulation_statics.gravity_effort` computes conditional gravity load
from current link COMs, mass priors and the actual hinge axis. Source stationarity
can constrain an opposing passive load; it cannot by itself distinguish friction,
elasticity, a detent or occluded support. Test the selected model with native
holding and finite-force actuation, without an undisclosed positional drive.

Fit independent material dimensions independently: changing a container radius
must not silently shrink its loose cloth, nor a pot fit shrink every leaf.
Keep the complete intrinsic material chart and quantity traceable across body
revisions. Source-consistent folds need finite thickness, resolved sampling and
native release/contact checks; repeated tightly packed pleats are not a generic
substitute for gathering or draping observed in the video.
When an input gives an exposed fraction, account for the material crossing the
opening and bending over the rim as well as the visible hanging tail. A half-
exposed sheet does not mean half its length hangs vertically. Compare visible
extent in several calibrated views, retain out-of-plane and occlusion uncertainty,
and revise hidden-length hypotheses explicitly. Do not stretch the hanging mesh
without updating and validating the complete intrinsic material chart.

Flexible response defaults to `task_and_semantics`: match the source as closely
as practical, require plausible material behavior and successful native task
interaction, and preserve exact local deformation/material-response errors as
diagnostics. Do not spend repeated rounds matching noncritical numeric responses
once semantics, physical plausibility and the task pass. The detailed quantitative
controls below are tools for diagnosing failures or explicitly requested precise
response, not universal prerequisites for this mode. Keep rigid precision and
actual flexible modeling; a rigid hull does not receive this exemption. See
`docs/agent/workflow.md` for the implemented policy fields.
Candidate hand-contact assays use this same policy and keep both the raw
material measurements and policy verdict. A passed friction interaction cannot
approve visibly excessive elasticity: object approval still requires a current
physical-plausibility review of the actual native response. Quantitative mode
continues to enforce the raw limits. Neither mode relaxes primary self/solid
contacts, changes the full material outline or grants a rigid substitute.
The native extraction/wiping fixture uses the same response policy and retains
raw area/strain peaks. Quantitative mode still aborts at its material limits;
approximate mode still rejects invalid geometry and actual contact failures,
and requires a review of the native material behavior before object approval.
Native passive/preparation probes and their geometry-seed adapter use that same
declared response policy. A local numeric strain excess remains visible in the
raw measurement and does not alone abort approximate-mode native motion.
This does not relax offline repair budgets: use the preparation tool's explicit
material inequalities and independent budget check when changing geometry.

Before a new task-object fit, inspect the current video and task to write
`agent/mechanisms.json` (version 1; see
[`assets/mechanisms.py`](../../src/calibrated_scene_reconstruction/assets/mechanisms.py)). Cover every task
object and physical link, including a single rigid `model` link. State mechanism
type, task relevance, observed/inferred/unknown evidence, representation and named
checks. Rigid-only tasks use an empty mechanisms list. Do not infer a joint just
from a seam. Primary and related mechanisms must pass; secondary mechanisms are
modeled where evidence allows and their uncertainty remains explicit.

Model real joints with literal `LINKS` (kind and exact Blender mesh names) and
literal `JOINTS`, or `JOINTS(params)` for parameter-dependent local anchors. Links
are unioned only internally. The output must preserve physical parts, actual
joint frames and limits; revolute limits use radians, prismatic limits metres.
An observed articulation need not be at mechanical zero. Optional `frame0` and
`frame1` are normalized link-local wxyz quaternions with local X along the
declared axis; supply both and an `initial_position` inside the real limits.
Keep the Blender link geometry in its captured state and encode that coordinate
with these frames. Isaac resets articulation coordinates to zero without its
serialized JointStateAPI, so the authoring tool emits it with zero initial
velocity. This is passive initial state, never a drive or pose animation.
Read hinge position in the joint frames, not just the relative body rotations.
Use native FEM/cloth with explicit material assumptions for flexible links;
never substitute a rigid mesh, animation or shape keys for solver deformation.

An asset can contain a real rigid joint tree and independent flexible/rigid
parts. Sharing its coordinate frame does not attach those loose parts. For a
folded sheet, preserve its intrinsic material dimensions rather than making its
captured fold the rest metric. Optional `CLOTH_REST(params)` declares per-cloth
rest points and typed stretch/shear/bend springs; see
[the rest-coordinate contract](../../docs/scene/object_mechanisms_plan.md).
Check initial contact, material area/rest metric, deformation and actual release
in addition to the task. Native backend fixture passes do not pass a video task.
For new cloth captures set `material_state_contract: 2` in `mechanisms.json`;
declare `material_area`, `rest_metric`, `strain_bounded` and
`no_initial_interpenetration` among each primary cloth deformation's checks.
Declare material validity assumptions and evaluate actual native node states.
Use uniform material sampling with Isaac5.1's authored total cloth mass; do not
make densely tessellated folds artificially heavier. Geometry regularization
such as `cloth_rest.project_shape` only initializes a shape and cannot replace
native self-contact, stability or task validation.
For a retained paper crease, optional `cloth_rest.bend_memory` changes only
named bending rest lengths with explicit source and construction basis. The
unfolded membrane metric and area stay unchanged. This unsigned distance
approximation still requires native fold/contact checks; it cannot certify
layer contact or keep a crease from inverting by metadata alone.

Check the initial membrane energy estimate as well as strain hotspots before
calling a source-hold failure a soft material. A source fit can store enough
elastic energy to move substantially when released. The estimate is diagnostic,
not a native stress measurement or a new threshold. Experimental internal
bending and finite-strength cohesive interfaces remain unregistered prototypes;
their synthetic or candidate passes cannot qualify task physics or standalone
USDZ loading. Retain natural-history/interface/source hashes, converged-force
controls, finite release, native coupling and runtime validation before adoption.
For a native particle candidate, use its actual distance-spring network for
that diagnostic, not continuum constants borrowed from another backend.
`check_particle_release.py` isolates native free release with explicit material
sampling, prestrain, damping and bending hypotheses. These synthetic controls
do not identify a video's material or qualify its task. In approximate mode,
authored and native geometry use the same independent local-bend review: keep
the raw distance failure and require positive extrusion Jacobians and
nonintersecting boundaries for every flagged local bend. Separate layers and
inverted folds remain failures. Padded particle envelopes are a separate
initialization diagnostic, not physical thickness; inspect native reset and
release before deciding whether an overlap is filtered or injects motion.
When sweeping a folded section along a curved path, inspect the frame sampling
as well as the centreline. Piecewise-constant normals can inject artificial
stretch between adjacent material rows. `cloth_curve.sweep(smooth_frames=True)`
is an explicit option for densely sampled smooth turns; it refuses unresolved
sharp corners and preserves the legacy segment-frame path. Keep the intrinsic
chart, independent material/contact checks and a fresh Blender/native rebuild.

When a candidate uses padded surface-FEM contact offsets, prepare its geometry
against the actual native rest envelope as well as physical material thickness.
Two nearby sheets need at least twice the cloth rest offset; a sheet and rigid
surface need their combined rest offsets. These numerical margins do not change
the declared material thickness or weaken physical checks. `cloth_project`
records both policies and rejects a seed that violates the preparation policy.
Its optional `--envelope-repair-limit-m` permits bounded offline clearance repair
while retaining all original material/self/solid checks; rebuild through Blender
and rerun source and native validation afterward. `--device cpu --linear-solver
sparse_direct` allows this geometry work to proceed while another job uses the
GPU. Never interpret a low-energy offline result as native stability or task
success; compare actual initialization and release with the unchanged candidate.
Approximate final response acceptance does not authorize stretching the material
to remove a modeling collision. Offline repairs retain explicit edge, principal
strain and area budgets from the seed and material declaration, and include
material inequality rows alongside contact constraints. Reject a contact-only
repair that exceeds those budgets and use intrinsic-length optimization or a
new fold proposal; retain failed geometry and rebuild/validate any replacement.

Declare the numerical profile in `simulation.json`. Particle cloth defaults to
PGS after native Isaac5.1 local-force and rigid-contact controls; pure rigid
profiles retain TGS. An explicit `solver` may override this only with native
validation. The selected solver, timestep and friction model must survive export
and runtime audit. TGS flexible scenes enable external forces each iteration;
PGS must disable that unsupported setting. Author no undeclared global particle
damping, lift or drag; spring damping remains the material dissipation.
Passing a simple crease/impulse fixture does not pass a folded multi-layer
object, source appearance, extraction, or wiping.
For particle contact evidence, bind the running material edge range, spring
topology, native spring/mass readback, rigid and particle contact radii,
neighbour budget and friction scale. A coefficient readback alone does not
establish a contact law. Keep matched force-response controls and their executed
sources; a passing control at another numerical profile cannot approve export.
Particle membrane response can depend strongly on timestep and sampling.
Measure the implemented response before interpreting authored spring constants
as a fabric modulus. In approximate mode, use those measurements to diagnose
visible softness or task failure; do not turn exact modulus matching into an
extra universal prerequisite. Bending, source drape and actual interaction still
need their own evidence.

For native surface cloth, explicitly use `backend: surface_fem` in both LINKS
and the mechanism declaration. Supply continuum E, Poisson ratio, thickness,
flexural rigidity and damping priors; never reuse particle spring constants.
Its uniform nodal mass model is validated with native nonuniform-mesh impulse
and rigid-contact controls, not a direct nodal mass readback. Inspect the
native traction response at the actual material resolution and numerical
profile before interpreting an authored modulus as implemented stiffness.
New surface material-state contracts require immutable per-prior-case
`membrane` and `flexure` controls in `agent/material_response.json`;
`apps/scene/check_surface_response.py` and `apps/scene/check_surface_flexure.py`
are independent diagnostics. Quantitative material acceptance requires current,
matched passing controls. Under `task_and_semantics`, retain quantitative
failures and measured effective response, then require semantic, physical
plausibility and actual task evidence; do not silently impose the quantitative
threshold as an extra gate. Gross numerical softness or unstable response still
needs correction. Large folds, real damping, source state, contact and task
checks remain separate.

If a full folded sheet fails while small contact controls pass, compare the
same geometry and material at finer native timesteps. Read actual nodal speed
and the passive probe's `time_resolution`; velocity-times-step relative to
physical thickness is diagnostic, not a stability certificate. Preserve failed
crossings and source-support failures even if a finer-clock trace is valid.
A fine-clock geometry construction still needs fresh unforced reload, source
state, real-finger task and export checks at the selected qualified production
profile. Check the actual compliant boundary against rigid partners as well;
positive tetrahedral Jacobians and cloth/stock proximity do not prove that
contents remain inside their container.
Before a holder-assisted hand trial, inspect an independent unforced assembly
release. A cloth material/contact pass does not imply that the container, moving panel or
other rigid components remain in the captured initial state. Measure full rigid
surface motion (including rotation) against the declared passive-response
bounds, and inspect joint motion, stored preload and support motion. Retain the
separate `rigid_passive_response` and `unforced_rigid_and_contact_pass` results
from the candidate probe. Neither certifies the initial material distribution:
a sheet can slide off its intended support while passing contact validity.
Check retained material, the requested initial articulation state and source
semantics separately. A test holder must not hide a scene stability failure.
Inspect the
`constitutive_authoring` audit: surface FEM uses `bend_reference`, while particle
cloth uses distance-based `bend_memory`. Declaring both does not apply both in
one backend or establish equivalent crease behavior. Native response validation
is still required for the actual material and contact configuration.
Explicitly requested contact criteria require current source-bound evidence.
Preserve raw source failures when an explicitly declared pose scope is relaxed;
continue every unrelaxed geometry, full-material and physics check.
verdict and continue every unrelaxed geometry, full-material and physics check.
Candidate cloth/SDF overrides that differ from the declared delivery numerics
remain diagnostic, even without an explicit hypothesis file. Declare a reviewed
profile before fresh qualification; never relabel an old diagnostic report or
reuse it to approve different exported contact settings.
Before optimizing a hidden folded packing, compare its visible drape extent
and layer organization with the source. Preserve the complete material chart
while allocating material between the visible region and hidden support. A
short visible flap can create unnecessary hidden stacks, closure pressure and
stored elastic energy. Treat every such packing as an inferred construction;
do not solve it by shrinking the sheet, inventing attachments or waiving a
visibly different material structure. Preparation-only poses need a completed
released construction and fresh initial-state checks before a task assay.
The zero-friction control must replay the positive trial's saved material target,
wrist pose and commands; nominal configuration equality alone is insufficient.
Before a pinch proposal, screen the open hand's full descent and rotation
against rigid objects, including the opened moving part, as well as the current flexible
surface. Check the closing fingers against rigid objects separately. Clear
endpoints do not imply a clear descent: moving part contact can block wrist tracking
without penetrating the moving part. Inspect actual wrist errors and native rigid
contacts before diagnosing cloth stiffness or friction. An explicit
`inside_finger` choice initializes that assignment at the safe hover pose before
simulation; never teleport the hand after reset to recover a failed approach.
If a bounded wrist correction follows material during closure, compute its
residual from actual native finger-pad transforms. Commanded joint angles do
not reveal contact-induced joint lag. Keep the original displacement/speed
limits, save every wrist command and replay that trace in the negative control.
Tracking a target does not itself prove two-pad capture or friction support.
The default `material_contact_mode: same_component` resolves both actual
interior contact regions after closure, then tracks those fixed source indices
at the corresponding fingers through hold. Folded layers need not share one
small intrinsic disk. Preserve the approach patch for direction checks, reject
different material components and keep explicit `single_patch` for inputs
requiring the original shared disk. Matched controls reuse the positive's
material regions; proximity alone still needs native retention and friction
evidence. A closing-force cap alone is not sustained preload. Qualify a source
fingertip fixture independently with an interior coupon, one-second stationary
loaded hold and identical zero-friction commands before changing the asset's
material. Keep source geometry/cooking fidelity, contact search radius and
physical penetration tolerance distinct; none may waive actual overlap.
Use the full declared support relation in native initialization as well as
Blender geometry checks. An end that is merely above its tray has not passed
the declared `material_contact` requirement. Measure actual triangle proximity
and material-end area without filling holes; loaded support and retention
after joint motion still require native evidence. Diagnose source-supported
packing and controller timing before changing cloth material.
Cap each fixture motor by the imported joint's actual finite effort limit;
one uniform requested budget must not override weaker distal joints. Read
back native per-joint limits and rerun the free-hand control after this changes.
For particle cloth, changing a rigid finger material alone is not a qualified
friction ablation. Use matched native rigid/self-contact controls at the running
spring network, sampling and timestep. A particle-wide base-friction change
also affects the container and table: the diagnostic hand route delays it
until after lift and requires whole-sheet separation from every non-hand
collider at every hold step. Preserve self-friction with independently tested
controls; retain the pre-intervention native state and exact motion replay.
This diagnostic route does not itself grant task or delivery acceptance.
A rigid-clear approach can still push a sheet off its support before closure.
Use the actual source hand colliders and post-opening material state; the sampled
proposal screen cannot certify moving-cloth clearance. Inspect native approach
contacts and material drift before attributing a lost grasp to low friction.
Expected contact with deformable material is not itself a failed approach:
finite native drives may displace the sheet while closing. Require predicted
capture of the selected material by both closed source pad envelopes; do not
raise the wrist merely until the open-hand path is empty and then close above
the sheet. Preserve contact-free approach as an explicit optional constraint.
Record predicted approach contacts separately and enforce actual native
penetration, material validity and two-distal-finger contact before lifting.
If open fingers push the target away before closing, screen a coupled final
descent and closure. Choose precontact height from the current sheet and source
hand geometry, screen intermediate wrist/joint poses against rigid objects,
and replay the same descent in the negative control. Sequential approach then
closure remains an explicit option. Neither geometric proposal proves a grasp.
Sample wrist-placement offsets at the physical sheet/contact scale. A fixed
millimetre grid can skip a complete thin-sheet clearance interval. The helper
reports its bounded resolution and any budget limitation; finer proposals never
increase collision tolerance or replace actual native capture and friction tests.
For a sheet, prefer an opposing-pad axis aligned with the actual local material
normal. A clear tangential closing path can push the sheet downward and leave
only transient fingertip-edge contact. Compare the same material node before
and after closure before changing friction or stiffness. The native assay
screens normal alignment; the geometry helper retains its legacy unfiltered
option. If aperture access requires it, propose a wider open preshape from
source joint limits and collision pads, then rerun the independent native hand
control. Do not alter the cloth material to compensate for an unsuitable grasp.
Whole-sheet near-surface counts at both fingers are only proximity witnesses.
Inspect the selected material against the actual opposing pad surfaces: contact
on two different pad shoulders can disappear as soon as the hand lifts. Do not
call this a loaded pinch or compensate by increasing the sheet's friction.
The optional `track_pinch_material` hand assay mode tests bounded wrist-only
feedback during closure (8 mm maximum correction, 10 mm/s). It tracks the
observed material and compensates the source hand's moving pad midpoint, never
writing cloth positions, forces or attachments. Every physics-step wrist target
is saved and replayed exactly in the negative control. This is a controller
hypothesis requiring fresh native tests; it grants no grasp, material or task
approval and does not replace the unforced initial-assembly check.
When a finger's middle segment displaces the sheet before the opposing distal
pad arrives, reconsider the CAD joint posture and source contact frame. Screen
alternative joint-space seeds and actual distal contact surfaces; retain the
original physical hand geometry and limits. Prefer a clear open pose with the
material between the fingers when available. The hand tool exposes
`closure_descent_m: 0` with `require_contact_free_approach: true` for sequential
descent and closure, alongside its coupled-descent route. Neither a static pose
nor a collision-free planned path replaces the native free-hand and task tests.
Source joint limits alone do not guarantee a collision-free hand posture.
Inspect movement of uncommanded neighbouring fingers when a free-hand test
misses its targets. Include every nonparticipating finger in self/contact/path
review; retracting a finger can resolve one interference and create another.
The CAD proposer accepts optional search bounds inside the unchanged source
limits. These restrict an optimizer, not the simulated joint limits. Inspect
the actual closed gap and normals: a least-squares result can remain infeasible
under those bounds. Reject such proposals before spending a native task trial.
The free-hand report also measures source pad references from actual native link
transforms. Inspect their signed normal gap and opposing normals, not only joint
angle tolerance: small joint errors can leave a thin sheet unclamped. These
reference distances are not minimum mesh gaps or normal-force measurements.
If the finite motor cannot reach its empty closing pose, revise and revalidate
the test controller before attributing slip to object friction. Preserve source
joint limits and contact geometry, disclose finite motor budgets, and repeat
the matched friction control; never compensate with cloth attachments.
Allow a bounded finite-drive settling interval after the closing ramp before
evaluating contact. The native assay records that duration in its protocol and
continues every physical check during the hold. Compare actual contact after
settling; commanding the final joint pose is not evidence it has been reached.
The matched zero-friction trial must retain the same settling commands.
Compare the executed command prefix and actual pre-lift states. Read and verify
friction on the native hand shape view, including its persistence during lifting;
an authored USD value alone is not native material readback.
Also verify effective pair friction with a native sliding control for the
installed backend. In Isaac5.1 surface FEM, tensor material writes can lose the
effective rigid material's multiply policy even though coefficients read back
as zero. Edit the existing rigid USD material without rebinding it, retain
multiply, and verify the native hand coefficients after physics steps. Changing
only the cloth tensor coefficient or rebinding a new rigid material can also
leave existing contacts unchanged. `check_surface_friction.py` retains native
sliding trajectories to distinguish these cases; parameter readback alone must
not certify a causal zero-friction control. Recheck on backend changes.

Treat maximum depenetration velocity as a numerical contact-correction budget,
not material damping. When moving parts or tools penetrate a flexible surface,
compare this budget with actual relative contact motion before changing cloth
stiffness. Retest moving contact, self-contact and unforced motion after changing
it. A closed-joint static pass does not prove an opened sheet stays supported;
check the actual material remaining at the intended grasp region. Diagnostic
viscosity adapters must not hide failures of the standalone native asset.
Independent `bend_angles` are currently diagnostic-only: the installed
Isaac5.1 authoring path failed matched positive/reversed native response
controls. Production rejects them. Disjoint per-triangle rest topology also
failed cooking. Keep capability evidence and use a validated representation;
schema presence alone cannot qualify a physical feature.
For a source-supported flexible part, optional `cloth_proposal` in the agent
tool API fits geometry against all current part views while retaining intrinsic
material, triangle/self-thickness and actual solid screens. It can use a smooth
material-coordinate field or individual nodes. A curved solid requires fresh
distance queries after motion; an initial tangent constraint is insufficient.
Tool `ok` means generation only: inspect `source_ok`, rebuild through the agent
model, then rerun every original source, native and task gate. A lower-energy
geometry proposal is not an identified material rest state. See the
[proposal tool contract](../../docs/scene/object_mechanisms_plan.md).
Optional
`cloth_rest.bend_reference` must preserve each triangle's membrane metric.
New material-state contracts also inspect the authored strain, self-crossing
and actual rigid penetration before World.reset. A settled or projected state
cannot hide an invalid exported start. Geometry preparation is only a candidate
and must be refitted to source images and independently checked without forces.

When a physical source observation names `part_prompt`, its link has its own
visible-part source constraints during Blender yaw selection, fitting and final
CPU evidence. Whole-assembly averages cannot approve a missing or wrongly
oriented small part. Retain independently acquired part labels, source-only
occlusion domains and the original thresholds. Static part agreement never
replaces unforced native stability or task checks.

Review source labels before changing geometry to reduce a residual. Independent
negative queries can mistakenly select a flexible task part; enclosing-part
labels can merge with white cloth under bright exposure. Source-part checks
reject major label conflicts and merged exterior/enclosing semantics before
fitting, require eight reliable views, and retain reasons for every unused
part view. Whole-object views remain unchanged.

An observation-only item can touch the target in image space. A connected mask
does not establish physical identity. For an item resting on a measured support,
declare `agent/observation_exclusions.json` and run
`python -m calibrated_scene_reconstruction.scene.objects.observation_exclusions <session> <object>`.
Proposals use a source RGB mask, registered cameras and the verified support;
conservative source-link foreground protection prevents projecting an item
through a bottle or other task object. Inspect every source overlay and resolve
its review before use. Exclusions make only those reviewed pixels unknown.
The source/mask/camera/support/review hashes bind subsequent fitting and export.
Never approve an exclusion from a candidate model's error image.

For coupled geometry, an optional `fit.initial_rigid_stage` names `parameters`,
independently observed `exclude_links` and a bounded `budget`. It initializes
the actual Blender body while source-labelled flexible regions are unknown;
its result is explicitly proposal-only. `fit.pose_anchor: primary_rigid` keeps
the body centre fixed as a loose flap changes shape or yaw. Optional
`fit.pose_parameters` selects `xy`, `yaw_deg`, `down_offset`; defaults preserve
the existing full optimization. Every final whole-object/part and native gate
still uses the complete object. Inspect internal support/cavity depth in
closeups before attributing a folded sheet's instability to material softness.

retention, pull controls and release. Deliver passive objects without test guides.
The existing task check returns per-mechanism named results; S5 rejects missing
required checks and stale geometry/physics evidence. Isaac Sim 5.1 is the primary
backend; a Newton-only result cannot stand in for its validation.

Also write `agent/physics_priors.json` before a new task-object fit. Infer a
material/construction hypothesis for every body and the task support. Include
nominal values, coherent ranges, SI units, source/basis and whether a value is
only a prior or constrained by observed motion. For rigid bodies estimate mass
or density (derive its counterpart from this geometry), contact friction,
restitution and passive damping. For native joints include friction and damping;
for flexible bodies include the native material constants and solver resolution.
State how mating materials combine, including contact friction and possible seals.

Use material/construction priors and available motion evidence. Static images do
not identify mass, friction or modulus; task success must not be relabeled as
real-property identification or used to silently narrow uncertainty. The native
task-check entry runs nominal and simultaneous low/high prior cases, preserves
their evidence, and requires all three to pass. These samples are a sensitivity
check, not an exhaustive proof over every parameter combination. Failures can
require better geometry, solver accuracy, mechanism understanding or new evidence;
do not change a plausible physical prior only to force the task through.

For new tasks also set `physical_observation_contract: 1` in `task.json` and
`observation_contract: 1` in `mechanisms.json`. Declare source-backed passive
shape observations (id, mechanism, link, check, source, basis, duration_s,
max_projection_motion_px and hashed session-relative evidence_files). Include
each observation's check in its mechanism's checks. Compare actual native
initial and unforced states with source images and initially visible material;
do this before any task actuation, without replacing the source reference by
a settled shape. A static source part that immediately collapses is a failure
even if whole-object fitting or extraction succeeds. Use source state-specific
evidence, review mistaken/ambiguous semantic part labels, and consider bending,
retained creases, layer contacts, support and clamping before selecting material
parameters. Static equilibrium constrains a coupled response, not a unique
modulus or mass. Do not pin the sheet, prescribe its pose, or turn it into a
rigid substitute to satisfy the observation. Native deformation and task checks
remain separate requirements.

Discover these response checks from the source before selecting final material
values: inspect an unforced interval, opening/release transitions and any supplied
manipulation clips. Record whether a free part holds a crease, droops, springs
back or stays supported. Separate camera motion, exposure/segmentation mistakes
and external contact from material response. Missing force measurements leave
the constants as priors. User feedback can trigger this review but must not
become a numerical stiffness target. User-confirmed identity and material
family are valid input evidence: identifying the complete unfolded reference
or specifying cloth-like fabric must constrain modeling and prior selection.
Do not discard these facts as subjective appearance feedback or continue a
paper-like fixed-crease model by default. Exact fibre construction, moduli,
wet adhesion and crease plasticity can remain uncertain.
Native passive checks retain intermediate samples
(at most50ms between native samples); both endpoints matching is insufficient.
Use the resulting geometry/initialization versus later-response diagnosis to
choose the next hypothesis and rerun source, native, task and sensitivity gates.

If an independently identified source sheet is a size reference, optionally
declare `agent/material_references.json` version1 with `references` entries:
`link`, observation-exclusion `region`, `relationship: "same_sheet_area"`,
`area_factors: [lower, upper]` and `basis`. The camera and verified support
project its source mask to a plane footprint; factors explicitly describe
boundary/plane and wrinkle uncertainty. The actual intrinsic chart area is
checked during whole-object fitting and final CPU QA. This is image-derived
relative geometry, never supplied true dimensions. Do not shrink material to
make a bad fold match the silhouette, or assume the footprint is exact unfolded
area. The reference remains absent from the delivered scene.

When the source contains the complete unfolded sheet, use version2 and
`relationship: "same_sheet_geometry"`, `source_extent: "complete"`, explicit
`linear_factors`, `aspect_factors` and `outline_difference_max` in addition to
area factors. The same reviewed mask/camera/support constrains both unfolded
extents, aspect and full area-normalized boundary. The actual intrinsic chart,
not the folded object's visible envelope, must satisfy these checks. Shape
normalization is only a comparison signature; no asset scale is fitted.
An equal-area narrow ribbon cannot replace a nearly square complete sheet.
Keep fold count, packing and layer contact as explicit construction hypotheses;
neither quantity agreement nor a source silhouette establishes a four-fold
packet or permanent crease memory. Use cloth-appropriate geometry preparation
when a procedural fold construction is unsuitable, then rebuild through the
agent model and rerun native/source/task checks with temporary fixtures removed.
The `inputs` tool exposes declared reference identities, their source image/mask,
material-family evidence, property priors and construction questions in
`modeling_facts`. Use these facts alongside the video; do not reinterpret a
confirmed full sheet as only an area hint. The optional `sheet_reference` tool
builds one uniformly sampled flat material surface and rectifies its actual
fabric photograph. Its `source_ok` covers the complete material only, never
the packed pose or task physics. The photograph retains source lighting; it is
not calibrated albedo, and correspondence to the exposed part remains to fit.

For physical pose preparation, `tasks.mechanisms.preparation.MaterialPreparation`
accepts explicit material-local finite-area grips, rigid-local paths and bounded
native hinge efforts. The candidate runner's `--preparation-plan` snapshots the
plan and force code, keeps real gravity/contact and records force release.
No cloth positions or permanent attachments are prescribed. Preparation is an
inferred construction experiment, not task success: rebuild its geometry through
the agent model and recheck the actual exported start, all source views and
unforced native response without temporary fixtures. Never export the seed pose
or treat a native-preparation `physical_state_pass` as source/task acceptance.
For folded sheets, preserve resolved layer offsets when transporting groups of
grips. Curved stacks have different layer arc lengths: permit longitudinal slip
instead of constraining every layer to the same material row at different radii.
`preparation_geometry` supplies actual-surface rays, group transport and local
offset-curve station proposals; these do not certify global sheet isometry.
Use explicit `release_ramp_s` and staged release when an abrupt removal of the
construction load causes a transient. Inspect the actual applied forces and the
fully released state. None of this changes the material law, proves a source
property, or replaces canonical native/task/sensitivity checks after rebuilding.

When part labels merge a cap, packaging or robot with the task material, use
`scene.objects.part_label_review`. Source-only text alternatives, pixel boxes
and source tracking produce proposals. Review every actual RGB/component sheet;
retain all visible exterior components and preserve raw labels and failed
queries. Bind `part_review` plus manifest/review hashes in the physical
observation. Reviewed exterior pixels may override a conflicting broad enclosing
query, while independent robot/background occlusions stay active. A merged or
hidden component cannot be declared verified. Never choose labels from rendered
errors or use part-view rejection to make geometry pass. Blender fit diagnostics
now include all independently observed part views.

Use `scene.objects.candidate_response` to inspect initial and sampled native
part motion against source images before exporting a candidate. It is diagnostic
only. Check new native material features with independent positive/negative
response controls; authoring an attribute or seeing a plausible video does not
show that the installed backend implements the intended constitutive behavior.

For folded material held between opposing solids, do not require one material
vertex to touch both solids. `tasks.mechanisms.seam_contact` measures rays through
the actual continuous sheet stack, with opposed solid normals and every layer
gap bounded by the existing contact tolerance. Retain the raw same-node count.
This is geometric proximity only: both originally contacted sides must retain
their material under unforced native physics, opening must release the seam,
and required causal friction-grasp tests still apply. Do not widen tolerances,
join disconnected plies, or infer clamp forces from the ray witness.

Do not interpret every near pair just beyond a 90-degree normal cutoff as
two penetrating cloth layers. Under `task_and_semantics`, native sheet checks
may use `cloth_clearance.measure(local_bending_review=True)`: keep the raw
mid-surface rejection, then test the actual thickness of an edge-connected
local bend with `sim.cloth_shell_clearance`. Every failed pair must be listed;
the local closed extrusion must have no intersections and a positive Jacobian
through its entire thickness. Remote folds, separate owners and incomplete
failure lists stay rejected. Strict geometry-proposal constraints and
quantitative mode remain unchanged. This geometric check does not identify
forces, approve elasticity, or replace native contact, semantic and task tests.
Extra contact padding can also enlarge the native self-collision exclusion
distance. Inspect `native_filter_diagnostic.padding_filter_tradeoff` before
changing either parameter; preserve physical thickness and the native minimum
filter bound, and rerun unforced stability as well as the failing interaction.

Before a full task run, audit the freshness and numerical parity of required
contact controls. `check_surface_pair_friction.py --profile` accepts an explicit
material/contact profile and two material-grid spacings. Bind the measured
minimum and maximum material-edge lengths; one scalar grid spacing cannot
qualify an anisotropic mesh. Generate each physical-prior case independently,
and compare matched zero/coefficient controls. A stale or mismatched result is
inconclusive; an unchanged sliding response is not proof of friction. Do not
drop a required contact effect merely to get a passing task report.

Cloth fits with a material-state contract now reject invalid authored states
before source scoring: intrinsic strain/area, required self-intersections and
solid penetration use the existing export-stage criteria and declared limits.
Inspect the preserved `initial_state/` measurements when a hypothesis fails.
An attractive silhouette cannot excuse an invalid sheet. Optional bounded
`fit.parameter_hypotheses` initialize source geometry without approval, preserve
the base-body centre, and still require every final source/native/task gate.
For a declared metric material thickness, check nonlocal finite-layer clearance
as well as triangle intersections. Fit fields can compress folded layers
without crossing triangles or exceeding membrane-strain limits. Preserve the
declared thickness and actual material chart; reject the invalid proposal
rather than changing the prior to make it fit. The clearance screen excludes
local material neighbours, so native contact and task checks remain required.
Near a tight fold, repeat the clearance check after exact surface refinement:
coarse shared-vertex exclusions can conceal finite-thickness conflicts. The
`assets.cloth_refine` helper wraps the existing uniform surface subdivision and
transfers intrinsic coordinates, natural fold geometry, components and UV seams
for surface FEM. It preserves the surface, not a native physics approval; repeat
mass, material, contact and task checks at the new resolution. Particle spring
laws require separate resolution calibration and cannot use this transfer.
Review an explicit natural fold at that same refined topology, independently
of the loaded mesh. `initial_state.Checked` records its intersections and
finite-thickness clearance in `natural_reference_geometry`. Exact isometry or
adequate flat-panel separation does not imply a resolved crease. If the natural
reference fails, explain intended preload/history or revise the inferred fold
construction before attributing the resulting motion to solver precision.
Rounded discrete corrugations can preserve the intrinsic metric, but their
history remains a hypothesis and still needs native release/task validation.
Use `cloth_sheet.build(spacing_m=[width_step, length_step])` when a narrow
crease needs more samples across its turn than along the sheet. Recheck the
complete source outline and material quantity after resampling; do not use a
narrower sheet to make it fit. `cloth_fold.rounded_layered` refuses unresolved
turns above its explicit angle bound. The older one-edge `layered` construction
is only a coarse proposal: subdividing its straight triangles does not round
the crease. Repeatedly halving a near-crease gap under exact subdivision is a
construction defect to investigate, not evidence for a thinner material.
For dense offline geometry preparation, `cloth_project --contact-pairs closest`
can reduce the optimization active set when storing every nearby pair exceeds
the bounded candidate budget. It still traverses every eligible pair for the
clearance verdict, and independently validates every accepted trial against the
full geometry contract. Keep the selection in the input record; a nearest-pair
constraint set alone is not a contact proof. Native validation is unchanged.
First consider a smaller `--maximum-step-m`: its search padding shrinks with
the step, so a full contact set can fit the budget. Inspect saved rejected
trials instead of repeatedly increasing iterations. Offline contact proposals
anticipate local-fold closure at60degrees; the physical checker still uses its
unchanged opposed-normal rule. This conservative guard prevents a tiny step
across90degrees from suddenly revealing a thickness conflict that was absent
from the optimization constraints. For nearby, non-opposed regions of one
continuous sheet, use a local angular inequality before the90degree boundary;
do not force adjacent material regions apart as if they were separate layers.
Already opposed or materially distant regions keep the full distance rows.
Contact penalties are geometry preparation
only and must not be confused with native material stiffness or friction.
Do not copy the supported drape into material memory or reduce thickness to
make this diagnostic pass. This review does not prohibit physically justified
prestressed objects or replace checks on their actual loaded states.
Inspect native cooking diagnostics before accepting a collision precision
budget. Isaac may clamp SDF resolution while the USD attribute retains the
requested value. The task entry rejects this unaccepted profile change;
request a supported profile and rerun all applicable checks, or choose a
different collider representation. Do not use requested resolution as measured
native contact accuracy.
Dense folded surfaces can exhaust the native contact buffer before a usable
state is produced. Retain the overflow as a failed trial; do not ignore dropped
contacts or lower mesh quality to hide it. An explicit bounded
`simulation.json.gpu_max_deformable_surface_contacts` value is authored and
audited in both validation and delivery. Check shared VRAM headroom and rerun
native validation after changing capacity; capacity alone is not convergence.
When a native finite-thickness test fails near a resolved fold, compare the
failed pairs' rest-space distances with the authored self-collision filter.
The hand assay records this diagnostic automatically. It is a hypothesis about
missing native contacts, not a direct readback of the engine's pair list or a
reason to waive the failure. Finer material sampling can put additional local
pairs inside the filter. Respect the installed backend's supported filter/rest
offset range, retain physical thickness, and test compatible mesh, crease and
numerical-envelope choices against full geometry and native controls.

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
S2 and delivery require it. See [the support contract](../../docs/scene/visual_robot_support.md).
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
`materials` (or existing physical `links`), source-only `part_review` and
hash-bound `evidence_files`. Select one of `materials` or `links`, not both.
Use link selection when different parts share a material, such as a transparent
moving panel and its fixed collar. Their separate boundaries and initial pose must
not disappear inside the whole object's silhouette.
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

For physics recordings, see [video physics](../../docs/video_physics.md) or
[replay calibration](../../docs/real2sim_physics.md). Their default route also
requires a reviewed Blender artifact. Their motion initialization does not
approve a primitive as the final model, and estimated scale remains uncertain.
When the user requests a task from its captured initial state, declare the
`task_motion` request in `render_motion_trace.py`'s manifest: bind the actual
result, state file, protocol, ordered samples and requested phases. A preparation
recording cannot substitute for that sequence. Preserve failed attempts as
failed; do not animate a missing grasp or lift. See the task-video contract in
[object mechanisms](../../docs/scene/object_mechanisms_plan.md).

For transparent bodies, separate observed opaque print from transmitted scenery.
Never bake a table, tape or a reflected/mirrored rear label into the front-wall
albedo. The experimental `scene.objects.blender.optical_print` tool proposes a
front-owned dark-ink layer from agent-reviewed source regions and unchanged
physical triangles. Inspect source regions, masks and native calibrated renders;
its output is unapproved. A `fix.look.optical_print` layer needs complementary
maps and a resolved source/mesh/hash-bound review. Unseen lettering remains
unknown. Native opacity and source-ray visibility must agree; printing never
removes solid collision geometry. See `docs/scene/object_review.md`.

For an opaque label on transparent material, `optical_print` also supports
`mode: opaque_patch` with independently reviewed source `polygons_xy`. Preserve
the whole backing and source colours, not just dark lettering. Isolated native
material sweeps are static diagnostics, not scene or physics approval.
Before unforced rigid diagnostics, inspect every colliding independent part
pair, including loose contents against each other. Save the authored failure;
settling cannot repair the exported initial state retroactively. The shared
`rigid_initial_state` screen samples actual closed surfaces in both directions
and preserves hollow cavities; its finite sampling is not an exhaustive
intersection certificate. Optical rays through glass are hypotheses with
thickness/IOR uncertainty. Review source identity and camera/geometry fit
before inventing bends or retention to reconcile an impossible source state.

For repeated thin shafts, verify whole-line correspondences separately from
physical tips. `landmark_sets.triangulate_lines` fits image-space line residuals;
`triangulate_line_sets` proposes unordered axes from reviewed distinct segments,
including partial views. Review duplicate masks and cross-view identities in
source RGB. Keep per-view errors, the original full-object cache and all native
checks. An infinite-axis proposal identifies neither finite length nor support.
Use `landmark_sets.line_plane_uncertainty` to inspect how endpoint pixel noise
propagates to the axis at the actual contact plane. Low image residual alone
does not establish a precise hidden endpoint. Its local independent-noise
estimate excludes calibration error and correlated nearby frames; retain that
limitation and do not turn it into a gate relaxation.
For a visible circular edge, `source_circle.fit_circle` independently estimates
the opening from calibrated edge arcs, with per-view residuals and positive-depth
checks. Label the same physical boundary in each image, leave occluded arcs
unobserved, and distinguish the top opening from an inner sleeve or reflection.
Keep edge labels and camera hashes with the proposal. Inspect omitted-view
sensitivity separately; neither a circular hypothesis nor a small residual
proves that an unseen inner wall supplies a required contact.
When a functional boundary is dark or washed out, use
`scene.capture.exposure.detail_brackets` on the original HDR source, with the
capture calibration and explicit detail crops. Compare positive as well as
negative exposure where relevant. The QA manifest binds the video, calibration,
conversion and extracted pixels; canonical images and their labels stay intact.
Exposure variants are the same observation, not extra views. An unresolved dark
interior remains unknown; do not infer a retention feature from task success.

For loose contents, separate user-confirmed support facts from a chosen contact
family. A confirmed floor contact does not imply that every shaft must also
touch a particular rim or the floor perimeter. Allow source-supported free
floor positions and inter-object contacts before rejecting straight geometry.
If the input explicitly permits approximate poses of secondary loose parts,
record the scoped requirement in `task.json.acceptance` and stop optimizing
their exact image poses. Preserve count, approximate source-derived dimensions,
independent rigid bodies, valid contacts and the primary manipulation passage.
Unknown interior construction remains unknown. Keep previous source failures
and require separate unrelaxed vessel/target geometry checks; a whole-assembly
appearance waiver must not hide an inaccurate opening or manipulated object.
The optional `check_rigid_assembly.py --force-probe-plan` applies a bounded
translation force at the actual native centre of mass, then releases it
completely. An explicitly declared orientation tool adds bounded torque, never
a joint or prescribed pose; both forces and torques vanish during release.
`--companion-plan` adds one current rigid asset with a hash-bound, rigid-only
initial experimental placement. It cannot resize or deform the source mesh.
The runner records actual native states, applied wrenches and contact penetration. This diagnostic
does not certify removal, insertion, full-scene support or delivery by itself;
inspect actual clearance and the complete trajectory before interpreting it.
Check tool capacity against each sampled native mass before a gravity-compensated
trial. A saturated tool that cannot hold the object is not controlled insertion.
Apply forces at the native centre of mass, and record force-free release.
Monitor every relevant rigid pair, including contacts between unactuated loose
parts. Keep peak-time native poses for comparison with actual triangles; a
large discrepancy with reported contact depth needs a numerical/collider
investigation. Do not silently discard either measurement. Contact sensors
must not change the scene's stabilization settings.
Save every native rigid timestep and run `apps/scene/check_rigid_trajectory.py`
against the unchanged source solids. Video-rate frames and the largest reported
contact alone can miss short intersections elsewhere. Report spatial sampling
and tolerance; this remains finite sampling, not a continuous collision proof.
Convex cooking may simplify thin cross sections even when its returned vertices
are a subset of the source. Read back the cooked hull and measure surface error.
Coordinate conditioning may change neither world geometry nor metric cooking
limits. Do not split SDFs at material boundaries unless each resulting solid is
closed; a Boolean-unioned shaft/head often has open material subsets.
For tool tests, an optional minimum-jerk target, end hold and release ramp can
avoid artificial start/stop impacts. Keep finite wrench bounds, record the
actual response and require a fully unforced final interval.
For an orientation fixture, use the complete native COM inertia tensor in the
body frame and rotate its principal axes into world space for damping. A thin
object's smallest inertia is not a valid substitute for its transverse moments;
that can create an underdamped test tool. Validate the fixture separately, bound
all torques and keep all actuation zero after release. Record the actual tensor.
For a straight insertion, `tasks.rigid.insertion_corridor` can propose a path
past existing contents. Its capsule/envelope geometry is a proposal only:
validate the full manipulated surface, opening and native release. A failed
straight-down press alone does not prove an impossible task; conversely,
finding a free proxy corridor does not prove the actual asset can be inserted.
Check the head, handles and other attached geometry throughout the sweep, not
only the insertion shaft. `swept_point_clearance` can screen sampled full-surface
paths against enclosing capsules; it does not approve moving-contact dynamics.
Check the requested stopping pose against the complete source surface as well.
A tilted flat-ended rod can intersect the floor even when its tip centre is
above it. `stop_before_plane` shortens a proposed translation using all vertices
and a disclosed support half-space; it never edits geometry or prescribes body
motion. Reject initially intersecting setups and retain the actual contact and
force-free release checks after planning. Increasing solver precision cannot
repair a tool target that intentionally presses a rigid surface through a floor.
Keep all assay placement changes on the rigid body's transform. Baking a rigid
rotation/translation into collider vertices can change its native SDF grid even
though the visible shape is unchanged; such a probe does not qualify the
delivered collider. Preserve source vertices/faces and validate the actual body
pose mapping in recorded trajectories. Check native SDF/convex cooking fidelity
against the source surface, not only the requested USD attributes. A numerical
contact correction requires measured error, explicit separation bounds and fresh
all-body contact/release/task tests; it is not permission to loosen the gates.
`fit.source_pose_groups` can describe indistinguishable rigid members with one
independently acquired collection mask. Each group requires an exact input
relaxation `source_pose:<object>/<group>`, a named dimension check in the
required native criterion, unrelaxed body-source checks and actual-solid
screening. Group masks must never be copied into purported per-member labels.
Raw source failures remain saved; absent or stale functional proofs fail.
Complete the finite solids, screen all pairs, and test unforced native motion;
a better source fit alone cannot approve retention or equilibrium. Hidden wall
and floor construction remain uncertain, not knobs to tune to task success.
Include mutual contacts when source-resolved shafts approach one another.
Keep alternative contact graphs as separate hypotheses; neither a near-contact
distance nor a fitted contact graph proves force balance. Test existing material
prior cases before attributing drift to friction. A source-derived property
bound must be identified independently of task success, with geometry ambiguity
and the raw failed cases retained.
When repeated native trials drift despite near-contact geometry, use a simple
positive/negative native control to test the numerical representation first.
`assets.rigid_statics.point_contact_balance` can then diagnose a supplied contact
graph using gravity, explicit masses/centres and friction priors. It enforces
equal/opposite internal forces and admits no adhesion or holding moments.
Its polygonal friction cones and assumed contact points are approximations:
feasibility is necessary under those assumptions, not proof of equilibrium in
Isaac. Do not create missing contacts or tune material priors merely to make
the linear solve succeed. Keep the native source-hold and full task checks.

Support parts need their own visible evidence. Review mistaken source instances
before changing geometry; `fit.part_identity_review` binds corrections to source
RGB and archived mask hashes and makes those pixels unknown. New support rounds
check declared material parts separately with actual whole-support occlusion.
A tabletop aggregate score cannot approve wrongly placed legs. Small source
fragments retain diagnostics while search weighting uses observed pixel counts.
See `docs/scene/visual_robot_support.md` for selective shape fitting, bounded
multi-start search and fixed-resolution final checks.
Preserve the source texture sampler through GLB-to-USD transfer; clamping a
repeated wood texture can turn most of a table into a flat edge colour. When
choosing a repeated source patch, reject compact marks rather than painting
them out or repeating screws across the surface. Keep original RGB and patch
provenance, and review unique visible marks separately.
Use the actual modeled support in native task validation. A diagnostic slab
cannot certify contact with the exported table. Preserve pre-union Blender
solids in support rounds; `collision_representation: source_convex` may use
them only when every source solid is closed and convex. Split large convex
polytopes into exact cells for the native vertex limit, never convexify a
concave table or fill the space beneath it. Read back cooked geometry and test
actual support contacts. Native invalid-material contact warnings are failures,
even when the process exits successfully. Numerical cooking fallbacks require
a revised declared profile and a new trial before any acceptance.

For new profiles shared by isolated object checks and a mixed rigid/flexible
scene, declare `external_forces_every_iteration` explicitly and compare its
actual native audit value, not only timestep, solver and iteration counts.
The legacy rigid-only SDK default can differ from flexible-scene authoring.
Keep legacy behavior when the key is absent, but do not transfer old isolated
passes to a differently configured scene. Diagnose unexplained contact preload
with separate normal/friction forces and native velocity/pose traces before
changing geometry or material; retain the original failed trials.


### Required understanding before modeling

Before fitting any object, declare `agent/object_understanding.json` as specified
in `docs/agent/workflow.md`. Inventory all semantic parts, their structures,
connections, material alternatives, expected mechanical responses, chosen
representations, physical links and source evidence. Task-secondary parts still
need semantic, geometric and physical review. Never substitute an outer hull
with a texture for characteristic visible structure. Never infer physical
rigidity from a rigid modeling shortcut. Use material/source priors to propose
appropriate non-task passive, contact and load/release response tests, including
alternative hypotheses and uncertainty. Preserve rigid-object accuracy too.

Candidate collider generation precedes native physical testing; OBJ/delivery
approval requires current `qa/object_understanding/<object>.json` with per-part
judgments. Flexible parts in approximate mode need a physical-plausibility
review and the native task, rather than independent exact-response benchmarks.
Rigid/quantitative obligations retain source-bound, measured Isaac Sim5.1
evidence. Tests use the delivered representation. Refresh all applicable shape,
structure, contact and task evidence after a relevant model change.

### Interior empty space and grasp direction

Review observed solid coverage and empty regions separately for every task-relevant
interior support. A correct outer silhouette or a visible small outlet cannot
justify a full hidden floor. Use source-bound `agent/geometry_checks.json` finite
segment probes (`version: 1`, per-object checks with link, starts, ends, occupied,
source path/hash and basis) to reject filled voids and missing support during
Blender iteration. Keep the exact boundary uncertainty and source-view review;
these probes are finite geometric prerequisites, not native task approval.

For an explicitly requested directional pinch, derive and review the intended
axis in the candidate body frame. Check the actual open pad line during planning and immediately before closure,
in addition to inside/outside assignment and full hand clearance. Its planar
projection is deliberate and must match the input interpretation. It does not
infer a sheet's longitudinal direction from its normal or from a radial fixture
frame. Reject unresolved axes and keep the actual pad trajectories. Record and
replay the same direction policy in friction controls. A failed grasp does not
identify an incorrect material law until source geometry and tool placement
have been checked. For top-down sheet grasping, preserve the explicitly requested approach,
descent, actual open-jaw capture/readiness and closure order.
Keep explicit coupled mode available; never override input-required sequencing.
Use one declared material direction frame in proposals and validation. Bind
fresh preclosure material association and the same commands in the friction
control. Expected rigid closure contacts require a source/task basis; actual
penetration checks still include every partner. Distinguish below-rim insertion
from above-mouth occupancy before changing cloth material parameters.

Closed articulation coordinates do not establish physical seating. Keep existing
cloth/solid seam witnesses and add source-declared rigid mating-surface checks
where the mechanism requires them. `agent/interface_checks.json` can declare
per-object first/second rigid links, reviewed face anchors (with topology hash)
and a physical gap bound. Measure them in native poses; preparation plans can require the same `release_conditions.interfaces` before
release. Bare rim sectors and the folded-cloth exit have different contact
requirements and must be reviewed separately. Gap probes do not replace solid
penetration or force-free retention. Build rigid seating geometry independently
of cloth layer counts or a preparation packing proposal. A modeled gap must not
be introduced merely to avoid closing contact; use native finite actuation and
report a failed closure instead.
For a folded sheet, prefer a reviewed intrinsic material direction over a
fixed world/body vector. The local material map determines its current tangent
at each candidate patch and again after the native approach. A flat-sheet PCA
or exterior-flap bounding box can identify width rather than the longitudinal
route. Justify the requested direction from chart/source correspondence and
verify alignment against saved native states. Fixed body-axis requests remain
appropriate when the task specifies that convention. Directional planning
uses the declared axis; actual contact, material validity and matched friction
controls remain required. Verify the declared alignment bound independently of
a looser experimental planning bound.

Mating requirements are state dependent. A preparation history is a construction
hypothesis, and a failed controller path alone does not establish physical
impossibility. All source, geometry, contact, material and requested task checks
still apply.
Reconcile supplementary-input part coverage with the semantic part inventory.
When a closeup reveals a previously hidden fixed support, add that identified
part to the appropriate geometry input's `invariant_parts` and bind its source
evidence. Geometry invariance does not mean the whole assembly has a fixed pose.
Do not limit closeup reconstruction to body/moving-part labels while silently leaving a
newly visible internal tray, partition or opening unconstrained. Keep moving or
unidentified parts in their explicit state/uncertainty scope.

A released native mid-surface may be nonpenetrating while its declared finite
thickness still lacks solid clearance. Do not mark that mesh admissible or lower
the thickness. `cloth_project --repair-physical-clearance` can propose a bounded
offline correction only with an explicit positive `--envelope-repair-limit-m`.
Its search guard retains material budgets, self contact, support and centre
penetration checks; the final result must pass the complete original physical
and numerical-envelope contracts. Rebuild with Blender and repeat source and
native checks. This option changes no simulation nodes, material properties or
acceptance thresholds and cannot approve the native trace that supplied the seed.

Declare an observed initial articulation interval once in the mechanism's
`initial_state: {position_interval: [lo, hi], basis: ...}` (radians or metres,
inside the joint limits). Reuse it for released proposals, complete task checks
and input-specific hand checks. Do not silently use a looser "closed" angle in
one checker. The initial joint interval does not replace calibrated source-pose
review, trapped-sheet contact or unloaded seating. Existing tasks without this
declaration retain their legacy defaults.

After temporary construction forces stop, remeasure release conditions throughout
the recorded unforced interval and at the final state. A moving part that was closed
under a drive but reopens after release has not met a retained closed-state goal.
Keep physics validity and construction-goal retention as separate verdicts;
`native_proposal` rejects a released seed that lost its declared conditions.

For exact source convex cells, coordinate conditioning can be changed without
changing the physical solid. `revolution_normalization_scale: axis_extents`
normalizes principal coordinates and restores their scales in USD. It requires
SDK-cooked geometry auditing and native GPU collision validation; never accept
CPU-only hulls around flexible objects. Native aggregate-pair overflows require
an explicit bounded `gpu_found_lost_aggregate_pairs_capacity` in the checked and
exported simulation profile. Do not ignore overflow messages or silently lose
contacts. Alternative timesteps remain numerical hypotheses until actual native
material/contact, task and runtime-profile checks qualify them.

A sheet merely lying above a support footprint is not necessarily supported.
When the input specifies an inner material end resting on a tray, declare
`initial_support.material_contact` with its intrinsic axis/end band, a bounded
physical gap and required material-area share near the actual upward triangles.
The authored gap cannot exceed twice the declared sheet thickness. Holes remain
empty, and this geometry screen does not prove force support. Check the complete
native articulation transition as well: the material must remain accessible
when a temporary support moves. Preserve and diagnose an early
slide-out before spending a full hand-closure trial. Do not add a hidden holder
or fill a real opening to retain the sheet.

If the user permits a deeper placement, increase retained material by feeding
the same complete sheet or adding a physically valid loose return above the
actual support. Consume added internal path from the exposed end; never resize
the sheet or substitute friction/material changes for placement. Preserve a
rim-crossing interior pinch region, closing clearance, real support holes and
the requested task. Compare an open-joint, all-force-free retention interval
before another complete grasp. `check_candidate_passive --retention-plan`
records body-local inside/above-mouth/outside area, material loss and at least
0.5s of released observation. Use explicit input/prior bounds, retain failures,
and keep the retention verdict separate from physical validity and task pass.

The last material band of a folded sheet need not touch the tray directly if
a nearby lower fold of that same connected sheet carries it. The opt-in
`initial_support.connected_folds` bounds both intrinsic material path and
end-to-real-triangle distance; preserve the original direct-contact metrics,
physical gap, strict routing and full self/solid checks. Tiny real point/line
contact is only a geometric candidate. Before delivery, every such link needs
hash-bound `support_retention` evidence in its object-understanding part review:
current full native states, source/physics/contact settings and actual released
retention, with no cloth guides, body holder, attachments or constitutive
ablation. `task_and_semantics` does not skip this proof. An open-state proof
still cannot replace the closed initial-state or actual contact-grasp tests.

A material-valid folded pose can still carry enough artificial membrane
prestrain to recoil on release. Inspect the unchanged-material release-energy
diagnostic and actual native motion, not only a strain ceiling or low RMS.
If direct geometry authoring/projection cannot form a relaxed packing, use the
existing `initial_state.Checked(..., construction_plan=plan)` path to begin
from a nearly isometric construction pose and physically form the drape with
bounded, disjoint finite-force handles. This entry keeps all material and
self/solid checks, emits `construction_state_admissibility` instead of source
initial-state approval, and preserves the original contract. Keep the source
mechanism declarations intact; do not delete their support conditions from a
candidate hypothesis. Put those original support/distribution conditions in
the preparation release checks, clear every guide/drive, then rebuild using
the ordinary source policy and rerun a fresh unassisted test. Construction
grips cannot qualify task friction or final retention.

For local finite-thickness bend review, triangle-centre graph distance is not a
mesh-independent material distance on anisotropic triangles. Use bounded paths
between the actual closest material witnesses, with segments inside connected
material triangles through vertex/edge sites. No path may jump a cut or another
sheet. Independent extrusion boundaries and positive director Jacobians remain
mandatory at the unchanged thickness. Shared larger extrusion witnesses may
cover overlapping local tests; a failed shared witness falls back to the original
per-pair test. Raw midpoint-distance failures remain recorded.

When the whole flexible envelope changes, Blender may recenter the full asset
while preserving primary-rigid placement. Fixed probe coordinates must follow
that frame change. Source segment checks may declare `origin_link` to translate
probe coordinates with a rigid link's bottom bounding-box centre; axes remain
model axes. Reanchoring never changes the source occupied/empty labels. Native
candidate probes inherit declared memory capacities unless explicitly testing
another budget, and retain overflow errors as failed experiments.

Review initialization energy before tuning a cloth's apparent softness or retrying
hand forces. A complete flat chart swept through two curvatures can introduce
large in-plane prestrain even when its outline, area and single-sheet topology
look correct. Compare group strains and the reported spring/continuum energy
with the object's mass and scene length scale; these are diagnostics, not native
energy measurements or universal rejection thresholds. Explain any intended
preload by source-supported loads/history (for example an inflated or stretched
object). Otherwise reduce geometric prestrain with bounded, contact-constrained
initialization at unchanged intrinsic coordinates before choosing material values.
Input-authorized flexible appearance tolerances do not justify an unexplained
loaded initial state. Numerical backend/spring changes need independent response
and contact controls; a previous control at different sampling, damping or padding
does not qualify a new scene. Retain failed native starts and distinguish them
from finger-closure failures.

In PhysX107.3 an articulation's parent/child links never collide with each other:
`physics:collisionEnabled=true` on that joint is insufficient. The native paired
joint-contact assay demonstrates the filtered control and effective alternatives.
`sim.contact_carrier` restores requested source contacts with an exact copy of
the original collider on a coincident fixed articulation component. Original
colliders are disabled; aggregate source mass, COM and full inertia are preserved,
and no mechanical DOF is added. Required-contact audits reject the flag-only
case, disabled component self-collision and filters that suppress the actual
pair. Check native carrier/owner pose parity and aggregate native inertia; retain
original disabled-pair policies and source collision cooking. A successful empty
moving-part seating control does not approve the cloth-pinched state or full task.

Choose the particle spring topology as well as its constants. A balanced square
network puts both cell diagonals in the membrane law. On an isometric fold along
the rendered diagonal, the opposite chord shortens: a flat traction calibration
therefore cannot establish bending behavior. For folded sheets,
`particle_lattice.chart(..., model='triangle_surface')` uses only actual triangle
edges for membrane response and keeps the opposite diagonal as a bending chord.
It retains real cuts and the immutable source chart. The previous balanced option
is preserved for explicitly qualified uses. New topology, damping, spacing or
padding requires fresh native response/contact controls and full task checks;
distance bending is still an approximation, not continuum flexural-rigidity
identification. Never fix a sampling hole by inflating physical sheet thickness.

When a Blender provider rebuilds an already centered metric proposal, carry its
opening metadata and joint anchors in that proposal's frame. Do not copy a
hard-coded opening from an older differently centered hypothesis. Reanchor the
primary rigid pose, and independently compare metadata with the actual geometry.

Treat an unmeasured sheet thickness as a material-class hypothesis, not a known
object dimension. An alternative inside its original uncertainty can be tested,
but preserve the interval and its prior-only provenance; do not claim that a
successful task identifies the true thickness. Change the actual native contact
envelopes consistently and refine the complete source chart for coverage, rather
than changing only a checker. Repeat geometry, response/contact and uncertainty
checks. `cloth_resample.surface` transfers a pose through actual material
triangles; extrapolation is explicit and bounded, and the requested new source
boundary/holes must be independently checked. Sampling proposals grant no native
or source approval.

For real articulated geometry, check the complete authored travel as well as
open and closed endpoints. `blender.articulation_path` samples actual closed
source solids bidirectionally with other joints held at capture state; record
its finite sampling scope and follow it with native motion/contact tests.
An initially plausible moving link can intersect its partner only midway through opening.
Never explain that collision impulse as a flexible material identification.
`Ctx.initial_articulation_state()` must be measured immediately after reset,
before settling or fixtures: native link poses must preserve the baked source
initial joint state. A collision flag or a final closed-seat measurement does
not establish this. Keep failed diagnostic traces failed.
`mechanism_pose.hinge_axis_proposal` can propose an axis offset while preserving
reference-pose rigid geometry. Offset and hidden hinge details remain explicit
source-reviewed hypotheses, not measurements. Recheck source views and native
contacts; never change a task-zone scale or a cloth material to hide a bad hinge.

Check particle contact coverage on actual deformed native triangles throughout
passive, hand and full task assays, using the current physical-case contact
radius. An authored dense mesh can open particle contact gaps after in-plane
stretch; approximate material-response policy does not waive those gaps.
Inventory every active non-hand collider for a friction ablation. Account for
source-preserving collision cells and fixed collision carriers by their verified
physical owner, rather than assuming each owner has one active /surface mesh.
Before a temporary closing fixture, inspect source-supported closed-state
geometry against the complete flexible object. A packing above the open mouth
may be impossible within the closed assembly. Do not tune material against an infeasible
closure command. Prefer an independently admissible closed initial hypothesis;
any permitted change of flexible fold pose remains conditional on the input's
actual native functional criterion, while rigid source geometry, full material,
real support, non-threading and collision constraints remain mandatory.

Compare the initial elastic energy with the object's weight before diagnosing
rapid flexible motion as a material problem. A doubly curved procedural sweep
can preload an otherwise inextensible sheet. Prefer an isometric seed and
explicit, bounded temporary physical preparation when that preload dominates.
`initial_state.Checked(..., construction_plan=...)` permits intermediate support
by declared finite-force grips active from initialization. It preserves all
material, self/solid geometry and thickness checks, retains the omitted final
support declaration in its contract, and writes only
`construction_state_admissibility`, never `initial_state_admissibility`.
This is no source, task, or delivery approval. Release every fixture, rebuild
with the ordinary source contract, and repeat unforced, source, semantic and
native task checks before adoption. Actual Sharpa acceptance still forbids
cloth forces/attachments; temporary preparation must not enter the final asset.

For vertical sheet preparation, identify which temporary grip bears the weight
and which only guides the material. Splitting upward gravity compensation between
upper and lower grips can compress the sheet and create artificial local folds.
Use explicit gravity shares and a bounded support ramp; retain actual force and
tracking measurements. Any gentle guiding tension remains a disclosed finite
fixture load, never a material adjustment or evidence for final task acceptance.

When baking native geometry back into an agent model, transform rigid corner
normals with their actual link poses and recompute cloth normals after deformation.
Carry the native trace's hash-bound material/representation pair into the fresh
geometry check; do not silently evaluate it with a stale canonical thickness.
The paired policy still preserves source support and all required mechanisms and
contact effects. A geometry proposal neither adopts those hypotheses nor grants
source, task or delivery approval; restore any temporary material before final
validation and keep the original material dimensions and provenance.

Numerical contact padding must not loosen the solid-penetration gate. Use the
same sampled physical sheet thickness and rigid discretization budget in passive,
source-response, hand and task checks. Keep the legacy offset-based fallback only
for existing particle assets that have no physical thickness declaration; do not
present that fallback as a measured thickness. Report padding separately and
retain actual penetration depths so numerical controls cannot approve bad contact.

After Blender recenters a construction model, rebase temporary grip paths with
`preparation_geometry.reanchor` against both actual rigid meshes. It verifies a
translation-only shape/topology match and adds an initial-target frame check;
do not reuse paths from the old model frame. Unchanged hinge coordinates need
no translation. Keep force release separate from construction-goal success:
clear every actuator and record subsequent native motion even if an inferred
packing goal failed. Preserve that failed goal and reject released acceptance;
diagnostic motion or an earlier valid geometry seed cannot turn it into a pass.

Changing the flexible backend/contact profile invalidates a previous packing's
numerical clearances. A material-thickness pass does not establish compatibility
with larger particle envelopes. Prefer native preparation on the final backend
from a low-prestrain seed; keep support tools until the closing mechanism carries
the load, then clear all tools. Optional `release_conditions.material_support`
rechecks the original actual support/routing and material-end contact contract
after release. Contact coupons alone never qualify whole-sheet dissipation or
source stability; reject gross stretching/oscillation before trying the task.

For insufficient insertion, review all-view relative orientation before refining
hidden packing. A well-fitted cylindrical host can hide a large direction error
in its loose material or insert. `orientation_review` supports explicit CPU
geometry work while GPU physics runs. Retain uncertain views and rerun actual
geometry checks after a direction proposal; source ranking is not a physics pass.
Prefer a directly authored deeper initial hypothesis over replaying an unseen
packing history. `cloth_feed.redistribute` preserves the supported spine end and
longitudinal arc distances while consuming more path inside and shortening the
free exterior portion. Preserve the complete chart and independently check
triangle strain, initial energy, self/solid gaps, original support and unforced
retention. A coarse source boundary, fixed angle or controller target is not
proof of a retained or graspable material state.

For a sheet that exits after opening, distinguish material below the rim,
material arched above the opening and actual supported material. Use actual
cavity triangles, partial shelves and inferred contents to plan a deeper loop;
`preparation_geometry.cavity_descent` only proposes temporary finite-force
handle targets. Keep a source-confirmed end on its declared support and never
thread an excluded outlet. Release all guides, rebuild and repeat unforced
closed/open and task checks with unchanged material. Nominal cavity depth is
not a free-space certificate.

A loaded contact pose may require a bounded physical-thickness repair without
unloading it to numerical rest padding (`cloth_project --contact-envelope
physical`). If independent per-ply corrections stall, `--clearance-solver
coupled` proposes simultaneous contact-compatible movements; full nonlinear
material, support and self/solid gates remain unchanged. Keep failed proposal
goals recorded and judge a fresh candidate against the actual input criteria.

Keep the same inside/outside source-finger assignment at the initial hover
and after opening, even when the input allows either finger. Interpolate tool
orientation continuously and bound free approach acceleration by actual source
mass and finite force reserve. Observe arrival and material capture before
closure. Record the joint-motion duration as a motion hypothesis; no fixed
fast timing or controller target can establish a successful grasp.

Before compensating for slipping with material parameters, recheck source
relations between the support, hinge and material exit. Circular hosts hide
relative-part pose errors. Solid/void probes at an inferred placement cannot
independently certify that placement. Add source-supported broad layout
relations (`geometry_checks` `radial_relation`) as well as occupied/empty
regions, retaining the old failed result. A fixed-leaf pose hypothesis must
preserve its rigid shape, mass, collision directions and consistent fixed-joint
frames; rebuild and repeat actual support, force-free opening and task checks.
Inside area alone is not load-bearing area or friction-force evidence.

For a loose sheet retained between contents and a container wall, inventory the
contents separately from the rigid shell. Record rigidity/compressibility and
packing as uncertain physical hypotheses; an unreviewed fused rigid fill must
not silently supply the task's retention. Test actual native compliant/cloth
coupling before changing representations. Preserve legacy rigid, solid and
particle paths. A coupon's external pressure fixture is not a delivered holder.

Require light-load retention and finite extraction, retaining exact slip,
material strain, volume Jacobians, native errors and contact distances. Use the
shared solid-contact error policy independently of numerical padding, retaining
the stricter half-thickness-envelope screen separately; do not invent a new
micrometre-level pass condition that silently exceeds the pipeline contract.
Matched friction counterfactuals must share their prepared state: preload at
the same nominal friction, then change existing contact materials before the
load and verify native coefficients after a real step. Zero friction from t0
is a different construction experiment. Compare a no-preload control as well.
Only a fresh full-sheet closed/open release, source/semantic review and the
requested actual contact task can approve a source scene.

For conforming `volume_fem` initial-state hypotheses, preserve the original
tetrahedral material reference separately from posed nodes via `VOLUME_INITIAL`.
Keep mass based on the reference volume, and require independent native USD
reload/topology/reference controls. Legacy explicit-state authoring remains
unqualified; existing default legacy representations stay available.

Repeated crossings during local packing are a reason to reconsider the whole
initial pose. Offline construction must retain the entire material chart,
fold history, volume reference and uncertainty. A common-station curve can
strain offset layers; independently matched parallel arcs can shear transverse
creases. Measure both, reduce prestrain with complete contact constraints, and
release afresh in Isaac. Include compliant solids in offline obstacle geometry
without freezing them in native physics. Construction, source, task and delivery
checks remain separate.

For continuum-sheet preparation, prefer the explicit `cloth_project --objective
triangle_metric` option when edge-error minimization leaves shear. Inspect
actual principal strain and stored energy; a small edge RMS alone can hide
substantial strain in narrow triangles. Keep the full original material and
all independent guards. Match initial equilibrium construction to the input's
actual articulation/contact state, with recorded finite temporary tools and
complete release before fresh source-state and task checks.

Save the actual triangle faces with every prepared sheet checkpoint, alongside
its positions and unchanged intrinsic material chart. When resuming a grid,
use `cloth_rest.triangulate_grid(..., faces=saved_faces)` to retain its checked
surface and spring connectivity. Selecting new diagonals from moved vertices
is remeshing: even unchanged positions can then fail finite-thickness contact.
A legacy point-only checkpoint has no transferable surface acceptance; recover
and bind its exact original topology or rerun all affected construction checks.

When feeding folded material past a partial shelf, a collision-free handle
centre is only a route proposal. Inspect neighbouring folds, held material and
available intrinsic length. Use passive-probe crossing locations in the original
material chart to locate the failure before changing material or force limits.
Coordinate preparation guides where needed, preserving complete material and
all physical checks; then remove every guide and rebuild the released source
state. Assisted geometry snapshots remain construction seeds until that passes.

When adding guides to a turning material region, preserve its intended frame
and layer ordering. Check relative target offsets, not only initial frames and
rigid clearance. `tasks.mechanisms.guide_frames.paths` shares a proper frame
and bounds chord shrinkage of sampled position targets. It supplies finite-force
guide paths, without imposing a rigid material patch or granting acceptance.
Inspect unheld opposing plies as well; do not assume a folded sheet is glued.
Use original material indices to distinguish layers and construction guides
from actual task contact. An inner and an outer ply can contact different
partners; declare those regions explicitly without claiming pressure from
proximity, and retain whole-sheet self contact, released hold and extraction.

Keep native cold initialization in the qualified task profile. A tiny reset
timestep can amplify contact-correction velocities; small initial displacement
does not establish stability. Check loaded, unassisted cold reloads, including
bounded resting motion, rather than inheriting a warm construction trace.
Installed Isaac5.1 rejects an independent selfCollisionFilterPose; use the
supported material bind-pose fallback only with consistent simulation/display
references, and reject native capability-fallback warnings. Audit the candidate
against its matching semantic inventory, including actual flexible material
motion in a declared rigid frame; never reuse a deployed rigid inventory for
a flexible candidate.
