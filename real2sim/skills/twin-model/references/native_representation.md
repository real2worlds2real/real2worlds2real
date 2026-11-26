# Native representations, material checks and contact controls

Maintained reference for [the modeling skill](../SKILL.md). Apply the sections relevant to the current representation and task; preserve all applicable local gates.

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
[the rest-coordinate contract](../../../docs/scene/object_mechanisms_plan.md).
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
release. A cloth material/contact pass does not imply that the bottle, lid or
other rigid components remain in the captured initial state. Measure full rigid
surface motion (including rotation) against the declared passive-response
bounds, and inspect lid opening, stored preload and support motion. Retain the
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
For an explicitly requested friction pinch of a native sheet, use the optional
`hand_cloth` tool with the current candidate, session hand plan and criterion id.
It runs free-hand preflight, positive and matched zero-finger-friction trials
sequentially. Only a verified criterion can be registered. This is input-condition
evidence, not task or delivery approval. Preserve raw source failures when an
explicitly declared pose scope is relaxed; use the separately verified effective
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
against rigid objects, including the opened lid, as well as the current flexible
surface. Check the closing fingers against rigid objects separately. Clear
endpoints do not imply a clear descent: lid contact can block wrist tracking
without penetrating the lid. Inspect actual wrist errors and native rigid
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
after lid opening still require native evidence. Diagnose source-supported
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
not material damping. When moving lids or tools penetrate a flexible surface,
compare this budget with actual relative contact motion before changing cloth
stiffness. Retest moving contact, self-contact and unforced motion after changing
it. A closed-lid static pass does not prove an opened sheet stays supported;
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
[proposal tool contract](../../../docs/scene/object_mechanisms_plan.md).
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

Contact mechanisms such as screw threads remain separate bodies with real mating
surfaces and no substitute joint. Fit visible functional features and shared
mechanism parameters while retaining all rigid accuracy thresholds. Unseen
threads/material properties remain inferred, even if task feasibility passes.
For threads check feature/SDF/offset/timestep precision, bounded-force screw-on,
contact-derived lead, retention, screw-off, release and pull-without-rotation.
Deliver passive objects; test guides/fixtures must not remain in the asset.
For screw closure, derive the actual seating surface from the generated profiles
and tighten with bounded torque until native contact sustains an angular stall.
Stopping above the hard seat does not establish tightening or retention. Measure
lead and lateral drift at functional axes, including offsets from asymmetric
body origins. Preserve separate case directories and partial failed traces.
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
State how mating materials combine, including thread friction and possible seals.

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
