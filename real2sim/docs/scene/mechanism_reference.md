# Advanced native mechanism and preparation reference

Read the relevant sections together with [the mechanism contract](object_mechanisms_plan.md). These are representation-specific controls and diagnostics, not universal evidence that every backend or task is qualified.

Preparation plans can declare optional `checkpoints`, evaluated against actual
native state before that step's fixture commands. Each entry requires exactly
`name`, `time_s`, `conditions` and a nonempty `basis`. Names must be unique;
times must increase and precede fixture release. Conditions use the same
interface as `release_conditions`: `maximum_hinge_angle_rad`, `container_state`,
`material_support` and `interfaces`. For example, an agent can inspect actual
material-end support before starting a closing motion. This is an explicitly
chosen construction condition, not a universal requirement that every object
follow the same preparation sequence.

An unmet checkpoint stops the candidate trial and retains the failed state;
it does not move material, delay the checkpoint or grant a task pass. Reports
include `checkpoints`, `scheduled_checkpoints`, `checkpoints_pass` and
`failed_checkpoint`; the runner separately records
`stopped_after_preparation_checkpoint_failure`. A released geometry proposal
requires every declared checkpoint to have passed. Independent release and
post-release conditions still apply.

Packing proposals must use the actual support surface when holes or separated
pieces matter. `assets.planar_support_placement.surface_placement` searches
translations of an unchanged convex material footprint over actual planar 2D
triangles. `clearance_m` and `step_m` are explicit; `max_candidates` bounds the
search. The optional `collision-geometry` dependency supplies Shapely. Triangle
union preserves apertures and disconnected regions. Exceeding the search budget
is inconclusive, not proof that no placement exists. The helper neither scales
the material nor certifies native support or contact.

`preparation_geometry.footprint_hull_placement` only screens the external hull;
it cannot establish support above an outlet. After Blender changes only the
model's centering translation, `preparation_geometry.reanchor` can rebase grip
paths against unchanged indexed rigid meshes. Changed geometry is rejected;
the rebuilt candidate needs its own mesh binding and native checks.

Construction performed with a different numerical profile remains a geometry
experiment. Candidates marked `requires_numerical_restore` can only supply a
`geometry_seed`; they cannot supply a released candidate. Restore a qualified
profile and rerun the physical, source and task checks before acceptance.

The agent's `native_cloth_proposal` tool can propose geometry from a complete
passing native trace after construction forces release. It bakes rigid-link
geometry and real joint frames in the selected primary rigid frame, preserves
the full intrinsic cloth and face ownership, and omits temporary fixtures.
Generation `ok` does not mean authored/source/native/task approval. Check its
separate `authored_ok`: native SDF tolerance can leave penetration that the
stricter authored check rejects. Rebuild through Blender, correct admissibility
where needed, and rerun every original gate without preparation fixtures.
If preparation fails on localized compressive strain, retain its material-space
hotspot and contact measurements. Compare an inferred packing route with a
wider gathering footprint or slower staggered fixtures, and test uniform mesh
refinement at the same physical sheet size and priors. Do not narrow the cloth,
add permanent paper folds, increase numerical thickness or relax strain limits
to hide an unresolved crease. These are construction/resolution controls, not
evidence that a material constant was identified.

New surface-FEM material-state contracts also require independent native
membrane and bending-response controls before task acceptance. The diagnostic
`apps/scene/check_surface_response.py` applies opposite finite edge tractions to
a flat strip, retains the transient, and compares interior extension with a
small-strain uniaxial continuum prediction. It accepts the declared physical
prior file/link and nominal/low/high case. No cloth positions or fixed nodes are
imposed. This is an isolated numerical check, not a source demonstration or a
measurement of the real object's properties.

`apps/scene/check_surface_flexure.py` separately measures a small-amplitude
free-strip bending frequency. It uses no gravity, external force, fixed nodes,
kinematic targets or positions written after reset. Compare actual resolved
oscillations with the slender-strip beam approximation; do not approve a
frequency from a single endpoint. Its damping is zero for this isolated control,
so real material damping still needs source and task validation.

Declare immutable results in `agent/material_response.json`, version1, with
`controls["object/link"]["nominal"|"low"|"high"]` containing `membrane` and
`flexure` entries, each with a session `qa/.../result.json` path and `sha256`.
The task checker requires the actual
native timestep/iteration budget, material values and tested resolution to
match. It checks the complete settled trace and current implementation hashes,
as well as the result flag; a successful mass/collision control alone cannot
qualify elasticity. Missing, stale or failed controls block task approval for
new surface cloth. Rigid and legacy contracts keep their prior rules.

Neither control qualifies large-fold behavior, self-contact, clamping, wet
adhesion or a packed initial pose. Preserve those source/native/task checks and
uncertainty trials.
If a finer mesh behaves softer at unchanged material values, first examine
numerical convergence instead of treating that response as an identified
stiffness. See the installed [OmniPhysics107.3 deformable
limitations](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/deformables_beta/deformable_beta.html).

Unforced source-state validation samples the entire declared interval at no
more than 50 ms spacing (or every native step when that is coarser). Initial,
intermediate and final native states are saved. An endpoint recovery cannot
erase a transient collapse. Diagnostics first distinguish source/initial-state
failure from later response mismatch. Agents inspect source ambiguity,
folds, clamping, support, thickness, self-contact and material hypotheses before
choosing a new candidate. A viewer's description such as “too soft” is a review
trigger, not a measured modulus. Keep source fidelity, strain limits, task
motion and prior sensitivity checks unchanged throughout that iteration.

`scene.objects.part_label_review` prepares source-only part identity reviews.
Text-query alternatives and positive/negative pixel-box annotations can resolve
white-on-white cap, packaging and sheet ambiguity. Source optical flow or source
detection bounds may propose boxes; they cannot approve masks. Proposals keep
the raw queries, source RGB hashes, annotation snapshots and visibility inputs.
An agent inspects every source component sheet and records retained visible
components in `review.json`. Bind its directory as observation `part_review` and
hash both review and manifest in `evidence_files`. Independently reviewed visible
foreground can supersede a broad enclosing-part query that merged the target;
merged or hidden material cannot receive that status. Source-confirmed material
seen through a transmitting shell needs `visibility: "direct_or_transmitted"`
in the component review and a separate source-backed optical contract. Identity
does not certify exterior depth. Independent
robot occlusions and background remain active. Existing object views are not
dropped to improve a fit score. Final Blender reports include every part view.

`python -m calibrated_scene_reconstruction.scene.objects.candidate_response` diagnoses an unapproved native
probe against current source pixels before export. A same-capture object round
supplies only the base-body pose proposal; actual per-link rigid poses and cloth
nodes are projected into the measured cameras. It records initial and sampled
part accuracy, visible-material motion, and source/native comparison images.
Geometry, initialization and later response failures have separate diagnoses.
This tool never approves a task or replaces native task and prior-case checks.

Validate a native material feature by a response experiment, not merely by
reading its authored USD attribute. Source response alone also cannot establish
that a parameter is implemented. Preserve failed capability controls and avoid
using a new constitutive feature until its effect is distinguished from an
appropriate negative control in the installed Isaac version.

Blender geometry search now applies the existing authored material-state checks
before scoring a cloth candidate. `blender.initial_state` measures intrinsic
strain/area, required triangle self-intersections, and solid penetration using
the same1e-6m authored-stage criterion. Each measurement is retained, including
rejected hypotheses. Actual mesh bytes bind the result, so regenerating an
artifact at the same cache path cannot reuse an old pass. The declared material
validity limits are part of round provenance and revalidated at export. Rigid
objects retain their existing path. This early rejection saves invalid source
searches; it does not replace the real native source-hold/task tests.

An optional `fit.parameter_hypotheses` list can seed up to32 bounded agent-model
parameter combinations. Every combination uses all unchanged source-object,
functional-part and material-reference constraints and the same authored-state
admissibility. These remain initialization proposals, with complete final CPU,
native and task gates required. A changed loose flap keeps the measured base
body at the same world centre during these comparisons.

The bounded fit searches only coordinates with nonzero ranges. Frozen shape or
pose coordinates are reconstructed outside the optimizer; they must not create
duplicate simplex vertices. An entirely fixed candidate receives one complete
evaluation, without a degenerate search. Parameter bounds remain construction
hypotheses: source evidence can require revising an overly narrow shape family
before changing a material estimate.

Yaw, bounded shape hypotheses and local refinement use the raw observation loss
plus normalized deficits of every whole-object and functional-part IoU/local
gate. The raw loss is still reported independently. This directs proposals
toward the remaining acceptance constraints instead of improving a mean score
while leaving a required local gate behind. Invalid authored geometry receives
no candidate acceptance. Final CPU measurements and all thresholds are unchanged.

Every new round snapshots its effective fit request, including conditional
shape hypotheses, optimized parameters, pose controls and model path. This
records actual runtime overrides rather than assuming the config on disk is
the request used by an isolated experiment. Propose coordinated parameter
seeds for distinct shape families: a changed bend profile with old twist and
cross-sectional curvature may be materially inadmissible before fitting.
Keep those failed seeds and revise the proposal; never increase the material
strain limits to make them searchable.

An agent may specify `fit.proposal_fixed`, a mapping of declared numeric
`FIXED` sampling settings, to search with a cheaper version of the same Blender
model. This is optional and has no implicit object-specific resolution rule.
Coarse geometry still receives source/material/admissibility checks, and its
builder refuses delivery artifacts. The final candidate is rebuilt with every
original `FIXED` value, reanchored to the same primary rigid-body placement,
and checked against all source views on CPU. Full material/contact checks,
native response and task gates remain required. Proposal metrics are preserved
separately; a good coarse fit cannot approve a failing full-resolution model.
The search budget reserves a full-resolution refinement phase (default up to60
evaluations; `full_refine_evaluations` may specify10..300 within that budget).
If the coarse optimum fails the full material/contact screen, bounded parameter
backtracking toward the initial full-model proposal seeks a checked seed and
records every failure. Then full-resolution refinement and exact CPU evidence
still decide acceptance. Failure to find any admissible full seed stops the
round; no strain limit, visibility domain or source label is changed.

An observation can declare `transmitting_links` and `transmission_basis` when
source RGB independently establishes a translucent shell. Bind the optical
review and its source images in `evidence_files`. The model must also contain
actual transmitting PBR faces for those same-assembly links. Part visibility
then uses a straight-ray proxy through those faces, preserving original face
IDs for visible-material motion. This is not a refraction, haze or transmittance
measurement. Opaque occluders still apply. Optical filtering never removes
faces from solid collision, strain, self-contact or task feasibility checks.

Every unforced source-state sample also records intrinsic strain, triangle
self-intersection and solid penetration using the shared `assets.cloth_state`
checks. Runtime penetration tolerance comes from the actual native contact
offset/discretization; it does not relax authored-state penetration limits.
Save each sampled state before continuing, so a later numerical failure does
not erase earlier physical or source discrepancies.

Candidate native probes snapshot their actual arguments, candidate bytes,
material/mechanism plans and runner source before initialization. Failed runs
retain their error and each completed native state. `candidate_response` emits
an explicit incomplete-experiment diagnosis instead of expecting a success
file or treating process exit status as a pass. A failure does not identify
stiffness. Older probes without a saved protocol cannot certify their settings.
Source-response diagnostics require checked solid penetration and contact
sampling as well as material strain and self-contact; optical transmission
does not bypass these checks.

`material_motion_profile` localizes transient movement along the intrinsic
material axes. Measured primary-rigid poses remove only assembly motion, leaving
every actual flexible node's displacement. Source-hold reports and candidate
diagnostics preserve the per-region displacement and worst sample. This helps
an agent inspect whether contact-side material moved or only the free region
bent. Regions are not automatically labelled as a clamp or free end, and their
motion does not identify stiffness, mass, friction or fold memory. The original
source reference and acceptance gates remain unchanged.

Material-state reports also retain the worst eight actual triangle strains,
their immutable material-coordinate centres and their supplied-state centres.
Inspect these locations before changing a modulus: an assumed sharp bend,
compressed layer spacing or a contact initialization can concentrate membrane
strain even when total area looks correct. These are strain measurements,
not stress/yield identification or automatically assigned clamp labels. The
existing strain and area limits remain unchanged.

The candidate probe also writes `initial_elastic_energy.json` for continuum
sheets. It estimates plane-stress StVK membrane energy from the unchanged
material chart, E, Poisson ratio and thickness. The reported whole-sheet speed
scale `sqrt(2 U / mass)` helps expose a highly stressed initialization despite
a small mean geometric error. It is not native stress readback, a predicted
velocity, an identified material property or a new acceptance threshold.
Compare authored and native initial states before attributing movement to
stiffness. A failed source hold still fails regardless of this estimate.

For folded sheets, an extraction fixture must state its material footprint.
A full-width material-edge strip can apply forces to multiple folded plies
at once even though it does not prescribe positions. The extraction check
therefore defaults to a compact material-coordinate patch, bounded to the
smaller of10mm and8% of material width, with the same depth bound. Patch
location and size remain fixture assumptions: they do not identify a source
grasp, prove physical gripper access, or validate a fold-count hypothesis.
Record selected material extents, nodes and mass. Never include distant
material intervals merely because their folded world positions are close.
`check_candidate_passive.py --legacy-full-edge-grasp` preserves an explicit
diagnostic comparison with the previous broad strip. A changed force fixture
does not fix or approve a failed passive hold, material law or construction.

When source appearance suggests cloth-like behavior but native motion looks
too soft, compare unchanged geometry/material under real gravity and an
explicit zero-gravity diagnostic. Motion without gravity can expose initial
stress, natural-curvature or contact problems; it cannot uniquely identify
their cause or a bending modulus. Disabling contacts is a separate negative
control and any resulting intersections remain failures, including at native
initialization. Keep actual native states and force/contact settings. A fitted
outline alone does not verify the number, spacing or continuity of folds.
Material-state reports also cache indexed surface connectivity: edge-connected
components, boundary/nonmanifold edges, duplicate triangles and inconsistent
winding. A continuous folded sheet and several disconnected overlapping
surfaces must remain distinguishable. These counts are diagnostics, not an
inference that the source really contains that number of folds, nor evidence
of layer adhesion or cloth-like appearance.

The optional source proposal loss includes a soft two-sided morphological
opening term for broad outline errors. It retains the global loss and every
view; exact CPU components and all acceptance thresholds remain unchanged.
Pose proposals can perform a bounded CPU polish when an approximate search
raster is near the source gates. This changes xy/yaw only, never height,
scale, source labels or geometry. Preserve the actual CPU evaluation trace.

`sim/discrete_bending.py` and `sim/cohesive_interface.py` are unregistered
constitutive experiments. They apply internal elastic impulses before native
membrane/contact steps, with no position writes or rigid attachments. The
linear solve must converge; an unconverged impulse is a failed experiment.
The latter uses an objective point/triangle interface frame, finite bilinear
traction/separation and irreversible debonding. Initial interfaces and all
strength/energy assumptions need explicit evidence and priors. Wet appearance
does not identify adhesive strength or justify permanently bonding layers.
Its simplified quadrature is not a validated mortar or fluid model. Both
modules require independent force/torque, native contact, release, mesh and
timestep controls, source response and task checks before any adoption.
Their native candidate flag archives the descriptor, natural angles, interface
data and imported force-source hashes. Neither prototype is installed by the
production loader or represented by a standalone USDZ; diagnostic success
cannot certify task/export/runtime support. A production extension would also
need common-loader registration, delivery serialization and reopening checks.

Native cloth authoring records `constitutive_authoring` in the stage audit.
Particle distance-based `bend_memory` and surface-FEM isometric `bend_reference`
are different descriptions; the audit names which one is authored and which
one is unused. Neither an authored attribute nor an alternative description
establishes equal physical behavior. Preserve native response controls and
unforced source checks. A mathematically valid ideal-fold reference rejected
by the installed native cooker is an unsupported prototype, not a task model.

The optional `cloth_rest.bend_angles` describes independent signed dihedrals,
but production authoring currently rejects it. In the installed Isaac5.1
adapter, matched flat/positive/reversed-angle controls had identical native
trajectories; an isometric `bend_reference` positive control did respond.
`check_surface_bending.py` retains explicit diagnostic modes for this comparison.
An accepted schema attribute is insufficient evidence of an implemented force.
Disjoint per-triangle rest coordinates also failed native creation. The
[107.3 implementation limits](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/deformables_beta/deformable_beta.html)
require matching simulation/rest topology. These are installed-path findings,
not claims about all backends or future versions. Keep their failed evidence;
do not use unsupported fields, a strained captured rest metric, or fixed nodes
to make a passive source check pass.

Geometry regularization must also check whole triangles. Vertex spacing alone
can miss an edge passing through another face. The optional preparation helper
can separate detected face pairs before native creation, then rechecks the
original material metric, triangle intersections and solid penetration. This
does not write a simulated pose or approve a relaxed replacement for the source
reference. Save the prepared geometry and its failed or passing measurements;
fresh source fidelity and native dynamics remain mandatory.

Candidate diagnostics stop at the first sampled physical failure by default,
saving the measured duration and every completed state. Explicitly continuing
an invalid diagnostic is allowed for investigation or a labelled preview, but
cannot become task evidence. Input records bind the candidate and actual mesh
bytes, alongside material/mechanism plans and the runner source.

Before interpreting a passive mismatch as a material problem, the agent can run
`python -m calibrated_scene_reconstruction.scene.objects.source_motion SESSION OBJECT PART_PROMPT RIGID_PROMPT
--exclude-prompt TRANSMITTING_PROMPT --output SESSION/qa/...`. It tracks existing
reviewed source pixels, compares calibrated epipolar residuals with an independent
rigid reference, and saves match boards and input hashes. Forward/back tracking
and local correlation filter correspondences; epipolar errors never filter a
motion verdict. Consistency does not rule out motion along epipolar lines, and
insufficient tracking is not a static pass. Review RGB before adopting a source
state assumption. No video resampling or source-view deletion occurs here.

`python -m calibrated_scene_reconstruction.scene.objects.source_landmarks SESSION OBJECT PART_PROMPT
--exclude-prompt TRANSMITTING_PROMPT --output SESSION/qa/...` reads measured
feature identities from the same capture's sparse reconstruction. Only direct
interior source-label support in multiple reviewed views, adequate camera
parallax and independent reprojection quality select landmarks. Candidate
surfaces never select these points. Sparse coverage cannot complete hidden
material, determine its rest state or identify mechanical properties. Preserve
track identities, source boards and provenance; use uncertainty and source RGB
review when proposing the next shape family. Every original silhouette,
material-area, native and task gate remains required.

For the installed Isaac5.1 surface-FEM adapter, the nodal-force mass model is
validated with PGS. Independent TGS impulse and rigid-contact controls failed;
a passing unforced contact probe does not make TGS eligible for task forces.
The force adapter reads the actual scene solver and refuses an unvalidated
profile before applying an impulse. Native capability diagnostics may still
test alternate solvers explicitly. Do not rescale mass or declare a task pass
to compensate for a failed backend control.

Authored sheet admissibility also checks nonlocal triangle clearance when the
material prior declares a thickness in metres. Zero triangle intersections do
not rule out overlapping finite layers. `initial_state.policy` binds the
nominal declared thickness into the cached contract; `cloth_state.check`
records closest gaps and rejects overlaps before source scoring. Never reduce
thickness to rescue an image fit. This geometric screen excludes shared-vertex
and nearby material regions and does not certify local fold thickness, native
contacts or task dynamics. Those checks remain separate. Geometry proposal
penalties may preserve layer spacing, but cannot create native pins or replace
the independent triangle clearance measurement.
`cloth_clearance.contact_witnesses` can supply closest point/triangle or
edge/edge features from failed pair examples for a geometry proposal penalty.
Refresh those features as geometry changes; their frozen barycentric weights
are not an acceptance check and must never become native attachments.

`python -m calibrated_scene_reconstruction.scene.objects.blender.cloth_proposal SESSION OBJECT CANDIDATE
--observation ID --output SESSION/qa/...` is an optional CUDA geometry tool.
The agent JSON interface exposes it as `cloth_proposal`. It reads the current
declared observation and physical thickness, fits the actual flexible triangles
in material coordinates, and queries the current closed rigid surfaces again
after motion. An initial tangent constraint alone cannot prevent finite motion
from entering a curved wall. The tool offers a smooth material-coordinate field,
individual nodes, or a smooth3D spatial field (`--mode spatial`). The latter
encourages nearby plies to receive similar geometry changes; it is not a native
constraint or a guarantee of physical bonding, correct folds or fixed spacing.
All modes archive full material, self-intersection, thickness and solid
screens. Rejected crossing pairs add proposal constraints using the last valid
state's approach side, never a normal inferred from an intersected pair.

`assets.layer_geometry` also supports explicit construction proposals with
source-reviewed material correspondence. Partial layer displacements can be
extended harmonically across unselected material strips; leaving the crease
strip fixed while moving panel interiors manufactures large seam strains.
Every component needs proposal anchors and all independent geometry/source
screens still apply. This interpolation creates no native attachment, does
not identify actual layer spacing, and cannot certify an invalid proposal.

Its image objective retains all views with an equal-view mean and median term;
pooling pixels alone can improve large views while worsening the unweighted
median acceptance metric. Exact CPU source gates remain unchanged. Minimum
membrane-energy proposals are separately marked unapproved and retain their
material chart; they do not identify a real rest state or material constant.
Tool completion (`ok`) means proposal generation only. Rebuild the proposal
through the agent's model, review every original source constraint, and run
unforced native, task and physical-prior sensitivity checks before export.
No tested native state, source view, rigid geometry, material prior or acceptance
threshold is overwritten by this tool.

Native cooking can reduce an authored SDF resolution without updating the USD
attribute. `PhysicsErrors` records these per-prim limits from the native log;
precision estimates use the reported bound. The task entry also rejects a
silently reduced collision profile. Author a supported resolution and repeat
the checks, or change the collider representation when the required detail
exceeds the supported limit. This does not turn ordinary warnings into solver
errors or certify SDF accuracy. See the installed-version
[SDF schema](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/schemas/physxschema.html)
and [collider guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/collision.html).

Asset-contact diagnostics may use a finite-force fixture built from the source
hand's actual fingertip collider geometry and masses. This isolates asset
contact feasibility from full-hand reach and robot control. Keep the full-hand
route available; label the fixture scope, source hashes, masses and force limits
in the result and video. Screen the complete closing path for unexpected rigid
contacts, not only its endpoint. An expected normal contact under finite-force
actuation can be declared with rigid link names and a source/task basis;
this changes command-pose planning only. Never exclude those partners from
actual native penetration checks. Refresh material correspondence after opening or other
motion; a material node selected earlier may have left the grasp region.

Declare the grasp direction's coordinate frame. A top-down pinch across a
container opening compares the fingers and material direction in the opening
plane; retain the full spatial angle as a diagnostic. A task that specifies a
spatial direction still needs that spatial check. Neither a direction check nor
near-surface samples prove a frictional grasp. Preserve passive-shape,
two-sided-contact, lift, material and matched-friction controls independently.
Explicit diagnostic recording may execute a real lift after failed grasp
witnesses to show the failure. It must retain the failed verdict and original
native states, and cannot create cloth attachments, nodal grasp forces or a
delivery pass. A prior-range sensitivity run is not an identified material or a
new accepted default.

`hand_cloth` defaults to `closure_mode: open_descent`: approach with open
fingers, finish the descent with open fingers, verify the actual open-jaw
capture region and open joints, then close at that position. The optional
`coupled` mode preserves simultaneous descent/closure for tasks that need it;
it cannot override an input-required open descent. The matched control binds
the mode, direction frame, duration and actual preclosure material indices.
`pinch_material_frame` consistently selects spatial or opening-plane direction
in proposal generation and actual validation. Before closure, associate the
actual jaw corridor with source-indexed interior material again; do not follow
an obsolete point after a sheet slides. Keep the initial proposal and the
preclosure association separate in the saved motion. No material position is
written by this operation.

Insertion checks distinguish below-rim material, material above the mouth and
their combined opening-column occupancy. Report normalized insertion depth;
the same occupancy can describe shallow or deep packing. Use source-supported
initial-state constraints rather than a universal fraction threshold. Compare
placement and tool trajectories at unchanged intrinsic dimensions, mass and
constitutive priors before diagnosing the material as incorrect. A force-limited
contact fixture can propose normal force from estimated load, effective friction
and an explicit safety factor; hardware limits still apply. This estimate is
uncertain and cannot identify material properties or replace native tests.

`material_contact_mode: same_component` tracks two actual interior contact
regions after closure. A folded sheet can touch opposite fingers at distant
intrinsic coordinates. Preserve the approach patch for direction checks and
the two contacted patches for lift/hold checks. They must belong to one
connected material component; each finger must retain its corresponding
source indices throughout the hold. The negative control reuses those indices.
`single_patch` retains the original shared-disk criterion when the input needs
that restriction. The acceptance checker recomputes contact association from
native states and source colliders; metadata alone cannot grant a pass.

Before diagnosing an asset from a source-fingertip fixture, run an independent
interior coupon pinch with the same material, sampling and numerical profile.
`apps/scene/check_fingertip_contact.py` records a loaded lift and at least one
second of stationary hold. Run its matched `--zero-friction-lift` control, then
verify with `python -m calibrated_scene_reconstruction.tasks.mechanisms.fingertip_qualification`. Short
upward motion may be a throw or a slipping pinch. Require sustained per-finger
contact, bounded hold drift, current source hashes, identical executed commands,
matching initial/prelift states and native zero-friction readback. A failed
tool control is not evidence that task cloth should be stiffer or stickier.

Keep a position controller's maximum force separate from sustained closing
preload. The optional diagnostic source-tip fixture uses bounded rigid-tool
loads for preload; it never actuates cloth nodes. Preserve the pair's position
control and use discrete-stable damping at the actual observation interval.
All imported hardware wrench limits and native penetration gates remain active.
Report the proximity-query range from contact offsets separately from physical
penetration tolerance. Widening a contact search must not permit deeper overlap.
Coupon qualification is limited to the contact tool: full-sheet source state,
passive stability, extraction and delivery still need their own checks.

Apply the complete declared support relation in both authored geometry and
native task initialization. `above_unthreaded` alone permits a hovering end;
when `material_contact` is declared, evaluate the specified end-band area near
actual support triangles as well. Do not drop that requirement in a hand or
source-fingertip diagnostic. Keep holes open and retain loaded-contact and
post-opening retention as separate native questions. The asset-contact assay
normally stops a lift after measured slip and loss of its fixed per-finger
material regions; recording the full failed attempt remains an explicit
diagnostic option and cannot create a task pass.
The lift's cubic smoothstep respects the hand plan's declared speed as its
peak speed: duration is `1.5 * distance / speed`. Dividing distance by speed
alone would exceed that speed by50%. The asset-contact fixture uses the same
planner as the articulated-hand route and records actual states; a slower
command does not relax any native penetration, material or retention check.
Asset-contact assays default to the same flexible-response and local-bend
policy as the articulated-hand route. Stop on nonlocal thickness, actual
crossing or solid-contact failure before tuning the grasp. The explicit
`--diagnose-thickness-residual` option retains raw failures while allowing
diagnostic motion; it never grants acceptance. A passing free contact coupon
does not pass a dense folded sheet's initialization or early release.
For Isaac5.1 surface FEM, `max_linear_velocity: -1` requests a solver-selected
default cap; it does not mean unlimited speed. Bind that numerical policy in
controls and task traces, and avoid interpreting capped motion as a material
stiffness observation.
`surface_contact_profile.padding_proposal` can suggest a bounded numerical
margin from a measured, separated native layer-gap failure. It refuses zero
gaps, an exceeded padding budget or a filter that would hide the witnessed
rest pairs. It preserves physical thickness and acceptance limits. Check the
new geometry envelope independently and qualify fresh matching controls before
adoption. The geometry projection CLI exposes `--hotspot-strain-scale` so a low
RMS cannot hide sparse interpolation spikes; material lengths and all contact
guards remain unchanged. A projected mesh is only a proposal until Blender
rebuilding, source comparison and native release pass.
Contact-tool qualification binds solver, sampling, material, contact offsets,
collision representation, closing-control mode and the tested upper normal
load. A light-load coupon cannot qualify a stronger pinch by extrapolation.
Control inputs retain their complete audit hashes. Runtime tool reuse also
compares a construction signature: task source text, lift height/speed, hold
duration and initial-distribution criteria do not change the contact tool.
Geometry, source posture, contact definitions, effort/friction priors and unknown
future fields remain bound. Changing task motion still requires a fresh task
trial; this separation grants no task or source approval.
For an initial closed clamp, a near-zero hinge angle is only an articulation
measurement. Verify actual retained material and support after all preparation
forces are removed. Use the existing preparation `release_conditions` for
container state, material-end support and mating interfaces; retain failures
after release. A finite hinge-only preparation can resolve contact geometry
without cloth forces. It does not authorize exporting its drive or claiming
that an otherwise loose sheet is clamped. Rebuild and repeat unforced source
and task checks on the resulting asset.

When shallow insertion causes a loose flexible object to slip, compare deeper
hidden placements before changing its material. `preparation_geometry.support_slide`
proposes bounded guide centres over actual support triangles. It samples the
supplied patch points along the path and refuses missing support or
an obstructing step; it never fills a partial tray or outlet. The agent infers
the feed direction from the connected material and real support geometry.
Moving radially toward the container centre is not always the direction that
feeds more material inside. Apply the targets only through temporary finite
preparation forces, let the real closure take the load, and release the guides.
Compare actual below-rim area/depth, end support and passive retention with the
previous placement. Keep intrinsic dimensions, mass, material and required
exposed grasp access. A sampled path or a moved controller target does not
establish valid full-sheet motion; rebuild and rerun unforced and task checks.

Reconstruction may instead author a deeper initial placement directly; it need
not reproduce an unobserved packing history. `assets.cloth_feed.along_spine`
samples farther along a complete inferred drape and transports existing section
offsets with proper rotations. It retains the full source outline and intrinsic
coordinates, including clipped edges. The simpler `along_columns` requires
common column endpoints and refuses clipped boundaries that would create fake
tails. Declare and bound any continuation, check its junction and all actual
material/contact/support budgets, then rebuild through Blender and validate a
fresh native release. These offline proposals cannot overwrite running cloth
positions or replace evidence that the final object actually stays in place.
When the inner end already has source-supported contact, prefer preserving that
end while redistributing the rest of the drape. `cloth_feed.redistribute` can
consume more path length inside the container and shorten the free exterior
portion at fixed longitudinal arc distances. It keeps the supported spine end,
uses proper section rotations and enforces an explicit free-end trim budget.
Unchanged centreline length does not prove full-sheet isometry: keep the actual
triangle strain, layer spacing and all ordinary contact checks.

When baking a rigid hinge pose, transform its collision directions together
with the mesh and joint frames. `mechanism_pose.hinge_candidate` carries a
declared revolution axis through that rotation; `hinge_proposal` alone changes
geometry only. `native_proposal` uses the same metadata transform for released
poses. Check source-volume/cavity preservation before starting native physics;
a stale axis can prevent decomposition of a correctly posed rigid source.

Construction handles on a folded sheet need spatial as well as material
selection. Two disjoint material patches may lie on the same spatial edge.
`preparation_geometry.spread_patches` selects separated handles from an
agent-declared region while retaining disjoint material indices. Use it for
temporary distributed support, not for choosing the opposing contacts of the
task pinch. If closure is obstructed, isolate the unchanged rigid assembly
first; with a functioning empty closure, fully clear the preparation aperture
and allow free material to feed while guiding the intended inner drape. Check
the actual phase conditions before closing, then remove every guide and repeat
source support, unforced stability and task checks.

For a loaded contact pose, distinguish physical thickness from the unloaded
solver padding. `cloth_project --contact-envelope physical` may propose a
bounded repair against the complete physical contract without spreading a
correct clamp to the unloaded rest envelope. The default remains
`native_rest`. Neither mode approves an asset: rebuild in Blender and reload
without guides, drives or state writes, then check support, release and the
task with unchanged material. Keep earlier construction-goal failures; use
the input's actual bounds when judging a fresh candidate rather than adding
an arbitrary stricter placement target.

When a drape slips out after opening, inspect where its interior material
actually lies and what carries its weight. Above-opening material is not
below-rim insertion or supported material. A source-confirmed end on a partial
tray can coexist with a deeper loop through the adjacent large void; it must
not be threaded through a separately excluded outlet. Before proposing that
route, query actual cavity triangles including shelves and hidden contents.
`preparation_geometry.cavity_descent` bounds a temporary handle against those
surfaces rather than trusting a nominal opening depth. It changes no material
or native state; the whole sheet, supported end, closure and force-free release
must still pass independently.

Contact fixtures must use the same inside/outside finger assignment for their
initial hover and post-opening target. Otherwise a sign-equivalent pinch axis
can cause a180-degree tool turn during the descent. Use the actual source pad
offsets, interpolate rotations on the shortest proper rotation path, and keep
the measured arrival/capture gates. `hand_motion.force_limited_seconds` gives
a cubic-translation duration from actual tool mass and finite force reserve;
it does not certify tracking or reach. Lid opening duration is an observed
motion or explicit bounded-prior input, recorded separately from the hold.

If bounded clearance repair stalls on contacting folds, independently pushing
one ply can violate its neighbour's thickness even when a collective motion
exists. `cloth_project --clearance-solver coupled` proposes a minimum-motion
step using sparse linearized inequalities for all selected contacts, including
currently nonpenetrating neighbours. The legacy Jacobi option remains. Both
use the same displacement budget and full nonlinear material, support,
self/solid checks; linear convergence alone grants no acceptance.

Validate source relations between parts as well as each part's outline. A
circular body and lid can hide a wrong tray/hinge/exit orientation. Reviewed
`geometry_checks` may declare a `radial_relation`: actual upward support-area
centroid versus a named attachment about the body origin, with a broad
source-supported angular interval. Preserve uncertainty; this is not an exact
angle or metric observation. Keep the solid/void probes too. A relative fixed
leaf proposal must move its shape and collision directions consistently and
rebase both fixed-constraint frames (`mechanism_pose.fixed_leaf_candidate`),
then rebuild and run source, native and task checks again. Do not approve a
wrong support layout by changing cloth friction or adding retention.

When permitted by the input, correct a sliding drape by feeding more of the
unchanged full sheet inside. A longer loose internal path shortens the exterior
end; it must preserve real cavity/support geometry and leave a middle pinch
region at the rim. Check open-state retention before retrying the complete
grasp, and distinguish below-rim material from material merely above the mouth.
The passive candidate app accepts `--retention-plan` with explicit distribution
requirements, `observation_start_s`, `minimum_duration_s` (at least0.5),
`maximum_inside_loss_fraction`, and optional `minimum_lid_angle_rad`. It measures
actual body-local snapshots after all forces clear. Failed physical validity,
incomplete runs, sparse observation, cloth guides or a fixed body cannot pass
this retention proof. A pass does not approve source appearance or the task.

Folded end support may explicitly use `initial_support.connected_folds`:
`maximum_material_path_m`, `maximum_end_distance_m`, and a source/prior `basis`.
It requires actual local contact with upward support triangles connected to
the end through a bounded intrinsic route. This never fills a hole or proves a
load-bearing contact. Keep raw direct-end contact results, strict routing and
all thickness/self/solid checks. Object-understanding reviews require a
`support_retention[link]` entry with `report` and hash-bound `evidence`, including
the complete native result, states, probe/candidate inputs, paired mechanism
and material plans, retention input and native error report. The read-only
support-retention gate checks the current known producer and current physical
mesh, native/material/contact profile and source files. It runs even when
flexible quantitative accuracy is relaxed to `task_and_semantics`. Closed
initial-state, accessibility and the actual contact task remain separate gates.

Separate source initial-state requirements from transient physical placement.
An admissible high-prestrain posed sheet is not a relaxed state. When geometry
projection fails to produce a usable packing, the existing
`initial_state.Checked(..., construction_plan=plan)` supports a nearly
isometric starting pose and real finite-force placement. Its output only has
`construction_state_admissibility`; it cannot approve the source initial
state. Keep the original source mechanism/physics plans, bound the actual
temporary handles and their motion over real geometry, and require original
support/distribution conditions after release. Rebuild the released proposal
with the ordinary source policy and reload it without fixtures before task
validation. Weakening a candidate's original support declaration remains an
error. Native retention and friction checks cannot reuse assisted preparation
as their proof.

Frictional retention between loose contents and a container wall is a possible
hidden packing hypothesis. Identify the contacting parts and the physical
source of normal pressure: packing, elastic bending, or contents compression.
Gap size or a large friction coefficient alone proves neither pressure nor
retention. Treat contents as a semantic part with separate material/rigidity
uncertainty; a rigid fill fused to the shell is an explicit local control
approximation, not validated compressible contents. Preserve confirmed routing
exclusions when adding a newly authorized hidden-placement hypothesis.

A collision-free guide-centre path does not show that adjacent folds can pass
the opening. Check the complete connected neighbourhood, material length and
which folds remain held before sending one patch below a partial shelf. Use
actual crossing locations in the original material chart to distinguish a
collapsed supported tail from a jam beside the moving patch. The passive
candidate probe reports these locations without changing its failure verdict.
`channel_feed.screen_route` can additionally query actual patch offsets under
translation. It keeps the rest of the connected material and deformation
unchecked; a blocked translation may need a different orientation or shape.
Coordinate temporary guides or revise the construction geometry; retain the
failed states and remove every helper before testing the delivered asset.

Added guide patches must share a coherent frame when an existing group turns.
Independently translated targets can pull distinct folds into the same space,
even when every path starts correctly and avoids rigid geometry. Compare their
relative source and target offsets before attributing collapse to the material.
`tasks.mechanisms.guide_frames.paths` constructs proper rotating/translated
target groups and reports a bound on interpolation-induced distance loss. The
targets remain bounded-force construction proposals; they never rigidify the
cloth or remove the complete native and released-state checks.
Also inspect nearby, intrinsically separate plies that are not guided. A
folded packet is not glued: moving one ply does not establish that the others
follow. Revise the construction or declare bounded guides for the affected
plies while preserving their relative frames. Contact conditions should name
the actual contacting regions: an inner and an outer ply may contact different
partners. This does not prove transmitted pressure; self-contact validity,
complete guide release, native retention and the requested task remain required.

For volumetric contents, save the native simulation and collision reference
meshes separately from their current nodal positions and velocities. A surface
snapshot alone cannot restore the interior material state or demonstrate that
an export preserves contact preload. The passive candidate probe records these
arrays without changing the native state. Any reconstruction of the initial
volume state still needs an independent export/reload comparison; recording
the arrays does not grant that check.

For an explicitly selected conforming `volume_fem` representation, Blender's
`VOLUME_INITIAL(p)` supplies the original tetrahedral material reference and
separate posed nodes. The skin must cover that same complete boundary. Native
USD reload controls compare recoil against an identical posed geometry whose
reference was incorrectly reset, and check actual topology/reference readback.
Mass uses the original reference volume and density. `linear_damping` is an
explicit native rate, never an alias of legacy Rayleigh damping. Existing
legacy/rope backends remain available; their explicit-state serialization has
not passed this control. Backend qualification alone grants no source, loaded
contact, whole-object, task or delivery acceptance.
`native_proposal` requires complete hash-bound volume traces for every extra
flexible part, preserves the reference through the common rigid-frame change,
and rejects missing states or recooked topology. Its rebuilt surface remains
a proposal requiring a fresh unforced native reload.

When local packing guides repeatedly create crossings, reconsider constructing
the complete initial pose offline rather than replaying an unseen packing
history. Preserve the full material chart, natural fold reference and volume
reference. Common longitudinal stations and independently matched parallel
arcs are alternative proposals: the latter can shear a transverse crease, and
the former can stretch offset layers through a turn. Measure full principal
strain, release energy, self thickness and all solid boundaries before fresh
native release. Include compliant contents as geometry obstacles in offline
projection without making them rigid in physics. Failed construction methods
do not justify altered material dimensions, hidden supports or relaxed task
verdicts.

For continuum sheets, `cloth_project --objective triangle_metric` minimizes
area-weighted Green-strain metric residuals in both material directions,
including shear. Narrow triangles can retain substantial principal strain
after relative edge errors become small. The existing edge objective remains
available; judge continuum preparation with actual principal strain and stored
energy, then independent source/contact/native checks. Both objectives preserve
the original chart, use bounded geometry steps and retain the same admissibility
guards. Neither supplies native forces or changes a rest state.

Construct equilibrium in the articulation/contact state specified by the input.
If temporary finite-force tools hold loose material while a real closure seats,
record their force/torque bounds, actual closure/contact checkpoints and complete
release. Export original material references with the posed state, rebuild via
Blender, and verify the closed initial state and its subsequent opening afresh.

`apps/scene/check_cloth_channel.py` tests an isometric bowed coupon against an
explicit source mesh using the supplied material/contact profile. It declares
the fixed rigid fixture and synthetic tangential gravity loads, stores native
states, and measures intrinsic strain, sampled solid clearance, self crossings
and native errors. Compare nominal/zero friction and a bow without preload;
require low-load retention and higher-load release rather than permanent
jamming. Its results never grant whole-object or task acceptance. The complete
unchanged sheet must still pass released closed/open stability, real contact
extraction and export/reload checks, with the contents approximation reviewed.

For full surface-FEM recoil isolation, `check_surface_release.py` preserves the
source mesh and intrinsic/natural reference, with zero gravity and no rigid
partners, attachments or state writes. Compare self contact on/off and retain
native warmup, COM, momentum, kinetic energy and membrane-energy estimates.
The automatic per-node velocity safeguard can alter net momentum during a
strained release. Compare an explicit bounded speed on the identical geometry
before interpreting whole-body drift as physical elastic recoil. The internal
deformation remains a separate measurement. A numerical comparison never
qualifies a changed profile for source contacts or delivery.

Full folded sheets also need timestep convergence: a coupon can pass while a
large sheet crosses thin layers during release. The passive probe records
actual nodal velocities and `time_resolution` (observed velocity times step,
relative to physical thickness and the native rest band). This is a diagnostic,
not CCD or a new acceptance threshold; sampled velocities can miss intervening
peaks. Compare identical geometry/material at finer clocks before changing
stiffness, friction or damping. A geometrically valid fine-clock trajectory can
still lose its source support and fail. If used to construct an initial pose,
preserve complete material/volume references and require a fresh unforced
reload, source state, qualified contact task and delivery checks at the actual
selected production profile.

Compliant contents must also pass actual boundary-versus-rigid penetration
checks, independently of cloth-versus-contents contact and positive volume
Jacobians. Both the passive and real-finger probes use the shared contact budget
and actual moving rigid frames. Include boundary vertices, edge midpoints and
triangle centres, excluding unused native allocation padding. Proximity alone
still proves neither pressure nor static retention.

`check_compliant_sheet_contact.py` tests the explicitly selected beta or legacy
volume-FEM / surface-FEM pair on Isaac5.1. A declared moving platen compresses a free soft block;
no volume or cloth nodes are pinned. Finite cloth-only impulses supply known
diagnostic tangential loads. Preserve raw material, tetrahedron Jacobian and
sampled soft-solid clearance failures, and compare identical nominal/zero
friction and no-preload cases. Fixed fixture boundaries and numerical rest
padding must not be reported as measured source packing or stock properties.
Both retention and finite extraction are required before proposing this
representation for a complete asset. Keep legacy solid/rope and particle
authoring unchanged until their separately selected replacement is qualified.
