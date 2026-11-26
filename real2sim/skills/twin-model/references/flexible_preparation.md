# Flexible preparation, grasping and release diagnostics

Maintained reference for [the modeling skill](../SKILL.md). Apply the sections relevant to the current representation and task; preserve all applicable local gates.

For a folded sheet, prefer a reviewed intrinsic `--pinch-material-axis` over a
fixed world/body vector. The local material map determines its current tangent
at each candidate patch and again after the native approach. A flat-sheet PCA
or exterior-flap bounding box can identify width rather than the longitudinal
route. Explicit `acceptance.criteria[].interpretation.pinch_material_axis` is
forwarded by the hand harness and enforced against saved native states; it must
be justified by the chart/source correspondence. Fixed body-axis requests remain
available for tasks that actually specify that convention. Directional planning
uses the input axis instead of imposing the default surface-normal proposal
filter; actual contact, material validity and matched friction controls remain
required. The verifier enforces the declared alignment bound independently of
a looser experimental planning bound.

Mating requirements are state dependent. A trapped-sheet initial condition
requires source-matched lid pose, actual sheet/solid seam contact and passive
retention. Unloaded bare seating is a different condition: declare
`phase: unloaded_closed` and test it after extraction, including zero-actuation
retention. Do not require an unloaded zero-gap condition through trapped
material. Native preparation history is one possible way to construct the
observed initial state; a failed controller path is not by itself proof that
the physical object or material is impossible. All source, geometry, contact,
material and requested task checks still apply.
Reconcile supplementary-input part coverage with the semantic part inventory.
When a closeup reveals a previously hidden fixed support, add that identified
part to the appropriate geometry input's `invariant_parts` and bind its source
evidence. Geometry invariance does not mean the whole assembly has a fixed pose.
Do not limit closeup reconstruction to body/lid labels while silently leaving a
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
the recorded unforced interval and at the final state. A lid that was closed
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
when a lid or other temporary support opens. Preserve and diagnose an early
slide-out before spending a full hand-closure trial. Do not add a hidden holder
or fill a real opening to retain the sheet.

If the user permits a deeper placement, increase retained material by feeding
the same complete sheet or adding a physically valid loose return above the
actual support. Consume added internal path from the exposed end; never resize
the sheet or substitute friction/material changes for placement. Preserve a
rim-crossing interior pinch region, closing clearance, real support holes and
the requested task. Compare an open-lid, all-force-free retention interval
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
lid seating control does not approve the cloth-pinched state or full task.

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
An initially plausible cap can intersect its bottle only midway through opening.
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
may be impossible under the lid. Do not tune material against an infeasible
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
closure. Record the lid-opening duration as a motion hypothesis; no fixed
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
