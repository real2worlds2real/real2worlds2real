# Agent workflow for reconstruction

Task objects default to agent-authored Blender models, fitted and revised against
the capture's calibrated silhouettes, openings and closeups. Template fitting is
an explicit fallback. Missing models or evidence stop the default route.

For the reconstruction entry points and dependency boundaries, see [reconstruction](../scene/reconstruction.md).

Read the skill needed for the current phase:

| Phase | Instructions |
| --- | --- |
| Inputs, graph roles, cameras and metric scale; S0/S1 | [Capture and placement](../../skills/twin-capture/SKILL.md) |
| Blender source, parameter/pose iteration, OBJ rounds and S2 | [Task-object modeling](../../skills/twin-model/SKILL.md) |
| Background, materials, task physics, S2–S6 and freezing | [Review and delivery](../../skills/twin-deliver/SKILL.md) |
| New mechanisms/settings, missing capabilities, reusable experience or harness upgrades | [Capability evolution](../../skills/real2sim-evolve/SKILL.md) |

Within object modeling, select and compose the
[experience modules](../../skills/twin-model/references/experience/index.md)
for vessels, insertion and gears. Their shared template separates
observations, parameter relationships, procedures, diagnostics and evidence.
New lessons update these modules and their index rather than adding one skill
per object or appending another case narrative to the phase instructions.

Use the [evolution workflow](evolution.md) when existing representations or tools
cannot satisfy the task, or an experiment yields a reusable procedure. Retrieve
matching experience, investigate the gap, validate a candidate and regress the
affected pipeline before adopting it. Update a discoverable skill/reference or
tool entrypoint; retain applicability, failed approaches and evidence. This
outer loop improves how reconstruction is performed. It does not replace local
scene refinement or native acceptance, and it ends at real2sim delivery.

Source code written for a session belongs in `configs/<name>/agent/`; runtime
rounds, evidence and exports belong in `sessions/<name>/`. A reconstruction has
no measured task-object or support dimensions. Legal board/robot calibration is
separate from development ground truth. For explicitly independent runs, use
only the original capture, legal calibration and the current workflow, without
old session notes, fitted geometry or models.

The agent-neutral JSON interface exposes stage execution, object iteration,
task checks, review recording, status and reports:

```bash
python -m calibrated_scene_reconstruction.agent.tool schemas
python -m calibrated_scene_reconstruction.agent.tool status --request request.json --output outputs/status
python -m calibrated_scene_reconstruction.agent.mcp_server
```

The status request is `{"session":"sessions/<name>"}`. Other request contracts are
returned by `schemas`. MCP uses the same input/output schemas plus an `output`
directory. Each invocation stores input hashes and append-only events under the
session's `agent_runs/`. Object rounds retain their own immutable snapshots and
events. Tool `ok` reports execution/numeric success; review/gate state is separate.

The automatic report reads each active Blender round and its current export,
rather than an older refinement cache. It checks rigid solids and flexible
surfaces according to their representation. A failed raw source-arrangement
check stays in the report; it becomes a labelled diagnostic only after the
current scoped input criteria actually pass. Missing or stale proofs remain
flags. Stage, OBJ and SUP review rows use the current gates, so a saved passing
record cannot hide a later invalidation. Report generation does not approve a
gate or change physics acceptance.
When later observation assembly appends auxiliary files to a native criterion's
evidence list, the gate accepts only an unchanged required prefix with all
additional files bound to the current verified mechanism hashes. It still
compares every criterion, default and relaxation verdict; removed/reordered
required files, changed checks or changed file contents fail. Native trial
recordings and their original acceptance snapshots are not rewritten.

Native task assays must use the assembled scene and its actual support. A
successful isolated collider experiment is diagnostic evidence, not a passing
S5. Use `tasks.rigid.native_assay` for bounded-force rigid tests; it records every
native pose and contact, including the support. Check the original mesh surfaces
as well as cooked collision contacts, and include an unforced settling interval.
Declare the observation horizon and terminal stability window before the run.
Inspect the whole release trace: a temporarily quiet interval can precede a
second contact rearrangement. Do not trim the recording to that quiet interval
or stop at its first passing sample. If the declared horizon is too short,
preserve its failure and rerun with a documented bounded longer horizon and a
sustained terminal window. Keep tolerances and material priors unchanged; extra
observation time cannot excuse persistent drift, bad support or penetration.
Declare external assay plans in `agent/task_check_inputs.json` as
`{"version":1,"files":{"qa/plan.json":"<sha256>"}}`; the task fingerprint and
trial snapshots preserve these inputs. Keep implementation files unchanged while
a trial is running. A modified implementation requires a new trial.

Collision normalization and decomposition must not change physical mass. Density
uses the closed source solid in metric coordinates, including USD transforms and
stage units; normalized cooking coordinates are not metres. Delivery checks
compare source geometry, collider transforms and the declared cooking recipe,
including mixed SDF and convex representations.

Review textures on the assembled scene, not only their PNG files. Support Boolean
unions preserve the active UV channel; collapsed UVs are rejected. Repeating
source material patches retain their original crop and remove smooth boundary
mismatch from the inferred tile. A material-only support revision can reuse an
object's contact fit only when both saved meshes are intact and their vertices,
faces, pose and support height are identical.

For print on transmitting surfaces, set the object's `fit.optical_print_source`
to an agent-reviewed source-region request. Texture rebaking reprojects that
request into the new atlas and marks it pending; review its source region before
use and inspect a native render before S4. `source_tint` material policies retain
sampled spatial details while applying a declared base-colour prior. They do not
identify absolute reflectance or replace independent image/geometry checks.

A current complete task result can directly supply explicitly requested input
criteria to a source-pose review. All required criteria, prior cases, saved
evidence and the current fingerprint must pass. This avoids requiring a second
copy of the same success proof, which would otherwise invalidate the first one.
Raw pose failures remain recorded; contact/material failures cannot be waived.

Use `apps/scene/render_rigid_trace.py` for multi-body native rigid recordings. Its
manifest names the session, checked USD/Z, `physics_poses.npz`, `native_result.json`,
input hashes, camera and output directory. Normal delivery requires current
passing task/package evidence; `diagnostic:true` is only an explicit preview.
Object poses are sampled from the recording without interpolation; camera motion
is presentation only.

Visible object structure needs a separate review from silhouettes and task
success. First list `physical_instances.<object>.geometry_features` in
`agent/modeling_plan.json` from the source-visible structure. Fitting cannot
silently drop these planned features. Declare matching `fit.geometry_features`
(`id`, `description`), or inherit them from that plan, and a session QA
`fit.geometry_review` path. The version1 review names `object`, the unchanged
`features`, `candidate_sha256` for `model.py` and `mesh.npz`, and source plus
untextured geometry views (`source`, `geometry_only`, `novel_geometry`,
`task_pose_geometry`). Each view records its session-relative `path` and SHA256.
`geometry_render` binds the actual neutral renderer's report (`path`, `sha256`);
its mesh, material policy and image hashes must match the claimed geometry views.
`view_checks` must give a passing verdict and reason for every neutral, novel and
task-pose view; a good front view cannot conceal a rejected side or operation view.
Each feature in `checks` needs `pass`, `reason`, `basis` (`observed`/`inferred`)
and the corresponding view ids. Missing, changed or rejected evidence blocks
export and OBJ acceptance. `apps/scene/render_geometry_review.py` renders an
actual candidate mesh with uniform material and no image textures; its rotations
are explicitly inspection poses, never native-motion evidence. The agent must
inspect those views: a valid signature does not recognize a correct structure.
Retain an object's characteristic structure through numerical fitting; do not
replace layers, holes or thin members by a textured outer envelope. Rebuild and
retest physical properties and contact after changing those structures.

When a rigid object contains visibly distinct regions, constrain them separately
with `fit.source_regions`. Each entry declares `id`, `prompt`, actual Blender
`materials`, a source-only `part_review` directory and its hashed `evidence_files`.
For example, a flower head and shaft remain one rigid body, but each must match
its own reviewed source masks. These constraints participate in fitting and
acceptance; a correct union silhouette cannot conceal the wrong proportions or
material boundary. Region masks must come from the input, never from the candidate
render. This complements neutral geometry review; neither substitutes for it.

Reconcile whole-object identity after reviewing part queries: a query can select
a neighbouring object, and retaining that query's old union would contradict the
part review. `scene.objects.object_label_union.prepare` composes independently
reviewed native-resolution part labels into a separate pending whole-RGB review.
Inspect every original frame and record `complete_visible_parts`, its source-only
basis and the verified identity before opting in with `fit.source_identity_review`.
Keep all visible parts, original frames, crop windows, raw queries and background;
do not use model errors to erase parts or invent labels. The loader binds the
source images, acquisition, cameras, original cache and part decisions. This
changes evidence identity only; it neither alters source thresholds nor approves
geometry. A blurred unresolved part remains explicitly unobserved in its part
review while the original whole frame remains in evaluation.
Part review also emits `source_detail_*.png`: original-pixel crops with surrounding
context and all positive fragments. Inspect these as well as the whole-scene
sheets; a narrow rod's false continuation onto a dark container can disappear
in a scene thumbnail. Enlargement uses nearest pixels and adds no new detail.

Source silhouettes sample the actual triangle union at calibrated OpenCV integer
pixel centres; part visibility uses rays at those same centres. Conservative
polygon drawing and positive antialias coverage expand thin boundaries and are
available only as explicit legacy diagnostics. GPU hard coverage accelerates
proposals; final acceptance remeasures with CPU geometry. The unchanged source
thresholds apply to every retained view. New round manifests include
`source_raster_contract` and immutable copies of both whole and visible-part
samplers. Export rejects a missing or changed contract: historical results remain
readable, but require a new measured and reviewed round for current approval.

The existing `calibrated_scene_reconstruction.objects.tool`, `calibrated_scene_reconstruction.objects.mcp_server` and `calibrated_scene_reconstruction.twin`
entries remain available. See [pipeline options and gates](../scene/twin_pipeline.md),
[architecture](../architecture.md), [video physics](../video_physics.md) and
[replay calibration](../real2sim_physics.md) for domain details.

Before any new object fitting, write `configs/<session>/agent/object_understanding.json`
(version1, `objects` mapping). Every reconstructed object needs `identity`,
`source_coverage`, `uncertainties`, hashed session-relative `source_evidence`, and
an inventory of `parts`. This includes reconstructed furniture and secondary parts,
not just manipulated objects. Each part declares `id`,
`structure`, `connection`, `material_hypothesis`, `material_alternatives`,
`expected_response` (rigid/flexible/articulated/uncertain), `response_basis`,
`representation` (rigid/deformable/articulated), `representation_reason`, `basis`
(observed/inferred), `connection_type` (integrated/bonded/joint/loose/support/uncertain),
`task_primary`, actual physical `links`, `source_evidence`,
and `geometry_feature`. These features must agree with the modeling plan and
structural review. A flower's thin petal surface is not proof of flexibility;
its actual native link kind and load response must support that choice.

Each part also declares `physics_tests` with unique ids, `purpose`, `scenario`,
and `metrics` (`name`, `unit`, finite prior `bounds`, `basis`). Required purposes
are passive_state, contact_response and material_response for rigid and quantitative
response reviews; articulated parts also require joint_response. Choose discriminating physical tests from the material
and construction hypotheses, including plausible alternative materials. A
material test must examine response to load and release; a density estimate or
successful task alone does not validate bending, elasticity or a connection.
Record uncertain hypotheses explicitly. Choosing a rigid approximation for a
potentially flexible part requires source/scale justification and measured
response within the declared approximation bounds. Declare its
`rigid_approximation.basis` and `error_metric`; that metric must be a measured
material-response obligation. Do not invent certainty from
a static video, or infer an identified stiffness from whether a task passes.

Flexible parts default to `response_accuracy: task_and_semantics`. Match source
shape and response as closely as practical, but keep exact local strain, shape
and material-response errors as diagnostics. Require correct semantic structure,
an actual flexible representation, a hash-bound `physical_plausibility` review,
valid scene contacts/stability and the requested native task before delivery.
Do not keep refining a noncritical numeric response after these conditions pass.
`task.json.flexible_accuracy` selects `task_and_semantics` or `quantitative` for
runtime material checks; an explicitly quantitative part retains its declared
intrinsic response tests. Raw failures remain recorded. Neither setting relaxes
rigid shape/scale accuracy or permits replacing flexible parts with rigid hulls.

The Blender source loop applies this default to independently labelled complete
flexible links too. It keeps the original full-object and part errors, checks
unrelaxed rigid geometry at the original thresholds, and preserves complete
material and initial-geometry checks. A missing independent view leaves the
default scope unavailable; it never removes that view from the raw fit. Purely
flexible assets retain source/shape review and material checks even though they
have no rigid source surface. Explicit input-specific source waivers keep their
existing separate contract.

A candidate with valid unrelaxed geometry may enter native tests through
`--allow-pending-source`; this does not approve OBJ or delivery. Default
qualitative source acceptance requires the current hash-bound semantic and
physical-plausibility review plus the complete current `qa/task_check.json`,
including applicable physical-prior trials. Candidate experiments, old native
results, missing parts and rigid substitutes cannot satisfy that requirement.

For explicitly secondary flexible details in the qualitative profile, runtime
connection/contact accuracy has a diagnostic review threshold:10% of the
detail's rotation-invariant span, capped at2mm after composing numerical error.
Exceeding it requests native-state semantic/physical review; it must not prevent
attempting the task solely on an incidental detail's exact response. Initial
authored intersections, nonfinite/collapsed material and topology failures
still fail. Complete local response/self-contact records are required; they
cannot approve delivery by themselves. Primary flexible contacts and all rigid
contacts retain their original limits. Qualitative secondary cloth declares
`self_contact_recorded`; quantitative/primary cloth keeps `no_self_intersection`.

If small features form a densely bonded composite, choose their representation
from construction/material evidence and an independent approximation-error
test. Preserve their actual visible geometry. Solver failure alone does not
justify labeling a flexible component rigid; broad free sheets must not be
replaced by a solid envelope. Derived semantic reviews consume native records
and are checked separately from the causal simulation-input fingerprint.

Static furniture declares `structural_priors` for `part_inventory`, `layout`,
`connections` and `support`. A rectangular four-leg table should have a coherent
paired layout and repeated construction unless source evidence supports an
exception. Review leg counts, symmetry, attachment and floor support along with
the tabletop. A declared reflection measurement is recomputed from actual
`source_solids.npz`; silhouette agreement cannot override a failed structure.
Static support parts may have empty intrinsic `physics_tests`; native scene
contact checks remain separate. SUP export and delivery require the current
semantic and neutral-geometry reviews, just like task objects.

`fit.run` rejects missing understanding and incompatible link kinds before
fitting, then checks the actual full build again. Every round snapshots the
understanding. Candidate collider export can precede native physical testing;
it does not approve delivery. Normal OBJ verification additionally requires
`qa/object_understanding/<object>.json`, containing version1, object, complete
unchanged `understanding`, current `candidate_sha256` for model.py/mesh.npz, pass,
and one entry in `parts` for every semantic part. Each entry has independent
semantics, geometry and representation judgments with pass/reason/hash-bound QA
evidence. Rigid/quantitative physics mappings cover every declared test. Each proof names a
hashed native report and evidence. The native report declares simulator
`Isaac Sim 5.1`, test_id, the same candidate hashes and understanding,
task_actuation:false, pass, hashed raw_evidence, hashed repository-relative
producer, and measurements mapping each name to value/unit. Verification checks
the measurements against the plan's bounds itself. This validates coverage and
provenance, not automatic semantic recognition; the agent must inspect the
source, actual geometry and native records honestly. Source silhouette,
structural review, physical priors and task feasibility remain separate gates.
Changing the inventory or physical proofs invalidates task fingerprints. Legacy
and template outputs need these reviews before renewed delivery; frozen assets
are not rewritten to make an old pass current.

If the input authorizes existing assets as blueprints, inspect them before
rebuilding the same object family. Keep a hash-bound reuse manifest with the
construction, complete material chart/topology, texture and physical priors
actually copied, plus the scope and failures of the old tests. Current video
evidence must still determine scale, placement, articulation and contact route.
For example, changing a sheet from resting on a tray to passing through an
aperture requires a new complete shape and native contact check, while its
compatible material and texture can be retained. Reuse does not transfer an old
task or delivery verdict. Independent reconstructions that prohibit previous
assets keep that restriction.

Native particle PBD and surface FEM require actual material components and real attachment
boundaries. `SURFACE_BINDINGS(params)` binds sheet vertices to a dynamic rigid
link; every disconnected sheet component must have a real root. Only root faces
against their actual rigid owner are filtered. Never filter away free-surface
contact or bridge separate petals/fabric with material springs. Initial checks
include intersections, finite-thickness clearance, immutable material strain,
and actual rigid-surface penetration. Support placement includes half the sheet
thickness rather than placing its midsurface inside the table.

Choose the backend from the material and the installed solver's demonstrated
behavior. Thin artificial fabric/plastic petals can use native particle PBD;
surface FEM remains a separate route with continuum parameters and its own
qualification. Do not transfer spring constants or a solver restriction between
these backends. An explicit TGS particle profile still needs the same native
contact, force/release and task checks as PGS. Material constants inferred from
static captures remain priors.

Before coupling explicit flexible roots to a generated metric rigid body,
`sim.source_moments` authors its COM and full inertia from the original closed
source solid and the declared mass hypothesis. Prior-case mass changes scale
that tensor. Collider cooking must not silently supply a different source mass
distribution. Externally scaled legacy instances retain their existing path.

Secondary approximate particle details can declare bounded sampling accuracy.
The continuous-thickness diagnostic allowance is limited by material
half-thickness, semantic component size and actual particle coverage. Preserve
the raw exact-thickness verdict; actual solid penetration, real roots and primary
task contacts keep their checks. Native response, semantic review and task
success are still required. Never convert a free sheet's nearby return surface
into a glued root merely to pass a clearance check.

The installed surface backend uses a separately qualified uniform-node mass
model, rather than readable per-node masses. Each candidate is checked for
component mean-mass bias. Split native bodies, or refine components in proportion
to actual material area; merging unequal sheets with equal node counts does not
preserve a homogeneous material. `surface_refinement.refine` preserves the
piecewise-linear surface and root boundary while adding conforming midpoints.
This is a construction tool; it does not approve native response. Report
within-sheet mass-centroid error and run independent load/contact controls too.

Mixed rigid/flexible task assays must restore a captured full native setup via
`Ctx.restore_assay_state`, and use `place_assay_assembly` for a single common
setup transform. During `native_assay.run`, placement and setup restoration are
prohibited. Rigid-driven trials record actual PBD/FEM nodes in `flexible_trace`;
`flexible_trajectory.measure` checks recorded material state, real root bonds,
self/cross-sheet contact and all actual rigid surfaces at a declared time
spacing. Its clearance remains sampled, not a continuous-contact certificate.
Rigid pose replay alone cannot visualize flexible response. Attached nodal-force
adapters remain blocked without independent coupled-force qualification.

Candidate assay summaries retain `raw_assay_pass` separately. If a flexible
approximation still requests native semantic review, `tasks.rigid.assay_status`
reports `pending_native_semantic_review` and leaves aggregate `pass` unset.
A final-window rest measurement or a relaxed secondary response screen cannot
approve loss of the object's shape. Review actual native node states across
initial settling and task motion, separate rigid motion from deformation, and
check that bound material retains its real attachment and plausible form.
These diagnostic assays never grant object or task acceptance by themselves.
For a matched material sensitivity test, the candidate assembly plan can name
`physics_prior_proposal`, a configuration-local JSON file with `version: 1`,
an explicit `basis`, and `overrides` grouped by `bodies`, `mechanisms`,
`supports` or another installed parameter group. Each existing parameter sample specifies
`value` and its unchanged `unit`. The adapter permits only `prior_only`
parameters within their original bounds and preserves validity requirements.
It refuses changes to identified estimates or simultaneous global low/high
sampling. Original priors, the complete candidate configuration, proposal and
receipt are immutable assay inputs. A useful response does not automatically
adopt that sample or identify the real material.

For a cloth backend comparison, `representation_hypothesis` names a
configuration-local JSON containing `version: 1`, `basis`, the complete
`mechanisms`, `physics_priors` and `object_understanding`, plus
`reference_candidates` mapping changed objects to session-relative candidate
descriptors. The diagnostic adapter currently permits only particle/surface-FEM
changes. It preserves all physical links, semantic/source evidence, mechanisms,
checks, validity limits and common mass/thickness/contact priors. The actual
coordinates, triangles, natural material chart, bonds, visual materials and pose
must match the reference exactly. Rebuild the selected backend and pass its
material-state checks; a particle-specific sampling allowance cannot qualify a
FEM surface. Both full source and candidate configurations are retained. New
constitutive parameters need independent priors in their own units; neither a
unit conversion from spring stiffness nor a successful task identifies them.

Review the solver's constitutive capability as well as material constants.
Particle distance springs have identical energy for opposite isometric fold
signs and do not implement `bend_reference` as signed natural curvature. This
does not prevent their use for ordinary cloth or establish that every fold will
invert. For formed sheets that must retain natural curvature, consider the
existing surface-FEM route and verify its actual release/contact response.
`assets.cloth_route_capability` records this distinction and a synthetic law
control without changing any threshold or granting native approval. Candidate
assembly authoring reports retain the per-link capability diagnostic.

Task-state stability must respect the degrees of freedom of the actual contact.
The default `assembly_measurements.release_stability` still measures every rigid
surface point. For an explicitly source-verified smooth circular shaft in a
round opening, `cylindrical_insertion_release` instead measures both source axis
endpoints and every insertion constraint throughout the final native window.
The task must leave azimuth unconstrained; keyed and other orientation-critical
interfaces cannot use this interpretation. Keep the raw whole-surface motion
verdict, require the other bodies' rest checks, and still check full actual
surfaces, zero applied wrench, materials, semantics and physical plausibility.
Declare and hash-bind this interpretation before the final native trials. It
adds no simulated joint, pin, damping or prescribed rotation, and does not relax
rigid geometry or penetration accuracy.

Particle readback registration checks index identity separately from physical
shape. Retain the established bounded-displacement check. Larger warm-up motion
can use an ordered minimum-distance assignment only when its uniqueness is
certified beyond float32 uncertainty; competing assignments or permutations
fail. Dense assignment certification is limited to512 nodes per body, while
the original cheap path remains available for larger meshes. No registration
method edits native positions or velocities or approves material/contact state.

Rigid trajectory checks can reuse signed-surface results only for exactly equal
relative transforms in both directions within the same fixed-source check.
Every recorded timestep still receives its measured result. This removes repeat
queries for resting bodies without rounding poses, reducing temporal coverage
or changing the penetration threshold. The uncached path remains available as
an independent comparison.

For a shared GPU, launch native work through `tools/run_shared_isaac.sh`.
It holds one local Isaac lease and checks free VRAM before startup; all-case
physical trials check again between serial children. The default admission
estimate is 8192 MiB plus 8192 MiB shared headroom, configurable with
`SCENE_RECONSTRUCTION_ISAAC_EXPECTED_VRAM_MIB` and `SCENE_RECONSTRUCTION_SHARED_GPU_RESERVE_MIB`. These are
estimates, not memory caps or reservations respected by other sessions. Monitor
actual usage; never terminate another session's jobs to create capacity.

Profiles with `initialization_hz` also require a standalone package startup
check. After packing, run `apps/scene/check_package_physics.py SESSION FILE.usdz`
through the Isaac launcher, then `view_usd.py`. The startup check plays two
seconds at the authored ordinary timestep, without the Python small-step reset
wrapper, and records actual native rigid/flexible states and cooking errors.
This checks native startup and finite state, not task feasibility or material
accuracy. The reopened-package proof and S6 bind its report and raw-state hashes.
Existing profiles without this option retain their delivery behavior; `--force`
can additionally check an ordinary declared profile. Failed attempts are retained.

Mechanism export snapshots the authored task subtree before settling the robot.
After stopping simulation it restores the task geometry, transforms and physics
attributes, while separately retaining the robot's saved pose. It then repeats
the passive audit before flattening. A settled flexible shape must not silently
replace the source initial state or its material rest geometry in a new package.
This initial-state preservation is enabled for mechanism contracts; legacy
settling behavior is unchanged. All exports also inspect source USD composition
errors before flattening, including unresolved prim targets in existing files.
Flattening alone can erase these errors. Repair incomplete referenced CAD, or
use the explicit `--no-robot` workspace option when the optional robot is outside
the requested task asset. `scene_tour.py --no-robot` renders that same scope.

For physical process reconstruction, use the optional `physics_*` tools described
in [the video process workflow](../physics/process_workflow.md) and
[physics/process.md](../physics/process.md). Plan the bodies, materials,
representations, mechanisms, parameter priors and whole-process observations
before fitting; do not reduce every video to a fixed task's parameter list.
For newly extracted video features, include `observation_review`: inspect source
overlays for channel identity across the process, independent camera coverage,
occlusion and timebase. Bind videos, camera and observation hashes in this
review; color alone must not let background or hand regions become targets.
Existing legacy observation adapters remain supported. Only fit an initial pose
from source pre-contact fit frames; retain the held-out split, fixed geometry
and all later motion checks. Missing critical cameras require observation
feedback, rather than a fitted physical explanation for camera errors.
The agent owns model revisions; the harness owns native Isaac Sim 5.1 stepping,
joint parameter search, held-out scoring and evidence capture. Retain existing
replay/video/soft tools. Fix geometry through a new reviewed modeling round,
never through hidden scale or support-height fitting. Report prior-only and
unconstrained quantities honestly. A numerical pass requires a separate review
of semantics, physics, whole-process reproduction and parameter evidence.
`physics_attach` binds the accepted result to a scene's S5/S6 delivery checks;
it does not overwrite frozen geometry or automatically patch legacy exporters.
