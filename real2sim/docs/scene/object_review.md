# Source-grounded object and support review

This is the current review contract. It replaces the pre-oversight document, whose
template examples are not current modeling instructions. In particular, never size
a hole from the inserted object's diameter
or discard difficult source views merely to obtain a successful fit/task.

## Authoritative interfaces

Read [the agent workflow](../agent/workflow.md),
[the modeling skill](../../skills/twin-model/SKILL.md) and
[the native gates](twin_pipeline.md). Implementations live in
`scene.objects`, `scene.support`, `scene.review`, `scene.twin` and their native
runtime adapters. Agent tool schemas are the executable request reference.

Object tools preserve each model source, parameter fit, evidence and export in
`sessions/<session>/objects/<object>/round_<n>/`. Support rounds have their own
history. Agent-authored models and plans live under `configs/<session>/agent/`.
Reviews append rounds rather than overwriting old failures.

## Review layers

| Layer | Evidence and required interpretation |
|---|---|
| Identity and semantics | Raw source, all physical parts, construction/material alternatives, connection types, support and task relevance |
| Scale and cameras | Legal calibration and registered metric views; no task-object ground-truth dimensions |
| Source labels | Independent source masks, component/region identities, query policies, occlusion and exposure review |
| Numeric fit | Current multi-view CPU evidence, median IoU at least 0.85, local difference at most 0.02, at least eight frames on the default route |
| Structure | Named source-visible features, neutral captured/novel/task-pose renders and per-feature review bound to source/mesh/render hashes |
| Interior geometry | Source-supported solid and empty regions, actual cavities/openings/contact features; silhouette alone cannot prove them |
| Support | Observed plane, thickness/edges/legs, full part semantics, actual contact and robot/table placement; current SUP gate |
| Materials/appearance | Correct physical-face ownership, texture/UV scale, optical interpretation, captured patterns and native assembled rendering |
| Native physics | Valid initial state, actual collider representation, mass/material priors, passive/material/contact controls and released behavior |
| Task | Complete actual native interaction in the assembled scene, relevant prior cases and explicit fixture/control limits |
| Delivery | Current OBJ/SUP and applicable S0–S6, dependency-complete USD/Z reload, native package checks and immutable freeze |

Numerical fitting, structural review, material plausibility and task feasibility
are different checks. None substitutes for another. A texture cannot replace
missing petals, slots, thin members or cavity walls. A high union silhouette
cannot hide incorrect region proportions. A successful isolated collider coupon
cannot approve an assembled task.

## Before fitting

Declare `agent/object_understanding.json` with object identity, source coverage,
uncertainties, hashed source evidence and a per-part inventory. Record structure,
connections, material hypotheses/alternatives, expected response, representation,
physical links, task relevance and discriminating physical tests. The semantic
inventory must describe the candidate's actual representation.

List visible `geometry_features` in `agent/modeling_plan.json`. Declare matching
`fit.geometry_features` and `fit.geometry_review`, or inherit the plan's features.
Use `fit.source_regions` for distinct source-visible regions; those masks come
from input RGB, never the candidate render. See the modeling references for the
exact source, view and neutral-render evidence contract.

For interior support/free space, use source-bound `agent/geometry_checks.json`
finite segment probes and explicit uncertainty. These are geometric prerequisites,
not native task approval. Preserve critical concavity in both visible and cooked
collision geometry. Source-supported alternative hypotheses can be tested; changing
an unseen floor or cavity solely to improve policy/task success is forbidden.

## Refinement

Inspect complete per-view evidence, worst views, part regions, neutral views,
openings, closeups and native recordings. Diagnose source identity, camera/scale,
geometry, representation, numerics and control separately. Preserve failed
candidates and raw measurements. Revise the relevant layer, rebuild and rerun all
affected checks. Optimizer convergence or GPU proposal scores are not acceptance;
final default numeric acceptance uses independent CPU coverage.

`fit.accepted` refers only to the current reviewed/exported round. Editing source,
geometry, pose, camera/mask evidence or required physical inputs invalidates affected
reviews. Do not silently keep a passing flag. New texture looks require fresh S4
review and cannot mutate a frozen session.

## Scoped acceptance and flexible parts

Read explicit `task.json.acceptance` before interpreting a scoped relaxation.
The implemented `scene.acceptance` contract retains raw/default failures and a
separate effective verdict. A relaxation activates only after all required input
criteria pass with current bound evidence. It cannot excuse stale inputs, invalid
material, primary contact failures, hidden assistance or changes to frozen assets.
Task success is not a physical-parameter identification method.

Flexible parts default to `task_and_semantics`: preserve source geometry and actual
flexibility, require semantic correctness, hash-bound physical plausibility, valid
scene contact/stability and the requested native task. Exact noncritical response
errors remain visible diagnostics. `quantitative` parts retain declared measured
response obligations. Rigid precision and primary flexible/rigid contact requirements
remain intact. Secondary-detail tolerance policy is implemented in
`assets.response_policy`; do not invent a new relaxed threshold in a report.

## Appearance, assembly and export

Review actual assembled views, not only texture PNGs or isolated mesh previews.
Check support UVs, repeating pattern scale, lighting/exposure, seams, baked shadows
and contact shadows. Distinguish opaque print on transmitting material from scenery
visible through it; texture baking cannot change collision solids. Source-region
and native-render review are required for optical print proposals.

Use current status to identify missing, stale and rejected records. Append OBJ/SUP
reviews through the shared agent interface. S6 checks current prerequisites and
package evidence; freeze only after native acceptance, and verify the freeze
before data collection. Post-freeze development geometry evaluation is separate
and cannot retroactively tune the independent reconstruction.

For detailed cases, read the model skill's [fitting](../../skills/twin-model/references/fitting.md),
[rigid/optical](../../skills/twin-model/references/rigid_and_optical.md),
[semantic/interior](../../skills/twin-model/references/semantics_and_interiors.md)
and [native](../../skills/twin-model/references/native_representation.md) references.
