# Semantic understanding, interior free space and grasp direction

Maintained reference for [the modeling skill](../SKILL.md). Apply the sections relevant to the current representation and task; preserve all applicable local gates.

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
axis in the candidate body frame. `check_hand_cloth.py --pinch-axis X Y Z`
checks the actual open pad line during planning and immediately before closure,
in addition to inside/outside assignment and full hand clearance. Its planar
projection is deliberate and must match the input interpretation. It does not
infer a sheet's longitudinal direction from its normal or from a radial fixture
frame. Reject unresolved axes and keep the actual pad trajectories. Record and
replay the same direction policy in friction controls. A failed grasp does not
identify an incorrect material law until source geometry and tool placement
have been checked. For top-down sheet grasping, `hand_cloth` defaults to open
approach, open descent, actual open-jaw capture/readiness, then closure.
Keep explicit coupled mode available; never override input-required sequencing.
Use one declared material direction frame in proposals and validation. Bind
fresh preclosure material association and the same commands in the friction
control. Expected rigid closure contacts require a source/task basis; actual
penetration checks still include every partner. Distinguish below-rim insertion
from above-mouth occupancy before changing cloth material parameters.
A captured closed-lid state requires native contact and force-free release
with the full sheet present. A synthesized closing path uses finite actuation;
a micro-open authored pose or a small hinge angle alone cannot certify the
initial state.

Closed articulation coordinates do not establish physical seating. Keep existing
cloth/solid seam witnesses and add source-declared rigid mating-surface checks
where the mechanism requires them. `agent/interface_checks.json` can declare
per-object first/second rigid links, reviewed face anchors (with topology hash)
and a physical gap bound. The wipe fixture measures them in native poses;
preparation plans can require the same `release_conditions.interfaces` before
release. Bare rim sectors and the folded-cloth exit have different contact
requirements and must be reviewed separately. Gap probes do not replace solid
penetration or force-free retention. Build rigid seating geometry independently
of cloth layer counts or a preparation packing proposal. A modeled gap must not
be introduced merely to avoid closing contact; use native finite actuation and
report a failed closure instead.
