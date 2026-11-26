---
name: generate-functional-cousins
description: Plan or generate optional post-twin variants with substantial prompt-directed geometry changes while preserving object function and articulated mechanism topology.
---

# Generate functional cousins

The user interface is natural language. Interpret quantities, shared wishes, per-variant requests and local changes yourself. Do not require the user to split their prompt into fixed shape/material/scope fields. The framework needs an explicit internal plan for scheduling, progress and resume, not a rigid user-facing form. Infer execution intent as well: planning-only requests set `execute=false` and do not start child jobs.

Planning runs have `generation_mode=cousin_planning`. Inspect `context/parent.json` and useful parent views/source. The twin is already accepted; do not rebuild it. Submit `submit_cousin_plan` with the inferred count, one concrete prompt per cousin and shared notes. There is no fixed count. An optional explicit count is a user override. If a necessary quantity is absent or contradictory, submit a concise question rather than inventing work.

Default plans should explore meaningful geometry, proportions, dimensions and design, beyond palette/texture swaps, while preserving functional identity. Let the user's prompt determine the scope. `geometry_change_expected` is an internal planning judgment: use false when a user specifically wants surface-only variation; do not use it to bypass an unconstrained request for substantial redesign.

Generation runs have `generation_mode=cousin`. Read `context/cousin_task.json`; it carries the original request, this variant's prompt, accepted parent structure and accepted sibling summaries. The parent images/source are contextual evidence, not pixel-matching targets. Reuse the editable parent Python or `parent_scene.blend` where useful. Original scale, pose and appearance are free to change where the prompt and function permit.

For each executing child, follow the construction and representation-specific review in [generate-editable-object](../generate-editable-object/SKILL.md), with this cousin contract defining what may change. Its reference-matching requirements apply only to features the user asks to retain. Use [improve-visual-realism](../improve-visual-realism/SKILL.md) for the requested appearance, and [review-articulation](../review-articulation/SKILL.md) when redesigned joints or clearances affect operation. Return those results to this child's function, structure and diversity review.

Keep the simulation asset kind and functional identity. For articulated mechanisms, preserve joint identifiers/roles, types and nearest-joint parent connectivity. Preserve additional mechanical loops, user-control roles and coupling ownership. Joint positions, mounting orientation, finite limits and anchors may adapt coherently to the redesigned geometry. Mesh topology and decorative part count need not match. Rigid objects can change nonessential details: for example a mug can become a handleless drinking glass while remaining a hollow, usable vessel.

The harness compares structure against the frozen parent and checks exact geometric reuse when geometry change is expected. Its canonical visible-vertex fingerprint is only an anti-clone diagnostic; a changed hash does not prove meaningful design diversity or maintained function. Judge those properties from the requested goals, editable structure, renders and real simulation evidence. Different material or object names do not establish geometric redesign.

Each child has an independent modeling/revision/target-simulation acceptance loop. Parent success and successful siblings do not certify the current child. Choose specialized tools, intermediate simulation and video review when useful; no fixed viewing schedule or per-draft movie is required. Fix target failures and call accept_result only after both hard checks and your visual/functional judgment support delivery.
