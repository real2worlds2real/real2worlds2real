---
name: generate-scene-cousins
description: Generate whole-scene cousins from an accepted parent using functional object redesign and optional mesh or Gaussian appearance changes while preserving recognizable layout.
---

# Generate scene cousins

The user supplies a natural-language request and optionally a count override. Astra interprets quantity, shared style, per-variant intent and which components should change. There is no fixed number or required prompt grammar. Object and scene parents use the same `real2sim-harness cousins` entrypoint; `generate --scene` can defer a requested cousin stage until twin acceptance.

During planning, inspect `context/parent_scene.json` and useful parent views/video. Use `submit_cousin_plan` with one prompt per requested scene. Preserve the raw request. Set `execute:false` for planning-only intent. Do not build during planning. Each child receives the plan, parent scene library and accepted sibling context.

During generation, reuse unchanged participant bundles from the parent manifest. Use `build_scene_cousin_object` when an operation object should be redesigned. Its `object_id` refers to the accepted scene's participant; `asset_id` resumes the retained functional worker. The existing object-cousin loop checks editable structure, current physics and visual evidence. Preserve articulated roles/types/connectivity, mechanism loops and cloth attachment ownership. Dimensions, metric joint frames, proportions and appearance may adapt.

Redesigned participants follow [generate-functional-cousins](../generate-functional-cousins/SKILL.md). Whole-scene assembly and runtime review follow [compose-interactive-scene](../compose-interactive-scene/SKILL.md). Use [improve-visual-realism](../improve-visual-realism/SKILL.md) for coherent object/environment appearance under the requested variant style; the parent's pixels are not a matching target for intentionally changed features.

The initial scene-cousin contract preserves operation participant IDs and simulation kinds. Add noninteractive decorative details in the background rather than silently replacing an operation target. A missing editable parent source allows reuse but blocks functional redesign until that source is supplied.

By default preserve the parent's recognizable room structure, relative object placement and support relationships. Keep operation-object, support and background placement anchors, plus support proxy dimensions, unchanged. Redesign object silhouettes, proportions, materials and style within those anchors. Check that changed dimensions still fit the support and leave room for articulated motion and manipulation. Do not invent rearrangement for diversity, including moving geometry inside an unchanged background transform.

User-requested layout variation is also valid scene scaling. During planning, infer this from the natural-language request and record its scope in each variant's internal `layout_change_request`; use null when the user did not request it. The user does not need to supply a separate flag or structured prompt. When layout changes are requested, follow their scope and revalidate supports, collisions and operation spaces. Numeric checks permit anchor changes for that plan and report them; Astra must judge whether the actual changes stay within the requested scope. Without such a request, numeric checks preserve manifest anchors and Astra inspects internal background geometry, recognizable room structure and clearances in final scene views/video.

For a requested three-dimensional room or task-area distractors, use the [spatial cousin review](../reconstruct-environment/references/experience/spatial-cousin-review.md) from the reconstruct-environment skill. Inspect the composed child from a translated camera and check obstacle proximity when relevant.

Background choices are optional:

- Mesh environments can be authored through `build_scene_background`; compact scenes may remain fully mesh.
- `run_gaussian_background` with `operation:"cousin"` performs CPU global or regional SH appearance grading on standard Gaussian ParticleFields. Scene cousins retain the captured spatial layout by default; spatial region transforms are available when the user's request calls for layout variation. This preserves the captured representation, including directional appearance. Read [the scene cousin contract](../../docs/SCENE_COUSINS.md) for its schema and limitations.
- Learned background generation/restyling needs a configured, validated backend. CPU grading does not synthesize new semantic textures or unseen surfaces. Choose mesh completion or an available appropriate backend when those are needed.

Coordinate background appearance, operation objects and support/collision proxies in the parent's scene frame. Replacing captured operation targets or moving requested background regions can expose holes or duplicate a mesh object; inspect and repair those consequences. Background appearance supplies no physical support. Use the task-level solver plan for interactions rather than independent engines per asset.

Reassemble and request current whole-scene simulation. Tools, viewing strategy and intermediate video are chosen by Astra. Final scene acceptance remains required: `accept_scene` checks current evidence, parent/library integrity, participant lineage, layout policy and exact geometry reuse, while Astra judges conformance to the user's scope, meaningful design and function. Exact fingerprints are not semantic diversity scores. Failed simulation feedback returns to this child for repair. A missing runtime is an unfinished retained scene, not delivery.

Resume a family without resetting counters or rebuilding accepted siblings. The explicit later execution request supersedes earlier planning-only wording while preserving design constraints. Operator runner overrides apply to new/unfinished scene workers, not to the parent receipt.
