---
name: plan-scene-generation
description: Plan a scene-generation task from images or video and a natural-language request, separating interactive participants, supporting collision geometry and visual environment while coordinating scene and object tools.
---

For shared scene reconstruction use [the environment workflow](../reconstruct-environment/SKILL.md). This skill supplies its native procedures.

The user's request defines the task. Explicitly named operation targets, parts, actions and exclusions take priority over inferred scene roles. When no target is specified, infer useful candidates from the evidence. A cabinet may be visual context in one task and an articulated target in another. Preserve this distinction in object-worker assignments and retain the original request as context.

## Evidence obligations for observed scene twins

For twins from photographs, multiview captures or video, retain these outcomes:

1. **Understand the observations.** Identify participant roles, supported
   structural priors and exceptions, changing object state, and uncertainties.
   Carry this understanding into participant assignments and editable models.
   Choose meaningful adjustable dimensions within that structure; segmentation
   supplies visible-boundary evidence, not the object's 3D construction.
2. **Establish camera/frame reliability.** Justify the cameras, scene coordinates
   and scale to the extent required by the claim. Dimensions assumed for camera
   fitting remain assumptions until checked by additional evidence. An anchor's
   model axes do not by themselves establish gravity or a shared support plane.
3. **Check the important entities and relations quantitatively.** Cover their
   visible geometry, layout, occlusion and support using the useful observations.
   One object's landmark fit does not verify the rest of the scene. Keep a
   concise record of entities/relations, source version, views, scope and findings.
4. **Diagnose discrepancies and verify repairs.** Invoke applicable tools for
   consequential mismatches. Preserve supported priors and recheck the actual
   current source after changes; old residuals cannot close a new candidate.
5. **Complete the requested final checks.** Inspect the composed scene in its
   target renderer and complete applicable physical/functional checks before
   claiming delivery. Report concrete missing evidence and unfinished criteria.

The agent chooses the tools, order and iteration path. Required evidence cannot
be silently omitted. Limited inputs narrow the supported claim; use their
available information without inventing hidden geometry or requiring an arbitrary
number of views. Numerical success and plausible images each cover only their
actual scope.

## Plan and select evidence routes

Establish a coarse spatial context: the scene frame, available scale evidence, important supports and rough participant placement. Use estimates explicitly when measurements are absent. This skeleton need not be a polished environment. Known participant dimensions can help refine the scene frame; choose the detailed order and tools from the available evidence.

Select shared evidence guidance by what is missing: [measure-reference-geometry](../measure-reference-geometry/SKILL.md) for supplied metric/board evidence, [recover-multiview-geometry](../recover-multiview-geometry/SKILL.md) for unregistered consistent views, and [align-reference-camera](../align-reference-camera/SKILL.md) for draft-to-photo alignment. When boundaries or support shape need numerical evidence, use [segment-reference-views](../segment-reference-views/SKILL.md) and, with justified cameras and planes, [recover-support-footprint](../recover-support-footprint/SKILL.md). These are conditional routes, not a required sequence.

For precision scene twins, establish a current-source visible-boundary baseline
for each important participant while there is still time to repair geometry.
The default is `fit_multiview_geometry` with `evaluate_only:true` on the saved
mesh/Blender source, selecting the participant's actual visual meshes and world
transforms. Read its specialist before preparing the request. Obtain the
observations needed for this check: reuse supplied masks, retain reliable manual
contours, or use `segment_reference_views` and inspect the intended instances.
Not having requested masks yet does not exempt a participant from this baseline.
Simple primitives and plausible overview renders still need verification.

Reuse an existing equivalent check only when it projects the actual current
mesh's visible boundary, identifies the participant, source/camera versions and
corresponding views, handles occlusion, and retains per-view/local differences.
Hand-marked corners, centers and axis-aligned extents can constrain a model or
camera fit; the optimizer's residual on those annotations is not this saved-source
boundary review. Do not replace an available framework check with a new partial
solver merely because its fitting residual is small. Review each participant's
own discrepancies; a joint average cannot close a poorly fitted object.

For a consistent multiview capture whose cameras were inferred from assumed
draft dimensions, also obtain a separate camera/structure cross-check through
the applicable [multiview tools](../recover-multiview-geometry/SKILL.md), unless
adequate independent camera evidence is already retained. Use feature geometry,
depth or a learned prior according to availability and the ambiguity; VGGT is
not individually mandatory. Residuals on the same fitted landmarks are not an
independent check. Read and interpret actual outputs in their camera/pixel frame;
a failed invocation does not establish reliable cameras. Retain concrete failures
and unresolved uncertainty rather than declaring the check satisfied.

For video, use complementary frames for a static arrangement and time-resolved
evidence for changing state; read the common methods before combining them.
Resolve mask identity failures with reviewed candidates, better prompts or
retained boundary annotations. A failed prompt does not verify the object or
remove its review obligation. Prioritize unresolved reconstruction accuracy over
additional isolated physics checks or presentation renders that cannot resolve
it. Do not leave the boundary baseline until the remaining budget cannot support
correction. Physical work that informs geometry or uses otherwise blocked time
can still happen earlier; all requested final checks remain required.

A consequential shape or layout conflict must trigger a targeted diagnostic
before further unsupported parameter guesses. For projected proportions, obtain
and inspect instance masks/contours and use bounded fitting where applicable.
For ambiguous height, depth or contact, check registered multiview/depth evidence
and the relevant base/support boundaries. Reuse adequate evidence and supported
structural priors. If several objects disagree coherently across views, investigate
the shared cameras, frame and support plane before changing each object's shape
independently. A low residual on an assumed anchor cannot establish those shared
quantities. Inspect backend readiness and actual outputs; a tool name in
the catalog is not proof that its runtime or weights are available. A concrete
limitation stays explicit and does not make an assumed repair verified.

Treat hand-tuned dimensions and placements as candidate hypotheses. When current
cameras, masks or landmarks can constrain a consequential discrepancy, prefer
the existing bounded fitter and require a quantitative check of the disputed
property before closing it, including after a manual edit. Visual approval or
a low residual on unrelated landmarks is insufficient. Preserve supported
priors and reject numerically improved candidates that violate them.

Refine interactive assets and the visual environment with their respective tools. These branches can progress independently once they have compatible frames, roles and approximate placement. Interactive assets use the complete object-generation/review pipeline and preserve editable structure, collision geometry, parameters and deformation bindings. Backgrounds can use code-generated meshes, geometry reconstruction or Gaussian methods according to capture coverage and the task. Shared image/video, camera, calibration and geometry tools serve both branches.

Each participant follows [generate-editable-object](../generate-editable-object/SKILL.md), using its representation-specific guidance and the original scene constraints. Carry reference-supported identity features and useful view pointers into participant assignments, and review them both locally and in the composed scene; an object can be too small or occluded in a scene overview to expose its shape errors. Read [compose-interactive-scene](../compose-interactive-scene/SKILL.md) when preparing manifests and whole-scene execution; individual acceptance does not establish scene interaction. If choosing a Gaussian background, read [prepare-gaussian-background](../prepare-gaussian-background/SKILL.md). For requested real-motion matching, use [calibrate-observed-physics](../calibrate-observed-physics/SKILL.md) and return the selected parameters to current scene execution and review.

Allocate detail by task relevance. Operation targets use the full object pipeline; a simple supporting tabletop may need only a quick mesh with correct placement and collision. Irrelevant surroundings may be simplified or omitted. Use Gaussian appearance when it contributes useful background fidelity and the capture supports it; do not require it for every scene.

For a reference-based scene twin, visual matching is a task requirement unless the user explicitly narrows the scope. Read [improve-visual-realism](../improve-visual-realism/SKILL.md) for participant and environment appearance. Compare corresponding reference views for layout, proportions, visible colors and surface details, material response, illumination and contact shadows. A geometrically simple support may still occupy much of the image and need careful appearance matching. Investigate whether a visible difference comes from geometry, materials, lighting, color management or export before choosing a repair; use [bake-reference-texture](../bake-reference-texture/SKILL.md) when observed surface detail warrants it. Carry this visual objective into participant assignments and review the composed scene in the target renderer. Passing geometry and physics checks does not establish visual fidelity. Address differences that matter to the requested twin, retain uncertainty in unobserved appearance, and describe consequential remaining differences at acceptance.

For a compact, carefully modeled room with several operable objects, prefer an editable mesh environment and mesh-based asset surfaces. This supports structural changes, functional variations and explicit physical geometry. Deformables and fluids retain their dedicated simulation representations and surface bindings. Gaussian background is an optional choice for complex appearance/coverage needs, not a required stage or a prerequisite to mesh generation.

Determine the manipulation region before finalizing a Gaussian background. Exclude or replace the original interactive objects so generated assets do not appear twice. Inspect the background revealed by motion, seams, occlusion and capture coverage. Generated unseen completion is inferred content, not measured recovery. Background appearance still needs explicit collision/support geometry where interactions occur.

Select a physics backend for the interacting task with `plan_scene_physics`: rigid/articulated, cloth and soft-body tasks default to PhysX. Fluid/strand tasks retain a Newton default for the shared interaction task. An explicit user backend choice takes priority; do not silently change it to suit an available adapter. Require supported solver/contact capabilities. Use the simulator adapter appropriate to the operator's installation; the planning flow imposes no Isaac Sim version.

Compose and review the whole scene, including its intended actions and reference views. Use a scene-level review camera when a view centered on one object would hide relevant context. Source, geometry, placement or parameter changes require affected checks to be rerun. Return failures and rendered evidence to Astra and repair the responsible layer. Program checks and Astra's visual/functional judgment are both needed before `accept_scene`.

These are planning dependencies and deliverable criteria. Astra chooses concrete tools, model providers, refinement order, useful intermediate simulation/video inspection and repair strategy. The current scope is credible appearance, editable/scalable assets and basic verified operability; fitting real physical parameters remains downstream work unless requested.
