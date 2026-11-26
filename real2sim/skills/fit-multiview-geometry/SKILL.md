---
name: fit-multiview-geometry
description: Inspect current geometry against registered silhouettes or fit bounded shape/pose parameters, preserving structural priors and single-view limits.
---

# Fit observed silhouettes

`fit_multiview_geometry` performs bounded numerical search over your own
`build_mesh(parameters)` function, shared by object and scene tasks. Prefer it
when registered silhouettes constrain a consequential shape or placement error.
It supplies automatic refinement beyond `inspect_silhouette`, a managed helper
which only scores an existing attempt; native clients can inspect retained
attempt reports and masks directly.

For a current candidate, first use `evaluate_only:true` with `mesh:"current.npz"`,
or `model_format:"blender", model_script:"current.blend"` and exact
`selected_objects` or `selected_roots`. No search parameters are needed for
these static inputs. This checks the actual saved geometry and world transforms
once, with source hashes, per-view masks and local residuals; it does not change
the source or certify cameras, scale, hidden geometry or acceptance. Select each
important scene participant separately. Ignore unrelated occluders where needed,
and inspect both the numerical differences and the corresponding RGB.
See [read-only review](../../docs/OBSERVATION_TOOLS.md#read-only-projection-review).

Use this measured discrepancy to decide whether to repair camera registration,
mask identity/visibility, structure or bounded geometry parameters. A center-only
landmark fit cannot replace the complete visible-boundary check. After editing,
repeat the inspection on the actual current source.

If evidence is missing, use [segment-reference-views](../segment-reference-views/SKILL.md) for instance masks, [recover-multiview-geometry](../recover-multiview-geometry/SKILL.md) for relative cameras, or [measure-reference-geometry](../measure-reference-geometry/SKILL.md) for supplied calibration/metric anchors. Reuse registered observations when already available. For an existing attempt, [review-silhouette](../review-silhouette/SKILL.md) helps distinguish framing and shape disagreement before choosing fitted variables.

Read [the request contract](../../docs/OBSERVATION_TOOLS.md#bounded-multi-view-fitting) before preparing the workspace JSON. Use one or more genuine observed masks, consistent undistorted camera coordinates and a small set of meaningful bounded geometry/pose parameters. A coarse mesh is usually enough to fit outer dimensions before adding details. Cameras and scale may use existing calibration/multiview tools; invented correspondences are not evidence.

Choose the model's geometric priors before optimization, using visible structure
and manufacturing/mechanism evidence. Encode them in `build_mesh(parameters)` or
the Blender builder so every candidate preserves them. For a rotational vessel
body, revolve one radial profile about a fixed axis and vary shared radii,
heights and transitions; do not fit left and right vertices independently.
Likewise preserve supported mirror symmetry, straight panels, uniform section
or repeated dimensions when appropriate. Apply a prior only to the relevant
part: a handle, spout or other supported asymmetric feature needs its own geometry.
Retain the prior's evidence and implementation with the model source. Inspect
the resulting geometry as well as the overlay. When image fit conflicts with a
supported prior, investigate camera, mask, occlusion and model expressiveness;
do not introduce unsupported deformation just to increase IoU. A remaining
projection mismatch must stay visible in the result and its quality verdict.

Do not confuse symmetry with a fixed shape template. A rotational body can have
a changing radial profile; an initial straight taper or default section is a
model assumption until the reference supports it. Choose shared profile
parameters that can express the visible shape while retaining the justified
symmetry. Constraining unsupported draft dimensions merely because they already
exist can prevent a correct fit just as excessive freedom can invent one.

With only one useful photograph, use this tool to correct visible projected
proportions and local outline errors under a fixed corresponding camera. Do not
discard useful single-view evidence merely because multi-view reconstruction is
unavailable. Retain camera-estimation assumptions explicitly; the fit cannot
recover the camera, hidden surfaces, depth or absolute scale. Hold unsupported
geometry parameters fixed in the model and disclose the chosen shape priors.
Additional crops/resolutions of the same image do not become new viewpoints.
Inspect `fit_scope`, per-view overlays and local differences, and review the
corrected source in RGB. A high projection score is not a complete 3D twin.

Call `fit_multiview_geometry(request_path="fit.json")`. Inspect per-view overlays, mean/per-view IoU, the chosen parameters and convergence status. Set `time_budget_seconds` for a bounded search within the task's remaining time. A stopped search returns its best evaluated candidate without a convergence claim; a hard timeout remains a failed operation and may expose a hashed `partial_candidate` requiring fresh review. Incorporate useful results into the editable Blender source, then recheck appearance, structure and simulation normally.

A poor fit can come from camera registration, masks or insufficient model freedom. Repair the relevant cause; do not silently discard difficult views. Hidden geometry, internal structure and calibrated physical scale are not certified by a silhouette score. Skip this fitter when current quantitative comparisons already establish the relevant geometry, or when the input cannot constrain its parameters; retain the actual evidence or limitation.

When registered views show a consequential shape discrepancy that your bounded
parameters can express, prefer this existing fitter over repeated unsupported
parameter guesses. A manual correction is a candidate hypothesis; when the
observations constrain the disputed property, verify it quantitatively in the
corresponding views before closing the finding. Review the reference's visible local features
after fitting; global IoU is not a substitute for that review, and changing
dimensions cannot repair a shading-normal error.

Choose observations that constrain the disputed property. A top-center landmark
can fit well while height relative to an uncertain base, silhouette or neighboring
placements remains wrong. Check the relevant boundaries and support assumptions,
and keep supported proportions and structural priors in the parameterization.
For scene participants, label unrelated occluders with the documented view
`ignore` mask; visible-mask fitting must not shrink hidden geometry to imitate
occlusion. Bound shared ratios/profile parameters where supported, and do not
use frames of changing object state as one fixed-geometry observation unless
the corresponding object poses/state are explicitly represented.
After applying or manually changing fitted parameters, evaluate the actual final
mesh with current cameras; recompute affected residuals rather than retaining
the optimizer's old statistics. Verify that the builder applied the chosen
parameters, including dependent dimensions and modifiers.

Read [additional numerical options](../../docs/RECONSTRUCTION_REFINEMENT.md#local-shape-feedback-and-direct-blender-fitting)
when global IoU hides a missing notch/handle, projected landmarks would help, or
the parameterization needs actual Blender modifiers. Local residuals and optional
quality targets expose small connected differences; they remain separate from
tool execution success and final acceptance. `model_format:"blender"` directly
evaluates your `build(parameters)` with the operator's Blender. Select relevant
visual objects, retain independent parts and keep the normal source/physics loop.
Use the existing pure mesh callback when it is sufficient. Optional projected
curves describe arbitrary features; they cannot measure hidden cavity depth.
