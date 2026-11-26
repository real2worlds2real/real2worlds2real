# Optional posed-observation tools

Retained rigid motion, convex probe clearance and per-participant support checks
also have a shared `audit_task_evidence` tool. Its separate
[numerical audit contract](EVIDENCE_AUDITS.md) does not require camera views.

Additional optional [reconstruction refinements](RECONSTRUCTION_REFINEMENT.md)
cover calibrated multi-video cameras, whole-map metric registration, local shape
feedback/direct Blender fitting, support edge hypotheses and rendered Gaussian cleanup.

These tools are shared by object and scene generation. Astra decides whether the available observations justify using them. They complement camera calibration, known dimensions, ChArUco, COLMAP, VGGT and the existing silhouette reviewer. They do not replace editable Blender modeling or final simulator acceptance.

Install `pip install -e '.[fitting]'` for fitting. Texture baking with existing UVs and footprint recovery use the normal host dependencies. Optional automatic UV unwrapping adds the `texture` extra (xatlas). No learned model weights or account credentials are required.

## Common request

Every request has `schema_version: 1`, a `frame` describing the common geometry/camera frame, and `views`. The rig-alignment operation below explicitly separates model and prior frames instead. Paths are relative to the request JSON. Agent calls accept files from the workspace or retained run, copy the request and all named inputs into an immutable evidence directory, and return reports plus image previews. They do not alter the original inputs. Limits are 100 views and 32 million total image pixels per request; images are at most 4096 pixels on each axis. Operation-specific limits are stated below.

```json
{
  "schema_version": 1,
  "frame": {"source": "Explain the registration and scale evidence, including uncertainty", "metric_verified": false},
  "views": [{
    "source": "Explain where this pose and intrinsics came from",
    "image_size": [640, 480],
    "intrinsics": [[600, 0, 320], [0, 600, 240], [0, 0, 1]],
    "world_to_camera": [[1,0,0,0], [0,1,0,0], [0,0,1,2], [0,0,0,1]],
    "image": "view.png",
    "mask": "mask.png"
  }]
}
```

The numeric pose above illustrates the schema; it is not a pose estimate for a real photograph. Use actual registered poses and consistent geometry coordinates. All geometry/depth numbers are expressed in metres, but `metric_verified: false` is valid when scale is an explicitly stated estimate. Setting the flag is not an accuracy certificate.

Cameras use **OpenCV** coordinates: +X right, +Y down, +Z forward; `world_to_camera` is a proper rigid 4x4 matrix. Integer image coordinates denote pixel centers. Input images/masks must already be undistorted; nonzero distortion is rejected. Resizing or cropping requires updated intrinsics. Existing PnP output has `world_to_camera_opencv` (3x4); append `[0,0,0,1]`. COLMAP camera records must be converted to pinhole intrinsics and the same undistorted pixel domain. Never mix Blender camera matrices, normalized VGGT coordinates and metric poses without explicit conversion/alignment.

Optional view fields are `ignore` (nonzero means do not use these pixels) and `depth` (finite nonnegative `.npy` camera-Z depth in metres, zero means unknown). RGB images are needed for texture baking. Binary masks are needed for fitting; their nonzero pixels are foreground. Footprint masks instead use three explicit class labels.

## Shared camera-rig alignment

Tool: `align_multiview_camera_rig(request_path="rig.json")`.

Use an already recovered shared relative-camera rig plus explicit corresponding
model points and image pixels to produce candidate cameras in the model frame.
The tool jointly fits **one positive similarity**, preserving the relative rig;
it does not independently solve or refine every camera. This differs from the
single-image `align_reference_prior` PnP helper and from independently measured
whole-map metric registration. Estimated model dimensions remain estimates.

Each view supplies `prior_world_to_camera` (rigid4x4 OpenCV pose in the shared
prior world), pinhole `intrinsics`, `image_size`, provenance `source`, paired
`model_points` (Nx3 in the model frame) and `image_points` (Nx2 in that displayed
image), and optionally `image`. Use2..64 views,4..2000 noncollinear pairs per
view and at most20000 total pairs. A translating camera baseline is required;
pure rotation cannot set model-to-prior scale. Images must be undistorted, and
intrinsics must have zero skew. Use supplied calibration when available. For
VGGT, extend each3x4 `extrinsics` pose with `[0,0,0,1]`; either use original-image
calibration or map learned intrinsics back with
`inverse(original_to_processed) @ processed_intrinsics`. Never mix processed
pixels/intrinsics with original-image coordinates.

This complete synthetic example uses two camera positions and known projected
points; its numbers are not estimates for a real task:

```json
{
  "schema_version": 1,
  "frame": {"source": "Synthetic model frame; task dimensions may be assumed", "metric_verified": false},
  "prior_frame": {"source": "Synthetic shared relative-camera frame"},
  "views": [{
    "source": "Synthetic rig view1",
    "prior_world_to_camera": [[1,0,0,0],[0,1,0,0],[0,0,1,2],[0,0,0,1]],
    "intrinsics": [[600,0,320],[0,600,240],[0,0,1]], "image_size": [640,480],
    "model_points": [[-0.1,-0.1,0],[0.1,-0.1,0],[0.1,0.1,0],[-0.1,0.1,0],[0,0,0.1],[0.1,0,0.1]],
    "image_points": [[290,210],[350,210],[350,270],[290,270],[320,240],[348.5714286,240]]
  }, {
    "source": "Synthetic rig view2",
    "prior_world_to_camera": [[1,0,0,-0.2],[0,1,0,0],[0,0,1,2],[0,0,0,1]],
    "intrinsics": [[600,0,320],[0,600,240],[0,0,1]], "image_size": [640,480],
    "model_points": [[-0.1,-0.1,0],[0.1,-0.1,0],[0.1,0.1,0],[-0.1,0.1,0],[0,0,0.1],[0.1,0,0.1]],
    "image_points": [[230,210],[290,210],[290,270],[230,270],[262.8571429,240],[291.4285714,240]]
  }],
  "max_reprojection_rms_px": 2,
  "max_evaluations": 300
}
```

Optional `robust_scale_px` defaults to2; `max_evaluations` allows10..2000.
Optional positive `max_reprojection_rms_px` must hold for **every** view and all
points must be in front for `quality_target_passed:true`. All views remain in
the objective and report. Inspect per-view/per-point reprojection, overlays,
optimizer status and independent evidence; `ok:true` only reports computation.
No quality target means `quality_target_passed:null`, not an automatic pass.

Outputs include `candidate_cameras.json` with model-world `world_to_camera`
records and `world_to_prior_similarity` in the report. Existing camera solutions
are never overwritten. Learned rig errors cannot be repaired by this common
transform; low error on fitted correspondences is not independent validation.
Compare candidate cameras and current-source geometry against the useful views
before adopting them. A poor joint fit should trigger investigation of the prior,
pixel mapping, model assumptions or correspondences, not unsupported deformation.

## Bounded multi-view fitting

Tool: `fit_multiview_geometry(request_path="fit.json")`.

Provide one or more posed masks and a self-contained Python `model_script` defining `build_mesh(parameters)`. The operation name is retained for compatibility; a single view is supported for visible-projection correction. It returns a dictionary with `vertices` (Nx3 world coordinates), `triangles` (Fx3 integer indices), and optional `uv` (Fx3x2 triangle-corner UVs). NumPy is available. The callable executes in an isolated child process with a bounded tool timeout; it may construct any agent-authored parameterization, rather than selecting a fixed object kit.

Cameras are fixed during the fit. Retain independently established registration
when available; explicitly label any estimated camera or scale in `source` and
`frame`. A result under an estimated camera is conditional on that camera, not
independent calibration. With one camera, fit only observed projection features;
hold unobserved dimensions fixed according to disclosed modeling assumptions.
The report returns `fit_scope`, provided view count, distinct camera-pose count,
camera provenance and `three_d_reconstruction_verified:false`. Repeated crops or
files at the same pose do not add a viewing direction. Multiple views can supply
additional constraints but their count alone does not establish reconstruction
accuracy. All views still participate in the objective and retained evidence.

Shape priors belong in the parameterized model. For example, an observed
rotational vessel body can use a revolved radial profile with one shared radius
per height. The fitter changes only exposed parameters; it does not add mesh
deformation variables. Preserve supported symmetry, straight sections and other
structural constraints in every generated candidate. Mask/camera error must not
be compensated by unsupported asymmetry. Record prior evidence with the model
source and inspect the fitted mesh; an IoU gate does not validate those priors.

Additional request fields:

```json
{
  "model_script": "parametric_model.py",
  "parameters": {
    "radius_m": {"initial": 0.04, "bounds": [0.02, 0.08]},
    "height_m": {"initial": 0.10, "bounds": [0.07, 0.15]}
  },
  "max_evaluations": 100,
  "time_budget_seconds": 120
}
```

Supports 1–12 scalar parameters and 10–1000 mesh evaluations. The local Nelder–Mead search uses normalized bounded parameters and the mean IoU over **all** supplied views; ignored pixels do not count. It returns the best observed candidate even when the budget expires, and reports convergence separately from tool execution success. Outputs include `parameters.json`, `fitted_mesh.npz`, per-view overlays and the complete evaluation trace. Mesh limits are 300,000 vertices and 100,000 triangles; a coarse fitting mesh is usually more practical than the final detailed surface.

Optional positive `time_budget_seconds` limits search wall time. The task entry
also caps it below the operator's process timeout, reserving report time. Search
stops between evaluations using the observed evaluation duration; an individual
slow model call can still exceed that estimate. A time-limited result reports
`time_budget_exhausted:true` and `optimizer_converged:false`, with its actual
best evaluated candidate. Improving candidates are retained atomically in
`search_progress.json` and its hashed mesh. If a model hangs or the process hits
the hard timeout, the operation remains `ok:false`; `partial_candidate`, when
present, identifies retained work requiring inspection and fresh source review.
It is not a completed fit or an accepted asset.

Use only when the reference views and masks constrain the chosen parameters. Inspect the per-view overlays and RGB, apply the chosen parameters to the editable Blender source, and run normal geometry/physics review again. Silhouette agreement cannot establish cavities, hidden details, measured scale, joint function or material correctness. A poor fit may indicate camera/mask errors or a parameterization that cannot express the shape; do not force a high score by dropping inconvenient views.

## Read-only projection review

Use the same `fit_multiview_geometry` operation with `evaluate_only:true` to
inspect a current candidate without parameter search. Keep the common frame and
registered `views` with observed `mask` and optional `ignore` fields. Supply one
of these static sources, with no `parameters`:

```json
{"evaluate_only": true, "mesh": "current-world-mesh.npz"}
```

```json
{"evaluate_only": true, "model_format": "blender",
 "model_script": "current-scene.blend", "selected_roots": ["can_placement"]}
```

The NPZ retains the common mesh arrays. Blender mode uses the operator-owned
executable and the existing task boundary; request JSON cannot choose a command.
Saved Blender sources are opened with automatic Python execution disabled.
Select exact mesh names with `selected_objects`, or named parents plus their
visible mesh descendants with `selected_roots`, never both. Missing selections
are errors. World transforms and evaluated modifiers are retained, without
rescaling, per-object alignment, source edits or an implicit optimizer.

The report has `operation:"silhouette_inspection"`, `evaluate_only:true`,
`evaluations:1`, `optimizer:null`, source/request hashes and selection, per-view
IoU, local missing/excess regions and overlays. `evaluated_mesh.npz` is a derived
snapshot, not a replacement asset. Optional existing `quality_targets` describe
only their chosen projection scope; `ok:true` reports computation success.
No optimizer convergence or simulation acceptance is claimed. Re-run affected
views after mesh, placement or camera changes. Keep a review for each important
entity; an unrelated object's landmark residual cannot close its finding.

For Python/Blender parametric callbacks, `evaluate_only:true` instead evaluates
the supplied initial parameters once, retaining the normal parameter contract.

## Observed texture baking

Tool: `bake_reference_texture(request_path="texture.json")`.

Provide posed RGB observations and a `.npz` `mesh` with the arrays above, including a nonoverlapping `[0,1]` UV atlas. Existing mesh geometry and camera poses must use the same frame. Overlapping UV islands that refer to different surface positions are rejected.

Additional fields: `uv_mode` (`auto` by default, uses existing UVs or generates them with xatlas; `existing` rejects missing UVs; `unwrap` generates a new atlas), optional `fill_unseen` (default false), `resolution` (32–2048, default 512), `mask_erode_px` (0–16, default 1), `visibility_tolerance_m` (positive, default 0.003), `color_outlier_threshold` (RGB distance, default 80), and optional `double_sided` (default false; requires consistent outward face winding otherwise).

The baker checks triangle self-occlusion with a perspective Z-buffer, optional real depth and masks, view obliqueness and relative sharpness. It blends robustly around a weighted color median. Unknown atlas texels remain transparent; coverage and per-view contributions are reported separately. It writes `texture.png`, unchanged `textured_mesh.npz`, `coverage.png` and numerical coverage data. Import the atlas into Blender's Base Color on the matching UV layer; preserve or procedurally complete unseen regions as explicitly inferred appearance.

Color fusion occurs in linear sRGB with robust outlier rejection. Sharpness is measured over the target region, rather than an unrelated background. Optional `fill_unseen: true` also writes `texture_completed.png` using push/pull interpolation, while `texture.png` and numerical coverage preserve observed versus unknown pixels. Completion is inferred and must not be called recovered hidden appearance.

This captures photographed appearance, including illumination, reflections and exposure. It does not solve intrinsic albedo, roughness or lighting. Separate scene occluders need `mask`, `ignore` or actual `depth`; the object's own mesh alone cannot identify an unrelated hand or foreground object. Inspect seams, transparency, printed details and specular surfaces before using the result.

The included `scripts/reference_mesh_bridge.py` exports one evaluated Blender mesh, its world coordinates in metres and its active UV layer. Run it with Blender `--background source.blend --python scripts/reference_mesh_bridge.py -- export --object ObjectName --output mesh.npz`. After baking, use `-- apply --object ObjectName --mesh textured_mesh.npz --texture texture.png --output textured.blend` with the original source loaded. Apply checks that geometry and UVs still match, packs the texture and saves a new editable file. It assigns one new base-color material to the selected object, with an editable neutral fallback for unseen texels; author other PBR properties as needed. Generated UVs can be transferred back when editable mesh topology corresponds; UV cuts inside a polygon require triangulation or UV preparation before export. For native Blender Smart UV Project, export with `--unwrap --prepared-blend uv-source.blend`, then apply to that retained UV-prepared source. Geometry changes still reject the transfer. It preserves other objects and source physics annotations.

## Support footprint recovery

Tool: `recover_support_footprint(request_path="support.json")`.

Provide registered three-class masks and a plane with `origin_m`, orthonormal `u` and `v`, `bounds_m: [[u_min,v_min],[u_max,v_max]]`, and positive `cell_m`. The supplied plane may be a desk, shelf, floor or other support; its pose/height must come from evidence or be explicitly estimated.

Mask defaults: support 255, occluder 127, background 0. Override with `support_value`, `occluder_value`, `background_value` if needed. `min_support_views` defaults to 1, and `max_background_fraction` to 0.25. Occluded/ignored and out-of-frame pixels do not vote for removal. Cells with no observations stay unknown; the tool does not fill hidden boundaries or create a collision mesh.

Optional `depth_consistency:{"absolute_m":0.02,"relative":0.01}` checks the
supplied plane against camera-Z depth in every registered view. This mode needs
at least two views, each with a `depth` file, and defaults to
`min_support_views:2` (values below two are rejected). A support label contributes
only where its depth agrees with the projected plane within
`max(absolute_m, relative * depth)`. Missing depth and foreground surfaces keep
the plane unknown/occluded; observed depth behind the candidate plane provides a
negative free-space vote, unless the mask/ignore label marks the sample occluded.
Occluder labels are never promoted to support by depth alone.

Depth is an input estimate or measurement, not automatically independent ground
truth. It must use the same raster, camera-Z convention, scale and coordinate
frame as the supplied calibration. Repeated views or a shared depth-model bias
can agree without proving accuracy. The retained grid includes separate
`depth_agreement_views`, `depth_foreground_views`, `depth_unknown_views` and
`depth_free_space_views`; these are depth diagnostics, while support votes also
require the semantic labels. Omitting this option preserves mask-only voting.

Optional `min_evidence_views` and `min_background_views` require enough actual observations before negative votes carve a cell. Both default to 1 and are bounded by the supplied view count.

Optional `surface_points` is a `.npy` Nx3 or `.npz` `points_m` cloud registered into the same frame. `plane_refinement: {"search_m":0.3,"bin_m":0.005,"min_points":100,"apply":true}` diagnoses the dominant point layer along the plane normal. Competing layers more than 3 cm apart are reported; ambiguous corrections are not silently applied. This is not an arbitrary plane-orientation estimator. `point_evidence: {"band_m":0.015,"block_m":0.05,"seed_uv_m":[0,0]}` optionally restricts the candidate to a connected point cluster's inferred hull near the plane; omit the seed to choose the largest cluster.

Optional `completion` accepts `close_radius_m`, `open_radius_m`, `extend_under_occlusion_m`, `min_occluded_views` and `occlusion_fraction`. Morphology and bounded growth under predominantly occluded cells affect a **separate candidate**, not the raw observations. `candidate.png` marks inferred additions yellow, and the numeric output includes `candidate_footprint` and `inferred_completion`. These are editable modeling hypotheses, not proof of a hidden boundary.

The returned `footprint.npz` contains states (0 unknown, 1 retained support, 2 rejected), support/background/occlusion counts, grid edges and the plane frame. `footprint.png` shows unknown gray, support green and rejected red. Use it to guide Astra's editable support geometry; judge unseen-edge assumptions separately. It estimates a footprint on the supplied plane, not the plane height or an arbitrary 3D surface.

## Standalone invocation and evidence

```bash
python scripts/reference_evidence.py silhouette_fit --request fit.json --output fit-result
python scripts/reference_evidence.py reference_texture --request texture.json --output texture-result
python scripts/reference_evidence.py support_footprint --request support.json --output support-result
```

Standalone requests are operator-owned and may reference local files. Agent calls restrict inputs to the current workspace/run and retain their hashes before execution. Success means the operation completed; score, convergence, coverage and remaining uncertainty must still be read. None of these reports replaces object/scene acceptance or provides an automatic physical accuracy claim.
