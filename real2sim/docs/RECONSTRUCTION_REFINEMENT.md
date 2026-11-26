# Optional numerical reconstruction refinements

These shared object/scene capabilities supply evidence and candidate edits. The
agent selects them when the input and task justify them. They do not impose a
generation order or replace source-matched simulator acceptance.

Use `real2sim-harness task schemas <operation>` and `real2sim-harness task call <run> <operation>
--request <arguments.json>` in an external coding-agent session. The call arguments
are `{"request_path":"workspace-relative-request.json"}`. Managed sessions share
the same domain implementations. Standalone observation operations use
`python scripts/reference_evidence.py <operation> --request ... --output ...`.

## Calibrated video cameras

`recover_video_cameras` / standalone `video_cameras` takes:

```json
{
  "schema_version": 1,
  "frame": {"source": "Describe the capture and scale evidence", "metric_verified": false},
  "streams": [{
    "video": "capture.mp4",
    "interval_seconds": 0.5,
    "max_frames": 250,
    "start_seconds": 0,
    "end_seconds": 60,
    "camera": {
      "image_size": [1920, 1080],
      "intrinsics": {"fx": 1500, "fy": 1500, "cx": 960, "cy": 540,
                     "distortion": [0, 0, 0, 0, 0]},
      "source": "Replace with actual calibration provenance"
    }
  }],
  "matching": {"window": 10, "cross_stride": 10, "loop_stride": 10, "max_pairs": 50000},
  "num_threads": 4
}
```

Numbers illustrate the contract, not a usable calibration for an arbitrary phone.
Alternatively each stream supplies ordered
`frames:[{"image":"a.png","time_seconds":0},...]` instead of `video`.
The video route chooses the sharpest frame in each time bucket; frames retain
source hashes and timestamps. Image storage is streamed rather than holding all
selected full-resolution frames in memory.
Decoder timestamps preserve variable-rate timing; unavailable/nonmonotonic
timestamps use an explicitly labeled nominal-rate fallback.

Each stream keeps its own camera. Omit `camera` to estimate OPENCV intrinsics on
that stream before joint matching; this is self-calibration, not a measurement.
Supplied and estimated cameras are used to undistort frames once. Sequential,
cross-stream and optional loop pairs are matched with COLMAP SIFT, then the joint
mapping keeps those intrinsics fixed. Outputs retain images, sparse geometry,
camera poses, calibration provenance, pair count and reprojection diagnostics.
Inspect stream coverage and rejected views; one registered subset does not align
the entire capture. Use a consistent static state; articulation/deformation can
invalidate a rigid reconstruction. No automatic absolute scale is claimed.

Existing `run_reference_geometry` remains available for ordinary unordered
images, including its optional build-dependent learned-feature backend.

## Metric registration of the whole reconstruction

`register_metric_reconstruction` / standalone `metric_registration` takes the
[common posed-observation contract](OBSERVATION_TOOLS.md#common-request), plus
either a known `board` in the [measurement format](MEASUREMENTS.md), or:

```json
{
  "correspondences": {
    "source": "Explain independent metric landmark evidence",
    "reconstruction_points": [[0,0,0],[1,0,0],[0,1,0]],
    "metric_points_m": [[0,0,0],[0.1,0,0],[0,0.1,0]]
  },
  "registration": {"tolerance_m": 0.005, "trials": 256}
}
```

For board mode, provide actual RGB views and registered cameras. The tool detects
corners, triangulates those seen from at least two useful baselines, rejects
negative-depth/poor-reprojection corners and estimates a robust positive
similarity into the board's metre frame. At least six triangulated board corners
are required. Explicit paired landmarks need at least three noncollinear points;
more independently observed landmarks are preferable.
Explicit 3D landmark registration may omit camera views; board mode needs them.

Returns `source_to_metric`, metres per source unit, inlier masks, **all** landmark
residuals and transformed cameras. Optional `surface_points` also writes a
transformed point cloud. Source poses and geometry stay unchanged. Apply the same
similarity exactly once to all relevant geometry/cameras before further fitting.
Board pose from a single image cannot independently scale an arbitrary object.
Local board agreement does not eliminate drift elsewhere in a long capture.

`calibrate_measurement_camera` also accepts `model:"full"|"constrained"`,
`subset_trials:0..32` and `detector_upsample:1|2`. Defaults preserve the full
camera fit. The constrained mode fixes the principal point/aspect and estimates
one radial term; consider it for weak calibration coverage, with the assumptions
made explicit. Random-half focal refits diagnose sensitivity, not a confidence
interval. Existing reprojection and viewpoint-diversity checks remain.

## Local shape feedback and direct Blender fitting

`fit_multiview_geometry` retains its ordinary `build_mesh(parameters)` callback.
Optional `local_weight:0..10` adds connected missing/excess-region area to the
all-view IoU objective; default zero retains the old objective. Local metrics are
reported regardless. `local_residual:{"min_fraction":0.003,"min_width_px":2}`
controls meaningful residual size and rejects thin registration fringes.

Optional `quality_targets:{"min_mean_iou":0.85,"max_mean_local_difference":0.02}`
reports whether requested numeric targets were met. `ok:true` means execution
completed; `quality_targets_passed` and optimizer convergence are separate.
These thresholds are task settings, not universal reconstruction guarantees.

Set `model_format:"blender"` when the model needs Blender geometry/modifiers.
The retained `model_script` then defines `build(parameters)` using `bpy`; the
operator's configured Blender executable is supplied by the task service.
Standalone requests additionally supply `blender_executable`. A persistent worker
evaluates actual transformed meshes without merging parts. Optional
`selected_objects:["body","handle"]` controls which meshes constrain the fit;
otherwise visible mesh objects contribute. Ordinary pure-mesh fitting remains
available. The worker runs inside the normal task execution boundary.

Either callback may return `features:[{"name":"rim","points":[[0,0,1],...]}]`.
The Blender callback returns this optional dictionary in addition to constructing
the scene. Arbitrary model curves/landmarks are projected into every view and
returned as images and coordinates. Optional
`observed_pixels_by_view:{"0":[[x,y],...]}` reports paired pixel residuals.
Supply genuine correspondences. A visible rim overlay cannot establish hidden
cavity depth, and fitting does not certify articulation or contact function.

### Optional image-landmark constraints

The fitting request may supply `landmarks:"landmarks.json"`. This constrains
existing named model `features` during optimization, rather than only drawing
an overlay after silhouette fitting. For example:

```json
{"schema_version":1,"source":"Describe visible point correspondence and annotation uncertainty",
 "observations":[{"view":0,"feature":"rim","point_index":0,
                  "pixel":[120,80],"covariance_px2":[[4,0],[0,4]]}]}
```

View indices refer to the request's registered views; `point_index` selects a
point in the named model feature (default zero). Coordinates and uncertainty are
in the original view's pixel domain. Supply symmetric positive-definite 2x2
covariances and genuine point correspondences, not arbitrary silhouette centers.
A single 2D landmark constrains a projection, not a unique 3D location. Include
additional views when needed for identifiability. Missing model points, invalid
uncertainty and behind-camera projections fail explicitly.

The optional landmark loss is the mean squared covariance-whitened residual per
coordinate, multiplied by `landmark_weight` (positive, default 1). Every supplied
observation contributes. The report retains directed pixel residuals and
`rms_sigma`; it also saves `fit_request.json`, including camera calibration and
fit settings, with the input fingerprints. An optional
`quality_targets.max_landmark_rms_sigma` sets a task-specific numeric gate.
Candidates meeting all requested gates take priority over lower-loss candidates
that fail a gate. Numeric targets do not replace visual and functional review.

`optimizer:"Powell"` is available as an optional bounded local search;
`"Nelder-Mead"` remains the default. Select based on parameter behavior and
execution budget. Neither method promises a global optimum or supplies missing
geometry/camera evidence. Omitting landmarks retains the silhouette objective.

## Support boundary hypotheses

`recover_support_footprint` keeps its arbitrary supplied-plane grid, raw votes,
point-height/cluster evidence and separate inferred completion. Optional
`edge_refinement` proposes **rectangular** boundaries in that plane:

```json
{
  "edge_refinement": {
    "bounds_uv_m": [[-0.5,-0.3],[0.5,0.3]],
    "evidence": "color",
    "search_m": 0.06,
    "step_m": 0.005,
    "probe_m": 0.02,
    "min_pairs": 30,
    "min_score": 0.3,
    "min_improvement": 0.05
  }
}
```

Choose `mask` or `color`; color needs RGB images as well as class masks. Each edge
compares inner/outer samples across views, excludes occluders/ignored pixels and
reports the offset sweep and supporting pair counts. A proposal must improve on
the original boundary with enough evidence. Color is estimated from the support
mask's interior in each view. Similar-colored neighboring objects may confuse it.
Raw footprints remain unchanged, and the optional rectangular proposal does not
replace nonrectangular modeling or certify hidden boundaries.
Omitted edge `bounds_uv_m` starts from the candidate footprint's bounding rectangle.

## Rendered Gaussian visibility and free space

`filter_gaussian_visibility` / standalone `gaussian_visibility` takes common
posed views, `gaussians:"source.ply"`, optional positive-similarity
`source_to_frame` (identity by default), and an explicit `filter`:

```json
{"filter":{"min_drawn_views":1,"max_free_space_fraction":0.7,"min_depth_views":5}}
```

For render visibility, a view supplies `particle_visibility:"values.npy"` and
`gaussian_source_sha256`. The array must contain one actual renderer value per
particle in that exact source order. `scripts/3dgrut_refinement_worker.py --mode
render` now exports these arrays when the renderer supplies `mog_visibility`.
Set `require_particle_visibility:true` in its render request to fail explicitly
when unsupported. Never substitute frustum flags for these values.

For free-space evidence, views supply camera-Z `depth` arrays and optional ignore
masks. Particles behind a surface are occluded, not free-space votes. Sufficient
views must see a particle in front of the surface before filtering. Optional
`depth_tolerances` are `near_ratio`, `far_ratio` and `absolute_m`. Gaussian-rendered
depth remains a model estimate; real depth can be supplied in the same frame.
Thresholds are explicit and require review.

Outputs include a candidate PLY, source hash, per-particle counts/reasons and keep
mask. Existing extent-aware spatial cropping, rotated exclusion boxes and native
ParticleField export remain separate and usable. Inspect holes and retained
surfaces, then prepare/export the chosen candidate and review the composed scene.

## Reference-conditioned restoration

The Gaussian refinement adapter now supplies actual nearest capture references
alongside proposed anchor views. Existing Fixer remains available. Optional
`scripts/difix_view_adapter.py` consumes them using local Difix weights; configure
it as the restoration runner using [the example](../examples/difix-view-runner.json).
No login or automatic weight download is performed. The reference choice uses
camera distance and direction as a heuristic, not a visibility certificate.

Both restoration choices retain original captures, depth-consistent anchor
propagation and the default geometry-frozen color update. Optional small joint
refinement remains available. Real-reference conditioning does not make generated
repair pixels observations. Choose/reject proposals from their returned evidence.

## Optional native SAM3 masks

Read [segment-reference-views](../skills/segment-reference-views/SKILL.md) when
pixel masks would help. The native command uses an operator-owned model environment
and local weights; it does not start a second coding-agent session. Prompted
instances, scores, masks, centroids and coverage stay available for agent choice.
Adjacent-frame IoU association is explicitly heuristic; it cannot establish
identity through occlusion or a camera jump.

Optional per-view `prompt_boxes` maps declared prompt names to `boxes` (1..32
displayed-image xyxy rectangles), matching integer `labels` (1 inclusion,
0 exclusion; defaults to inclusion), and optional boolean `box_only`.
For example, `{"prompt_boxes":{"carton":{"boxes":[[20,10,180,130]],"box_only":true}}}`
lets caller visual understanding specify an approximate target region when text
grounding fails. `box_only` supplies the model's visual prompt instead of the
identity string. Every proposed instance and spatial annotation stays retained;
the box does not clip the result or certify that a mask is correct.

For the shared `segment_reference_views` operation, set
`REAL2SIM_SEGMENTATION_RUNNER` in the operator's launch environment to
[an operator configuration](../examples/segmentation-runner.json). Its explicit
`read_paths` grant only installed model/runtime access inside the task boundary.
Both managed and external entries retain inputs, verify output-image hashes and
mask domains, and expose candidates. Task requests cannot select the host command.
The native command remains available without this wrapper. See
[official SAM3 usage](https://huggingface.co/docs/transformers/v5.20.1/en/model_doc/sam3)
and optional `requirements/segmentation.txt`; model weights are separately supplied.

Reference backend status checks this operator runner rather than importing a
`sam3` module in the coding agent's Python. `configured_not_executed` means the
configuration and executable were found; it does not verify weights, inference
or mask quality. A retained successful run is required for those claims. VGGT
status reports a missing checkpoint as a file prerequisite; it does not inspect
or infer the operator's account approval.
