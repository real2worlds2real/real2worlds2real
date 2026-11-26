# Optional metric measurements

The visual, scale-constrained and geometry-observed levels share tools. They describe available evidence, not different generation pipelines or automatically certified accuracy. The default image/video route remains available without metric input.

The optional [posed-observation tools](OBSERVATION_TOOLS.md) add bounded fitting of agent-authored meshes, visibility-aware photo texture baking and occlusion-aware support footprints. They share the available camera/frame evidence across objects and scenes. They do not replace the measurement levels or promote estimated scale into verified measurements.

## What the tools provide

| Evidence | Shared tool | What it establishes |
| --- | --- | --- |
| Dimensions explicitly stated in a user prompt | `declare_measurements` | Retains exact quoted numeric dimensions and task tolerances before modeling |
| Known ChArUco board and diverse calibration photos | `calibrate_measurement_camera` | Camera intrinsics, distortion, reprojection residuals, uncertainty and view-diversity checks |
| Board visible in an object reference | `measure_reference_board` | Detected corner observations and camera pose in the metric board frame |
| Saved Blender asset plus dimensions / observed surface points | `review_measurements` | Actual evaluated geometry bounds and exact point-to-triangle residuals |
| Any task | Read `measurement_context.json` / task `measurements/inputs.json` | Which evidence was supplied, its provenance and its limitations; native source/task access supplies this directly |

The tools complement existing PnP, COLMAP, VGGT, local surface inspection and video reading. They do not require replacing the code-generated Blender representation with a reconstruction model.

## Natural-language dimensions

Users can simply state a dimension, for example `height 12 cm`, in their objective. Astra can retain it through `declare_measurements` before the first build. The host checks the quoted source against the original request and checks numeric units. An estimated common object size cannot be promoted to measured evidence. A requested tolerance describes acceptance of this task; it is not an estimate of sensor accuracy.

## Optional measurement input file

For camera/board data, multiple measured dimensions, or point observations, supply a local JSON file:

```bash
real2sim-harness generate --reference /path/to/object.jpg --measurements /path/to/measurements.json \
  --blender /path/to/blender --auth api-key --model gpt-6-astra
```

Example schema (include only the evidence you actually have):

```json
{
  "schema_version": 1,
  "dimensions": [
    {"part": null, "axis": "Z", "value_m": 0.12, "tolerance_m": 0.002, "source": "user-provided height"}
  ],
  "board": {
    "type": "charuco", "squares_x": 7, "squares_y": 5,
    "square_length_m": 0.03, "marker_length_m": 0.022,
    "dictionary": "DICT_4X4_50"
  },
  "calibration_images": ["calibration-01.jpg", "calibration-02.jpg", "calibration-03.jpg", "calibration-04.jpg", "calibration-05.jpg"]
}
```

File paths resolve relative to this JSON. The input file and referenced data are retained in the ignored run directory with hashes. Changes invalidate evidence rather than silently changing an accepted task. Values here illustrate the format; they are not parameters to assume for an unknown board.

An already calibrated camera can replace the calibration photos:

```json
{
  "image_size": [1280, 960],
  "intrinsics": {"fx": 1000, "fy": 1040, "cx": 640, "cy": 480, "distortion": []},
  "source": "calibration for these optics and displayed resolution"
}
```

Put this under `camera` in the measurement file. Resolution must match the displayed, EXIF-corrected reference. Intrinsics are in pixels; distortion follows OpenCV. Changed optics, image crops or resizing require the appropriate recalibration or explicit coordinate transformation. Known physical board dimensions must match the actual print, not just the intended print setting.

## Board frame and object frame

Board pose alone does not establish an arbitrary object's depth, size or location. With `asset_to_board:null`, board measurements do not install an asset reference camera. An explicit, justified rigid 4×4 transform can map asset metres into the board frame; only then is a corresponding camera installed for Blender review. Do not assume the object lies on the board plane.

The detector returns annotated actual corners. For distorted input, comparison images are undistorted with the same intrinsics before comparing to Blender. Focal-length asymmetry is represented through pixel aspect; raw distorted pixels are not treated as an ideal pinhole projection.

Calibration rejects insufficient distinct views, excessive reprojection residuals and limited orientation diversity. These are conservative checks, not a complete condition-number or observability certificate. The default calibration tool uses at least five usable views and an 8-degree board-normal span. Low residuals alone do not establish a trustworthy calibration. `calibration.py` exposes diagnostics and a standalone Python API for other controlled calibration protocols.

## Observed metric geometry

Add `surface` to the same file:

```json
{
  "path": "observations.npz",
  "source": "metric depth observations; specify acquisition and scale basis",
  "measured_to_asset": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
  "tolerance_m": 0.002,
  "max_points": 100000
}
```

Use `.npz` with a finite `points_m` array of shape N×3, or `.npy` with that array directly. Coordinates must already be metric. The explicit rigid transform registers the observations into the asset's construction frame. Identity is valid only when those frames are known to coincide. Scale fitting, reflections, shear and ICP are not silently performed; this prevents registration from hiding a dimension error.

The review opens the current `.blend` in Blender, evaluates modifiers, measures visible asset geometry and excludes fixture/ground/simulation-proxy meshes. Dimension targets use world-axis extents: `part:null` means the whole asset; otherwise select an exact object name or `real2sim_measurement_part` tag. The source pose and chosen axes matter.

Observed points are compared to the exact evaluated triangles using a BVH. The report includes p95, maximum, RMSE and within-tolerance fraction, along with a colored residual panel and retained numeric residuals. A deterministic sample bounds cost; sample counts are disclosed. The default declared surface gate checks p95 against the supplied tolerance, so inspect the maximum and outlier fraction as well. Partial observations do not check hidden geometry, completeness or topology. Learned relative-depth predictions need an independently justified scale/frame before they can be used as metric evidence.

## Review and acceptance

Declared dimension/surface targets require a successful `review_measurements` in addition to existing visual and simulator checks. The gate checks the current source hash, immutable input hash and retained report hash. Geometry changes require remeasurement and new simulator evidence. Board-only or camera-only input does not invent a measurable object dimension or force a surface check.

The review measures and suggests useful scale ratios; it never silently rewrites geometry. Astra can revise the editable construction, including joint anchors, colliders and surface bindings, then retest. Physics parameters remain a separate downstream calibration concern. Cousins do not automatically inherit the twin's exact measurement targets because their intended size and shape may change.

See [OpenCV ChArUco calibration](https://docs.opencv.org/4.13.0/da/d13/tutorial_aruco_calibration.html) and [board detection](https://docs.opencv.org/4.13.0/df/d4a/tutorial_charuco_detection.html). The implementation reuses the installed OpenCV package; it adds no model weights or account dependency.
