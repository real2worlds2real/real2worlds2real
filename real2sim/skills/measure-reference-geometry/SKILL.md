---
name: measure-reference-geometry
description: Use known dimensions, ChArUco calibration or metric surface observations to constrain editable geometry and review residuals. Use when the input contains measurement evidence or precision is requested; ordinary uncalibrated photos can stay on the visual route.
---

Use one shared measurement interface; evidence determines the achievable scope. A known dimension anchors that dimension, a calibrated board anchors its own frame, and observed metric surfaces support local geometric checks. None certifies unseen geometry or real material physics.

- Read workspace `measurement_context.json` when present and the task's `measurements/inputs.json` to see supplied dimensions, camera/board data, calibration images and observed surface points. `project_access.json` locates the readable task root. Inputs remain host-retained and hash-checked by the measurement/calibration and acceptance code. For optional native analysis, the shared `measurement.inspect_inputs(store)` function also verifies retained hashes and summarizes the evidence level; no separate model query tool is required. Do not ask for evidence already supplied.
- A user may simply state “height 12 cm” in the objective. Before the first build, `declare_measurements` can retain an array encoded in `dimensions_json`: `[{"part":null,"axis":"Z","value_m":0.12,"tolerance_m":0.002,"source":"height 12 cm"}]`. `source` must quote the actual request exactly, and the value/unit must match it. A tolerance is a task setting, not a sensor accuracy estimate. Never promote a common-size guess to a measured constraint.
- When a known ChArUco board and same-optics calibration views were supplied, `calibrate_measurement_camera` detects corners and fits intrinsics/distortion. Review reprojection error, angular diversity, rejected views and estimated intrinsic uncertainty. Repeated near-identical views do not reliably calibrate a camera.
- `measure_reference_board` detects board corners in one original reference. With `asset_to_board:null`, it returns only board-frame camera pose and an annotated image. A board in the picture does not establish an arbitrary object's depth or position.
- Supply an `asset_to_board` rigid 4×4 transform only when its registration is justified by observations or declared setup. It maps asset coordinates in metres into the board frame. The tool then installs an aligned camera. `inspect_camera_alignment` compares against an undistorted reference when necessary. Do not compare a distortion-free Blender projection directly to distorted pixels.
- `review_measurements` reads the saved attempt's actual evaluated geometry. Dimension checks measure world-axis bounds for the whole asset (`part:null`), an exact mesh object name, or a `real2sim_measurement_part` tag. Declare the intended axes and object pose; a world-axis extent is not an orientation-independent width.
- Surface inputs are measured points in metres with an explicit rigid `measured_to_asset` registration. The tool reports exact closest-triangle distances for a retained, bounded sample, along with p95, maximum, RMSE and within-tolerance fraction. It performs no scale fitting or ICP. A partial cloud cannot verify completeness, topology or unobserved surfaces.
- Use residuals to revise the editable construction, then review again. Scaling must include functional geometry, joint anchors, collision proxies and deformation bindings; regenerate affected simulation evidence. The tool suggests scale ratios but does not silently rescale the asset.
- When dimension/surface constraints exist, acceptance requires a successful source-matched measurement review in addition to the normal simulator gates. With no such constraints, the existing acceptance path remains available. Cousins do not inherit the twin's exact size/surface targets automatically because the user may intentionally redesign them.

For video-derived observations, select sharp overlapping views of one consistent object state. A deforming/opening sequence is not automatically one rigid multiview reconstruction. VGGT depth is learned evidence; it becomes metric measurement only with a justified scale/frame and independent checks.

For weak phone calibration, optional constrained fitting, random-half sensitivity
and2x marker detection complement the default full camera model. For a whole map,
use optional `register_metric_reconstruction` with observed board triangulation or
independent3D landmarks; board pose alone does not register it. Read
[the numerical refinements](../../docs/RECONSTRUCTION_REFINEMENT.md) only when this
evidence is present. Keep assumed camera/shape priors distinct from measurements.
