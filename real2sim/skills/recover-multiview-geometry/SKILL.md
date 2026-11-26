---
name: recover-multiview-geometry
description: Recover and audit relative cameras plus sparse geometry from multiple consistent real views without treating the reconstruction as the final simulation asset.
---

# Recover multi-view geometry

Use this skill when at least two distinct references show one object in a consistent state with useful viewpoint overlap.

Inspect actual references with native media tools; managed sessions can also use `inspect_references`. Duplicate frames provide no geometry. Do not combine views in which an articulated or deformable object changed state unless the selected backend explicitly models dynamics.

Call `run_reference_geometry` with `sift` first for textured views with moderate overlap. For low texture, large viewpoint changes or a failed SIFT run, use `aliked` only when backend status reports COLMAP `learned_features: true`; that route combines learned ALIKED features with LightGlue. Native coding clients can inspect operator configuration or `real2sim_harness.reference.geometry_backend_status()` in the task's analysis environment; managed interfaces may expose the optional `inspect_references` helper. Do not require that helper in the external task interface. Interpret registered-image count, 3D point count, track length and reprojection error together. A low reprojection error on only two registered images is not evidence that every view is aligned.

When feature geometry is inadequate or a learned camera/depth cross-check would
resolve a consequential ambiguity, read `use-vggt-prior` and check the configured
backend. It covers missing prerequisites, visual evidence and the separate
alignment step needed before its arbitrary world frame can become Blender
reference cameras.

When a backend supplies a shared relative-camera rig in an arbitrary frame and
you have explicit model-to-image correspondences across the consistent views,
use `align_multiview_camera_rig` to jointly align the whole rig to that model
frame. Read [its contract](../../docs/OBSERVATION_TOOLS.md#shared-camera-rig-alignment).
It preserves relative rig geometry, returns candidate cameras and per-view
reprojection, and leaves existing camera solutions untouched. Assumed model
dimensions do not become independent metric evidence. Inspect disagreements
before adopting the candidate; one similarity cannot repair bad relative poses.
Use independent measured landmarks/boards with metric registration instead
when actual whole-map metric calibration is the requirement.

Treat the result as camera and coarse-shape evidence for Astra's editable Blender construction. Do not present a sparse point cloud as the final mesh, articulation or physics asset. Metric scale still requires a known dimension or other scale anchor.

For ordered or multiple videos, optional `recover_video_cameras` supplies per-stream
calibration/self-calibration, undistortion and sequential/cross-stream/loop matching.
For a known board or paired metric landmarks, `register_metric_reconstruction`
estimates a similarity for the complete reconstruction and returns registered
cameras. Read [the numerical contracts](../../docs/RECONSTRUCTION_REFINEMENT.md)
when these cases apply. A single board pose and a globally registered map are
different evidence. Neither low reprojection residual nor a board at one location
certifies distant geometry. Choose views, matching budget and registration scope
from the actual input; these operations are optional in both object and scene work.
