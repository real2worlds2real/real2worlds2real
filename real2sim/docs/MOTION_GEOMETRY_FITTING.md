# Optional motion geometry and initial estimates

These helpers connect measured motion to experiments. They neither choose an
experiment nor impose a motion taxonomy. The external agent may use existing
geometry, another estimator, or its own analysis instead.

## Registered pixels to a plane

`project_motion_to_plane` consumes an observations file produced by
`prepare_motion_observations`. It shares the existing OpenCV camera convention:
undistorted pinhole images, world-to-camera transforms, world translations in
metres. It does not run camera recovery or infer scale.

Example workspace request:

```json
{
  "schema_version": 1,
  "observations": "motion/observations.json",
  "camera": {
    "world_to_camera": [[1,0,0,0],[0,1,0,0],[0,0,1,2],[0,0,0,1]],
    "intrinsics": [[100,0,50],[0,100,50],[0,0,1]],
    "image_size": [100,100],
    "source": "synthetic example; replace with registered camera evidence"
  },
  "plane": {
    "origin_m": [0,0,0], "u": [1,0,0], "v": [0,1,0],
    "frame": "support-plane",
    "source": "synthetic plane; replace with measured support evidence",
    "point_on_plane_assumption": "tracked surface landmark lies on this plane"
  },
  "metric_evidence": {"source": "synthetic known dimensions", "metric_verified": true},
  "residual_scale_m": 0.005
}
```

For a moving camera, replace `camera` with `cameras_by_frame`, an object mapping
original decoded frame indices (string keys) to registered camera records. Each
track must retain matching `frame_indices` and `image_size`. Missing cameras,
unobserved samples, intersections behind the camera and near-parallel rays stay
missing. One fixed camera is rejected for unresolved moving-camera input. A
`registered` image track must actually be expressed in that camera's pixel domain;
labeling it registered does not compensate camera motion.

Plane axes must be orthonormal; resulting coordinates are `[u,v]` in metres.
The original observation hash, camera/plane request and metric-evidence statement
are retained. `metric_verified` is the caller's evidence status, not a certification
made by the projection tool. Scale uncertainty remains relevant even with a valid
projection. Match geometry, camera and metric registration in the same world frame.

Choose the plane for the **tracked observable**, not merely the nearest support.
A cup's visible silhouette centroid generally does not lie on the tabletop;
projecting it onto that plane introduces height and rotation bias. A persistent
landmark at a justified height can use a parallel plane. General 3D motion,
deformation, rolling and varying landmark height require a suitable different
measurement model. Predicted/interpolated samples are not promoted to measurements.

## Time windows and an optional fitting function

`motion_fitting.select_window(observations, start_s, end_s, track_ids=...)` selects
`[start_s,end_s)` without rebasing the clock, filling gaps or smoothing. Retain the
full observations so alternative windows and held-out checks remain possible.

`estimate_coulomb_slide` is an optional native Python wrapper for a caller-chosen
function with the `calibrated_scene_reconstruction.physics.video.fit.fit_slide(t, xy)` return convention:
`p = [x0,y0,heading,v0,deceleration]`, `cov` a 5-by-5 covariance, and optional
alternative-model diagnostics. An operator-installed implementation can be passed
directly; this repository does not vendor or require it, install it automatically,
or import arbitrary Python symbols from request JSON.

```python
from real2sim_harness.motion_fitting import estimate_coulomb_slide

# fitter is an explicitly imported callable chosen by the agent/operator.
report = estimate_coulomb_slide(
    observations, "landmark", (release_time, end_time), fitter,
    fitter_source="installed package, revision and function",
    assumptions={
        "horizontal_support": "measured plane/gravity relationship",
        "free_response": "reviewed post-release interval with no further contact",
        "translation_dominant": "landmark and orientation evidence",
        "physical_clock": "original capture timestamps/playback-rate evidence",
    },
    relative_scale_std=None, relative_clock_std=None,
    output="slide-initial-estimate.json",
)
```

At least eight observed metric 2D samples are required by this bridge. It estimates
effective kinetic friction `a/g` under the stated assumptions and retains initial
speed separately. It does not infer mass, static friction or restitution. An
effective contact coefficient need not equal one material's coefficient under a
simulator's friction-combination rule.

Abrupt-stop and piecewise-speed alternatives are retained when supplied by the
fitter. A better alternative generates a warning to inspect obstacles or external
intervention, not an automatic friction acceptance. Fit covariance alone excludes
camera/scale/model uncertainty. Optional relative scale and clock standard
deviations add first-order terms `mu*s_scale` and `2*mu*s_clock`; their quadrature
assumes independence and still excludes model mismatch. Unknown uncertainty stays
unknown. `stop_time`, if returned, is relative to the first fitted sample, while
`initial_state_estimate.time_s` remains on the original observation clock.

Use this as a hypothesis for an agent-designed target-backend experiment, then
inspect actual trajectories, residuals and video. The agent can propose parameters
without this helper or select another fitting/search method. Synthetic projection
and fit-interface tests establish numerical behavior, not real-video accuracy.
