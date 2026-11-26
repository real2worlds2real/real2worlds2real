# Separate camera motion from object motion

Use this guidance when a recording follows the target or changes viewpoint.
A stable object location in the image can conceal substantial world motion.

Estimate camera registration from static background features, excluding the
moving target, hands, reflections and other independently moving regions.
Review feature coverage, forward/backward tracking consistency and residuals
across the selected interval. A large inlier count in a small patch is not
sufficient evidence for the whole frame. Preserve failed or occluded intervals
as missing rather than silently bridging them with predicted target motion.

A homography can register a genuinely planar support surface. It does not
recover general camera translation or remove parallax for elevated object
features. For appreciable depth variation, use calibrated camera poses and
appropriate 3D landmarks. Keep camera conventions, image crops, intrinsics,
frame indices and timestamps consistent with the observation interface.

Recover target motion separately after registration. Match projected model
features to the same observed features: a visible contour centroid is not the
body origin, and a point above the support plane is not on that plane. For a
rotationally symmetric body, a persistent label or other asymmetric feature may
constrain heading; otherwise retain orientation ambiguity. Check both position
and rotation when the observed interaction makes them relevant.

If the asset already has a justified scale, reuse its geometry and scale
provenance. An assumed model dimension remains a conditional scale, not a new
measurement. Do not compensate registration or scale errors by fitting contact
coefficients. Inspect registered background features and projected target views
alongside motion residuals before deciding which hypothesis to revise.

Use the existing observation and projection contracts in
[Motion geometry and fitting](../../docs/MOTION_GEOMETRY_FITTING.md).
Tracking and camera estimation may be authored in the task workspace; this
reference does not add an automatic moving-camera reconstruction tool.
