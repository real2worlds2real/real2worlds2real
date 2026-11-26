---
name: recover-support-footprint
description: Recover observed desk, shelf or floor footprint from registered multi-view class masks while preserving occluded and unseen regions as uncertainty.
---

# Recover support footprint

`recover_support_footprint` combines registered support/occluder/background masks on a supplied plane. Use it optionally when clutter hides support boundaries and multi-view evidence can improve Astra's support geometry. It is shared by scene and object work; a simple support can still be authored directly.

Read [the request contract](../../docs/OBSERVATION_TOOLS.md#support-footprint-recovery). Supply a justified or explicitly estimated plane frame and grid bounds. Camera/image coordinates must match the undistorted observations. Label occluders separately from background; erasing a mug into background would incorrectly cut away the desk beneath it.

Call `recover_support_footprint(request_path="support.json")`. Inspect the grid, vote counts and preview: green is supported, red rejected, gray unobserved. Occluded/out-of-frame observations do not vote against support. Unobserved cells remain unknown rather than becoming fabricated evidence.

Optional common-frame `surface_points` support plane-height refinement and clustered footprint evidence. The height diagnostic reports competing layers and does not apply an ambiguous correction. Optional morphology and bounded extension under predominantly occluded cells produce a separate candidate footprint; yellow completion pixels are inferred, while raw mask votes remain available. Choose these aids when justified by the input instead of treating them as mandatory steps.

Use the recovered footprint as numerical guidance for editable mesh/support proxies. You choose any justified completion of unseen edges and revalidate placement and contacts in the target simulator. This tool does not estimate plane height, build arbitrary 3D surfaces or certify measured dimensions.

For a plausibly rectangular boundary missed by segmentation, optional
`edge_refinement` compares inner/outer samples using masks or RGB support colors.
Read [the edge contract](../../docs/RECONSTRUCTION_REFINEMENT.md#support-boundary-hypotheses).
Inspect offset sweeps, view votes and actual images; neighboring surfaces of the
same color can mislead it. The result is a separate rectangular hypothesis, so
raw unknown/observed cells and nonrectangular modeling remain available. Supplied
plane orientation is not an automatically estimated arbitrary plane.
