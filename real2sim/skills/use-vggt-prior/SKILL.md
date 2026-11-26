---
name: use-vggt-prior
description: Inspect VGGT-Omega camera/depth evidence, align a shared rig or one reference camera to a dimensioned draft, and retain model/scale assumptions.
---

# Use a VGGT-Omega prior

Use this skill when shape or viewpoint ambiguity remains after visual inspection, especially for low-texture references where feature matching is weak.

Inspect the actual references with the client's native media tools; managed
interfaces may also expose the optional `inspect_references` helper, which is
absent from the external task schema. Call `run_reference_prior` at resolution
512; it reports missing checkpoint/backend prerequisites instead of providing a
prior when unavailable. Do not present that status as completed inference.
Each returned panel contains the exact model input, relative depth and confidence
in that order. Use depth discontinuities and consistent structure as evidence;
discount reflective, transparent, textureless or low-confidence regions. The
prediction is not a final mesh, metric scale or physics asset.

Before a dimensioned Blender draft exists, use the panels to guide construction.
Never copy VGGT-Omega extrinsics directly into Blender because the learned world
frame and scale are arbitrary.

For a static multiview capture with a shared rig, prefer
`align_multiview_camera_rig` when explicit model-to-image correspondences are
available across views. It maps the whole rig to the model frame with one
positive similarity, retaining relative poses. Read
[the request contract](../../docs/OBSERVATION_TOOLS.md#shared-camera-rig-alignment)
for original/processed pixel mapping, arbitrary-frame extrinsics, candidate
cameras and per-view residuals. Use supplied intrinsics when available; otherwise
map the learned intrinsics to the same pixel domain as the observations. Inspect
the output and independent evidence before adopting the candidate. It does not
refine an incorrect relative rig or certify assumed dimensions.

For a single view or independently posed product photographs,
`align_reference_prior` remains a per-image option. It uses the learned
intrinsics in original pixels and explicit model-point/pixel PnP, not the whole
relative rig. Prefer rigid, unambiguous landmarks at varied depths. Hidden shape
and assumed dimensions remain uncertain; do not use a changing part or specular
highlight as a fixed landmark. Set tolerances from image/annotation uncertainty,
not by loosening them until a fit passes.

After selecting cameras, render again and follow
[camera review](../align-reference-camera/SKILL.md), including current-source
boundary review for a scene. Preserve failed/disagreeing candidates. Independent
PnP may remain useful after a failed joint alignment, but does not close the
shared-camera uncertainty merely because it fits its own annotations well.
