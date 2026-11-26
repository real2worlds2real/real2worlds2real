# Depth of an observed thin surface

**Module id:** `observed-surface-depth`

**Scope:** Thin or partially occluded material with calibrated views and sparse depth or stereo matches. For a bonded leaf or petal, also use [flexible surfaces](flexible-bonded-surfaces.md).


## Observations

Keep the original pixels, masks, camera parameters and match ownership. Inspect both ends of a correspondence. A foreground mask can contain a match to the background. Label a stripe or folded margin as visible material only. Retain complete outlines separately and review ambiguous regions across views.

## Representation and parameters

For a locally single-valued surface, fit a robust local affine depth to positive metric samples. Record the image-space support radius, depth units, retained sample range and rejected points. Bound a proposed depth to the local retained range. Mark queries outside the support radius as inferred even when a finite value can be produced.

## Procedure

1. Check camera registration and correspondences on textured foreground and background features. Keep reverse-match, reprojection and positive-depth diagnostics where available.
2. Compare the old and bounded proposals on the same samples. Retain measured coverage, unsupported continuations and failed views. Avoid forcing two layers of a folded surface into one depth chart.
3. Reproject the actual candidate triangles into calibrated views held back from the fit. Inspect tips, occlusion order, source silhouettes and the reconstructed area. Review both missing and excess material where a complete outline is visible.
4. For a hanging sheet over a rim, inspect the recovered crease, continuous hidden tail, shell clearance and lid sweep. If manipulation is requested, run its material and contact checks on the assembled asset.

## Checks and failure diagnosis

Include a sloped strip, an isolated background match and an unsupported tip. Local fits can diverge at the tip while matching the input silhouette. A majority of background samples can also produce a smooth, positive surface at the wrong depth. Use another camera view to distinguish those cases.

## Tools

- [multiview geometry](../../../recover-multiview-geometry/SKILL.md)
- [silhouette review](../../../review-silhouette/SKILL.md)

## Reuse limits

A retained sample range describes observed values. It supplies neither a statistical confidence interval nor evidence for an unseen layer. Assess changes in smoothing, geometry and contact on the current asset.
