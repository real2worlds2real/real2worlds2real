---
name: review-silhouette
description: Diagnose shape, pose, or framing mismatch using calibrated multi-view masks and silhouette metrics after a Blender attempt.
---

# Review silhouette evidence

Use this skill when `run_blender` reports silhouette evidence and a mismatch may come from outer shape, pose or camera framing. Do not use canonical-view IoU to judge an uncalibrated real photograph.

Call `inspect_silhouette` for the relevant attempt. Compare views rather than optimizing only the mean:

- IoU measures overlap;
- mask area reveals scale or depth errors;
- bounding box and centroid reveal framing, orientation or joint-pose errors.

Change geometry only when the disagreement is consistent with the RGB references. A high silhouette score does not validate internal cavities, part hierarchy, materials or physical behavior. Preserve improvements that pass other gates.

When registered masks exist and visible shape is disputed, inspect their
per-view overlays and local residuals rather than relying only on the mean IoU.
Review identity-bearing features inside the outline in aligned RGB crops too:
surface slope breaks, seams and shading normals can be wrong with nearly
unchanged masks. For those discrepancies use
[improve-visual-realism](../improve-visual-realism/SKILL.md) and local diagnostic
renders; further optimization of outer-mask IoU may not address their cause.

For camera disagreement, consult [align-reference-camera](../align-reference-camera/SKILL.md) or [measure-reference-geometry](../measure-reference-geometry/SKILL.md) when a known board is available. If registered masks support a bounded shape/pose update, [fit-multiview-geometry](../fit-multiview-geometry/SKILL.md) supplies optional refinement. Once shape is adequate, return to the object's appearance and functional review; [improve-visual-realism](../improve-visual-realism/SKILL.md) addresses material and lighting differences that silhouettes cannot measure.
