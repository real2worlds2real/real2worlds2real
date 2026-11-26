---
name: align-reference-camera
description: Recover and review a real reference photo's camera from a dimensioned Blender draft and explicit 3D-to-pixel correspondences, retaining geometry and scale assumptions.
---

# Align a reference camera

Use this skill when canonical renders no longer suffice to compare a Blender draft with a real photograph.

For a reference-based twin, establish useful corresponding reference/render views
before claiming that visible proportions and features match. Reuse supplied
calibration or a justified existing camera rather than solving it again. If
alignment is missing and viewpoint uncertainty obstructs that review, perform
camera alignment; do not silently use a convenient canonical view instead.
Several product photographs may have independent cameras, backgrounds and
object poses. Align useful views individually when joint reconstruction is not
supported; multiple photographs alone do not justify an SfM reconstruction.

Inspect image orientation, dimensions and available camera metadata before
assuming focal length. Use native image/file inspection in an external coding
session; `inspect_references` is an optional helper in interfaces that expose it,
and is omitted from the external task schema. Prefer an EXIF focal prior when
present. Without one, treat the focal estimate as uncertain and use a calibrated
or learned-intrinsics backend when available.

With known or explicitly assumed draft dimensions, choose visible model landmarks and their image pixels, then call `solve_reference_camera`. Use at least six well-spread points when possible. Include points across multiple depths; an entirely coplanar set is fragile. Coordinates must refer to the model's actual world frame and the displayed image orientation. Assumed dimensions remain assumptions; fitting their projections does not measure them independently.

Reject or revise solutions with large reprojection error, implausible camera placement, or correspondences that only cover a small image region. A single image does not establish absolute scale: retain a user-provided dimension or other metric anchor.

For a static multiview scene, check the shared camera relationships against
independent features or useful multiview/depth evidence, using
[recover-multiview-geometry](../recover-multiview-geometry/SKILL.md) when needed.
A small landmark residual on one assumed shape cannot verify the camera rig or
the other participants. An anchor object's axes define a convenient model frame;
they do not establish gravity or a supporting plane's orientation. Keep those
relationships uncertain until observed constraints support them.

Run Blender again after solving. Compare the photograph, corresponding RGB
render and visible silhouette. In an object interface exposing
`inspect_camera_alignment`, use it for the retained attempt. For a saved scene,
inspect the RGB natively and use the current-source `evaluate_only` review in
[fit-multiview-geometry](../fit-multiview-geometry/SKILL.md) with registered masks
when boundary evidence is needed. Diagnose shared framing/perspective errors
before changing individual shapes; localized disagreement can also arise from
mask identity or occlusion and does not by itself prove a geometry error.

Inspect recognizable interior features as well as the silhouette. Camera
alignment cannot certify that creases, seams, surface normals or materials are
correct; route localized surface disagreement to
[improve-visual-realism](../improve-visual-realism/SKILL.md). If the observations
cannot support a stable camera, retain that limitation and use the strongest
justified qualitative comparison without presenting it as calibrated scoring.
