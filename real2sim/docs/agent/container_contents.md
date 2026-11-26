# Containers and loose contents

A container can match its own silhouette while its modeled rim hides contents
that are visible in the source. Check the actual assembled projection before
changing the contents, their material, or source labels. Compare the missing
pixels with independent source RGB and identify the first actual occluding
triangle. A dark visible surface is not an annotation error merely because the
current model misses it.

`scene.objects.blender.context_feedback.score` evaluates an opaque parent and
its declared contents using their actual candidate meshes and poses. It keeps
all source views, masks and original per-object thresholds. Each child must
already declare the parent through `source_support_occlusion` and
`support_seed`. Transparent or refractive containers need optical evidence;
this opaque-depth helper deliberately rejects them.

```python
from calibrated_scene_reconstruction.scene.objects.blender.context_feedback import score

review = score(session, "container", container_candidate,
               {"contents": contents_candidate})
# Each candidate: {"mesh": actual_npz_path, "pose": actual_pose}
```

The result binds actual files and poses and reports all original numeric
metrics. It remains `accepted: false`: it supplies proposal feedback, not an
automatic coupled optimizer, semantic approval or native physics evidence.
Revise explicit shape hypotheses, then rebuild and repeat both source checks.
Do not edit a previously completed round or silently replace its context mesh.

For rigid assemblies on a sloped support or inside a vessel, opt into actual
triangle placement in the object's `fit` configuration:

```json
{
  "support_seed": {
    "object": "container", "round": 2, "surface": "interior_floor"
  },
  "support_placement": {
    "mode": "actual_triangles", "sample_spacing_m": 0.002
  }
}
```

Omit `support_seed` to use the verified modeled task support. `uppermost` and
`interior_floor` select actual upward intersections; holes remain empty. The
initializer samples actual downward object triangles and recomputes vertical
placement after shape or planar-pose changes. Parent meshes, poses and the
metric frame are bound to the round. This option currently requires explicit
rigid links; the existing plane and flexible initialization routes remain
available. The common vertical placement does not independently arrange loose
parts or certify support, stability or absence of penetration.

An audited `source_pose_seed` can preserve a source-derived planar initializer;
it transfers only XY and yaw. Support height is rebuilt. Declare editable model
data under `fit.assets` so each round snapshots the actual geometry seed, UVs
and construction residuals. Retain meaningful shape parameters and their
provenance, and rerun checks after editing them.

For physical validation, check actual material solids rather than filled
bounding boxes. Release all parts independently in the intended Isaac profile,
without runtime pins or pose corrections. Measure carried surface displacement
and solid penetration, then reproject the unforced final state against the same
source views. Native stability alone cannot approve a changed arrangement.

For declared low-load rigid contact checks, `tasks.rigid.contact_probe` is an
optional diagnostic fixture. It records every body pose, actual tool motion,
contact force, penetration and native mass. Its kinematic mode uses native
kinematic targets: `set_world_pose` teleports and is not an equivalent contact
driver. A sphere pressed between rigid supports can still generate large loads
even at low commanded speed. Retain such failures; do not raise the load guard
or change the object's material to make the fixture pass.

The optional `actuator.mode: bounded_force` uses a dynamic spherical tool with
explicit mass, spring stiffness, damping and force limit. Only the diagnostic
tool receives gravity compensation; target gravity and material stay unchanged.
This remains experimental until a matched native trial demonstrates finite
contact, bounded motion, actual withdrawal and unchanged target source surfaces.
`tasks.rigid.intrinsic_response` computes the existing normalized drift,
contact-depth and rigid-deformation metrics from those independent records.
Zero deformation verifies the rigid representation under that load; it does
not identify the real object's stiffness. Keep actual section geometry and mass
priors bound to each report, and require the original semantic and source checks.

Keep the tested numerical profile explicit. A diagnostic at another solver,
timestep, contact offset or SDF resolution does not qualify the final scene.
Reconcile the actual export settings and repeat the relevant native checks.
Likewise, a Blender texture node alone does not establish Isaac appearance.
Use the normal texture export route, bind its atlas to the exact physical mesh,
preserve each link's face-UV range, and inspect the reopened native scene.
For an existing producer-authored atlas, `blender.authored_look.create` can
transfer the immutable round's explicitly declared image and corner UVs without
resampling. It verifies the round and asset hashes and orders UVs by the actual
physical parts, rejecting omitted or duplicate faces. This optional single-atlas
path leaves the default multiview baker unchanged. Its result still requires
native appearance review; an atlas transfer report is not object acceptance.

Source labels also need independent review. A shadowed fruit edge or small stem
can be absent from the foreground occluder labels while the parent plate mask
correctly omits it. Treating that pixel as visible background would ask the
plate model to grow a hole. Inspect the original RGB before revising a label;
never copy a candidate silhouette or residual into the ground truth. Source-only
colour or topology proposals are still proposals: review each changed region,
retain the old labels and original numeric results, and preserve observed parent
positives and real exterior background. A reviewed visibility revision requires
new measured rounds and new source fingerprints.

Renderer changes also invalidate previous source approval. Exact integer-centre
triangle coverage can differ from positive antialias coverage at thin boundaries.
Keep both historical measurements, rerun every original view with the current
source raster contract, and require the native released state to pass it too.
