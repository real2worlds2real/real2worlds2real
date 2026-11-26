# Flexible surface diagnostics

These tools diagnose a candidate. They do not approve its source geometry,
semantics, physical response or task. Keep the original rigid accuracy limits,
all current source views and the actual native contact failures.

## Check source identities first

A material region and an object silhouette are different targets. For example,
a container query may include its contents in one view but exclude them in
another. Inspect original RGB and masks before changing geometry or assigning
material groups. Do not use rendered residuals to draw source boundaries.

[`part_label_polygons`](../../src/calibrated_scene_reconstruction/scene/objects/part_label_polygons.py)
accepts explicit source-pixel annotations when text queries cannot separate
visible materials. It reads no candidate mesh or fit score. The config-relative
JSON has `version: 1`, `object`, and a `frames` mapping. Each changed frame has
its actual `rgb_sha256` and one or more `polygons`, each containing `xy` pixel
vertices, `label` (0 or 1), and a source-based `basis`. Coordinates use the
original image, with x to the right and y down. Unlisted frames retain their
pixels; removing an entire previously nonempty source view is rejected.

Example command; replace the session, object and existing source paths:

```bash
PYTHONPATH=src envs/recon/bin/python -m calibrated_scene_reconstruction.scene.objects.part_label_polygons \
  sessions/session_name object_name \
  --proposal qa/source_parts/object_name/round_1 \
  --annotations agent/objects/object_name/source_polygons.json
```

The tool creates a new archived proposal and a separate component-review round
under `qa/source_parts/<object>/`. Source RGB, annotation, producer, original
query and parent proposal hashes are preserved. Known independent visibility
restrictions still apply. The returned `review.json` is pending. Review its
components against RGB and bind that review and manifest through the existing
`part_review` / `evidence_files` contract. Old labels and whole-object evidence
are retained. This process corrects source identity; it is not a shape waiver.

Rigid-only initialization retains independent source regions via
[`rigid_supervision`](../../src/calibrated_scene_reconstruction/scene/objects/blender/rigid_supervision.py).
Omitting a flexible link rebases global face offsets while preserving local
material/optical annotations. Rigid positive labels take precedence over a
conflicting flexible occlusion label. Final full-object source checks remain
required, including views unavailable for the initialization proposal.

## Check rigid structure before adding flexible details

A shared rigid owner does not prove that its visible constituents have a valid
arrangement. Distinct stems, bars or enclosure walls can cross even when every
individual mesh is watertight and the aggregate inertia is finite. Preserve
source identities and check the actual solid relationships before fitting their
attached sheets. Declare intentional joints, fused seams and buried supports
explicitly; do not apply a blanket no-overlap rule to legitimate assemblies.
A swept-radius distance can guide a rod proposal, but actual triangle/contact
checks and complete source validation remain necessary. Source-fit improvements
that introduce invalid intersections are rejected, not promoted.

## Check mass and contact sampling before fitting material stiffness

Native uniform node masses can overweight densely sampled tips, folds or small
pieces. Audit actual material area with
[`surface_mass.distribution`](../../src/calibrated_scene_reconstruction/assets/surface_mass.py); this
checks disconnected component masses and reports the area/node centroid error.
It does not certify within-sheet mass distribution or material behavior. The
Blender builder currently enforces its component check for surface FEM. Particle
candidates need an explicit corresponding audit too; particle authoring alone
does not establish area-consistent masses.

For a surface with known length/width parameterization,
[`strip_sampling.resample`](../../src/calibrated_scene_reconstruction/assets/strip_sampling.py) takes a
finite `(length_rows, width_columns, 3)` source grid and physical spacing. The
grid uses a-b-c / a-c-d triangle splits. Cross-section node counts follow physical
width. `target_vertices=round(source_area / spacing**2)` can allocate counts
across homogeneous disconnected strips. This is a geometry utility, not an
automatic material or mass approval.
Include any subsequently added basal or seam material and its nodes in that
allocation. A fixed number of extra nodes per piece can disproportionately
overweight the smallest pieces; inspect the completed mesh after additions.

The report records every actual root-edge node, sampled bidirectional geometric
deviation, area and node counts. Bind the whole declared root edge, then check
the actual export. The geometric deviation is sampled, not a Hausdorff proof.
Rebuild in Blender, check material/source geometry, audit actual mass, and run
native release. Do not add fixed constraints to free material nodes.

Resampling must preserve visual material identities too. An all-vertices-inside
rule can erase a thin stripe after triangle spacing changes. For an explicit
rectangular material chart, `material_region_transfer.from_grid` maps target
face centres back to the source triangles and reports actual area shares and
entirely lost regions. This does not change the physical mesh or certify a
coarse unresolved stripe; inspect coverage, exported USD subsets and the native
render. Generic remeshing without that chart needs an equivalent identity map.

Check whether the observed connection is a point, edge hinge or bonded area.
For a declared clamp, retain the existing `surface_bindings[].clamp` physical
footprint check. Width and area bounds apply independently to each connected
material sheet; another sheet's large patch or distant anchors cannot hide an
unbound, edge-only or undersized patch.
[`attachment_conditioning.audit`](../../src/calibrated_scene_reconstruction/assets/attachment_conditioning.py)
additionally examines the actual bound nodes of each connected sheet. Its
rotational sensitivity reports the RMS anchor displacement caused by a small
rigid rotation, including weak nearly collinear attachment directions. Spread
between different sheets cannot conceal a weak individual attachment. This is
diagnostic, not a new threshold or proof of native clamping. A finite base must
be supported by connection evidence and actual material; extra free-sheet pins
are not a repair. Native comparison can still reject a geometrically sound
attachment hypothesis.

[`cloth_sampling`](../../src/calibrated_scene_reconstruction/assets/cloth_sampling.py) measures holes in
particle contact coverage. The existing bounded secondary-surface allowance
is not a guarantee of face contact: separate sheets may still cross between
particles. Preserve the actual native intersection and physical-thickness
checks even when the sampling diagnostic passes.

## Compare the implemented response

`cloth_rest.surface_chart` supplies intrinsic triangle geometry and natural
curvature for a shell hypothesis. Its spanning-tree stretch/shear categories
are topology bookkeeping, not a calibrated particle material model. If particle
contact or bending cannot represent the part, compare an explicitly declared
native surface-FEM candidate using the same source geometry and mass. Supply
continuum E, Poisson ratio, thickness, flexural rigidity and damping priors;
never reinterpret raw particle spring values as continuum constants.

Use the existing `apps/scene/check_surface_response.py` and
`apps/scene/check_surface_flexure.py` for independent material controls.
Authored stiffness and implemented response can differ with mesh spacing,
timestep and solver iterations. Keep failed controls and effective responses.
A static view does not identify a material modulus; retain uncertain material
class branches, including natural and manufactured alternatives where needed.

`apps/scene/check_candidate_passive.py --assembly <object> --surface-link <link>`
records actual source candidate release; the default surface link remains
`wipe`. Compare the same geometry/material at finer clocks before attributing
a failure solely to softness. A `--no-gravity` counterfactual helps distinguish
gravity-driven motion from preload or initialization, but cannot approve an
unforced real-gravity state. None of these controls approves the source asset
or replaces its task test.
Apply the declared flexible-response policy when reviewing those records.
For explicitly secondary, qualitative details, retain raw small contact and
local-response failures and inspect their actual visual/physical significance
as described in [object review](object_review.md). They are not additional
universal quantitative gates. Initial interpenetration, collapse and excessive
visible sag remain modeling problems, and rigid precision stays unchanged.

## Declare the passive reference frame

[`passive_response.evaluate`](../../src/calibrated_scene_reconstruction/tasks/mechanisms/passive_response.py)
keeps existing metre-valued `rigid_drift_m` and `surface_drift_m`. It also
implements `normalized_unforced_drift` with unit `object_extent`:

- Rigid parts explicitly declare `reference_frame: "world"`. Motion includes
  every rigid surface point and body rotation; a self-reference is rejected.
- Flexible parts explicitly declare `reference_link` naming an actual rigid
  part. Motion is measured for the same indexed native nodes in that frame.
- [`passive_extent`](../../src/calibrated_scene_reconstruction/tasks/mechanisms/passive_extent.py) uses the
  maximum world-axis span of all actual initial object surface points. The
  denominator remains fixed throughout the trace. Missing parts, invalid native
  samples and a zero/nonfinite extent are errors.

The original declared bounds apply to the resulting ratio. Unsupported or
unframed metrics remain failures. Returning to the initial pose at the end does
not erase an intermediate excursion. A rigid support can pass while its leaves
or fabric fail; report both independently.

When a rigid-only proposal scores better than the complete object, compare its
actual world-space surfaces and source visibility domain before changing the
rigid shape again. Wrong flexible geometry can hide the correct rigid region.
Source-occluder masks used to initialize the rigid part are not permission to
remove those surfaces or carry the proposal score into full-object approval.
Retain the complete object's score and separate any diagnostic removal of
predicted occluders from an actual reconstruction candidate.

A small absolute root displacement can still allow a large rotation when the
bonded patch is small. Inspect actual bound-node motion in the owner frame and
the spatial distribution of motion across each connected sheet. A nearly rigid
blade rotating around its base calls for checking source-supported base shape,
connection conditioning and native attachment behavior before increasing the
whole blade's stiffness. Short gravity transients alone do not identify a
material's equilibrium response. The existing physical/semantic policy and
current source checks remain required; these diagnostics do not grant approval.
