---
name: generate-editable-object
description: Task-level workflow for generating one editable operable object, selecting physical representation and specialist skills, and iterating through source-matched simulation acceptance.
---

# Object generation

For a shared reconstruction plan use [the object workflow](../reconstruct-object/SKILL.md). This skill supplies its native procedures.

Understand the input images/video and user request, including optional dimensions
and later cousin requests. If this is a scene participant, retain the relevant
original scene constraints and identity while focusing on the assigned object.
Declare an inferred asset contract when requested by the task. Keep observed
features separate from hidden-side assumptions. Optional camera, metric and
geometry aids serve an identified uncertainty; they are not mandatory for every
object. Original videos remain available beyond initial reference frames.

When continuing an existing draft, inspect its saved contract and actual
mechanism before selecting a representation. Independently moving parts may
be contact-driven rather than joint-driven; motion alone does not establish an
articulated contract. Retain conflicts between the draft and task requirements
for correction rather than treating inherited metadata as reference evidence.

For a reference-based twin, visual fidelity is part of completion unless the
user explicitly narrows the task. Before modeling, retain a compact set of
identity-bearing visible features with reference-image or crop pointers: major
proportions, changes in surface slope, openings, seams and distinctive finishes
as supported by the observations. Keep these features available during review.
Compare the supplied observations with corresponding rendered views, including
complementary views when multiple useful photographs are supplied. Canonical
hero renders alone do not establish reference matching. When pose or framing
prevents a meaningful comparison, use
[align-reference-camera](../align-reference-camera/SKILL.md) or existing justified
camera evidence before attributing the difference to shape. When object
boundaries are uncertain, use
[segment-reference-views](../segment-reference-views/SKILL.md) or a suitable
task-local mask method. Registered shape disagreement should trigger
[review-silhouette](../review-silhouette/SKILL.md) and, when the observations and
parameterization support it,
[fit-multiview-geometry](../fit-multiview-geometry/SKILL.md).
One useful reference image still supports visible-projection correction with
that fitter; retain fixed-camera assumptions and do not claim hidden shape or
metric reconstruction from a successful single-view fit.
For a precision scene participant, coordinate the parent scene's current-source
boundary baseline with its actual placement and camera frame; local physics and
canonical presentation renders cannot discharge that scene comparison.
Use reference-supported geometric priors when choosing the editable model:
rotational or mirror symmetry, straight surfaces and shared section dimensions
should constrain candidates where applicable. Fit within that model rather than
deforming visible sides independently to match a photograph. Preserve supported
asymmetric parts separately and investigate camera/mask errors when they conflict
with a strong prior. An improved image score cannot override those requirements.

Read
[improve-visual-realism](../improve-visual-realism/SKILL.md) when developing and
reviewing the twin's appearance, including local shape and normal diagnostics
when a visible feature is missing despite a plausible outer silhouette.
Investigate material, lighting and color
management when highlights, shadows or saturation differ; a plausible material
label or a passing geometry/physics check does not establish visual matching.
Use [bake-reference-texture](../bake-reference-texture/SKILL.md) when observed
surface detail warrants it. Choose methods and iteration depth from visible
differences and the user's objective; retain uncertainty where the observations
cannot separate material from illumination. Check appearance in the target
renderer before acceptance. Review the retained visible features explicitly;
repair consequential observable mismatches before claiming visual completion.
If a required comparison or repair cannot be completed, report that criterion
as unmet rather than substituting a physics pass or overall resemblance.
For clipped or occluded inputs, compare the corresponding visible regions;
hidden geometry uncertainty does not excuse an unreviewed visible outline.
After a local material/normal repair, return to the reference comparison and
check which independent shape or finish findings remain.
Record consequential remaining differences and their effect on the requested
delivery. Tool choice remains evidence-dependent; invoking every listed tool is
not a completion requirement.

Use editable Blender construction with clear semantic parts and materials.
In a native coding session write `build_scene()` in a workspace Python file and
pass it to `run_blender`; a limited direct API transport instead submits the
source through its build tool. Use metres/Z-up and the task's semantic front.
Set `real2sim_asset_kind`, `real2sim_front_axis`, `real2sim_up_axis`, `real2sim_units`; required
part terms need clear object names. Use the selected representation's metadata
and runtime contract rather than mimicking behavior with an unrelated model.

After the representation is known, read the applicable specialist guidance:

| Chosen representation or need | Skills to consider |
| --- | --- |
| Rigid or articulated object | `review-simulation`; `review-articulation` for joints and limits |
| Cloth / sheet | `review-deformation` or source-bound `simulate-deformable-behavior`; `refine-deformable-surface` for detailed appearance |
| Elastic solid | `simulate-volumetric-solids` |
| Rope / cable | `simulate-slender-bodies` |
| Rigid mechanism driven by mating contacts | `simulate-mating-rigid-parts` |
| Articulation coupled to flexible surfaces | `simulate-articulated-cloth` |
| Particle liquid / material | `simulate-particle-materials` |
| Granular solid / sand | [simulate-granular-materials](../simulate-granular-materials/SKILL.md) |
| Prefractured impact response | `simulate-impact-fracture` |
| Supplied dimensions, a calibration board or metric surfaces | [measure-reference-geometry](../measure-reference-geometry/SKILL.md) |
| Draft-to-photo camera mismatch | [align-reference-camera](../align-reference-camera/SKILL.md) |
| Consistent overlapping views without registered cameras | [recover-multiview-geometry](../recover-multiview-geometry/SKILL.md); [use-vggt-prior](../use-vggt-prior/SKILL.md) when learned evidence is useful and available |
| Instance boundaries needed for comparison or fitting | [segment-reference-views](../segment-reference-views/SKILL.md) |
| Registered silhouette disagreement | [review-silhouette](../review-silhouette/SKILL.md) to diagnose; [fit-multiview-geometry](../fit-multiview-geometry/SKILL.md) for optional bounded refinement |
| Reference-based twin appearance | `improve-visual-realism`; `bake-reference-texture` when observed surface detail warrants it |
| Transient behavior uncertainty | `inspect-video`, `review-dynamic-contact` as useful |
| Real observed motion must be physically matched | [calibrate-observed-physics](../calibrate-observed-physics/SKILL.md); ordinary operability does not require calibration |

Inspect actual construction and rendered feedback and revise the editable source.
Intermediate simulator checks and movie rendering are chosen when useful, rather
than required on every draft. Fluids and contact-driven deformation may need
earlier runtime feedback because their simulated states determine appearance.

Before delivery, complete the selected representation's **source-matched target
simulation and visual/functional acceptance**, plus any required metric or local
surface review. A successful Blender build or numerical-only intermediate check
does not satisfy final acceptance. Diagnose failed exports or replay at their
responsible layer. Unsupported adapters do not establish operability. Preserve
passive geometry/parameters separately from demonstration fixtures and controls.

Use `accept_result` only for evidence-supported delivery. A requested cousin
stage follows the accepted twin and revalidates each variant; do not invent extra
variants. Follow the user prompt when interpreting which properties may change.
