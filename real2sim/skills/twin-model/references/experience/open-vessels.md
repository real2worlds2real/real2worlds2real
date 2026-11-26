# Open vessels

**Module id:** `open-vessels`

**Scope:** rigid cups, bottle bodies and hollow containers; axisymmetric
construction only when supported by the current source.


## Observations

Identify the exterior, opening, rim, bottom and any visible inner wall separately.
Use calibrated silhouettes and source-labelled mouth edges. Distinguish the
opening from an inner sleeve, reflection or printed ring. A single exterior
silhouette cannot identify wall thickness or a hidden inner floor.
Record unseen wall/base thickness and rounding as hypotheses with bounds.

## Representation and parameters

For a supported rotational shape, revolve a **closed material cross-section**:
underside → outer wall → rim → inner wall → inner floor. The resulting material
solid is watertight while the cavity remains open. Check the section for
self-intersection; watertight triangles alone do not establish valid walls.

Couple outer/inner profiles through positive wall thickness and a base below the
inner floor. Fit visible top/bottom diameters, height and taper from this capture.
Add neck/shoulder profile changes when observed. Do not force a cylinder, outward
taper or rotational symmetry onto an incompatible cup. Handles and asymmetric
attachments require explicit geometry and can invalidate a radial-only fit test.
Declare an `OPENING(params)` consistent with the actual exported cavity, including
its functional frame, rim and inferred/observed depth.

## Procedure

1. Fit exterior and visible opening independently enough to expose disagreement;
   retain per-view uncertainty. Inspect sections as well as outside renders.
2. Generate the complete material boundary, retaining the cavity and bottom.
   Recheck mesh validity, inner/outer separation and opening after export.
3. For nesting, compose with [rigid insertion](rigid-insertion.md). Compare the
   moving object's outer section against the receiving cavity's inner section
   across depth and the sampled insertion path. Mouth clearance alone is not
   sufficient. Include the handle/full envelope if present.
4. For two vessels of the same family, compare the fitted tapers: within fit
   noise they are equal, so the nested contact is a conforming cone and the
   seat height is set by the assumed wall thickness. Look in close frames for
   what would seat the pair with clearance instead (rim lip, stacking shoulder
   or rib, foot ring) and model only what is visible. State what was not
   resolvable, usually the whole interior.
5. Test sensitivity to uncertain wall/base/rounding choices. Report nominal and
   conservative cases separately; the task does not measure the hidden wall.

## Checks and failure diagnosis

| Finding | Inspect before adjusting parameters |
| --- | --- |
| Good exterior fit, failed nesting | Interior taper, base height, wall uncertainty and complete insertion path |
| Closed mesh, invalid material | Cross-section self-intersection, inward/outward winding, zero-thickness rim |
| Fit works only for one azimuth | Polygon sampling, asymmetry and whether radial checks are applicable |
| Task succeeds after resizing the cavity | Whether a source-derived boundary was changed to satisfy the task |
| Nested vessel never rests; contact force grows | Two vessels of nearly equal taper form a self-locking cone seat: follow the self-locking taper diagnosis in [rigid insertion](rigid-insertion.md) |

Geometric path sampling establishes only model clearance at its sampled scope.
Dynamic settling, stable support, friction and contact claims need applicable
native tests and released-state evidence. Preserve source fidelity independently.

## Tools

- [vessel meridian validation](../../../../src/calibrated_scene_reconstruction/assets/vessel_meridian.py)
- [Observed circular boundaries](../../../../src/calibrated_scene_reconstruction/scene/objects/source_circle.py)

## Reuse limits

Reuse the section construction, parameter dependencies and diagnostic order.
Re-estimate dimensions, pose, appearance and material hypotheses from the new
input. Old thresholds or thickness ranges do not define the next cup. Flexible
walls, snap lids and seals require additional models and evidence.
