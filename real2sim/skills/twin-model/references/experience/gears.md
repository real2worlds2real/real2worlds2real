# Gears and post-mounted gear assemblies

**Module id:** `gears`

**Scope:** observed external spur-gear hypotheses, mounting bores and gear pairs.


## Observations

Separate tooth count, tip/root envelope, plate thickness, hub and through bore.
Count visible periodic features across views and preserve ambiguous tooth-count
hypotheses. Observe the aperture separately from the hub silhouette. Pressure
angle, hidden underside and entry chamfer may be priors rather than observations.
Do not infer bore or tooth geometry by forcing the assembly to succeed.

## Representation and parameters

For an appropriate spur-gear hypothesis, represent actual tooth flanks/roots,
plate, hub and bore. A smooth disc or decorative tooth texture cannot establish
functional meshing. Preserve positive material between bore, hub and tooth root.

The existing unshifted external-gear construction couples tooth count `z`, module
`m`, pitch diameter `m*z` and tip diameter `m*(z+2)`. These are model assumptions,
not universal formulas for worn, shifted, helical or nonstandard teeth. Pressure
angle and profile choice must be explicit. For a requested compatible gear train,
a shared module can be a structural hypothesis; estimate it from observed gear
geometry and keep per-gear residuals/count uncertainty.

Mounting-bore clearance is a separate constraint from tooth compatibility.
Preserve current source-derived post/gear centers instead of adjusting them until
a simulated rolling score passes. Resolve inconsistent geometry as a hypothesis
or observation problem.

## Procedure

1. Inspect tooth-count and profile alternatives against source edges. Fit the
   visible rim/height/hub and separately reviewed bore from current observations.
2. If compatible meshing is requested, compare independent pitch/module estimates
   before sharing one parameter. Preserve disagreement rather than averaging it
   away without review.
3. Export actual solids and examine tooth roots, bore continuity and collision
   approximation. A profile can fit an image yet fail to retain usable teeth.
4. Compose with [rigid insertion](rigid-insertion.md) for each gear/post mounting.
   Seating on a post does not demonstrate pairwise meshing.
5. For a pair, inspect phase, center distance and the actual exported tooth
   outlines over rotation. A prescribed tooth-count rotation ratio is a
   geometric screening condition; it is not contact-driven transmission.
6. When physical rolling/transmission is required, run current native contact
   trials with the declared drives/fixtures and inspect motion, penetration and
   residual motion after relevant drives are released.

## Checks and failure diagnosis

| Finding | Inspect before changing parameters |
| --- | --- |
| One mesh phase fits, rotation jams | Full tooth-period sweep, profile/root interference, backlash and colliders |
| Looks right, cannot mount | Actual bore, post section, hub/seat and full insertion path |
| Pair works only after moving posts | Source center observations and shared-module/tooth-count hypothesis |
| Rendered teeth disappear physically | Collision simplification, feature resolution and contact offsets |
| Forced rotation looks correct | Whether motion was prescribed instead of transmitted by contact |

Keep mounting and meshing results separate, including negative cases. Do not
copy an old tooth count, pressure angle, bore radius or success threshold into a
new source as a measurement.

## Reuse limits

Reuse profile hypotheses, compatible-parameter relationships and separated
mounting/meshing diagnostics. New tooth families, materials, helical gears or
compliant teeth need additional modeling and native evidence. A compatible
render is not measured geometry or proof of torque transmission.
