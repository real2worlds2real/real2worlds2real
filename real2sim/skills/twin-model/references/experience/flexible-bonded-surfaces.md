# Flexible surfaces attached to a rigid part

**Module id:** `flexible-bonded-surfaces`

**Scope:** A petal, artificial leaf or sheet has a bonded root and a free surface. Use [observed surface depth](observed-surface-depth.md) when the initial shape is uncertain.


## Observations

Identify each continuous piece of material, its visible free edge and its attachment footprint across views. One segmented patch may cover two overlapping leaves, or only the coloured stripe of one leaf. Mark obscured roots and reinforcing ribs as inferred unless visible or confirmed.

## Representation and parameters

Keep the material's stress-free shape, its loaded initial pose, thickness, mass distribution, bond footprint and collision surfaces distinct. For a reinforced leaf, describe the rib and the surrounding blade separately if they bend differently. A loose stem in a vessel needs its own free contacts.

## Procedure

1. Run an unforced release in the intended simulator. Record the rigid owner's world motion, the root-to-owner gap and the tip motion relative to the owner starting at reset.
2. If the surface jumps immediately, compare diagnostic trials with gravity disabled and with just the owner contact disabled. Preserve the geometry, bond and other contacts. Inspect the collision envelope, including any welded contact carriers, before tuning stiffness.
3. For PhysX particle cloth with explicit rest points, bind the deformable endpoint at the indexed material rest point. Compute the rigid endpoint from the corresponding **posed** vertex in the rigid actor's local frame. Audit initial coincidence in posed world coordinates.
4. Inspect triangles within and between sheets, including face interiors near a rim. After a shape refit, recheck the rest shape and any mass derived from area. Restore full contacts and check release, loading and the complete requested interaction.

## Checks and failure diagnosis

Compare a case where rest and posed coordinates coincide with one where they differ. Using the posed coordinates for both endpoints can pass the first control. A stable core can conceal a drifting leaf. Inspect each free surface. A single fitted silhouette or a stationary tip cannot establish material response.

## Tools

- [source-surface comparison](../../../simulate-deformable-behavior/source-surface.md)
- [dynamic contact review](../../../review-dynamic-contact/SKILL.md)

## Reuse limits

The endpoint convention is specific to that representation. Check the native attachment convention for FEM or another solver. Re-estimate stiffness, bond geometry and thickness from current observations and trials.
