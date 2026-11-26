# Review room depth and task obstacles

**Module id:** `spatial-cousin-review`

**Scope:** Use the room-depth check when a cousin calls for a three-dimensional environment. Use the obstacle check when the request calls for clutter near the manipulation. Apply either independently and preserve the requested layout scope.


## Observations

Identify the support surface, pickup region, carried object, landing footprint and gripper approach. For room layout, identify furniture fronts, opening directions, chair use areas and passages.

## Representation and parameters

Inspect the composed stage in its metric frame. Use separate geometry for furnishings whose depth matters. A panorama may supply lighting. Select the camera baseline, motion samples and clearance bounds for the current scene and object.

## Procedure

1. Render the delivery view and a laterally translated view with the same orientation and intrinsics. Sample visible near and far features on the requested room surfaces. Project points from the first camera's actual rendered depths into the second camera and compare their predicted and observed displacements. Exclude newly occluded samples.
2. For interfering clutter, check both clearance and proximity. Measure the smallest gap to the sampled swept object and gripper and the distance from a fixture footprint to the operation. Inspect support, the arm approach, and contact at pickup and placement.
3. For requested layout variations, inspect clear floor space in front of oriented cabinets and drawers, chair use areas and continuity of passages. Reopen the composed scene to check actual placements and inspect the unlabelled render.

## Checks and failure diagnosis

Use a stationary image behind moving foreground geometry as a negative depth control. Test whether moving all obstacles to the table perimeter is detected. For access, place a fixture in a drawer's opening area or across a passage and check that the conflict is detected.

## Tools

- [scene cousin generation](../../../generate-scene-cousins/SKILL.md)
- [scene assembly](../../../compose-interactive-scene/SKILL.md)

## Reuse limits

A camera rotation alone cannot establish parallax. Static clearance along one replay covers only that sampled motion. Claims about moving obstacles, replanning or policy robustness require the corresponding trials.
