# Hinged lids and free-sheet extraction

**Module id:** `hinged-lids`

**Scope:** Attached covers, source-captured closed states, finite opening and extraction of a free sheet from its container.


## Observations

Distinguish the captured articulation state from a constructed closing path. Inspect separate cover/body boundaries, actual sheet/solid seam contact, retained material and any hidden support. Hidden structure and material response remain hypotheses.

## Representation and parameters

Keep distinct rigid links, real joint frames and limits. Separate panel thickness from raised rims and recesses; inspect material volume, COM and gravity torque. Preserve the complete intrinsic sheet metric. Loaded material pinch and unloaded bare seating are different conditions.

## Procedure

Check the actual initial joint state immediately after reset. Use finite construction actuation and then remove its forces; a small joint angle alone does not establish seating or clamping. Inspect the full opening path and force-free material retention before the requested grasp and extraction. Test unloaded seating separately after extraction.

The [procedure details](hinged-lids-reference.md) retain the existing implementation-specific guidance.

## Checks and failure diagnosis

Retain failed passive-release and approach traces. Inspect actual support, stored preload, opening speed, rigid contact and wrist tracking before changing material. A material/contact pass is not proof of retained graspable material, scene stability or full task feasibility.

## Tools

- [Opening, extraction and wiping assay](../../../../src/calibrated_scene_reconstruction/tasks/mechanisms/wipe.py)
- [Native hand interaction tool](../../../../src/calibrated_scene_reconstruction/agent/hand_cloth.py)
- [opening acceptance](../../../../src/calibrated_scene_reconstruction/apps/reconstruction/hinged_acceptance.py)
- [articulation statics](../../../../src/calibrated_scene_reconstruction/assets/articulation_statics.py)
- [preparation](../../../../src/calibrated_scene_reconstruction/tasks/mechanisms/preparation.py)
- [interface geometry](../../../../src/calibrated_scene_reconstruction/assets/interface_geometry.py)

## Reuse limits

Do not transfer old geometry, joint states, packing, material priors or task passes to a new source. Construction and diagnostics do not qualify the delivered asset; repeat source, material, native and task checks.
