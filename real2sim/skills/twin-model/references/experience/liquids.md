# Liquids and native surface reconstruction

**Module id:** `liquids`

**Scope:** Source-visible resting liquid, declared contained liquid or separate spills, native particle dynamics and finite cloth absorption.


## Observations

Identify the liquid boundary across calibrated views; distinguish it from a label or reflection. Record uncertainty in the amount and cavity. A fitted level does not solve refraction or identify material optics.

## Representation and parameters

Declare non-mesh components separately from rigid/cloth links, with an actual container link or independent task-object owner. Use explicit density, viscosity and surface-tension priors. Physical cavity volume and the independently checked native particle-volume mapping determine amount; dense seed layout is not a physical pass.

## Procedure

Reconcile input inventory, fit the visible boundary against the actual cavity, and seed the inferred physical volume. Check native settling, mass, level, contact and containment. Compose the real particle systems and their simulation owners; inspect the combined native surface through the actual container material. Bound and record any explicit initialization step.

The [procedure details](liquids-reference.md) retain the existing implementation-specific guidance.

## Checks and failure diagnosis

Reject omitted components, invisible or empty native surfaces, unintended particle motion, mass loss and unqualified native evidence. Surface smoothing only changes visual surfaces/normals and needs boundary, volume and source comparison. Absorption needs actual contact, capacity and noncontact controls; removed liquid mass must enter the native absorber.

## Tools

- [Declaration and evidence rules](../../../../src/calibrated_scene_reconstruction/assets/continuum_components.py)
- [Level fitting](../../../../src/calibrated_scene_reconstruction/scene/objects/liquid_level.py)
- [planar film](../../../../src/calibrated_scene_reconstruction/scene/objects/planar_fluid.py)
- [FCC seeds](../../../../src/calibrated_scene_reconstruction/assets/fluid_lattice.py)
- [Native dynamics](../../../../src/calibrated_scene_reconstruction/sim/native_fluid.py)
- [composition and surfaces](../../../../src/calibrated_scene_reconstruction/sim/fluid_reference.py)
- [quality diagnostics](../../../../src/calibrated_scene_reconstruction/sim/fluid_quality.py)
- [absorbing sheet](../../../../src/calibrated_scene_reconstruction/sim/cloth2d_wipe.py)

## Reuse limits

Neither static images nor a task pass identify real liquid parameters. Prior uncertainty, calibration validity, numerical resolution and installed-backend controls remain explicit. Do not substitute a static mesh for requested native dynamics.
