---
name: simulate-granular-materials
description: Construct and review granular solids such as sand or beads using passive native PhysX particles, with editable sampling, material assumptions and rigid-tool contact.
---

# Granular materials

Use this when discrete grains, a changing pile, or rigid-tool redistribution
matters. Choose the target backend from the user's scene. For a shared PhysX
scene, use `real2sim_harness.physx_granular`; read the [native API and runtime requirements](../../docs/PHYSX_MATERIAL_CAPABILITIES.md#solid-grains).
The existing Newton MPM route remains available for a compatible explicitly
chosen material/solver. Do not silently switch a requested backend.

Preserve the reference's support/container geometry, occupied volume, grain
scale and appearance. Retain editable source geometry or sampling code, initial
points and explicit physical/numerical assumptions. Distinguish visible grain
radius, rest/contact distance, sampling density and per-grain mass. Refining
sampling must not silently change total material mass or container capacity.
PBD cohesion/friction is an assumed model, not a measured sand identity.

Materialize a passive solid particle set in the caller's existing GPU-dynamics
PhysX world. Containers and tools require matching native colliders. Test
settling and an informative local interaction; a static seed render does not
show particle behavior. Retain count/finite/leakage/support checks and actual
motion. Look for tunneling, excess separation, clumping, artificial motion
limits and the wrong pile response. Adapt sampling, contacts or material
settings based on the demonstrated cause, preserving the physical clock.

Keep task controls, target patterns and robot policies outside the passive
material. Inspect current-source native images/trajectory, not just the score.
Record validation with its source/configuration; a liquid/rigid acceptance
receipt does not automatically establish granular behavior. Follow the task's
configured completion requirements without fabricating a standard receipt.
