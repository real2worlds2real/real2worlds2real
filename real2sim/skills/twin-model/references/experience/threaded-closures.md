# Threaded closures

**Module id:** `threaded-closures`

**Scope:** rigid mating screw closures such as bottle or cup lids.


## Observations

Identify neck and cap, opening direction, functional axes, seating surface,
finite thread region and visible lead-in/runout. Inspect the cap interior and
neck closeups. Mark unseen profile/material details as inferred. Fit visible
functional features without using successful screw motion as a dimension label.

## Representation and parameters

Compose with [open vessels](open-vessels.md) for the shell and material boundary.
Keep neck and cap as separate solids with real mating surfaces. A female thread
is a mating groove with clearance, not a duplicate external ridge inside a cap.
The current contact route must not replace engagement with a screw joint.

Couple axial pitch, start count, handedness, phase, radial profile, clearance and
finite thread extent. In the current `helical_profile`, a helical branch advances
`pitch * starts` axially per revolution, with direction set by handedness; do
not confuse axial pitch with multi-start lead. The generator accepts start count
and handedness, but that API support does not qualify every variant physically.
Threads/materials hidden from the source remain inferred after a task pass.

Relate mesh/SDF resolution, contact offsets and timestep to functional feature
size. Retain both inner/outer surfaces and positive wall thickness. Do not carry
a previous cap's offsets or resolution to a new scale without checking them.

## Procedure

1. Inspect a section, cap interior and neck before dynamic trials. Check mating
   profile interference, phase/alignment, wall thickness and actual hard seat.
2. Compare matched and incompatible profile hypotheses. Distinguish pitch/lead
   mismatch, radial interference and runout collision from insufficient applied
   torque or solver artifacts. Preserve failing controls.
3. Run bounded-force/torque screw-on. Measure axial progress versus rotation at
   the functional axes, including offsets from asymmetric body origins. Motion
   must derive from contact; a commanded helix is only an illustration.
4. Derive the seat from the generated profiles and tighten with bounded torque
   until native contact sustains angular stall. Stopping above the seat is not
   evidence of tightening or retention.
5. Check passive retention with cap wrenches/guides removed, screw-off, complete
   release and pull-without-rotation. A rotation-held axial pull is a different
   trial from passive retention. Keep separate case directories and failed traces.
6. Deliver passive objects without temporary fixtures or guides. Follow current
   mechanism/task gates; a result in another backend does not qualify the
   required Isaac backend.

## Checks and failure diagnosis

| Finding | Inspect before changing geometry or friction |
| --- | --- |
| Turns but will not advance | Mating pitch/lead/handedness, actual contact and axial alignment |
| Jams near entry or exit | Finite lead-in/runout, radial clearance and full pose/contact trace |
| Tool stops before release | Torque saturation, net lifting load, lateral drift and contact separation |
| Retains only under a drive | Applied wrenches/guide state; repeat genuine unforced retention |
| Unrealistically easy motion | Missing contact, collider approximation or prescribed motion |

For a finite-thread exit failure, inspect saved pose/wrench traces before fitting
material or geometry. Test-tool gravity compensation and finite viscous damping
may be used without an axial position target; report the net lifting force and
derive a bounded tangential torque budget from the actual cap radius. This is a
tool budget, not a measured material property. Scale guide damping to actual
mass/inertia and timestep. A torque-limited stall does not prove no geometric exit.
It may stop a failed diagnostic early, never approve release. Preserve material
uncertainty and rerun every physical-prior case after revising the tool.

The [procedure details](threaded-closures-reference.md) retain declaration examples,
source limitations, native checks and implementation references.

## Tools

- [thread geometry](../../../../src/calibrated_scene_reconstruction/assets/thread_geometry.py)
- [Native contact trials](../../../../src/calibrated_scene_reconstruction/tasks/mechanisms/thread_contact.py)
- [thread diagnostics](../../../../src/calibrated_scene_reconstruction/tasks/mechanisms/thread.py)

## Reuse limits

Reuse the coupling, diagnostic order and test phases, not a past radius, pitch,
pose or friction. Snap fits, deformable seals and unqualified thread variants
need additional experiments. Current [twin-model](../../SKILL.md) and delivery
gates remain mandatory for the newly reconstructed assembly.
