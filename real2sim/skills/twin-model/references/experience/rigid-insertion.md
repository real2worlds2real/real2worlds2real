# Rigid insertion and seating

**Module id:** `rigid-insertion`

**Scope:** peg/hole, shaft/bore, axial nesting and rigid assembly paths.


## Observations

Identify moving and receiving parts, observed aperture, shaft section, functional
axes, expected direction and final support. Distinguish an observed mouth from
an unseen internal bore/floor. A visible circular rim does not measure blind
depth, entry chamfer or internal contact surfaces. Preserve those uncertainties.

## Representation and parameters

Keep parts separate. Represent the receiver's cavity and the complete moving
body, including shoulders, handles, collars or a gear hub. Use source-supported
geometry and functional frames, not arbitrary object origins. A peg modeled
lying along local X needs an explicit transform into its insertion axis.

Track moving envelope versus receiving free space along the whole path. Radial
clearance, depth, orientation, floor/seat height and approach corridor are linked
but distinct. The scalar hole-minus-shaft radius test applies only to a suitably
aligned circular section; it does not establish collision-free motion or seating.

## Procedure

1. Fit moving and receiving geometry from current source evidence. Check the
   opening and shaft section at sufficient resolution before optimizing poses.
2. Propose an approach corridor using the complete moving envelope and relevant
   obstacles. For cup nesting compose with [open vessels](open-vessels.md); for
   post-mounted wheels compose with [gears](gears.md).
3. Check the path, not only the endpoint. Retain full-resolution triangle and
   collider comparisons when thin features or neighboring objects matter.
4. If the claim is physical insertion/seating, execute bounded or gravity-driven
   native trials. Disclose fixtures and temporary guides. Remove the relevant
   guide/drive before claiming unforced retention or a stable released state.
5. Review depth, actual seating surface, lateral displacement, tilt, penetration
   and release. Distinguish geometric accessibility, native task feasibility
   and real object accuracy. Robot grasp/IK is required only by the task contract.

## Checks and failure diagnosis

| Finding | Inspect before changing geometry |
| --- | --- |
| Mouth fits, insertion jams | Inner profile, shoulders, full-body path and sampled/cooked collision geometry |
| Endpoint is valid but motion intersects | Intermediate trajectory and obstacles; a teleport is not insertion |
| Guided pass, failure after release | Fixture dependence, support contact, mass/COM and actual seating |
| Unexpected lateral drift or tilt | Functional-frame transform and body origin, not only nominal placement |
| Success requires a deeper/wider hole | Whether unseen construction was fitted to the desired task outcome |
| Seated part keeps creeping in; contact force and penetration grow at rest | Whether the seat is a self-locking taper (below) before touching time step, iterations or collider resolution |

### Self-locking tapers

A conical or wedge seat whose half-angle `alpha` is below the friction angle
(`tan(alpha) < mu`) is self-locking: with ideal rigid contact its static normal
force has no unique value. In native rigid trials this showed as a seated part
that never rests: the summed normal force of the pair grows by orders of
magnitude above the part's weight while the reported penetration and the
insertion depth creep. Compute `tan(alpha)` from the fitted taper and compare it
with the whole declared friction interval before interpreting such a trace.

Diagnose in this order, on a saved copy of the assembled stage:

1. Read force and penetration over the whole trace, not only the terminal
   window; a quiet first seconds can precede the growth.
2. Control: friction below `tan(alpha)` on the pair. If the part then rests at
   the Coulomb equilibrium force `W / (sin(alpha) + mu cos(alpha))`, the wedge is
   the cause. This control is not a candidate setting; do not adopt a friction
   value the material prior does not support.
3. A smaller time step, more solver iterations, a finer SDF, zero restitution,
   a depenetration-velocity limit or disabled CCD did not remove the growth in
   the recorded case. Scene-level solver stabilization hides it and is not a
   physical model.
4. Real thin-walled parts rest in such a seat through wall compliance. Declare
   it: `contact_stiffness` and `contact_damping` on the rigid links in
   `agent/physics_priors.json` ([contract](../../../../src/calibrated_scene_reconstruction/assets/compliant_contact.py)).
   Derive the interval from the part's material and wall (for a thin shell, the
   ring stiffness under an axisymmetric line load), then map it to the native
   per-contact-point spring using the measured number of contact points of the
   actual colliders. Do not pick the interval from which values pass. Check that
   the reported penetration scales with force over points times stiffness.
5. Check the solver with matched controls at the stiff end of the declared
   interval. In the recorded case TGS still accumulated force for the stiffest
   prior (failing at 240 Hz and 960 Hz, resting at 480 Hz: not a convergence
   pattern), while PGS with the same compliant contact rested for nominal, low
   and high priors at 240 Hz and 480 Hz and diverged at 120 Hz. PGS without
   compliance did not build force but did not rest either. Declare the solver
   and step in `simulation.json` with these controls as its basis, keeping the
   step safely finer than the coarsest one that failed.
6. Rerun the unchanged task check for nominal, low and high priors, and every
   intrinsic test of the bodies whose contact model or numerical profile
   changed. The native model ignores restitution for a compliant material: say
   so in the review. Read the force trace as well as the rest criterion: a
   terminal-window motion just inside the threshold can hide growing force.

The resulting insertion depth and locked-in contact force depend on the release
(drop height) and on these priors. They describe the model, not the real pair.

Keep failed controls and current-source checks. Unseen depth remains a hypothesis
even when a peg seats in that assumed model. Numeric tolerances must come from
the current fidelity contract and numerical resolution, not an old session.

## Tools

- [insertion corridor](../../../../src/calibrated_scene_reconstruction/tasks/rigid/insertion_corridor.py)
- [Round openings](../../../../src/calibrated_scene_reconstruction/assets/round_opening.py)
- [source-circle fitting](../../../../src/calibrated_scene_reconstruction/scene/objects/source_circle.py)
- [Compliant contact priors](../../../../src/calibrated_scene_reconstruction/assets/compliant_contact.py)

## Reuse limits

Reuse the decomposition into path, contact and released-state checks. Refit
geometry/poses and rerun the actual scene. Rotation-coupled contacts require additional checks; deformable feeding, press fits and friction pinches are not qualified by
this rigid clearance procedure alone.
