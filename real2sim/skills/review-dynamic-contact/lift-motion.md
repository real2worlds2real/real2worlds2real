# Unexplained motion during lifting and holding

Use this reference when a flexible asset shakes, buzzes or repeatedly pops during
a smooth lift or stationary hold. This applies to cloth, sheets, compliant solids
and strands when the task calls for such motion; it is not a stillness rule for
wind-driven or deliberately shaken objects.

A successful extraction, low stretch and no detected penetration can coexist
with visibly unnatural dynamics. A previous accepted standalone asset can also
behave differently after scene conversion. Treat either as a real unresolved
defect rather than attributing it to realism, light material or intentional wind
without evidence.

## Evidence you can investigate directly

`project_access.json` gives the readable framework source and this task's records.
Use native file/search tools and write analysis programs in the workspace as
needed. A dedicated check tool is not required. Locate actual controller keys,
effective physics parameters, retained states, timestamps, bindings and rendering
code; compare authored settings with the runtime that actually produced the movie.

- Separate prescribed grasp motion from the ungrasped region. Inspect settling,
  lift, departure from supports/openings, and a sufficiently long stationary
  controller interval. Stopping the grasp need not stop a free sheet instantly,
  but continuing large oscillation needs an explanation.
- Choose measurements that answer the uncertainty: per-vertex displacement,
  speed, lateral excursion, acceleration, decay, or consistency with the input.
  Average per-vertex motion before aggregating only when that is the intended
  quantity: opposing folds can cancel in a centroid trace. Use actual timestamps
  and distinguish world, participant and camera frames.
- Sparse video samples can alias rapid flutter. Request dense original-frame
  intervals or crops where useful, and correlate them with saved numerical
  states. Frame-rate trajectories cannot exclude faster substep instability.

## Competing causes

Check the evidence before choosing an intervention. There is no fixed repair
sequence or universal quiet-speed threshold.

| Candidate cause | Useful evidence |
| --- | --- |
| Continued or discontinuous excitation | Actual interpolation, velocities, release/hold logic, forces and time offsets; inspect runtime code rather than keyframe endpoints alone. |
| Conflicting constraints or rest contact | Pin ownership, opposing supports, initial intersections and clearances. |
| Excessive contact repulsion | Interaction distance versus physical thickness, rest-fold separation and control-edge length; distinguish physical contact distance from detection-only reach. |
| Numerical or material mismatch | Physical substep size, iteration convergence, effective stretch/bending damping and stiffness; solver-specific coefficients are not interchangeable. |
| Integration changed the dynamics | Standalone versus shared-world solver construction, defaults, contact distances/materials, gravity, transforms and controller semantics. |
| Appearance/replay error | Whether actual nodes move the same way; detailed follower binding, normals and renderer sampling can produce apparent vibration independently. |

Small controlled counterfactuals can distinguish causes without repeatedly
rendering a long movie. Keep unchanged inputs fixed and retain results. Turning
off contact is a diagnostic, not a delivery fix when contact is required. A
numerical/material change requires fresh matching simulation and review. Do not
filter cached trajectories, freeze the free surface, conceal intervals, or raise
thresholds merely to pass. Keep chosen physical assumptions editable for later
calibration.

When reporting the result, distinguish demonstrated improvement, remaining
uncertainty and the actual viewing/measurement coverage. Access to records and
skills helps investigation; neither guarantees that every defect was found.
