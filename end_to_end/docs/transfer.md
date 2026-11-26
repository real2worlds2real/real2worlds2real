# Transfer objective and simulator integrity

The system optimizes readiness for the real task. It does not optimize simulation
success in isolation. The agent does not operate the real robot; its final package
and report explicitly leave real validation pending.

## Evaluation can improve simulation

Policy failures can expose missing geometry, contact mechanisms, articulation,
sensor noise, occlusion, latency, controller behavior or deployment coverage.
Use them to improve simulation under a causal transfer hypothesis. New physical
measurements are useful but not required for every improvement. A source-supported
model correction, a physically justified mechanism, an independent native control,
or a conservative uncertainty distribution can support a change.

Record the failure, proposed cause, expected real benefit, changed assumptions,
unchanged invariants and discriminating comparison before the experiment. Compare
the previous baseline and new candidate under fixed development conditions and
report both task and transfer checks. A realistic correction may lower simulated
success; do not reject it solely for that reason.

| Transfer-oriented change | Simulator shortcut to reject |
|---|---|
| Correct a source-supported cavity or missing contact surface | Widen the hole, shrink the collider or move the support because insertion failed |
| Model uncertain friction with justified priors and sensitivity controls | Tune friction/mass/stiffness to the successful policy and call it real identification |
| Add deployed observation delay, noise and occlusion | Give the policy perfect object poses or future frames unavailable on the robot |
| Match real actuator limits, IK/servo saturation and timing | Add invisible forces, welds, teleportation or unbounded control to complete a trial |
| Add plausible deployment cousins and correlated randomization | Delete hard scenes, failures or inconvenient seeds to raise the score |
| Improve collider cooking against the unchanged source surface | Make collision geometry easier while retaining a misleading visual mesh |
| Learn recoveries from qualified expert labels | Copy policy actions into expert labels or reset a failed episode without reporting it |
| Use a fresh held-out protocol after further development | Repeatedly tune on a test set and continue calling it unseen |

## Invariants across the loop

1. Keep raw observations, legal calibration, reviewed semantics and physical
   uncertainty distinguishable from inferred hypotheses. Independent reconstruction
   cannot import old fits or withheld geometric ground truth. Development-only
   post-freeze geometry evaluation remains isolated from the current reconstruction.
2. Keep native material/contact validity, actual body/link geometry, realistic
   forces and the user's task definition. Bounded construction/test fixtures must
   be disclosed, released when required, and absent from passive delivered assets.
3. Preserve the exact deployment observation/action interface: frames, units,
   ordering, action type, timestamp alignment, normalization, clipping, rate,
   latency, history/chunk handling and sensor availability. The same I/O artifact
   is checked at data, training, evaluation and delivery boundaries.
4. Keep training, development validation and sealed holdout separate. Every final
   declared scenario is reported, including crashes and failures. Dataset statistics
   and model selection use development data only.
5. Record why a sim change should transfer and what independent comparison might
   disprove it. Keep the old result. Do not confuse a scalar reward improvement
   with evidence that a physical hypothesis is true.

## Human-operated real handoff

Deliver the selected checkpoint and checksums, preprocessing/normalization,
camera and robot calibration assumptions, exact control interface, software and
asset versions, resource/latency measurements, simulator settings and all dataset/
training/evaluation lineage. Include known sim-to-real gaps and expected operating
conditions, physical limits and a staged human test protocol with observable stop
conditions. Do not invent operating limits absent from the robot/task contract.

The report distinguishes: source reconstruction evidence; qualified native task
physics; closed-loop simulated policy performance; deployment-interface readiness;
and **real-world performance not yet measured**. If the human later supplies real
results, preserve them as attributed observations and create an explicit next
experiment. The controller never silently changes its claim to real validated.
