---
name: data-policy-loop
description: Collect task-qualified simulation data, train deployment-compatible policies, evaluate closed-loop performance and refine data, training or simulation using transfer-oriented evidence.
---

# Data, policy and evaluation loop

Read the current oversight contract, [module contracts](../../docs/modules.md)
and [transfer rules](../../docs/transfer.md). Use the mature pick/place
path only for a matching task. Demonstration retargeting is partial; its exported
plans are not validated robot episodes. Novel insertion, deformable or articulated
tasks require a qualified expert and task adapter before large collection.

## Data loop

Verify current twin gates and freeze. Keep the task geometry and physical priors
fixed across training variants unless the contract explicitly requests realistic
object diversity. Every cousin needs coherent geometry, appearance, contacts,
camera/support/robot relations and protected-task checks. Sample plausible joint
distributions; correlated construction/material variables are not independent
uniform knobs. Record the deployment rationale for ranges and distributions.

Qualify the expert with the same finite robot control and contact model. Inspect
planning, approach, grasp, interaction, release and recovery. Keep failed trials.
Run a small recording and exporter compatibility check before scaling. Confirm
command versus measured-state labels, timestamps, next-frame shifts, camera
ordering, transforms, action clipping and normalization using round-trip samples.

Export success-only data explicitly with `--only-success`, or separately label
recovery demonstrations under the agreed failure policy. The raw collector can
contain failures. Split by the contracted scene/object/source unit before fitting
statistics or augmentation; do not randomly split near-identical frames. Record
per-stage and per-stratum usable coverage, not only total episode count.

## Training loop

Choose a model from the user's I/O, task horizon, multimodality, compute, latency
and available prior data. Existing LeRobot DP, official DP and OpenPI adapters
have different representations and environments; names alone do not establish
compatibility. For a new model, implement and verify the adapter and action
decoder before training. Pin dataset manifests, code, configurations, seeds,
normalization and pretrained checkpoints. Inspect loss, action decoding and a
short rollout; a lower offline loss is not task success.

Checkpoint selection uses validation only. Package the deployed observation
and control path with the checkpoint. Do not expose simulator object poses,
perfect depth or segmentation to a policy unless that channel is available with
the declared quality on the real robot.

## Evaluation and feedback loop

Evaluate actual closed-loop rollouts at the intended frequency with bounded
controls. Keep traces for successes and failures, report counts and every required
stratum, and use the agreed success statistic. Test observation noise, delay,
timing jitter, contact/model uncertainty and environment shifts within their
justified scope. Do not relax success definitions or omit hard seeds.

Use failure timing and causal controls to choose a response:

- Missing coverage or recoverable off-distribution states: targeted collection or
  DAgger with a qualified expert at those states; preserve policy and expert
  labels, provenance, mixture weights and failed recoveries.
- I/O or timing errors: fix preprocessing/controller parity and rebuild affected
  exports/checkpoints before attributing the failure to model capacity.
- Policy/model limitations: compare training/model changes on fixed validation
  conditions; retain the previous baseline.
- Simulator omissions: improve the model under a transfer hypothesis and source/
  physical consistency checks. Conservative uncertainty or additional realistic
  difficulty may improve transfer while reducing simulated success.

`real2sim/tools/policy/fail_states.py` and collector start-state options are useful for
supported rigid tasks, not a universal DAgger implementation. New task state
serialization, recovery experts and sampling mixtures must be implemented and
tested explicitly. Never relabel the policy's unsuccessful action as an expert
action or leak holdout traces into training.

Use the sealed holdout only after the development loop is complete. Deliver
deployment readiness evidence and unresolved real-world gaps to the human;
simulation success does not verify the real robot.
