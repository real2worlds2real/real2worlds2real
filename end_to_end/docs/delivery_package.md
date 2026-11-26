# Delivering a simulation policy

`simulation_policy_accepted` ends model interaction after sealed evaluation. It
does not complete the outer `delivery` module. Package the accepted result,
independently inspect the handoff, and submit the ordinary delivery review.

The generic helper supports the currently executable `astra_dense_bc_v1` NPZ
format, including ridge and MLP BC. Other checkpoint formats need an explicit
packaging adapter; the helper does not silently convert or relabel them.

Run the helper as the command in a normal `delivery` work packet. It can live
outside the frozen repository, with dependencies imported from that repository:

```sh
PYTHONPATH=/frozen/repository/src:/frozen/repository/policy_development \
  /qualified/python -B /external/tools/package_bound_policy.py \
  --binding /run/world_binding.json \
  --agent-report /run/completed_agent/report.json \
  --native-config /run/qualified_native_config.json \
  --output /run/outer/attempts/delivery-0001/package
```

Use actual registered paths and the next owned delivery-attempt directory.
Pin the helper, binding, report and configuration in the work packet. This
command never installs software, starts a simulator, calls a model, changes the
scene, runs a policy, opens an unfinished holdout, or approves delivery.

Before reading the final report, the helper requires current accepted data,
training, development evaluation and holdout under the same binding. Both
evaluations must identify the accepted training checkpoint. It verifies the
completed agent history and current final receipt, the exact native execution
configuration, the frozen export and all declared file dependencies.

The new output directory contains:

- `policy.npz`: the byte-identical accepted checkpoint.
- `policy_io.json`, `native_config.json`, `world_binding.json` and
  `final_agent_report.json`: the executed contracts and completed evidence.
- `normalizer.npz` and `normalization.json`: exact mean/scale arrays extracted
  from the checkpoint, without refitting.
- `dependencies.json`: path, size and SHA-256 for external scene exports,
  assets, source code, execution packets and evidence. The actual launcher,
  working directory and environment are retained by those pinned packets.
- `HANDOFF.md` and `manifest.json`: limitations, file identities and the
  outstanding delivery-review obligations.

Large assets remain on the worker. This is a **content-pinned external closure**,
not a self-contained portable archive or a runtime installation. Preserve its
recorded file locations; relocation requires explicit rebinding and
requalification. Simulator/Python installations must be recreated and checked
with the exact-launcher preflight. Hashes alone do not prove runtime usability.

The package is always marked `robot_deployable: false`,
`real_validation: pending` and `delivery_accepted: false`. It may use privileged
state observations or initialized fixtures. Repetition identifiers can induce
new reset samples within a world; they do not establish new-scene
generalization. The handoff must retain the adapter's actual assumptions and
release behavior.

After execution, independently inspect copied checkpoint/normalizer parity,
native controller/I/O parity, the available external dependencies and the human
handoff. Author all checks required by the current delivery plan, using the
package files as evidence; supply the copied checkpoint and `policy_io.json`
in the ordinary controller review. No packaging result substitutes for those
reviews. Keep final evaluation closed and real-robot operation separately
authorized.

The focused tests use explicitly synthetic native acceptance fixtures. They
validate packaging and rejection behavior, not robot or simulator performance.
