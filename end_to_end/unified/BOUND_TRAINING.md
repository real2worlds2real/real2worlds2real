# World-bound dense behavior cloning

`bound_training.py` is an executable training adapter for the unified harness.
It fits actual ridge regression or a small NumPy MLP to accepted demonstrations.
Its numerical objective follows `peg_worker.train`, generalized to the declared
observation/action dimensions. It does not import the Peg teacher, insertion
features, action scaling, calibration or success definition.

This is a narrow, inspectable learning route for vector observations and vector
actions. It is not ACT, Diffusion Policy, a vision encoder, a sequence policy or
evidence that a robot task succeeds. Other policy families should implement the
same world-bound worker interface using their own qualified data decoders.

## Input contract

The world must have a current native freeze and accepted data module. Training
runs inside the unified controller's `train` work packet. The worker checks the
current attempt, request hash, dependency identities and accepted data files both
before and after fitting. A task label without these prerequisites is rejected.

The native version 1 episode manifest additionally identifies the dense format:

```json
{
  "version": 1,
  "format": "astra_dense_bc_v1",
  "task_id": "the_exact_bound_task",
  "world_sha256": "the_exact_world_digest",
  "policy_io_sha256": "digest_of_the_complete_bound_policy_io_object",
  "observation_fields": ["measured_joints", "measured_object_estimate"],
  "action_fields": ["joint_command"],
  "action_alignment": "the_exact_bound_alignment_description",
  "episodes": [
    {
      "id": "demonstration-001",
      "group": "a_declared_training_group",
      "seed": 1,
      "source": "attributed_source_episode",
      "split": "train",
      "success": true,
      "label": "expert",
      "artifact": "/absolute/path/to/episode-001.npz"
    }
  ]
}
```

Every NPZ contains `observations` of shape `[T, observation_dimension]` and
`actions` of shape `[T, action_dimension]`, with finite floating point values.
Columns concatenate the named one-dimensional fields in the exact frozen I/O
order. Actions are actual expert command labels at the contracted alignment;
this trainer does not convert measured states into commands or infer offsets.
Image fields, action chunks and dimensions above the bounded CPU interface are
rejected. The manifest and every episode must already appear in the current
accepted data evidence. Formatting a file does not qualify its provenance.

Only episodes marked `train` are fitted. Validation data remain excluded from
normalization and optimization. Holdout data cannot appear in this manifest.
The trainer fits observation mean/std from training rows, with a scale floor of
`1e-6`; action values retain their declared units. The checkpoint packages those
statistics and decoder semantics for the independent I/O parity review.

## Register and invoke

Register an `astra_bound_backend_v1` manifest through `backend_register`. The
task/backend IDs must equal those in `world_bind`; there is no hardcoded task.
Its `operations.train` command uses the following fields, with real absolute
paths substituted before registration:

```json
{
  "method": "ridge_bc",
  "argv": ["/absolute/python", "/absolute/repo/policy_development/bound_training.py",
           "--binding", "{binding}", "--request", "{request}", "--result", "{result}"],
  "cwd": "/absolute/repo",
  "env": {"PYTHONPATH": "/absolute/repo/src", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"},
  "inputs": ["/absolute/repo/policy_development/bound_training.py",
             "/absolute/repo/end_to_end/src/astra_e2e/oversight/policy_bridge.py",
             "/absolute/repo/end_to_end/src/astra_e2e/oversight/evidence.py",
             "/absolute/repo/end_to_end/src/astra_e2e/oversight/controller.py"],
  "timeout_seconds": 300
}
```

Use `method: "mlp_bc"` for the small MLP. Register the changed manifest as a new
revision. Call `policy_execute` with `operation: "train"` and method parameters:

- Ridge: `seed` (default 0), `ridge_alpha` (default 0.001, range 1e-8–100).
- MLP: `seed`, `hidden_dim` (default 24, range 1–512), `updates` (default 500,
  range 1–5000), `learning_rate` (default 0.009, range 1e-6–0.1).
- Optional `dataset_manifest` must equal the accepted data review's exact path;
  it cannot select an unreviewed alternative.

## Outputs and next step

The worker writes a new `policy.npz`, `policy_io.json`, `training_metrics.json`,
`receipt.json` and `result.json` beside the owned job request. It never replaces
an existing result. The checkpoint records the world, task, I/O, request, data
acceptance, source implementation and training episode identities. All receipt
artifacts have content hashes. `predict()` decodes this checkpoint in the same
declared vector order; the task runtime remains responsible for bounded control
and actual closed-loop rollout.

The receipt schema is `astra_bound_policy_receipt_v1`. It is distinct from the
legacy repair-worker receipt: execution authority and budgets come from the
native oversight attempt, not a claimed legacy grant/protocol envelope.

A low training error only establishes a fitted model. Inspect the artifacts,
provide independent train-module checks, and translate/review the receipt before
an evaluator can use the checkpoint. Development and sealed confirmation require
a separately implemented world-bound rollout adapter. No training receipt
contains fabricated success rates or auto-generated passing reviews.

The CPU tests fit actual models to explicitly synthetic numerical data, verify
prediction accuracy and dynamic dimensions, exclude adversarial validation rows,
and reject stale, foreign-world, holdout and unpinned inputs. Native gates are
mocked in these tests; the tests do not establish scene or manipulation success.
