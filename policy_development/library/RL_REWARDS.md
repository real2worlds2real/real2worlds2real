# Choosing rewards for residual RL

The station's `train_residual` tool trains a bounded residual on a frozen base policy or program. Choose the shaping terms from the [reward catalog](rewards/) using the observations and labels supplied by the task's station. Each catalog entry describes its inputs, formula and cautions.

Use the controller's `reward_search` tool to look up related terms and recipes in the [work library](WORK_LIBRARY.md). A literature formula can be used as an `expression` term after translating it to the features supported by the [reward runtime](../runtime/rewards.py).

## Training configuration

Supply a `reward_terms` list in the arguments to `train_residual`. For example, for a task whose station labels an object as `cup`:

```json
[
  {"id": "reach_palm_to_label", "weight": 1.0, "params": {"label": "cup"}},
  {"id": "success_sparse", "weight": 10.0}
]
```

Use the label from your task binding. The first term uses the palm position and the labelled point cloud. The learner applies the sparse bonus when the station reports success at the end of an episode. The [training adapter](../tools/train_residual.py) checks term names and expression syntax, applies the campaign's remaining training budget, and records the chosen terms in the candidate configuration.

## Evaluating a candidate

Assess the trained candidate against the task's fixed success criterion on the development partition. Training reward and rollout success measure different quantities. Freeze the selected candidate before the sealed evaluation and retain the checkpoint, evaluation traces and station binding for handover. See the [station workflow](../docs/station_harness.md).
