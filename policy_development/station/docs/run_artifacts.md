# Run artifacts

Every evaluated episode leaves a run directory `runs/<name>/trials/<id>/` (station evaluation runner, `eval_runner.py`):
- `outcome.json`: `program_ok`, `task_success`, `termination_reason`, `steps_executed`, `error`
- `source/<file>`: the evaluated program or policy file; `source/program-result.json`: the program's top-level `result`
- `episode/steps.jsonl`: one line per control step: `step`, the station's proprioception keys (`info()["proprio"]`,
  by default `q29`, `palm_pos`, `palm_quat`), `tactile_n` when the station has touch, and what was sent: `target` (29
  absolute joint targets, `cloud_q29` stations) or `action` with `obs`, the observation vector of that step (layout in
  `info()["observation_layout"]`; `state_vector` stations, such as a real2sim twin station)
- `media/keyframes.npz`: the point cloud and labels every 20 steps (`step`, `cloud`, `labels`), on stations that
  observe a cloud

The success rule never appears in these files; `termination_reason` is the station's verdict.
