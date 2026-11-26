# Policy development

Part 2 of the repository: policy development in simulation. An agent chooses, runs and compares methods on a **station** (a simulation twin that
serves one task through the robot's own interface and hides its success judge), freezes one deliverable, gets one sealed
evaluation, and hands the deliverable to the task's real-robot chain. Design and commands: [docs/station_harness.md](docs/station_harness.md).

A station is either a hand-authored task or a twin accepted in [real2sim](../real2sim/README.md), taken over by its
delivery record (`python -m astra_station new-twin-task --delivery <record>`); [the chain](../end_to_end/docs/chain.md)
runs both parts in order. This part imports nothing from the other two.

This directory holds only what is **task-independent**. Everything that names a task (task specs and documents, the
Isaac task backends, real-robot chains, task deployments) lives in a separate **task directory** that you supply. No
task directory ships with this release; the [station harness](docs/station_harness.md#what-is-generic-and-what-a-task-supplies)
describes its format. A real2sim twin needs only a registry directory to write into: `new-twin-task` creates its task
spec there. Site facts (hosts, paths) live in `policy_development/configs/station/site.*.json`.

```
 task registry (<task directory>/tasks/<task>/task.json)     real2sim twin (delivery record of an accepted scene)
        |                                                          | new-twin-task (src/astra_station/twin_task.py)
        v                                                          v
 binding (pins station code, sealed partitions, docs, data, the twin's delivery record and files)
        |
 campaign: budgets, candidates, datasets, hash-linked ledger          <- library/ (experience cards, reward terms, work library)
        |                                                  ^
 decision loop: one fresh agent session per step  --tool-->  tools/  (method adapters, receipts)
        |                                                  |  run in the station root's sandbox: runtime/, station/client/
 freeze -> unseen check -> sealed held-out (coordinator) -> handoff to the task's real chain (registry or package builder)
                                                                         |
                                                     real_robot/station_worker.py + the task's observer (<task directory>/real/<task>)
```

| Directory | What it is | Runs where |
|---|---|---|
| [`src/astra_station/`](src/astra_station) | Controller: site, binding, tool catalog, campaign, runner (decision loop), agent transports, experience memory, twin task specs; CLI `python -m astra_station` | controller host, stdlib only |
| [`tools/`](tools) | One adapter per method tool (`evaluate`, `import_real`, `record_demos`, `train_act`, `synthesize_program`, `research_session`, `train_residual`, `compose`); `_adapter.py` = argv contract, receipts, station and sandbox access; `subsession.py` / `subtool.py` = confined worker sessions; `prompts/` = their briefs | controller host |
| [`runtime/`](runtime) | Code that runs inside the sandbox or inside a candidate: `rollout_runner.py` (training rollouts, demos, real-worker cost), `diagnostics.py`, `program_policy.py`, `compose_policy.py`, `residual_policy.py` + `residual_learner.py` + `rewards.py`, `kinematics.py` + `poe_kinematics.py` (identified arm models), `act_policy_station.py` + `act_prep.py` | station sandbox / candidates / training host |
| [`station/client/`](station/client) | The station's client side, mounted at `/station` in the sandbox: `station.py` (protocol client), `program_runtime.py` (**the one implementation of the program contract**, also used by the real worker), `eval_runner.py` (sealed evaluation runner) | sandbox |
| [`station/server/`](station/server) | Station server: `core.py` (protocol, sealed partitions, usage policy, ledger), `backend.py` (task backend contract), `serve.py` + `station_up.sh` (launch from a task spec), `twin.py` (real2sim twin backend), `partitions.py` (sealed partitions of a new task) | simulation hosts |
| [`station/coordinator/`](station/coordinator) | `task_tool.py`: workspaces, station hosts and relay, bubblewrap sandbox, policy-development tools, sealed held-out; task facts from the registry | controller host |
| [`station/install.py`](station/install.py) | Assemble / compare a station root from this repository and your task directory | operator |
| [`real_robot/`](real_robot) | `station_worker.py`: the generic real-robot station worker (a deliverable exactly as the station runs it; the task's observer builds the observation) | robot workstation |
| [`library/`](library) | `cards/` (experience cards), `rewards/` (reward terms), `work_library.py` (reviewed literature + lab rewards); docs `WORK_LIBRARY.md`, `RL_REWARDS.md` | controller host |
| [`configs/station/`](configs/station) | Site files (`site.*.json`: station root, hosts, compute, the GPU rule of the sandbox) and tool manifests (`tools/`) | controller host |

## Interfaces a task chooses from

| `policy_io.interface` | Observation | Action | Hold (stay still) | fk / ik |
|---|---|---|---|---|
| `cloud_q29` | labelled point cloud, q29, palm pose, tactile pads | 29 absolute joint targets | measured joints | if the station offers them (`info["kinematics"]`) |
| `state_vector` | `obs` vector (`info["observation_layout"]`), q29, palm pose | `action_dim` deltas in [-1, 1] | zeros | no |

Tool manifests list the interfaces they support (`policy_development/configs/station/tools/*.json`), so an agent only sees applicable methods.
