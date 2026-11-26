# The chain: video to simulation, policy development, preparation for the real robot

The repository has two parts. Each has one contract with the outside:

| Part | Ends in / starts from | Written by |
|---|---|---|
| [real2sim](../../real2sim/README.md) | a **delivery record** for an accepted twin | `real2sim-harness deliver` or the reconstruction workflow's delivery step |
| [policy development](../../policy_development/README.md) | a **task binding**; then a frozen deliverable and one sealed evaluation | `python -m astra_station ...` |

Native deliveries use `astra_twin_delivery_v1`. A workflow that tracks model revisions emits `astra_twin_delivery_v2`,
which also identifies the files and check receipts for each revision. Both the reconstruction harness and the policy
station verify these records. See the [delivery contract](../../real2sim/docs/DELIVERY.md).

`python -m astra_e2e.chain` joins them. It keeps one run directory, tells the agent what the current stage needs, and
records a stage as done only after reading the files that stage's part wrote. It embeds no model and operates no robot.

```text
observations ──> real2sim ──> station task ──> policy development ──> sealed evaluation ──> real preparation
  video,          twin built    task spec and     campaign: methods      once, on starts       hand-over for
  task in words   and accepted  binding from the  chosen, run, compared; nobody has seen       human-operated
                  (Part 1)      delivery record   one deliverable frozen (Part 2)              validation
                                (Part 2)          (Part 2)
```

For new runs, [select the reconstruction modules](reconstruction.md). The agent can deliver one object, a complete scene, or continue through policy development. The legacy pinned-route request is also supported.

## Run it

```sh
source env.sh                                        # repository root
python -m astra_e2e.chain schemas                    # the request contract and the stage list
python -m astra_e2e.chain init   --run <new directory> --request request.json
python -m astra_e2e.chain next   --run <run>         # what to read and what to run now
python -m astra_e2e.chain advance --run <run> [--delivery <record>] [--execute]
python -m astra_e2e.chain status --run <run>
```

[`configs/workflow/chain.request.example.json`](../../configs/workflow/chain.request.example.json) shows a request: the task in words, the
observation files, the delivery goal, and what policy development needs (site file, task id,
environment factory, policy I/O, campaign name). `station.setup` is optional: the station author's interaction setup
for the factory (robot, base, target, thresholds), needed when the delivered twin states no robot placement.
[Worked factories for both routes](../../real2sim/examples/station/README.md).

The chain matches every requested observation to the delivered scene by SHA-256, so a renamed or copied capture is accepted.
To reuse an older scene intentionally, add `"reuse_delivery": {"sha256": "<SHA-256 of the delivery JSON>", "reason": "<why this scene applies>"}`
to a new request. The stage record and handover preserve the reason and any unmatched observations. This also supports historical
deliveries that predate capture provenance.

| Stage | Done when | `--execute` does |
|---|---|---|
| `observations` | at `init`: every observation file exists and is hashed | — |
| `real2sim` | `advance --delivery <record>`: the record verifies (every pinned file re-hashed), matches the module plan (or the explicitly pinned legacy route) and requested delivery level | never: building the twin is the agent's work in `real2sim/` |
| `station_task` | the task spec names this delivery record, the coordinator's sealed files exist, the binding pins the record | writes the task spec from the record and binds the task |
| `policy_development` | the campaign runs on this binding and froze a deliverable whose snapshot matches its hash | creates the campaign; never runs its decision loop |
| `sealed_evaluation` | the campaign's sealed result exists | runs it, once |
| `real_preparation` | `real_preparation.json` is written | hands the frozen snapshot to the task's real-robot chain if the task names one, then writes the package |

Two steps are open-ended and stay with the agent: building the twin, and the campaign's decision loop
(`python -m astra_station run --state <state>`). The chain prints both commands in `next`.

## What the chain refuses

- A twin that changed after it was accepted, at any later stage (the record, the binding and the station each re-hash it).
- A delivery record from another route than the request names, or one that delivers a single object: a station needs
  a scene with a robot interaction setup.
- Inventing the coordinator's sealed files (start partitions, evaluation token). They are prepared outside the
  campaign's reach.
- A campaign on another binding, or a frozen snapshot that differs from the hash the campaign recorded.
- A second sealed evaluation, or overwriting any record of the run.

## What the last stage states

`real_preparation.json` holds the frozen deliverable by hash, its policy I/O, the simulation evidence (twin, binding,
development and sealed results) and what is open before the robot moves: deployment I/O qualification, the cell's
limits and an operator, plus anything the twin's record does not state (a robot placed in the scene, physics
calibrated against real motion). Its scope is `simulation_twin` and `real_validation` is `pending`. The chain claims
nothing about the real robot; see [transfer rules](transfer.md) and [the delivery package](delivery_package.md).

## Relation to the oversight controller

The chain is the route through the two parts. [The oversight controller](README.md) and
[the unified agent tools](unified_harness.md) remain for projects that need module-level requirement tracking,
budgets and feedback loops across reconstruction, data collection and training on a reconstruction-bound policy path
(`unified/`). They use the same twins and can run alongside a chain.
