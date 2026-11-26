# Agent entry point

The objective of this repository is **real2sim2real**: observations of a real scene become a simulation, a policy is
developed in it, and the result is prepared for validation on the real robot. Simulation success is intermediate
evidence, never a claim of real-robot success. All maintained code, comments, instructions and documentation are
written in English.

## Read by scope

| The request is about | Go to |
|---|---|
| Building a twin, cousins, or calibrating physics from photos or video | [`real2sim/AGENTS.md`](real2sim/AGENTS.md) and [its skill index](real2sim/skills/README.md). Run that part's commands from `real2sim/`. |
| Developing a policy on a station (a hand-authored task or an accepted twin) | [`policy_development/README.md`](policy_development/README.md) and [the station harness](policy_development/docs/station_harness.md) |
| The whole way: from a video to a policy prepared for the robot | [`end_to_end/skills/end-to-end/SKILL.md`](end_to_end/skills/end-to-end/SKILL.md) and [the chain](end_to_end/docs/chain.md) |
| Collection, training or evaluation on a reconstruction-bound policy path | [`end_to_end/skills/data-policy-loop/SKILL.md`](end_to_end/skills/data-policy-loop/SKILL.md) |
| An object, mechanism, task, policy or backend the tools do not support | [`end_to_end/skills/research-transfer/SKILL.md`](end_to_end/skills/research-transfer/SKILL.md); for reconstruction capabilities, [`real2sim/skills/real2sim-evolve/SKILL.md`](real2sim/skills/real2sim-evolve/SKILL.md). Investigate and validate a capability; do not silently downgrade the task to an existing demonstration. |

`source env.sh` (repository root) puts the three source roots on `PYTHONPATH`: `real2sim/src` (packages `real2sim_harness`,
`calibrated_scene_reconstruction`), `policy_development/src` (`astra_station`), `end_to_end/src` (`astra_e2e`). An editable installation may
point at another worktree; do not relink a shared environment to select a checkout.

## Contract and decision ownership

User requirements define the objective. Resolve material ambiguity at intake: source, task, robot, policy I/O,
deployment conditions, precision, generalization, evaluation, compute and execution authority. Record the decisions
before jobs start. Afterwards make research and implementation choices autonomously inside that contract, without
repeatedly asking for routine decisions. If new evidence makes the contract impossible or requires new real-world
access or authorization, report the concrete blocker; never invent a user answer or mark the goal achieved.

No stage may silently weaken another's checks. The scene's gates, the station's sealed partitions and the single
sealed evaluation stay as they are; a top-level approval never replaces local evidence. Policy evaluation may improve
the simulation only through a stated causal hypothesis with controls that preserve physical and source consistency.
Do not alter geometry, friction, collision shapes, controllers or success definitions merely to raise a simulator
score. See [transfer rules](end_to_end/docs/transfer.md).

## Work and evidence

- Work in the requested branch or worktree. Preserve other worktrees and existing uncommitted changes.
- Keep the partition: Part 1 and Part 2 do not import each other, and neither imports `end_to_end`. What crosses
  between them is a file with a schema: the delivery record, the task binding.
- A record that was accepted, frozen or sealed is never overwritten. Create a derivative and a new run instead.
- Hashes establish provenance, not truth. Look at the actual images, motion, geometry and logs before writing a
  positive review.
- No physical robot operation and no external publication are part of this repository's tools. Prepare a
  human-operated hand-over and keep real validation pending.
- State which checks ran on a simulator, a GPU or a physical robot when reporting results.

These instructions do not require a particular model provider, or more than one agent.
