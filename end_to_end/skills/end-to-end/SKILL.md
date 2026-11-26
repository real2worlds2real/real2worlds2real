---
name: end-to-end
description: Plan and oversee a task-specific real2sim2real workflow from user intake through reconstruction, data collection, policy training, closed-loop simulation evaluation and human-operated real deployment handoff.
---

# End-to-end oversight

Own the user's whole objective, not a sequence of scripts. Read
[the system contract](../../docs/README.md),
[transfer rules](../../docs/transfer.md) and
[module contracts](../../docs/modules.md). The external agent performs
reasoning and research; `astra_e2e.oversight` provides durable execution and
evidence control. A ready module is work to perform, not an automatic pass.

The [unified harness](../../docs/unified_harness.md) exposes this
controller and the integrated Astra learning library through one JSON tool
surface. Discover registered backends, bind the current accepted native twin,
and use supervised bound policy jobs. A legacy tool's historical task success
does not establish compatibility with this source or world. The same orchestration
applies to any registered task; Peg is only a possible validation example.

## The two parts and the chain between them

The repository is partitioned as the work is: `real2sim/` builds and accepts the twin, `policy_development/` develops a
policy on it in simulation, and this layer joins them. For a request of the form "from this video, build the scene,
develop a policy for this task and prepare it for the robot", drive [the chain](../../docs/chain.md):

```sh
source env.sh
python -m astra_e2e.chain init --run <new directory> --request request.json
python -m astra_e2e.chain next --run <run>          # what to read and run for the current stage
python -m astra_e2e.chain advance --run <run> [--delivery <record>] [--execute]
```

For new requests omit `route` and follow [module selection, handoffs and experience](../../docs/reconstruction.md). Use one object revision per participant and one environment revision for the room. Retrieve skills by module and mechanism. Use the native receipt of each selected operation and recheck current sources before delivery.

Follow `next`: in the real2sim stage read that part's skills and build the twin there, then hand its delivery record to
the chain; in the policy-development stage run the campaign's decision loop there. The chain checks each part's own
records before it lets the next stage start, and its last stage only prepares a hand-over. Use the oversight
controller below when the project also needs module-level requirements, budgets and feedback across reconstruction,
data collection and training.

## Intake once, before dependent execution

Read the supplied sources and deployment requirements. Consolidate consequential
questions into the initial discussion: task and success/failure conditions;
embodiment and controller; actual observations/actions, dimensions, frames, units,
rates, timing and normalization; precision-critical features and uncertainty;
deployment variation; data/expert availability; model choices; evaluation splits
and targets; resources and authorized actions. Reuse answers already supplied.
Record any safe assumptions explicitly. Do not guess a missing physical dimension
or treat an example configuration as the user's intended contract.

Create an intake JSON using `schemas` and the example in
`configs/workflow/intake.example.json`. `intake` lists missing/inconsistent
requirements. Resolve all `unresolved` entries and semantic ambiguities before
`init`. Machine schema completeness is necessary, not proof of understanding.

## Customize and execute

1. Choose routes from actual capabilities and task constraints. Justify model,
   policy family, precision, collider representation, native physics checks,
   cousin design, randomization, expert, data coverage and feedback method.
   Include custom checks in each module's `requirements`; these add to mandatory
   checks. Do not substitute more cousins for missing task-contact fidelity.
2. Initialize an isolated run; inspect `plan.json` and `next`. Translate each
   module contract into the actual session/model/training/evaluation configs and
   a work packet. Pin implementation and configuration files as command inputs.
   Consult [the adapters](../../docs/adapters.md); verify flags from
   current code. Do not launch a lab-specific script on a different task blindly.
3. Execute the module's nested proposal/check/refine loop. Investigate numerical,
   semantic and native failures separately. Keep the local review and frozen
   evidence requirements. Collect a report only after inspecting the artifacts.
4. Submit the report through `review`. Execution `ok` is not acceptance. Respect
   current upstream hashes and preserve failed attempts. Continue ready modules
   independently without requesting routine approval again.
5. Diagnose validation failures by cause. Use targeted data, corrected expert
   labels, curriculum, model/training changes, DAgger, or a justified sim update.
   A `feedback` revision requires the transfer hypothesis, expected real benefit,
   anti-hack analysis, evidence and an ablation plan. It invalidates the affected
   descendants; it does not change the agreed success threshold.
6. Once validation and transfer checks pass, evaluate the sealed holdout once.
   A failure closes this experiment. Preserve it; any further development uses
   a new contract with a new untouched holdout, never a renamed copy of the old
   test. Do not use development-only geometry ground truth to alter a frozen
   independent reconstruction.
7. Package the policy, preprocessing, controller contract, assets, software,
   training lineage, evaluations and deployment gap register. State
   `simulation-qualified deployment candidate; real validation pending`.

## Exploration, learning and stopping

When a capability is absent or the familiar route fails for a new mechanism,
apply [research-transfer](../research-transfer/SKILL.md). Decompose the unknown,
actively search the web for related work, methods and primary references using
available tools within the agreed authority. Revisit that search during validation
when results expose a new gap or contradict the hypothesis. Record sources and
separate published claims from local evidence; test a minimal discriminating
prototype and integrate it behind the same module contract. A stub, animation,
synthetic fixture or successful isolated demo cannot qualify a full task.

Use `learn`/`recall` to retain scoped procedures, failures, counterexamples and
validation evidence. Reuse procedural knowledge, never old fitted assets as
independent source evidence. Skills are amended only after validation and a
regression check; candidate findings stay labelled as candidates.

Budget exhaustion is `unmet`, not success. Preserve the best candidate, blockers,
consumed resources and next discriminating experiment. Do not loosen a gate to
finish. Physical-robot access or new authority cannot be invented in pursuit of
autonomy. These are exceptional blockers, not reasons to ask about routine work.
