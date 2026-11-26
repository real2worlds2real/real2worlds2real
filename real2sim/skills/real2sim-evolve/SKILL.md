---
name: real2sim-evolve
description: Extend real2sim tools, harness or skills for unsupported reconstruction tasks and retain reusable procedures for new mechanisms or settings after validation and regression. Use when local refinement reveals a capability gap or yields reusable experience.
---

# Real2sim capability and experience evolution

Two outputs are useful: a new implementation/tool, and knowledge of how to
handle a mechanism or setting next time. A working implementation alone does
not preserve its assumptions, failed approaches or validation procedure.
Scope ends at reviewed simulated-world delivery, including task physics and
cousins. Continue to use the existing phase skills and native scene gates.

## Recognize the gap and retrieve experience

Read the task and current source before diagnosing a missing capability.
Separate observation ambiguity, representation limits, parameter failure,
backend limitations, missing tool coverage and missing procedural knowledge.
Ordinary local parameter refinement does not need an evolution record.
Do use this workflow for a new mechanism/setting or a shared pipeline upgrade.

First read the matching [experience modules](../twin-model/references/experience/index.md)
and run `recall` with meaningful
context filters using the [ledger interface](../../docs/agent/evolution.md).
Use mechanism/setting labels rather than one asset name: `rigid_insertion`,
for example, can cover both shafts and pegs. Record backend/version, material
regime and observation conditions when they limit applicability. Partial exact
filters help discover related work; matching a label does not prove transfer.

The index composes vessel, insertion and gear methods and supplies a
[module template](../twin-model/references/experience/module-template.md).
Extend the matching module for a new diagnostic/setting; add a module for a
distinct reusable method. Register it in that index instead of growing the
phase skill with another case narrative. These references remain inside the
existing skills; a separate skill is only needed for a distinct workflow.
Use the stable module id as `context.experience_module` in new ledger records.
This is an existing string context field, not a new tool or schema requirement.
For composite work record the primary changed module and document the others;
bind every modified module/index file in the validation inputs.

For a shared module, name its semantic operations in `required_capabilities` and cite their native outputs.
A procedure observed on another capture or implementation still needs the current backend and task checks.

## Develop and test

1. Describe the gap, competing hypotheses, expected result and acceptance
   criteria. Preserve the initial failure and baseline. Use an isolated
   experiment/session; do not mutate a frozen delivery.
2. Research unfamiliar methods or unexplained failures using available primary
   papers, official docs and original implementations. Save sources, versions,
   applicability and rejected alternatives. Literature is a hypothesis source,
   not validation of the installed backend.
3. Author a minimal candidate. Keep one-task scripts local; extract reusable
   geometry, domain logic and agent orchestration to their existing layers.
   New shared tools need actual implementation, contracts and discovery wiring.
   For a new setting, expose relevant configuration and supported ranges rather
   than embedding the first case's numbers as universal defaults.
4. Before shared integration, freeze a `propose` record with baseline evidence,
   implementation files and planned regression/integration commands. Prototype
   history can predate this record. If validation requirements change, create
   a new proposal referencing the old one; do not rewrite its plan.
5. Run each planned `check`. Pin implementation, tests, configuration and all
   relevant transitive inputs; list produced evidence in `artifacts`. Include
   layer checks for shared code, protocol tests for tools, and skill validation
   plus a realistic usage scenario for instruction changes. A new skill scenario
   may be reviewed by the same external agent; multiple agents are not required.
6. Compare old behavior and the new case under unchanged acceptance conditions.
   Include failed/mismatched controls for functional claims. Set
   `native_required: true` for new or changed contact, collision, material or
   simulator behavior, and run the actual native checks. CPU fixtures and
   animations cannot establish physical success. Inspect rendered/motion/log
   evidence before writing an adoption review.

## Preserve what was learned

Use the module's common sections to record the mechanism/setting, observable
cues, representation choice, parameter dependencies, procedure, checks/failure
diagnostics, implementation/evidence and reuse limits. Distinguish inferred internal structure from observed
structure. Keep exact source dimensions/poses and case-specific settings in the
case record, not the reusable recipe. For independent reconstruction, never use
old fitted assets, numerical ground truth or old approvals as new evidence.


Update the intended skill/reference/config interface before final checks so its
exact content is tested. Use `promote` only after all required checks are current
and an agent has reviewed semantics and negative results. Its entrypoints must
be part of the checked implementation. Adoption is a scoped procedural finding,
not scene delivery approval or proof of universal generalization.

On later tasks, retrieve the procedure, verify applicability, adapt from the
current source and rerun affected native/scene checks. Save counterexamples as
new candidate records and refine the procedure. Modified implementation or
evidence makes a previous adoption stale. If native evidence is unavailable,
retain a candidate with the missing experiment instead of promoting it.
