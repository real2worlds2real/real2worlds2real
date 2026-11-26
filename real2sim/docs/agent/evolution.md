# Real2sim capability evolution

The external agent can extend the harness and learn reusable procedures for new
mechanisms/settings using [real2sim-evolve](../../skills/real2sim-evolve/SKILL.md).
The workflow is **gap → isolated candidate → regression and integration checks
→ semantic review → scoped adoption → retrieval and validation on a new task**.
It borrows research and procedural-memory ideas from the separate `end-to-end`
branch, without importing its policy/data/real-deployment controller.

Reusable object/mechanism knowledge lives in the
[experience modules](../../skills/twin-model/references/experience/index.md),
read on demand by the existing phase skills. Each module has a stable id and
the same observation/representation/procedure/checks/evidence/reuse sections.
New findings update a matching module or add one using the indexed template.
New records conventionally set `context.experience_module` to that id; the CLI
already supports arbitrary string context keys. A context file such as
`{"experience_module":"rigid-insertion"}` retrieves that module's local records.
Older mechanism-only records remain valid and use their original filters.
Include changed module/index files in `implementation`, and module entrypoints
in the adoption review, so checked instruction versions are bound to evidence.

`calibrated_scene_reconstruction.agent.evolution` implements a standalone Python/JSON ledger and check
runner. It does not embed an LLM, write skills automatically, modify native
gates, commit code, or grant scene approval. The existing MCP tool registry is
unchanged; invoke this outer workflow through its CLI.

## Records and lifecycle

```text
outputs/real2sim_evolution/<id>/
  proposal.json               immutable scope, baseline hashes and validation plan
  checks/000001-<check-id>/
    request.json              command and input hashes before execution
    command.log               combined stdout/stderr
    result.json               outcome, artifacts and provenance; absent if interrupted
  adoptions/000001.json       scoped agent review and exact checks it accepted
```

Use the current checkout on `PYTHONPATH`; the `objects` extra supplies JSON
Schema validation. No simulator import or GPU is needed for the ledger itself.
Commands below are real; request/context files must be authored for the task:

```sh
PYTHONPATH=src envs/recon/bin/python -m calibrated_scene_reconstruction.agent.evolution schemas
PYTHONPATH=src envs/recon/bin/python -m calibrated_scene_reconstruction.agent.evolution propose outputs/real2sim_evolution /absolute/path/proposal.json
PYTHONPATH=src envs/recon/bin/python -m calibrated_scene_reconstruction.agent.evolution check outputs/real2sim_evolution/<id> <check-id>
PYTHONPATH=src envs/recon/bin/python -m calibrated_scene_reconstruction.agent.evolution promote outputs/real2sim_evolution/<id> /absolute/path/review.json
PYTHONPATH=src envs/recon/bin/python -m calibrated_scene_reconstruction.agent.evolution status outputs/real2sim_evolution/<id>
PYTHONPATH=src envs/recon/bin/python -m calibrated_scene_reconstruction.agent.evolution recall outputs/real2sim_evolution /absolute/path/context.json
```

`schemas` is the authoritative request contract. `propose` requires:

| Field | Meaning |
| --- | --- |
| `id`, `module`, `change_kind` | Unique experiment id; reconstruction module; tool, skill, setting, procedure or framework |
| `context` | Nonempty string mapping of applicable mechanism, backend/version, material regime or observation setting |
| `gap`, `hypothesis`, `acceptance` | Failure/capability need, proposed explanation and explicit success criteria |
| `baseline` | Absolute paths to existing baseline/failure evidence; hashes are historical, not required to stay current |
| `implementation` | Absolute paths of all changed/reused implementation, configuration and skill files to validate; new files may be created after proposing |
| `native_required` | Whether native simulator evidence is necessary for this claim |
| `checks` | Named planned commands: `id`, `kind`, `argv`, `cwd`, `inputs`, `artifacts`, `timeout_seconds` |

Every proposal needs regression and integration checks; native claims also need
a native check. Use explicit argv arrays and absolute cwd/input/artifact paths.
`inputs` includes transitive implementation, fixtures and settings relevant to
the check. `artifacts` names produced reports/images/traces for integrity checks
and agent inspection; it may be empty for a CPU test whose log is sufficient.
The runner always fingerprints all declared `implementation` files as well.
Commands run without a shell and have a timeout; exit failure or changed inputs
during execution fail the check. Check commands should be foreground commands;
the runner is not a job scheduler or sandbox. Select the existing shared Isaac
launcher for native work and stay within the task's authorized resources.

`promote` requires a review with `summary`, `applicability`, `limitations`,
`negative_results`, `reuse_procedure` and `entrypoints`. Entry points must be
checked implementation files, with discovery links in the phase workflow or
experience index. Preserve original failures and explain their resolution.
Latest attempts of **all** planned checks must pass with current input, log and
artifact hashes. The review is appended, never substituted for earlier history.

| State | Use in later tasks |
| --- | --- |
| `candidate` | Hypothesis or partial result; not an adopted procedure |
| `adopted` | Current checks and agent review support the stated applicability only |
| `stale` | Inputs/evidence changed or a new check attempt needs review; revalidate before reuse |

`recall` accepts a nonempty subset of exact context filters, such as
`{"mechanism":"rigid_insertion"}`. It returns candidate/stale entries too so
negative evidence remains discoverable. Inspect the status and limitations;
broader context matching is not a generalization certificate. New tasks must
produce their own geometry, physics and S0–S6/OBJ/SUP evidence.

## What the agent still owns

The runner checks execution and provenance, not the semantic truth of a command
label, the completeness of declared dependencies or the physical validity of a
render. The agent chooses adequate tests, compares the baseline, inspects actual
results, and declares native checks when required. A zero exit code alone is not
acceptance. Do not weaken gates or source fidelity to turn a failure into a pass.

Keep concise reusable procedures in tracked skills/references or configuration
interfaces; keep case numbers and heavy artifacts with the experiment. Local
ledger records are not automatically shipped with a clone. A clone can read the
tracked recipe but must obtain the evidence or validate anew. Do not describe
a one-case finding as broadly validated, or reuse old fits in independent runs.
Changed validation plans require a new proposal with the old record included
in baseline evidence. Frozen scene changes require derivative sessions.

Validation for this ledger:
