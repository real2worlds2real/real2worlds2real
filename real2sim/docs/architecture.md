# Code ownership and execution architecture

This document describes current ownership and the dependency target. The prior
proposal is preserved in `archive/pre_oversight/docs__architecture.md`; it is not
the execution manual. Use the repository audit tool for current file counts and
dependency findings rather than historical counts.

## Domains

| Domain | Responsibility |
|---|---|
| `agent.harness` | Unified tools, current world binding, supervised generic policy adapters and cross-task release/rollback |
| `agent.oversight` | Whole-chain contract, module plans, execution/evidence, feedback and knowledge; no embedded LLM or robot operation |
| `agent` | JSON/CLI/MCP transport, session locks, provenance and explicit stage composition |
| `scene` | Capture, metric scale, object/support fitting, backgrounds, cousins, appearance, native review and freeze |
| `assets` | Mesh/material/representation contracts and reusable geometry measurements |
| `robots` | Specifications, articulated CAD, pure kinematics, task frames and viewpoints |
| `grasping` | Grasp synthesis, trajectory and placement proposals |
| `physics` | Real replay/video observations and physical identification |
| `tasks` | Task environments, finite-control protocols, outcomes and policy-environment adapters |
| `datagen` | Recording, dataset export, demo retarget proposals and videos |
| `policy` | Training/model adapters, serving clients and generic rollout |
| `sim` | Native Isaac/PhysX and Newton runtime implementation |
| `core` | Shared domain-neutral paths, storage, media, geometry and numerical adapters |
| `evals` | Isolated post-freeze development geometry evaluation |
| `rl` | Reserved namespace; no general RL trainer |

The integrated `policy_development/` package preserves the existing Astra
learning/reward/experience implementations from main in isolated child processes.
The frontend's `rl` namespace remains unimplemented; this is distinct from the
legacy task-specific RL backends and the new world-bound vector BC trainer.
See [the unified harness](../../end_to_end/docs/unified_harness.md) for admission and evidence
contracts. A registered legacy method does not imply a new scene is supported.

Apps start the backend before runtime-sensitive imports and compose prepared
inputs. Tools supervise jobs. Shell pipelines remain native entry points; the
oversight contract determines their role in the entire experiment.

## Dependencies

Core imports no domain. Assets and robots consume core. Scene, grasping, physics,
datagen, policy and tasks may consume core/assets/robots; tasks may use sim. Sim
consumes core/assets/robots. Agent composes domains without native backend startup.
Apps perform runtime composition. Pipeline domains do not import `evals` or
retired `attic` implementations as live logic.

Keep domain dependencies acyclic.

Reusable geometry has been separated from backend placement: cloth clearance,
self-contact, local shell checks and signed surface distance live in assets;
state visibility lives in core geometry; cuDSS lives in core; articulated CAD
computation lives in robots. Identity-preserving aliases retain old imports.
Fingerprint lists name canonical implementation files rather than only wrappers.

## Boundary contracts

- Scene produces reviewed geometry, poses, materials, cameras and native-ready
  descriptions; runtime composition consumes prepared inputs.
- Tasks expose reset, observations, action application, outcomes and success to
  the generic policy loop. Action decoding must match the deployed controller.
- Recording emits command/state/camera evidence. Export preserves timestamps,
  label meaning and dataset lineage; training owns normalization and model state.
- Oversight consumes current artifacts and native gates. It adds task requirements
  without rewriting local acceptance or inventing semantic approval.
- Knowledge stores scoped procedures and validation evidence, not fitted assets
  for reuse as independent source observations.

## Entrypoints and migration

Canonical paths include `pipeline/twin.sh`, `pipeline/environment.sh`,
`apps/datagen/episode.py`, `tools/datagen/collect_place.py`, `tools/policy/`,
`physics/replay`, `physics/video` and `agent/oversight`. Old recon/real2sim/vidphys,
demo/export, root apps/tools and renamed modules retain compatibility entries.
The machine map is `src/calibrated_scene_reconstruction/compatibility.json`; see [compatibility](compatibility.md).

Preserve local round histories, freezes, oversight history and dataset/checkpoint
lineage. A frozen contract change needs a derivative session/run.
