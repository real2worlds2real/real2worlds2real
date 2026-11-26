# Agent tools

`python -m calibrated_scene_reconstruction.agent.tool schemas` returns the shared JSON contracts for stage, object, support and physics tools, review recording, status and reports. `python -m calibrated_scene_reconstruction.agent.mcp_server` exposes the same contracts over MCP stdio; MCP additionally requires an output directory.

Stage execution reports execution and current gate state separately. Numerical success is not an agent's visual approval. Reviews append new rounds; accepted object reviews bind the current source, mesh, fit round and observation hashes. S6 requires all applicable preceding stages and current object rounds to pass.

Each tool run owns a session lock and an exclusive run directory with input fingerprints and append-only events. Agent-authored code belongs in `configs/<session>/agent/`; evidence, round snapshots and exports belong in sessions. No tool reads development ground truth to judge a reconstruction round.

The three phase skills are [capture](../../skills/twin-capture/SKILL.md), [model](../../skills/twin-model/SKILL.md) and [delivery](../../skills/twin-deliver/SKILL.md). Independent reconstructions read raw captures, legal calibration and current instructions; old notes, fitted models and pose caches are excluded.

The model skill reads composable [experience modules](../../skills/twin-model/references/experience/index.md)
for vessels, insertion and gears. These are references within
the existing skill, with a common template for subsequent experience; they are
not additional tool registrations or per-object skill packages.

[Real2sim evolution](../../skills/real2sim-evolve/SKILL.md) is the cross-phase
skill for capability gaps, new mechanisms/settings and harness improvements.
Its [CLI ledger](evolution.md) preserves candidate experiments, executes planned
checks, records scoped adoption and retrieves reusable experience. It is separate
from scene tool execution and cannot approve a scene or rewrite a skill by itself.

The object-only CLI and MCP interface retain their old tool names through compatibility entries. The full workflow interface uses `object_` prefixes. Model-fitting and iteration strategies remain in scene, while JSON transport and orchestration belong to agent.
