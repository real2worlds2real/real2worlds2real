# Task environments

| Task | Environment and success | Recording and evaluation |
|---|---|---|
| Rigid grasp/place | `tasks/rigid/runtime.py` owns the environment; `succeeded` retains collection outcomes. `tasks/rigid/success.py` retains the separate plate evaluation criterion. `task_checks.py` provides measurements for session-authored checks. | `apps/datagen/episode.py`; recorder in datagen, generic runner in policy, rigid adapter in `tasks/rigid/evaluate.py`. |
| Garment folding | `tasks/lehome/appearance.py` supplies the challenge environment's look; the challenge retains task dynamics and success. `tasks/lehome/policy.py` accepts an injected policy client. | `apps/tasks/lehome_run.py` forwards challenge modules. LeHome collection/evaluation helpers live under `tools/tasks/`. |
| Rope manipulation | `tasks/rope/letters.py`, `mpc.py` and `appearance.py` define goals, planning and appearance. Newton solver/control implementation lives in `sim/newton/rope.py`. | `apps/tasks/rope.py`, `rope_letter.py`; Newton recording/video in `sim/newton/`; planner evaluation helpers under `tools/tasks/`. |

Backend imports occur in the corresponding runtime environment. The LeHome launcher has lightweight help before external path setup. Newton applications require their upstream environment for backend imports and parser construction; `tools/runtime/run_newton.sh --help` describes the container launcher without starting it.

The LeHome app calls `register_policy(client_factory)` after loading its external stack. The task adapter does not import a policy implementation. Old `deformable` modules remain forwarding paths; the old registered-policy adapter preserves automatic registration for existing consumers.

Scene gates read `qa/task_check.json`. They do not import task environments. The app that runs a session's `agent/task_check.py` composes the runtime and writes that result before the gate checks it.
