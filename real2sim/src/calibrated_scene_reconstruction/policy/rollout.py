"""Task-independent policy construction and rollout over an injected environment."""
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
import json


class PolicyEnvironment(Protocol):
    def observe(self, step: int, runner) -> tuple: ...
    def step(self, action) -> None: ...
    def measure(self) -> Any: ...
    def succeeded(self, outcome) -> bool: ...


@dataclass(frozen=True)
class Rollout:
    steps: int
    outcome: Any


def run_rollout(runner, environment: PolicyEnvironment, max_steps: int, check_every: int):
    outcome = None
    for step in range(max_steps):
        images, state, goal, extras = environment.observe(step, runner)
        action = runner(images, state, goal, **extras)
        environment.step(action)
        if (step + 1) % check_every == 0 or step == max_steps - 1:
            outcome = environment.measure()
            if environment.succeeded(outcome):
                break
    return Rollout(step + 1, outcome)


def make_runner(args):
    from .runners import PolicyRunner, ServedDPRunner, ServedOpenpiRunner

    ck = Path(args.policy)
    tcfg_f = ck.parent / "train_config.json"
    tcfg = json.loads(tcfg_f.read_text()) if tcfg_f.exists() else {}
    if args.dp_server:
        runner = ServedDPRunner(args.dp_server, args.policy_action_steps)
        use_goal = False
    elif args.openpi_server:
        runner = ServedOpenpiRunner(args.openpi_server, args.policy_action_steps)
        if runner.fps != args.fps:
            raise SystemExit(f"--openpi-server: the policy's dataset is at {runner.fps} Hz, this run at {args.fps}")
        use_goal = False
    else:
        runner = PolicyRunner(ck, image_size=tcfg.get("image_size"))
        if args.policy_action_steps:  # the action queue is rebuilt from it at every reset
            runner.policy.config.n_action_steps = args.policy_action_steps
        if args.policy_inference_steps:
            runner.policy.diffusion.num_inference_steps = args.policy_inference_steps
        use_goal = "observation.environment_state" in runner.policy.config.input_features
    return runner, use_goal
