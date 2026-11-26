"""Policy-stage scheduler without robot or simulator I/O.

Construct one StageGraph per episode with a versioned JSON spec, a mapping of
policy IDs to inference runners, and verified observation/action/signal metadata.
Use bindings={stage_id: {"id": policy_id, "kind": method}} to select an allowed
method for one stage. A residual binding also names its base_id. Its runner
returns the composed action in the task contract and identifies its base hash.

Each tick receives an integer frame ID, observations keyed by contract, and
signal readings tagged with the same frame ID. Apply only Step.action from an
"action" event. Clear queued commands and hold on all other events. A host
performs a requested stage reset, then supplies the matching episode-local
reset token on fresh frames while the retry guard is confirmed. "succeeded"
and "failed" are sticky episode outcomes. Mode "trace" is for synthetic
scheduler tests; simulation requires independently qualified adapters,
signals, action bounds and frozen policies.
"""
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from operator import eq, ge, le
import re


OPS = {"eq": eq, "ge": ge, "le": le}
METHODS = {"act", "dp", "rl", "residual_rl", "mpc", "servo", "ridge_bc", "mlp_bc"}


@dataclass(frozen=True)
class Action:
    contract: str
    values: tuple


@dataclass(frozen=True)
class Reading:
    tick: int
    value: object


@dataclass(frozen=True)
class Step:
    event: str
    stage_id: str
    policy_id: str
    action: Action = None
    reason: str = ""
    reset_token: str = ""


def _positive(n):
    return type(n) is int and n > 0


def _require(ok, reason):
    if not ok:
        raise ValueError(reason)


class StageGraph:
    """One instance per episode. runner.act(obs) returns Action; reset() clears all policy state."""

    def __init__(self, spec, controllers, runtime, *, mode="simulation", bindings=None):
        spec, runtime = deepcopy(spec), deepcopy(runtime)
        self.mode = mode
        _require(mode in ("simulation", "trace") and runtime["mode"] == mode, "mode mismatch")
        if mode == "simulation":
            _require(spec["status"] == "verified_for_simulation", "stage specification has no simulation qualification")
            _require(re.fullmatch("[0-9a-f]{64}", runtime.get("evidence_sha256", "")), "missing environment evidence")
        self.stages = {s["stage_id"]: s for s in spec["stages"]}
        _require(len(self.stages) == len(spec["stages"]), "duplicate stages")
        self.bindings = {sid: dict(stage["policy"]) for sid, stage in self.stages.items()}
        for sid, binding in (bindings or {}).items():
            _require(sid in self.stages, "unknown stage override")
            self.bindings[sid] = dict(binding)
        self.controllers = dict(controllers)
        self.runtime = runtime
        self.signals = spec["signals"]
        self.action_contract = spec["action_contract"]
        action_meta = runtime["actions"][self.action_contract]
        self.width = action_meta["width"]
        _require(_positive(self.width) and self.width == spec["action_width"], "action widths differ")
        self.bounds = tuple(zip(action_meta["lower"],action_meta["upper"]))
        _require(len(action_meta["lower"]) == len(action_meta["upper"]) == self.width
                 and all(isfinite(lo) and isfinite(hi) and lo <= hi for lo,hi in self.bounds),
                 "invalid action bounds")
        self.invariants = spec.get("invariants", [])
        for condition in self.invariants:
            self._validate_signal(condition)
        self.sid = spec["entry_stage"]
        seen = set()
        cursor = self.sid
        while cursor is not None:
            _require(cursor in self.stages and cursor not in seen, "invalid or unbounded stage transition")
            seen.add(cursor)
            stage = self.stages[cursor]
            binding = self.bindings[cursor]
            limits = stage["limits"]
            _require(all(_positive(limits[k]) for k in ("enter_ticks", "run_ticks", "reset_ticks", "attempts")), "invalid stage limits")
            _require(binding["kind"] in METHODS and binding["kind"] in stage["training"]["methods"], "unsupported stage method")
            self._check_controller(cursor)
            _require(stage["action_contract"] == self.action_contract, "stage needs a verified action adapter")
            for key, table in ((stage["observation_contract"], "observations"), (self.action_contract, "actions")):
                capability = runtime[table][key]
                _require(capability.get("verified") is True, "unverified runtime contract: " + key)
                if mode == "simulation":
                    _require(re.fullmatch("[0-9a-f]{64}", capability.get("evidence_sha256","")), "missing adapter evidence: " + key)
            for guard in (stage["enter"], stage["exit"], stage["retry_guard"]):
                _require(_positive(guard["ticks"]) and bool(guard["all"]), "empty or invalid guard")
                for cond in guard["all"]:
                    self._validate_signal(cond)
            for condition in stage.get("invariants", []):
                self._validate_signal(condition)
            cursor = stage["on_success"]
        _require(seen == self.stages.keys(), "unreachable stage")
        self.last_tick = None
        self.phase = "enter"
        self.since = None
        self.streak = 0
        self.attempt = 1
        self.token = ""
        self.terminal = None

    def _validate_signal(self, condition):
        key, op, target = condition
        expected = self.signals[key]
        live = self.runtime["signals"][key]
        _require(op in OPS and live.get("verified") is True and live.get("privileged") is False,
                 "signal unavailable or privileged: " + key)
        _require(live.get("kind") == expected["kind"], "signal type mismatch: " + key)
        if expected["kind"] == "bool":
            _require(op == "eq" and type(target) is bool, "invalid boolean guard")
        else:
            _require(expected["kind"] == "number" and type(target) in (int,float) and isfinite(target), "invalid numeric guard")
        if self.mode == "simulation":
            _require(expected["availability"] == "verified" and live.get("source") != "synthetic_trace"
                     and live.get("source") == expected["source"]
                     and re.fullmatch("[0-9a-f]{64}",live.get("evidence_sha256","")),
                     "missing qualified source: " + key)
        else:
            _require(live.get("source") == "synthetic_trace", "trace-mode signals require explicit trace provenance")

    def _check_controller(self, sid):
        s = self.stages[sid]
        binding = self.bindings[sid]
        runner = self.controllers[binding["id"]]
        _require(runner.kind == binding["kind"] and runner.observation_contract == s["observation_contract"]
                 and runner.action_contract == s["action_contract"], "policy/adapter contract mismatch: " + sid)
        if self.mode == "simulation":
            _require(re.fullmatch("[0-9a-f]{64}", getattr(runner,"artifact_sha256","")),
                     "missing frozen policy artifact: " + sid)
        if runner.kind == "residual_rl":
            base_id = binding.get("base_id")
            _require(base_id and base_id in self.controllers and getattr(runner,"base_id",None) == base_id,
                     "residual requires its registered base")
            base = self.controllers[base_id]
            _require(base is not runner and base.kind != "residual_rl"
                     and base.observation_contract == runner.observation_contract
                     and base.action_contract == runner.action_contract, "base/residual adapters differ")
            if self.mode == "simulation":
                _require(getattr(runner,"base_sha256",None) == getattr(base,"artifact_sha256","")
                         and re.fullmatch("[0-9a-f]{64}",base.artifact_sha256), "frozen residual base mismatch")

    def _reset_policy(self):
        binding = self.bindings[self.sid]
        self.controllers[binding["id"]].reset()
        if binding["kind"] == "residual_rl":
            self.controllers[binding["base_id"]].reset()

    def _step(self, event, *, action=None, reason=""):
        return Step(event, self.sid, self.bindings[self.sid]["id"], action, reason, self.token)

    def _terminal(self, event, reason):
        if self.phase == "run":
            try:
                self._reset_policy()
            except Exception as error:
                event, reason = "failed", reason + "; policy reset failed: " + str(error)
        self.terminal = self._step(event, reason=reason)
        self.phase = "terminal"
        return self.terminal

    def _satisfied(self, conditions, signals):
        satisfied = True
        for name, op, target in conditions:
            sample = signals.get(name)
            _require(isinstance(sample, Reading) and sample.tick == self.last_tick, "stale or missing signal: " + name)
            value = sample.value
            valid = type(value) is bool if self.signals[name]["kind"] == "bool" else (
                type(value) in (int,float) and isfinite(value))
            if not valid:
                raise ValueError("missing or invalid observation of " + name)
            satisfied = OPS[op](value, target) and satisfied
        return satisfied

    def _guard(self, guard, signals):
        satisfied = self._satisfied(guard["all"], signals)
        self.streak = self.streak + 1 if satisfied else 0
        return self.streak >= guard["ticks"]

    def _expired(self, tick, ticks):
        return tick - self.since >= ticks

    def _timeout(self, reason):
        if self.attempt >= self.stages[self.sid]["limits"]["attempts"]:
            return self._terminal("failed", reason + "; attempts exhausted")
        self._reset_policy()
        self.phase, self.since, self.streak = "reset", None, 0
        self.token = f"{self.sid}/{self.attempt}"
        return self._step("reset_requested", reason=reason)

    def tick(self, tick, observations, signals, *, reset_ack=None):
        """Provide one fresh frame per tick. Only 'action' may be executed by the host."""
        if self.terminal is not None:
            return self.terminal
        if type(tick) is not int or tick < 0 or (self.last_tick is not None and tick <= self.last_tick):
            return self._terminal("failed", "non-monotonic observation tick")
        if self.last_tick is not None and tick != self.last_tick + 1:
            self.streak = 0
        self.last_tick = tick
        if self.since is None:
            self.since = tick
        stage = self.stages[self.sid]
        limits = stage["limits"]
        try:
            if self.phase == "reset":
                if self._expired(tick,limits["reset_ticks"]):
                    return self._terminal("failed","reset acknowledgement timed out")
                if reset_ack != self.token:
                    self.streak = 0
                    return self._step("await_reset")
                if not self._guard(stage["retry_guard"], signals):
                    return self._step("await_reset")
                self.attempt += 1
                self.phase, self.since, self.streak, self.token = "enter", None, 0, ""
                return self._step("retry_ready")
            if self.phase == "enter":
                if self._expired(tick,limits["enter_ticks"]):
                    return self._timeout("entry guard timeout")
                if not self._guard(stage["enter"], signals):
                    return self._step("await_entry")
                self._check_controller(self.sid)
                self._reset_policy()
                self.phase, self.since, self.streak = "run", tick, 0
            _require(self._satisfied(self.invariants + stage.get("invariants", []), signals), "stage invariant lost")
            if tick-self.since > limits["run_ticks"]:
                return self._timeout("stage execution timeout")
            if self._guard(stage["exit"], signals):
                next_sid = stage["on_success"]
                if next_sid is None:
                    return self._terminal("succeeded","all stage completion conditions verified")
                self._reset_policy()
                transition = self._step("transition",reason=next_sid)
                self.sid, self.phase, self.since, self.streak, self.attempt = next_sid, "enter", None, 0, 1
                return transition
            if self._expired(tick,limits["run_ticks"]):
                return self._timeout("stage execution timeout")
            if self.streak:
                return self._step("confirming_exit")
            key = stage["observation_contract"]
            if key not in observations or observations[key] is None:
                raise ValueError("required observation missing: " + key)
            out = self.controllers[self.bindings[self.sid]["id"]].act(observations[key])
            if (not isinstance(out, Action) or out.contract != self.action_contract or len(out.values) != self.width
                    or any(type(x) not in (int,float) or not lo <= x <= hi for x,(lo,hi) in zip(out.values,self.bounds))):
                raise ValueError("invalid controller action")
            return self._step("action",action=out)
        except Exception as error:
            return self._terminal("failed",str(error))
