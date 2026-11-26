"""Durable outer-loop controller. Agent decisions require explicit evidence.

Commands run synchronously in owned process groups. Immutable history snapshots
retain failed attempts and invalidated downstream results. This is provenance
and scheduling, not a security sandbox or an autonomous reasoning model.
"""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from .contracts import MODULES, PACKET_SCHEMA, REPORT_SCHEMA, validate
from .evidence import (compare, dataset_manifest, digest, evaluate_episodes, file_hash, fingerprint,
                       native_twin, policy_io, unchanged)
from .planning import build_plan, descendants, feedback_targets


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


class OversightBusyError(ValueError):
    """The writer lock was unavailable before this operation reserved work."""


class Controller:
    def __init__(self, root):
        self.root = Path(root).resolve()

    @classmethod
    def create(cls, root, spec):
        plan = build_plan(spec)
        sources = fingerprint(spec["source"]["files"])
        for workdir in spec["authority"]["allowed_workdirs"]:
            if not Path(workdir).is_absolute() or not Path(workdir).is_dir():
                raise ValueError("Allowed workdirs must be existing absolute directories")
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=False)
        (root / "history").mkdir()
        (root / "attempts").mkdir()
        write_new(root / "plan.json", plan)
        state = {"plan_sha256": digest(plan), "sources": sources, "created": time.time(),
                 "cycle": 1, "jobs": 0, "holdout_consumed": False, "closed": False,
                 "modules": {m: {"status": "pending", "attempts": [], "accepted": None} for m in MODULES}}
        ctrl = cls(root)
        ctrl._save(state, "created")
        return ctrl

    @contextmanager
    def lock(self):
        with (self.root / ".lock").open("a") as f:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise OversightBusyError("Another oversight operation owns this run") from exc
            try:
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)

    def _load(self):
        previous = None
        state = None
        for path in sorted((self.root / "history").glob("*.json")):
            row = read(path)
            recorded = row.pop("sha256")
            if digest(row) != recorded or row["previous"] != previous:
                raise ValueError("Oversight history integrity failure")
            state, previous = row["state"], recorded
        if state is None:
            raise ValueError("Missing oversight history")
        plan = read(self.root / "plan.json")
        if digest(plan) != state["plan_sha256"]:
            raise ValueError("Plan changed; create a new run with explicit lineage")
        if not unchanged(state["sources"]):
            raise ValueError("Raw source changed; create a new source revision")
        return plan, state

    def _save(self, state, event, **details):
        files = sorted((self.root / "history").glob("*.json"))
        row = {"previous": read(files[-1])["sha256"] if files else None, "time": time.time(),
               "event": event, "details": details, "state": state}
        write_new(self.root / "history" / f"{len(files):06d}.json", {**row, "sha256": digest(row)})

    def _current(self, module, plan, state, seen=None):
        seen = {} if seen is None else seen
        if module in seen:
            return seen[module]
        seen[module] = False
        row = state["modules"][module]
        if row["status"] != "accepted" or not row["accepted"]:
            return False
        acceptance = row["accepted"]
        if not unchanged(acceptance["files"]):
            return False
        for dep in plan["modules"][module]["depends_on"]:
            if not self._current(dep, plan, state, seen):
                return False
            if state["modules"][dep]["accepted"]["id"] != acceptance["dependencies"][dep]:
                return False
        if module == "twin":
            try:
                native_twin(acceptance["native_session"])
            except (ValueError, OSError, KeyError):
                return False
        seen[module] = True
        return True

    def status(self):
        with self.lock():
            plan, state = self._load()
            seen = {}
            current = {m: self._current(m, plan, state, seen) for m in MODULES}
            ready = [m for m in MODULES if not current[m] and
                     state["modules"][m]["status"] in ("pending", "failed", "invalidated") and
                     all(current[d] for d in plan["modules"][m]["depends_on"])]
            exhausted = self._budget_reason(plan, state)
            return {"plan_sha256": state["plan_sha256"], "cycle": state["cycle"],
                    "jobs": state["jobs"], "complete": all(current.values()),
                    "completion_claim": plan["completion_claim"] if all(current.values()) else None,
                    "closed": state["closed"], "budget_stop": exhausted,
                    "needs_revision": [m for m in MODULES if state["modules"][m]["status"] == "accepted" and not current[m]],
                    "ready": [] if exhausted or state["closed"] else ready,
                    "modules": {m: {"state": state["modules"][m]["status"], "current": current[m],
                                    "attempts": len(state["modules"][m]["attempts"])} for m in MODULES}}

    def progress_snapshot(self):
        """Read completed journal records without taking the writer lock.

        This is monitoring, not an acceptance or scheduling decision. In
        particular it does not revalidate source files, native gates or current
        dependencies. A partially written newest record is left to its writer;
        corruption in a completed record is never silently ignored.
        """
        paths = sorted((self.root / "history").glob("*.json"))
        previous = latest = None
        incomplete_tail = False
        for index, path in enumerate(paths):
            try:
                row = read(path)
            except json.JSONDecodeError:
                if index != len(paths) - 1:
                    raise ValueError("Incomplete nonterminal oversight history record")
                incomplete_tail = True
                break
            recorded = row.pop("sha256")
            if digest(row) != recorded or row["previous"] != previous:
                raise ValueError("Oversight history integrity failure")
            latest, previous = row, recorded
        if latest is None:
            raise ValueError("No completed oversight history snapshot is available")
        state = latest["state"]
        return {"schema": "oversight_progress_snapshot_v1", "record_sha256": previous,
                "recorded_event": latest["event"], "recorded_time": latest["time"],
                "incomplete_tail": incomplete_tail, "cycle": state["cycle"], "jobs": state["jobs"],
                "closed": state["closed"], "holdout_consumed": state["holdout_consumed"],
                "scope": "Recorded progress only; source, gates and acceptance currency are not revalidated.",
                "modules": {name: {"state": item["status"], "attempts": len(item["attempts"]),
                                    "recorded_acceptance": bool(item["accepted"])}
                            for name, item in state["modules"].items()}}

    def _budget_reason(self, plan, state):
        budget = plan["contract"]["budget"]
        if time.time() - state["created"] >= budget["wall_seconds"]:
            return "wall_seconds"
        if state["jobs"] >= budget["max_jobs"]:
            return "max_jobs"
        return None

    def _ensure_ready(self, module, plan, state):
        if state["closed"]:
            raise ValueError("Run is closed; retain the result and start a new contracted experiment")
        reason = self._budget_reason(plan, state)
        if reason:
            raise ValueError("Execution budget exhausted: " + reason)
        row = state["modules"][module]
        if row["status"] not in ("pending", "failed", "invalidated"):
            raise ValueError("Module is not ready; review its attempt or record a feedback revision")
        if len(row["attempts"]) >= plan["contract"]["budget"]["max_attempts_per_module"]:
            raise ValueError("Module attempt budget exhausted; do not weaken acceptance")
        missing = [m for m in plan["modules"][module]["depends_on"] if not self._current(m, plan, state)]
        if missing:
            raise ValueError("Missing or stale dependencies: " + ", ".join(missing))
        if module == "holdout" and state["holdout_consumed"]:
            raise ValueError("Holdout has already been consumed")

    def run(self, packet):
        validate(packet, PACKET_SCHEMA)
        with self.lock():
            plan, state = self._load()
            module = packet["module"]
            self._ensure_ready(module, plan, state)
            spec = plan["contract"]
            if not spec["authority"]["execute_local_jobs"]:
                raise ValueError("Local execution was not authorized at intake")
            if state["jobs"] + len(packet["commands"]) > spec["budget"]["max_jobs"]:
                raise ValueError("Packet exceeds remaining job budget")
            inputs = {}
            for command in packet["commands"]:
                cwd = Path(command["cwd"]).resolve(strict=True)
                if not any(cwd.is_relative_to(Path(p).resolve()) for p in spec["authority"]["allowed_workdirs"]):
                    raise ValueError("Command cwd is outside the intake workdirs")
                if command["timeout_seconds"] > spec["budget"]["command_timeout_seconds"]:
                    raise ValueError("Command timeout exceeds the intake budget")
                selected = command["env"].get("CUDA_VISIBLE_DEVICES", ",".join(spec["budget"]["gpu_ids"]))
                if not set(filter(None, selected.split(","))) <= set(spec["budget"]["gpu_ids"]):
                    raise ValueError("Command requests GPUs outside the intake allocation")
                paths = [Path(p) if Path(p).is_absolute() else cwd / p for p in command["inputs"]]
                paths += [cwd / arg for arg in command["argv"] if (cwd / arg).is_file()]
                inputs.update(fingerprint(paths))
            row = state["modules"][module]
            number = len(row["attempts"]) + 1
            directory = self.root / "attempts" / f"{module}-{number:04d}"
            directory.mkdir()
            write_new(directory / "packet.json", packet)
            attempt = {"number": number, "directory": str(directory), "inputs": inputs,
                       "owner_pid": os.getpid(), "owner_start": self._process_start(os.getpid()),
                       "dependencies": {d: state["modules"][d]["accepted"]["id"]
                                        for d in plan["modules"][module]["depends_on"]},
                       "execution_ok": False, "results": []}
            row["attempts"].append(attempt)
            row["status"] = "running"
            if module == "holdout":
                state["holdout_consumed"] = True
            self._save(state, "attempt_started", module=module, attempt=number)
            try:
                for index, command in enumerate(packet["commands"]):
                    remaining = spec["budget"]["wall_seconds"] - (time.time() - state["created"])
                    if remaining <= 0:
                        raise ValueError("Wall-clock budget exhausted")
                    state["jobs"] += 1
                    self._save(state, "command_started", module=module, command=index)
                    env = {**os.environ, **command["env"]}
                    env["CUDA_VISIBLE_DEVICES"] = command["env"].get(
                        "CUDA_VISIBLE_DEVICES", ",".join(spec["budget"]["gpu_ids"]))
                    # Never inherit another editable checkout's package selection.
                    from ..layout import source_roots
                    own = os.pathsep.join(str(p) for p in source_roots(command["cwd"]))
                    if own:
                        env["PYTHONPATH"] = own + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
                    for prefix in ("ASTRA_E2E", "R2S2R"):  # Old workers read the previous names.
                        env[f"{prefix}_OVERSIGHT_RUN"] = str(self.root)
                        env[f"{prefix}_OVERSIGHT_PLAN_SHA256"] = state["plan_sha256"]
                    with (directory / f"command-{index:03d}.log").open("x") as log:
                        child = subprocess.Popen(command["argv"], cwd=command["cwd"], env=env,
                                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        attempt["child_pid"] = child.pid
                        self._save(state, "command_spawned", module=module, command=index, pid=child.pid)
                        try:
                            code = child.wait(timeout=min(command["timeout_seconds"], remaining))
                        except BaseException:
                            self._terminate(child)
                            raise
                        # A packet must supervise its services instead of leaving background children.
                        self._terminate(child)
                    attempt["results"].append({"command": index, "returncode": code})
                    if code:
                        raise ValueError(f"Command {index} failed with exit code {code}")
                attempt["execution_ok"] = True
                row["status"] = "awaiting_review"
                self._save(state, "execution_completed", module=module)
            except BaseException as exc:
                row["status"] = "failed"
                attempt["error"] = str(exc)
                if module == "holdout":
                    state["closed"] = True
                self._save(state, "execution_failed", module=module)
                raise
            return {"module": module, "attempt": number, "execution_ok": True,
                    "accepted": False, "directory": str(directory)}

    @staticmethod
    def _process_start(pid):
        try:
            # Linux /proc starttime distinguishes PID reuse; comm may contain spaces.
            return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
        except FileNotFoundError:
            return None

    def recover(self, module, reason):
        """Mark an orphaned attempt failed only after its owner and process group exit."""
        if module not in MODULES or not reason.strip():
            raise ValueError("Recovery requires a valid module and an explanation")
        with self.lock():
            plan, state = self._load()
            row = state["modules"][module]
            if row["status"] != "running":
                raise ValueError("Only an interrupted running attempt can be recovered")
            attempt = row["attempts"][-1]
            if self._process_start(attempt["owner_pid"]) == attempt["owner_start"]:
                raise ValueError("Attempt owner is still running")
            if attempt.get("child_pid"):
                try:
                    os.killpg(attempt["child_pid"], 0)
                except ProcessLookupError:
                    pass
                else:
                    raise ValueError("Attempt process group is still running; inspect and stop that owned job first")
            row["status"] = "failed"
            attempt["error"] = "Interrupted: " + reason
            if module == "holdout":
                state["closed"] = True
            self._save(state, "interrupted_attempt_recovered", module=module, reason=reason)
            return {"module": module, "status": "failed", "preserved_attempt": attempt["number"]}

    @staticmethod
    def _terminate(child):
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait()

    def accept(self, report):
        validate(report, REPORT_SCHEMA)
        with self.lock():
            plan, state = self._load()
            module = report["module"]
            row = state["modules"][module]
            if row["status"] != "awaiting_review" or report["attempt"] != len(row["attempts"]):
                raise ValueError("Report must refer to the current completed attempt")
            if report["plan_sha256"] != state["plan_sha256"]:
                raise ValueError("Report refers to a different plan")
            attempt = row["attempts"][-1]
            if not unchanged(attempt["inputs"]):
                raise ValueError("Implementation or inputs changed during the attempt")
            for dep, identity in attempt["dependencies"].items():
                if not self._current(dep, plan, state) or state["modules"][dep]["accepted"]["id"] != identity:
                    raise ValueError("Dependency changed during the attempt")
            required = plan["modules"][module]["checks"]
            expected = {c["id"] for c in required}
            if set(report["checks"]) != expected:
                raise ValueError(f"Report checks must match the plan: {sorted(expected)}")
            artifacts = set(report["artifacts"])
            for check in report["checks"].values():
                artifacts.update(check["evidence"])
            for failure in report["failures"]:
                artifacts.update(failure["evidence"])
            if module in ("validation", "holdout") and report["evaluation"]:
                artifacts.update(e["trace"] for e in report["evaluation"]["episodes"])
            files = fingerprint(artifacts)
            if module in ("data", "train", "validation", "holdout", "delivery"):
                policy_io(plan["contract"], report["policy_io"])
                files.update(fingerprint([report["policy_io"]]))
            if module == "data":
                count, episode_files = dataset_manifest(plan["contract"], report["dataset_manifest"])
                if report["checks"]["usable_episodes"]["value"] != count:
                    raise ValueError("Usable episode count disagrees with the dataset manifest")
                files.update(episode_files)
                files.update(fingerprint([report["dataset_manifest"]]))
            if module in ("train", "validation", "holdout", "delivery"):
                if not report["checkpoint"]:
                    raise ValueError("Policy acceptance requires an actual checkpoint artifact")
                files.update(fingerprint([report["checkpoint"]]))
                if module != "train":
                    trained = state["modules"]["train"]["accepted"]["checkpoint_sha256"]
                    if file_hash(report["checkpoint"]) != trained:
                        raise ValueError("Evaluation/delivery checkpoint differs from the accepted training result")
            failed = [c["id"] for c in required if not compare(c, report["checks"][c["id"]]["value"])]
            evaluated = None
            if module in ("validation", "holdout"):
                evaluated = evaluate_episodes(plan["contract"], module, report["evaluation"])
                if not evaluated["pass"]:
                    failed.append("evaluation_target")
            elif report["evaluation"] is not None:
                raise ValueError("Only evaluation modules accept evaluation episodes")
            if module == "twin":
                expected_session = Path(plan["contract"]["source"]["session"]).resolve()
                if not report["native_session"] or Path(report["native_session"]).resolve() != expected_session:
                    raise ValueError("Twin evidence must refer to the contracted native session")
                native_twin(report["native_session"])
            directory = Path(attempt["directory"])
            write_new(directory / "review.json", report)
            files.update(fingerprint([directory / "review.json", directory / "packet.json",
                                      *directory.glob("*.log")]))
            accepted = not failed and not report["failures"]
            row["status"] = "accepted" if accepted else "failed"
            row["accepted"] = {"id": digest({"report": report, "files": files}),
                               "files": {**attempt["inputs"], **files},
                               "dependencies": attempt["dependencies"],
                               "native_session": report["native_session"],
                               "checkpoint_sha256": file_hash(report["checkpoint"]) if report["checkpoint"] else None} if accepted else None
            if module == "holdout" and not accepted:
                state["closed"] = True
            self._save(state, "review_recorded", module=module, accepted=accepted,
                       failed_checks=failed, evaluation=evaluated)
            return {"accepted": accepted, "failed_checks": failed, "evaluation": evaluated,
                    "suggested_targets": feedback_targets(report["failures"])}

    def feedback(self, revision):
        """Invalidate causes and descendants only with a documented transfer rationale."""
        required = {"targets", "diagnosis", "transfer_hypothesis", "expected_real_benefit",
                    "anti_hack_review", "ablation_plan", "evidence"}
        if set(revision) != required or not all(revision[k] for k in required):
            raise ValueError("Feedback needs targets, diagnosis, transfer rationale, anti-hack review, ablation and evidence")
        if not isinstance(revision["targets"], list) or not set(revision["targets"]) <= set(MODULES) - {"holdout", "delivery"}:
            raise ValueError("Feedback targets must be development modules")
        with self.lock():
            plan, state = self._load()
            if state["holdout_consumed"] or state["closed"]:
                raise ValueError("Consumed holdout cannot feed this development loop; create a new experiment and holdout")
            if state["cycle"] >= plan["contract"]["budget"]["max_cycles"]:
                raise ValueError("Feedback cycle budget exhausted")
            files = fingerprint(revision["evidence"])
            affected = set().union(*(descendants(m) for m in revision["targets"]))
            for module in affected:
                state["modules"][module]["status"] = "invalidated"
                state["modules"][module]["accepted"] = None
            state["cycle"] += 1
            self._save(state, "feedback_revision", revision=revision, evidence=files,
                       invalidated=sorted(affected))
            return {"cycle": state["cycle"], "invalidated": sorted(affected)}
