"""The end-to-end chain: observations -> real2sim -> policy development in simulation -> preparation for the real robot.

The repository has two parts with one contract each:

* **real2sim** (Part 1) ends in a *delivery record*. Native deliveries use ``astra_twin_delivery_v1``.
  Workflows with model revisions use ``astra_twin_delivery_v2`` to carry their check receipts.
* **policy development** (Part 2) starts from a *task binding* and ends in a frozen deliverable with one sealed
  evaluation (``python -m astra_station ...``).

This module joins them and nothing more.  It keeps one run directory with the request, an append-only hash-linked
ledger and one evidence record per stage; it tells the agent what the current stage needs (``next``); and it records a
stage as done only after reading the files that stage's part wrote (``advance``).  It embeds no model: the agent that
builds the twin and the agent that develops the policy are the ones the two parts already describe.  It operates no
robot: the last stage prepares a hand-over for human-operated validation and leaves real validation pending.

    python -m astra_e2e.chain schemas
    python -m astra_e2e.chain init    --run RUN --request request.json
    python -m astra_e2e.chain status  --run RUN
    python -m astra_e2e.chain next    --run RUN
    python -m astra_e2e.chain advance --run RUN [--delivery RECORD] [--execute]

``advance --execute`` runs the deterministic steps of the current stage itself (write the task spec, bind, create the
campaign, run the sealed evaluation, hand off).  Without it the chain only checks for their results.  The two
open-ended steps are never started here: building the twin, and the campaign's decision loop.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from astra_station import binding as binding_mod
from astra_station import twin_task
from astra_station.campaign import Campaign
from astra_station.common import Ledger, digest, file_hash, now, read, tree_hash, validate, write_new
from astra_station.site import Site

from .layout import REPO

SCHEMA = "astra_chain_v1"
STAGES = ("observations", "real2sim", "station_task", "policy_development", "sealed_evaluation", "real_preparation")
ROUTES = ("scene_pipeline", "task_workspace")

REQUEST_SCHEMA = {
    "type": "object",
    "required": ["name", "task", "observations"],
    "properties": {
        "name": {"type": "string", "pattern": "^[a-z][a-z0-9_]{1,40}$"},
        "task": {"type": "string", "minLength": 8},
        "observations": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        "route": {"enum": list(ROUTES)},
        "reuse_delivery": {
            "type": "object", "required": ["sha256", "reason"], "additionalProperties": False,
            "properties": {"sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                           "reason": {"type": "string", "minLength": 1}},
        },
        "goal": {"enum": ["object", "scene", "real_preparation"]},
        "cousins": {"type": "integer", "minimum": 1},
        "require": {"type": "array", "items": {"enum": ["metric_scale", "robot_placed", "real_physics_calibrated"]}},
        "robot": {"type": "string"},
        "station": {
            "type": "object",
            "required": ["site", "task_id", "factory", "policy_io"],
            "properties": {
                "site": {"type": "string"},
                "task_id": {"type": "string", "pattern": "^[a-z][a-z0-9_]{1,40}$"},
                "factory": {"type": "string"},
                "policy_io": {"type": "object", "required": ["action_dim"]},
                "num_envs": {"type": "integer", "minimum": 1},
                "max_steps": {"type": "integer", "minimum": 1},
                "setup": {"type": "object"},
            },
        },
        "campaign": {
            "type": "object",
            "required": ["name", "run"],
            "properties": {"name": {"type": "string"}, "run": {"type": "string"}, "budget": {"type": "object"},
                           "caps": {"type": "object"}, "decision": {"type": "object"}},
        },
        "real": {"type": "object"},
    },
}

SKILLS = {
    "scene_pipeline": ["real2sim/skills/twin-capture/SKILL.md", "real2sim/skills/twin-model/SKILL.md",
                       "real2sim/skills/twin-deliver/SKILL.md"],
    "task_workspace": ["real2sim/skills/use-real2sim/SKILL.md", "real2sim/skills/plan-scene-generation/SKILL.md",
                       "real2sim/skills/compose-interactive-scene/SKILL.md"],
}


class ChainError(ValueError):
    pass


class Chain:
    def __init__(self, run):
        self.run = Path(run).resolve()
        if not (self.run / "chain.json").is_file():
            raise ChainError(f"{self.run} is not a chain run (no chain.json)")
        self.request = read(self.run / "chain.json")
        self.ledger = Ledger(self.run / "ledger.jsonl")

    # ------------------------------------------------------------------ creation
    @classmethod
    def create(cls, run, request: dict) -> "Chain":
        run = Path(run).resolve()
        if run.exists() and any(run.iterdir()):
            raise ChainError(f"{run} is not empty; a chain run is never overwritten")
        validate(request, REQUEST_SCHEMA)
        if "cousins" in request and "route" in request:
            raise ChainError("A requested cousin family needs a module plan")
        observed = {}
        for raw in request["observations"]:
            path = Path(raw)
            if not path.is_absolute() or not path.is_file():
                raise ChainError(f"observation must be an absolute path to a file: {raw}")
            observed[str(path.resolve())] = file_hash(path)
        goal = request.get("goal", "real_preparation" if "station" in request or "campaign" in request else "scene")
        if goal == "real_preparation":
            if "station" not in request or "campaign" not in request:
                raise ChainError("A policy handover requires station and campaign")
            site = Path(request["station"]["site"])
            if not site.is_absolute() or not site.is_file():
                raise ChainError(f"station.site must be an absolute path to a site file: {site}")
        elif "station" in request or "campaign" in request:
            raise ChainError("Station and campaign settings require the real_preparation goal")
        request = {**request, "goal": goal}
        (run / "stages").mkdir(parents=True, exist_ok=True)
        record = {"schema": SCHEMA, "created": now(), **request}
        write_new(run / "chain.json", record)
        chain = cls(run)
        chain.ledger.append("created", request_sha256=digest(record))
        chain._done("observations", {"files": observed, "task": request["task"]})
        return chain

    # ------------------------------------------------------------------ state
    def stage_file(self, stage: str) -> Path:
        return self.run / "stages" / f"{stage}.json"

    def done(self, stage: str) -> dict | None:
        path = self.stage_file(stage)
        return read(path) if path.is_file() else None

    @property
    def goal(self):
        return self.request.get("goal", "real_preparation")

    @property
    def stages(self):
        return STAGES if self.goal == "real_preparation" else STAGES[:2]

    def current(self) -> str | None:
        for stage in self.stages:
            if not self.stage_file(stage).is_file():
                return stage
        return None

    def _done(self, stage: str, evidence: dict) -> dict:
        record = {"schema": "astra_chain_stage_v1", "stage": stage, "completed": now(), **evidence}
        write_new(self.stage_file(stage), record)
        self.ledger.append("stage_completed", stage=stage, record_sha256=digest(record))
        return record

    def status(self) -> dict:
        self.ledger.verify()
        current = self.current()
        return {"run": str(self.run), "name": self.request["name"], "task": self.request["task"],
                "stages": {stage: ("done" if self.stage_file(stage).is_file() else "current" if stage == current else "waiting")
                           for stage in self.stages},
                "current": current,
                "claim": (f"accepted {self.goal} delivered" if current is None and self.goal != "real_preparation" else
                          "prepared for human-operated real validation; real validation pending" if current is None
                          else "in progress; no result is claimed for stages that are not done")}

    # ------------------------------------------------------------------ paths of Part 2
    def _site(self) -> Site:
        return Site(self.request["station"]["site"])

    def _task_dir(self) -> Path:
        return self._site().tasks_dir / self.request["station"]["task_id"]

    def _campaign_state(self) -> Path | None:
        record = self.done("policy_development") or {}
        if record.get("state"):
            return Path(record["state"])
        pointer = self.run / "campaign_state.json"
        return Path(read(pointer)["state"]) if pointer.is_file() else None

    # ------------------------------------------------------------------ what the current stage needs
    def next(self) -> dict:
        stage, r, station = self.current(), self.request, self.request.get("station", {})
        env = "source env.sh          # repository root: the three source roots on PYTHONPATH"
        if stage is None and self.goal != "real_preparation":
            return {"stage": None, "delivery": self.done("real2sim")["delivery"]}
        if stage is None:
            return {"stage": None, "do": ["The chain is complete. Give stages/real_preparation.json to the operator; "
                                          "real validation is theirs to run and to report."]}
        if stage == "real2sim":
            if "route" not in r:
                from .reconstruction import Workflow
                return {"stage": stage, "goal": self.goal, "task": r["task"], "open_ended": True,
                        "observations": self.done("observations")["files"], "workflow": Workflow(self).status(),
                        "read": ["end_to_end/docs/reconstruction.md", "real2sim/skills/reconstruct-object/SKILL.md",
                                 *(["real2sim/skills/reconstruct-environment/SKILL.md"] if self.goal != "object" else [])],
                        "do": ["Use object and environment revisions to plan modeling. Recall applicable procedures. Run the next step, "
                               "inspect results and revise on failure. Advance after native delivery."]}
            route = r["route"]
            deliver = ("real2sim-harness deliver --session sessions/<name> --output <absolute path>/<name>.delivery.json"
                       if route == "scene_pipeline" else
                       "real2sim-harness deliver --run <accepted run> --output <absolute path>/<name>.delivery.json")
            return {"stage": stage, "part": "real2sim/ (run its commands from that directory)", "read": SKILLS[route],
                    "do": [f"Build and accept the twin of: {r['task']}",
                           "Observations: " + ", ".join(self.done("observations")["files"]),
                           ("Deliver the accepted object and its assets." if self.goal == "object" else
                            "Deliver the accepted scene and its native evidence."),
                           deliver,
                           f"python -m astra_e2e.chain advance --run {self.run} --delivery <that record>"],
                    "open_ended": True}
        if stage == "station_task":
            site = self._site()
            partition = site.root / "sim" / "sealed" / f"{station['task_id']}_partitions.json"
            return {"stage": stage, "part": "policy_development/", "read": ["policy_development/docs/station_harness.md"],
                    "needs": {"environment factory": station["factory"],
                              "sealed start partitions (coordinator)": str(partition),
                              "sealed evaluation token (coordinator)": str(site.root / "sim" / "sealed" / "heldout_token")},
                    "do": [env,
                           "Author the environment factory if it does not exist: <file.py>:<function>(source, task, num_envs, "
                           "run_dir) -> one environment per station env. Worked examples for both real2sim routes: "
                           "real2sim/examples/station/README.md.",
                           f"python policy_development/station/server/partitions.py {partition}      # coordinator, once",
                           f"python -m astra_e2e.chain advance --run {self.run} --execute"]}
        if stage == "policy_development":
            state = self._campaign_state()
            return {"stage": stage, "part": "policy_development/", "read": ["policy_development/README.md"],
                    "do": [env,
                           f"python -m astra_e2e.chain advance --run {self.run} --execute      # creates the campaign"
                           if state is None else f"python -m astra_station run --state {state}      # the agent's decision loop",
                           f"python -m astra_e2e.chain advance --run {self.run}      # after the campaign froze a deliverable"],
                    "open_ended": True}
        if stage == "sealed_evaluation":
            return {"stage": stage, "part": "policy_development/",
                    "do": [env, f"python -m astra_e2e.chain advance --run {self.run} --execute      # coordinator: once"]}
        return {"stage": stage, "part": "end_to_end/", "read": ["end_to_end/docs/transfer.md", "end_to_end/docs/delivery_package.md"],
                "do": [env, f"python -m astra_e2e.chain advance --run {self.run} --execute"]}

    # ------------------------------------------------------------------ advancing
    def verify_twin(self):
        from .reconstruction import Workflow
        accepted = self.done("real2sim")
        if not accepted:
            raise ChainError("No accepted reconstruction yet")
        path = Path(accepted["delivery"])
        if file_hash(path) != accepted["delivery_sha256"]:
            raise ChainError("The accepted delivery record changed")
        record = twin_task.verify_delivery(path)
        self.check_delivery(record)
        if "workflow_sha256" in accepted:
            selected, identity = Workflow(self).delivery()
            if selected != path or identity != accepted["workflow_sha256"]:
                raise ChainError("The accepted module plan changed")
        if "observation_provenance" in accepted:
            self._observation_provenance(record, accepted["delivery_sha256"], planned="workflow_sha256" in accepted)
        return {"delivery": str(path), "delivery_sha256": accepted["delivery_sha256"], "current": True}

    def advance(self, *, delivery=None, execute=False) -> dict:
        stage = self.current()
        if stage is None:
            raise ChainError("the chain is complete")
        if stage != "real2sim" and "route" not in self.request:
            self.verify_twin()
        return getattr(self, f"_advance_{stage}")(delivery=delivery, execute=execute)

    def check_delivery(self, record):
        if "route" in self.request and record["route"] != self.request["route"]:
            raise ChainError(f"the request names route {self.request['route']}, the record comes from {record['route']}")
        if self.goal == "object":
            if record.get("level") != "object" or not record.get("assets"):
                raise ChainError("An object request needs an accepted object with deliverable assets")
        elif record.get("level") != "scene" or not record.get("scene"):
            raise ChainError("the record delivers a single object; a station or scene request needs a scene")
        for key in self.request.get("require", []):
            if (record.get("statements") or {}).get(key) is not True:
                raise ChainError(f"Required delivery evidence missing: {key}")

    def _advance_real2sim(self, *, delivery, execute):
        if "route" not in self.request:
            from .reconstruction import Workflow
            workflow = Workflow(self)
            with workflow.lock():
                planned, identity = workflow.delivery()
                if delivery and Path(delivery).resolve() != planned:
                    raise ChainError("Use the native delivery of the current plan")
                return self._accept_delivery(planned, {"workflow_sha256": identity})
        if not delivery:
            raise ChainError("give the twin's delivery record: --delivery <record written by real2sim-harness deliver>")
        return self._accept_delivery(Path(delivery).resolve(), {})

    def _observation_provenance(self, record, delivered_sha, *, planned):
        # A completed plan checks all inputs against native captures and pinned step evidence,
        # including supplementary observations from other routes (Workflow.delivery).
        observations = self.done("observations")["files"]
        if any(not Path(path).is_file() or file_hash(path) != sha for path, sha in observations.items()):
            raise ChainError("An input observation changed after chain initialization")
        source_hashes = record.get("observation_sha256", [])
        validate(source_hashes, {"type": "array", "items": {"type": "string", "pattern": "^[0-9a-f]{64}$"}},
                 "delivery.observation_sha256")
        missing = [] if planned else sorted(set(observations.values()) - set(source_hashes))
        reuse = self.request.get("reuse_delivery")
        if reuse is not None:
            if reuse["sha256"] != delivered_sha or not reuse["reason"].strip():
                raise ChainError("The reuse declaration must identify this delivery record and explain its use")
        elif missing:
            raise ChainError("The delivery does not identify every requested observation. Supply the matching delivery "
                             "or declare intentional reuse with the delivery file's SHA-256 and a reason in a new request.")
        provenance = {"status": "explicit_reuse" if reuse else "matched", "missing_sha256": missing}
        if reuse:
            provenance["reason"] = reuse["reason"]
        if planned:
            provenance["source"] = "module_plan"
        return provenance

    def _accept_delivery(self, delivery, lineage):
        record = twin_task.verify_delivery(delivery)
        self.check_delivery(record)
        delivered_sha = file_hash(delivery)
        provenance = self._observation_provenance(record, delivered_sha, planned="workflow_sha256" in lineage)
        notes = []
        statements = record.get("statements") or {}
        if self.goal == "real_preparation" and statements.get("robot_placed") is not True:
            notes.append("The record does not state that a robot is placed in the scene; the environment factory must place it. "
                         "Declare the placement in the request's station.setup: it is recorded as the station author's.")
        if statements.get("real_physics_calibrated") is not True:
            notes.append("Physics is not calibrated against real motion; a simulation result says nothing yet about transfer.")
        if (record.get("task") or {}).get("text") not in (None, self.request["task"]):
            notes.append(f"The twin was built for the task text {record['task']['text']!r}; the chain's request says "
                         f"{self.request['task']!r}.")
        return self._done("real2sim", {"delivery": str(delivery), "delivery_sha256": delivered_sha, "route": record["route"],
                                        **lineage, "observation_provenance": provenance,
                                        "twin": record["name"], "scene": record["scene"], "acceptance": record["acceptance"],
                                        "statements": statements, "notes": notes})

    def _advance_station_task(self, *, delivery, execute):
        station, twin = self.request["station"], self.done("real2sim")
        site, task_dir, binding_path = self._site(), self._task_dir(), self.run / "binding.json"
        factory_path, separator, factory_name = station["factory"].rpartition(":")
        if not separator or not factory_name.isidentifier():
            raise ChainError("the environment factory must name <absolute file.py>:<function>")
        factory_file = Path(factory_path)
        if not factory_file.is_absolute() or not factory_file.is_file():
            raise ChainError(f"the environment factory {factory_file} does not exist (absolute path required)")
        spec_path = task_dir / "task.json"
        if not spec_path.is_file() and not execute:
            raise ChainError(f"no task spec at {task_dir}; run with --execute to write it from the delivery record")
        expected = twin_task.spec_from_delivery(Path(twin["delivery"]), station["task_id"], station["factory"], station["policy_io"],
                                                self.request.get("robot", "vega_right_sharpa"), station.get("num_envs", 4),
                                                station.get("max_steps", 600), station.get("setup"))
        if not spec_path.is_file():
            twin_task.write_task(expected, task_dir, REPO / "policy_development/station/server/twin.py")
        spec = site.task_spec(station["task_id"])
        if (spec.get("twin") or {}).get("delivery") != twin["delivery"]:
            raise ChainError(f"the task spec at {task_dir} names another twin than this chain's delivery record")
        # Compare the generated contract, leaving site-specific additions such as labels and resource settings intact.
        for section, keys in (("policy_io", expected["policy_io"]), ("twin", expected["twin"]),
                              ("tags", ("robot",)), ("station", ("num_envs", "partition_kind"))):
            actual = spec.get(section) or {}
            for key in keys:
                if key not in actual or actual[key] != expected[section][key]:
                    raise ChainError(f"task spec {section}.{key} differs from the request or its delivery")
        for key in ("task_id", "station_task", "text"):
            if spec.get(key) != expected[key]:
                raise ChainError(f"task spec {key} differs from the request or its delivery")

        # These are the arguments the twin station actually receives. Extra station options may be interleaved.
        parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False, exit_on_error=False)
        parser.add_argument("--task", default=station["task_id"])
        parser.add_argument("--num_envs", type=int, default=expected["station"]["num_envs"])
        parser.add_argument("--twin-session", default=expected["twin"]["session"])
        parser.add_argument("--twin-max-steps", type=int, default=600)
        parser.add_argument("--twin-partition-kind", default=expected["station"]["partition_kind"])
        parser.add_argument("--twin-factory")
        parser.add_argument("--twin-delivery")
        parser.add_argument("--twin-task-spec")
        try:
            required, _ = parser.parse_known_args(expected["station"]["args"])
            effective, _ = parser.parse_known_args(spec["station"].get("args", []))
            # station_up.sh resolves these placeholders before the backend sees its arguments.
            required.twin_task_spec = str(spec_path)
            if effective.twin_task_spec:
                effective.twin_task_spec = effective.twin_task_spec.format(spec=spec_path, spec_dir=task_dir)
        except (argparse.ArgumentError, ValueError, KeyError, IndexError) as exc:
            raise ChainError(f"invalid task spec station.args: {exc}") from exc
        for key, value in vars(required).items():
            if getattr(effective, key) != value:
                raise ChainError(f"task spec --{key.replace('_', '-')} differs from the request or its delivery")
        missing = [name for name in site.sealed_files(spec) if not (site.root / name).is_file()]
        if missing:
            raise ChainError(f"the coordinator has not prepared the sealed files under {site.root}: {missing}")
        if not binding_path.is_file():
            if not execute:
                raise ChainError("the task is not bound; run with --execute")
            binding_mod.create(self.request["station"]["site"], station["task_id"], binding_path)
        bound = binding_mod.verify(binding_path)
        if bound["site"] != str(site.path) or bound["task_id"] != station["task_id"]:
            raise ChainError("the binding belongs to a different station site or task")
        if (bound.get("twin") or {}).get("delivery_sha256") != twin["delivery_sha256"]:
            raise ChainError("the binding pins another delivery record than the one this chain accepted")
        return self._done("station_task", {"task_spec": str(task_dir / "task.json"), "task_spec_sha256": file_hash(task_dir / "task.json"),
                                            "factory": station["factory"], "factory_sha256": file_hash(factory_file),
                                            "binding": str(binding_path), "binding_sha256": bound["binding_sha256"],
                                            "validation_scope": bound.get("validation_scope")})

    def campaign_step(self):
        from astra_station.runner import Runner
        state = self._campaign_state()
        if self.current() != "policy_development" or not state:
            raise ChainError("Create the bound campaign at the policy_development stage first")
        campaign = Campaign(state)
        if campaign.cfg["binding_sha256"] != self.done("station_task")["binding_sha256"]:
            raise ChainError("The campaign binding changed")
        if campaign.s["frozen"] or campaign.s["ended"]:
            raise ChainError("The campaign decision loop has ended")
        return Runner(campaign).step()

    def _advance_policy_development(self, *, delivery, execute):
        campaign, state = self.request["campaign"], self._campaign_state()
        if state is None:
            if not execute:
                raise ChainError("no campaign yet; run with --execute to create it")
            created = Campaign.create(self.done("station_task")["binding"], campaign["name"], campaign["run"],
                                      campaign.get("budget"), campaign.get("caps"), campaign.get("decision"))
            write_new(self.run / "campaign_state.json", {"state": str(created.dir)})
            self.ledger.append("campaign_created", state=str(created.dir))
            return {"stage": "policy_development", "status": "campaign created; run the decision loop, then advance again",
                    "state": str(created.dir)}
        c = Campaign(state)
        if c.cfg["binding_sha256"] != self.done("station_task")["binding_sha256"]:
            raise ChainError("the campaign runs on another binding than this chain's")
        frozen = c.s.get("frozen")
        if not frozen:
            raise ChainError(f"the campaign has not frozen a deliverable; run `python -m astra_station run --state {state}`")
        if tree_hash(c.dir / "frozen") != frozen["tree_sha256"]:
            raise ChainError("the frozen snapshot differs from the hash the campaign recorded")
        unseen = (c.s.get("unseen") or {}).get("facts") or {}
        return self._done("policy_development", {
            "state": str(c.dir), "candidate": frozen["candidate"], "kind": "program" if frozen["program"] else "policy",
            "tree_sha256": frozen["tree_sha256"], "reason": frozen.get("reason"),
            "development": [f"{e['success']}/{e['n']}" for e in frozen.get("dev", [])],
            "unseen_check": unseen.get("success"), "used": frozen.get("used")})

    def _advance_sealed_evaluation(self, *, delivery, execute):
        c = Campaign(self._campaign_state())
        if not (c.dir / "heldout.json").is_file():
            if not execute:
                raise ChainError("the sealed evaluation has not run; the coordinator runs it once with --execute")
            c.heldout()
        result = read(c.dir / "heldout.json")
        n = result.get("heldout_n", result.get("n"))
        return self._done("sealed_evaluation", {"result": str(c.dir / "heldout.json"), "result_sha256": file_hash(c.dir / "heldout.json"),
                                                 "success": result.get("success"), "n": n, "terms": result.get("terms"),
                                                 "scope": "simulation on the delivered twin; one run, not repeatable for this deliverable"})

    def _advance_real_preparation(self, *, delivery, execute):
        c = Campaign(self._campaign_state())
        twin, task, dev, sealed = (self.done(s) for s in ("real2sim", "station_task", "policy_development", "sealed_evaluation"))
        spec = self._site().task_spec(self.request["station"]["task_id"])
        handed = None
        if (spec.get("real") or {}).get("handoff") or c.site.cfg.get("real_handoff"):
            if not execute:
                raise ChainError("the task names a real-robot chain; run with --execute to hand the deliverable to it")
            handed = c.handoff()
        package = {
            "schema": "astra_real_preparation_v1", "name": self.request["name"], "task": self.request["task"],
            "deliverable": {"snapshot": str(c.dir / "frozen"), "tree_sha256": dev["tree_sha256"], "candidate": dev["candidate"],
                            "kind": dev["kind"], "entry": c.s["frozen"]["policy_file"]},
            "policy_io": spec["policy_io"],
            "simulation_evidence": {"twin": twin["twin"], "delivery": twin["delivery"], "delivery_sha256": twin["delivery_sha256"],
                                    "observation_provenance": twin.get("observation_provenance", {"status": "unrecorded"}),
                                    "binding_sha256": task["binding_sha256"], "development": dev["development"],
                                    "unseen_check": dev["unseen_check"], "sealed": f"{sealed['success']}/{sealed['n']}"},
            "handed_to_real_chain": handed,
            "real_constraints": spec.get("real_constraints", []),
            "open_before_the_robot_moves": [
                "Deployment I/O qualification: the policy's observations and actions on the real interfaces, at the real rate.",
                "Workspace, speed and force limits of the real cell; an operator present and able to stop the robot.",
                *twin["notes"],
            ],
            "validation_scope": "simulation_twin", "real_validation": "pending",
        }
        write_new(self.run / "real_preparation.json", package)
        return self._done("real_preparation", {"package": str(self.run / "real_preparation.json"),
                                                "package_sha256": file_hash(self.run / "real_preparation.json"),
                                                "handed_to_real_chain": bool(handed), "real_validation": "pending"})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="astra-e2e", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("schemas")
    sub.add_parser("catalog")
    p = sub.add_parser("plan")
    p.add_argument("--run", required=True)
    p.add_argument("--request", required=True)
    p = sub.add_parser("step")
    p.add_argument("--run", required=True)
    p.add_argument("--id", required=True)
    p.add_argument("--arguments", required=True)
    p.add_argument("--evidence", action="append", default=[])
    for name in ("recall", "learn"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True)
        p.add_argument("--request", required=True)
        p.add_argument("--state", help="shared experience directory (default: private to this chain)")

    p = sub.add_parser("init"); p.add_argument("--run", required=True); p.add_argument("--request", required=True)
    for name in ("status", "next", "verify", "campaign-step"):
        p = sub.add_parser(name); p.add_argument("--run", required=True)
    p = sub.add_parser("advance"); p.add_argument("--run", required=True); p.add_argument("--delivery")
    p.add_argument("--execute", action="store_true", help="run this stage's deterministic steps (never the open-ended ones)")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "schemas":
            from .reconstruction import PLAN
            out = {"request": REQUEST_SCHEMA, "module_plan": PLAN, "stages": list(STAGES), "delivery_record": twin_task.DELIVERY_SCHEMA}
        elif a.cmd == "catalog":
            from .reconstruction import catalog
            out = catalog()
        elif a.cmd in ("plan", "step"):
            from .reconstruction import Workflow
            workflow = Workflow(Chain(a.run))
            out = workflow.choose(read(a.request)) if a.cmd == "plan" else workflow.perform(
                a.id, read(a.arguments), a.evidence)
        elif a.cmd in ("recall", "learn"):
            from .workflow_memory import memory
            out = memory(Chain(a.run), a.cmd, read(a.request), a.state)
        elif a.cmd == "init":
            out = Chain.create(a.run, read(a.request)).status()
        elif a.cmd == "status":
            out = Chain(a.run).status()
        elif a.cmd == "verify":
            out = Chain(a.run).verify_twin()
        elif a.cmd == "campaign-step":
            out = Chain(a.run).campaign_step()
        elif a.cmd == "next":
            out = Chain(a.run).next()
        else:
            out = Chain(a.run).advance(delivery=a.delivery, execute=a.execute)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError, ImportError, SystemExit) as exc:
        if isinstance(exc, SystemExit) and not isinstance(exc.code, str):
            raise
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
