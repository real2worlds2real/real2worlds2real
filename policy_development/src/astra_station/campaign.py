"""Campaign controller: budgets, candidates, datasets, tool execution, freeze, unseen check and the sealed evaluation.

State lives in the station root's campaign directory (`campaigns/<campaign>/astra/<station task>_<run>/`, the same layout
the baselines use, so the shared sealed held-out tool applies unchanged).  Every state change is also an event in a
hash-linked ledger.  The controller never chooses a method: the deciding agent does (see runner.py); the controller
validates, executes registered adapters, measures and records.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import time
from pathlib import Path

from . import binding as binding_mod
from . import catalog
from .common import Ledger, digest, now, read, tree_hash, validate, write, write_new
from .site import REPO, SRC, Site

DEFAULT_BUDGET = {"model_calls": 40, "wall_hours": 10.0, "train_episodes": 2000, "dev_evaluations": 20, "gpu_hours": 12.0}
DEFAULT_CAPS = {"train_act": 4, "train_residual": 2, "synthesize_program": 3, "research_session": 2, "compose": 4}
NAME = r"[a-z][a-z0-9_]{1,40}"
REF_KEYS = ("candidate", "base", "source", "teacher", "stage_a", "stage_b", "start_from")


INFRA_FAILURE = "station unavailable"           # _adapter.Job.ensure_station: the simulator did not come up (host busy, out of memory)


class CampaignError(ValueError):
    pass


class IneligibleCandidate(CampaignError):
    """The candidate lacks the evidence required to freeze this revision."""


class Campaign:
    def __init__(self, state_dir):
        self.dir = Path(state_dir)
        self.cfg = read(self.dir / "campaign.json")
        self.ledger = Ledger(self.dir / "astra_ledger.jsonl")
        self.binding = binding_mod.verify(self.cfg["binding"])
        if self.binding["binding_sha256"] != self.cfg["binding_sha256"]:
            raise CampaignError("binding differs from the campaign's")
        self.site = Site(self.binding["site"])
        self.task = self.site.task_spec(self.binding["task_id"])
        self.tools = catalog.load(self.site.tools_dir, REPO, self.site.cfg)
        catalog.check_pins(self.tools, self.cfg["tool_pins"])
        self.ws = Path(self.cfg["workspace"])
        self.s = read(self.dir / "state.json")

    # ------------------------------------------------------------------ creation
    @classmethod
    def create(cls, binding_path, campaign: str, run_id: str, budget: dict | None = None, caps: dict | None = None,
               decision: dict | None = None) -> "Campaign":
        b = binding_mod.verify(binding_path)
        site = Site(b["site"])
        task = site.task_spec(b["task_id"])
        bud = {**DEFAULT_BUDGET, **(budget or {})}
        ws, meta = site.workspace(campaign, task, run_id, bud["train_episodes"])
        d = site.root / "campaigns" / campaign / "astra" / f"{task['station_task']}_{run_id}"
        if (d / "campaign.json").exists():
            raise CampaignError(f"campaign state {d} already exists")
        tools = catalog.load(site.tools_dir, REPO, site.cfg)
        cfg = {"schema": "astra_station_campaign_v1", "campaign": campaign, "run_id": run_id, "created": now(),
               "binding": str(Path(binding_path).resolve()), "binding_sha256": b["binding_sha256"], "workspace": str(ws),
               "station_meta": meta, "budget": bud, "caps": {**DEFAULT_CAPS, **(caps or {})},
               "decision": decision or site.cfg.get("decision", {"transport": "claude_cli", "model": "claude-opus-5-5"}),
               "tool_pins": catalog.pins(tools)}
        write_new(d / "campaign.json", cfg)
        write(d / "state.json", {"used": {k: 0 for k in bud}, "cap_used": {}, "candidates": {}, "datasets": {}, "steps": [],
                                 "retrieved": [], "consecutive_retrievals": 0, "frozen": None, "unseen": None,
                                 "memory": [], "started": time.time(), "ended": None})
        Ledger(d / "astra_ledger.jsonl").append("created", config_sha256=digest(cfg))
        return cls(d)

    # ------------------------------------------------------------------ state
    def save(self):
        write(self.dir / "state.json", self.s)

    def left(self, key) -> float:
        used = self.s["used"].get(key, 0)
        if key == "wall_hours":
            used = (time.time() - self.s["started"]) / 3600.0
        return self.cfg["budget"][key] - used

    def cap_left(self, cap_key) -> int | None:
        cap = self.cfg["caps"].get(cap_key)
        return None if cap is None else cap - self.s["cap_used"].get(cap_key, 0)

    def ready_tools(self) -> dict:
        out = {}
        for k, v in self.tools.items():
            m = v["manifest"]
            if m["status"] != "ready" or not catalog.applies(m, self.task["policy_io"].get("interface")):
                continue
            ck = m.get("cap_key")
            if ck and self.cap_left(ck) is not None and self.cap_left(ck) < 1:
                continue
            out[k] = v
        return out

    # ------------------------------------------------------------------ tool execution
    def _resolve(self, args: dict) -> dict:
        cands, dsets = {}, {}
        refs = [args[k] for k in REF_KEYS if k in args] + list(args.get("candidates", []))
        for name in refs:
            if name not in self.s["candidates"]:
                raise CampaignError(f"unknown candidate {name!r}; known: {sorted(self.s['candidates'])}")
            c = self.s["candidates"][name]
            cands[name] = {"kind": c["kind"], "path": str(self.ws / c["path"]), "tree_sha256": c["tree_sha256"],
                           "entry": c["entry"], "method": c["method"]}
        for item in args.get("datasets", []):
            name = item["name"] if isinstance(item, dict) else item
            if name not in self.s["datasets"]:
                raise CampaignError(f"unknown dataset {name!r}; known: {sorted(self.s['datasets'])}")
            dsets[name] = {"path": str(self.ws / self.s["datasets"][name]["path"]), **{k: v for k, v in self.s["datasets"][name].items() if k != "path"}}
        for key in ("name",):
            if key in args:
                if not re.fullmatch(NAME, args[key]):
                    raise CampaignError(f"{key} must match {NAME}")
                if args[key] in self.s["datasets"] or any(c == args[key] or c.startswith(args[key] + "_") for c in self.s["candidates"]):
                    raise CampaignError(f"name {args[key]!r} is already used")
        return {"candidates": cands, "datasets": dsets}

    def execute(self, tool_id: str, args: dict, *, host: dict | None = None) -> dict:
        """Run one registered tool.  `host` carries controller-only fields (e.g. the unseen check) never set by an agent."""
        catalog.check_pins(self.tools, self.cfg["tool_pins"])
        binding_mod.verify(self.cfg["binding"])
        if tool_id not in self.tools:
            raise CampaignError(f"unknown tool {tool_id!r}")
        t = self.tools[tool_id]
        m = t["manifest"]
        if m["status"] != "ready":
            raise CampaignError(f"tool {tool_id} is not executable: {m.get('status_note', '')}")
        if not catalog.applies(m, self.task["policy_io"].get("interface")):
            raise CampaignError(f"tool {tool_id} does not support this task's policy interface {self.task['policy_io'].get('interface')!r}")
        validate(args, m["arguments"])
        ck = m.get("cap_key")
        if ck and self.cap_left(ck) is not None and self.cap_left(ck) < 1:
            raise CampaignError(f"the campaign's cap for {ck} is used up")
        for key in ("train_episodes", "dev_evaluations", "gpu_hours", "wall_hours"):
            if not host and self.left(key) <= 0:
                raise CampaignError(f"budget {key} is used up; freeze a candidate or stop")
        resolved = self._resolve(args)
        self.s["jobs"] = self.s.get("jobs", 0) + 1
        job_id = f"{self.s['jobs']:03d}_{tool_id}"
        jd = self.dir / "jobs" / job_id
        jd.mkdir(parents=True)
        request = {"schema": "astra_station_request_v1", "job_id": job_id, "tool": tool_id, "arguments": args, "resolved": resolved,
                   "host": host or {}, "workspace": str(self.ws), "job_dir": str(jd), "state_dir": str(self.dir),
                   "station_meta": self.cfg["station_meta"], "task_id": self.binding["task_id"],
                   "budget_left": {k: self.left(k) for k in self.cfg["budget"]}, "site": self.binding["site"],
                   "decision": self.cfg["decision"]}
        rsha = catalog.request_sha(request)
        write_new(jd / "request.json", request)
        if ck:
            self.s["cap_used"][ck] = self.s["cap_used"].get(ck, 0) + 1
        self.save()
        self.ledger.append("tool_started", job_id=job_id, tool=tool_id, request_sha256=rsha)
        argv = [a.replace("{binding}", self.cfg["binding"]).replace("{request}", str(jd / "request.json"))
                 .replace("{result}", str(jd / "receipt.json")) for a in t["argv"]]
        t0 = time.time()
        with open(jd / "adapter.log", "w") as log:
            p = subprocess.Popen(argv, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                 env={**os.environ, "PYTHONPATH": str(SRC), "ASTRA_JOB_DIR": str(jd)})
            try:
                rc = p.wait(timeout=m["adapter"]["timeout_seconds"])
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                rc = 124
        hours = (time.time() - t0) / 3600.0
        receipt = None
        if (jd / "receipt.json").exists():
            receipt = read(jd / "receipt.json")
            try:
                validate(receipt, catalog.RECEIPT_SCHEMA)
                if receipt["request_sha256"] != rsha or receipt["tool"] != tool_id:
                    raise ValueError("receipt belongs to a different request")
            except ValueError as e:
                receipt = {"status": "failed", "error": f"malformed receipt: {e}", "usage": {}, "created": {}, "facts": {}}
        if receipt is None:
            receipt = {"status": "failed", "error": f"adapter exited {rc} without a receipt; see {jd / 'adapter.log'}",
                       "usage": {}, "created": {}, "facts": {}}
        usage = dict(receipt.get("usage") or {})
        usage.setdefault("gpu_hours", 0.0)
        for k, v in usage.items():
            if k in self.s["used"]:
                self.s["used"][k] += v
        created = self._register(receipt.get("created") or {}, tool_id, args, job_id) if receipt["status"] == "completed" else {}
        ck = m.get("cap_key")
        refunded = bool(ck and receipt["status"] == "failed" and INFRA_FAILURE in (receipt.get("error") or "")
                        and not (receipt.get("usage") or {}).get("train_episodes") and not (receipt.get("usage") or {}).get("dev_evaluations"))
        if refunded:          # the station never came up: the method was not tried, so the attempt is not counted
            self.s["cap_used"][ck] = max(0, self.s["cap_used"].get(ck, 0) - 1)
        result = {"tool": tool_id, "status": receipt["status"], "facts": receipt.get("facts", {}), "created": created,
                  "usage": usage, "wall_minutes": round(hours * 60, 1)}
        if receipt.get("error"):
            result["error"] = receipt["error"][:1500]
        if refunded:
            result["note"] = f"infrastructure failure (station unavailable); the {ck} attempt was not counted"
        self.save()
        self.ledger.append("tool_finished", job_id=job_id, tool=tool_id, status=receipt["status"], usage=usage,
                           created=created, receipt_sha256=digest(receipt))
        return result

    def _register(self, created: dict, tool_id: str, args: dict, job_id: str) -> dict:
        out = {"candidates": [], "datasets": [], "evaluations": []}
        for c in created.get("candidates", []):
            p = Path(c["path"]).resolve()
            if self.ws.resolve() not in p.parents:
                raise CampaignError(f"candidate {c['name']} is outside the workspace")
            if c["name"] in self.s["candidates"]:
                raise CampaignError(f"duplicate candidate {c['name']}")
            entry = c.get("entry") or ("policy.py" if c["kind"] == "policy" else "program.py")
            if not (p / entry).is_file():
                raise CampaignError(f"candidate {c['name']} lacks its entry file {entry}")
            self.s["candidates"][c["name"]] = {
                "kind": c["kind"], "path": str(p.relative_to(self.ws.resolve())), "entry": entry, "tree_sha256": tree_hash(p),
                "method": c.get("method", tool_id), "tool": tool_id, "job": job_id, "parent": c.get("parent"),
                "params": c.get("params", args), "note": c["note"], "evals": []}
            out["candidates"].append(c["name"])
        for d in created.get("datasets", []):
            p = Path(d["path"]).resolve()
            if self.ws.resolve() not in p.parents:
                raise CampaignError(f"dataset {d['name']} is outside the workspace")
            self.s["datasets"][d["name"]] = {"path": str(p.relative_to(self.ws.resolve())), "episodes": d["episodes"],
                                             "frames": d.get("frames"), "domain": d.get("domain"), "note": d["note"],
                                             "tool": tool_id, "job": job_id, "tree_sha256": tree_hash(p)}
            out["datasets"].append(d["name"])
        for e in created.get("evaluations", []):
            c = self.s["candidates"].get(e["candidate"])
            if c is None:
                raise CampaignError(f"evaluation of unknown candidate {e['candidate']}")
            if tree_hash(self.ws / c["path"]) != c["tree_sha256"]:
                raise CampaignError(f"candidate {e['candidate']} changed after registration")
            row = {k: e.get(k) for k in ("starts", "n", "success", "terms", "per_episode", "diagnostics")}
            row["job"] = job_id
            if tool_id == "evaluate" and e["starts"] == "train":
                row.update(real_time=e.get("real_time"), tree_sha256=c["tree_sha256"])
            c["evals"].append(row)
            out["evaluations"].append({"candidate": e["candidate"], "starts": e["starts"], "success": e["success"], "n": e["n"]})
        return out

    # ------------------------------------------------------------------ host tools
    def _legacy_timing(self, name, tree_sha256, job_id):
        """Read a pre-release timing measurement from its original evaluation request and receipt."""
        if not re.fullmatch(r"[0-9]+_evaluate", job_id):
            return None
        folder = self.dir / "jobs" / job_id
        try:
            request, receipt = read(folder / "request.json"), read(folder / "receipt.json")
        except (FileNotFoundError, ValueError):
            return None
        if (request.get("arguments", {}).get("candidate") != name or request["arguments"].get("starts") != "train"
                or request.get("resolved", {}).get("candidates", {}).get(name, {}).get("tree_sha256") != tree_sha256
                or receipt.get("request_sha256") != catalog.request_sha(request) or receipt.get("status") != "completed"
                or receipt.get("tool") != "evaluate" or request.get("job_id") != job_id):
            return None
        for evaluation in receipt.get("created", {}).get("evaluations", []):
            if evaluation.get("candidate") == name and evaluation.get("starts") == "train":
                return evaluation.get("real_time")
        return None

    def _freeze_timing(self, name, candidate):
        cost = (self.task.get("real") or {}).get("worker_cost")
        if candidate["kind"] != "program" or not cost:
            return []
        evidence = []
        for evaluation in candidate["evals"]:
            if evaluation["starts"] != "train":
                continue
            job = evaluation["job"]
            if "tree_sha256" in evaluation:
                if evaluation["tree_sha256"] != candidate["tree_sha256"]:
                    continue
                timing = evaluation.get("real_time")
            else:
                timing = self._legacy_timing(name, candidate["tree_sha256"], job)
            if timing is None:
                continue
            if (not isinstance(timing, dict) or type(timing.get("program_steps")) is not int
                    or timing["program_steps"] <= 0 or timing.get("budget_ms") != cost["step_budget_ms"]
                    or type(timing.get("steps_over_budget")) is not int):
                raise IneligibleCandidate(f"{name}: training timing in {job} is incomplete or uses a different step budget")
            if timing["steps_over_budget"] != 0:
                raise IneligibleCandidate(f"{name}: training timing in {job} reports {timing['steps_over_budget']} steps over budget")
            evidence.append({"job": job, "real_time": timing})
        if not evidence:
            raise IneligibleCandidate(f"{name}: evaluate this program on training starts to measure its worker step budget before freezing")
        return evidence

    def freeze(self, name: str, reason: str) -> dict:
        if self.s["frozen"]:
            raise CampaignError("already frozen")
        c = self.s["candidates"].get(name)
        if c is None or not any(e["starts"] == "dev" for e in c["evals"]):
            raise IneligibleCandidate("freeze needs a candidate with a development evaluation in this campaign")
        if tree_hash(self.ws / c["path"]) != c["tree_sha256"]:
            raise IneligibleCandidate("candidate changed after registration")
        timing = self._freeze_timing(name, c)
        snap = self.dir / "frozen"
        if snap.exists():
            raise CampaignError("frozen snapshot already exists")
        shutil.copytree(self.ws / c["path"], snap)
        rec = {"candidate": name, "policy_file": c["entry"], "program": c["kind"] == "program", "tree_sha256": c["tree_sha256"],
               "reason": reason, "frozen_at": now(),
               "used": {**self.s["used"], "wall_hours": round((time.time() - self.s["started"]) / 3600.0, 2)},
               "dev": [e for e in c["evals"] if e["starts"] == "dev"]}
        if timing:
            rec["worker_timing"] = timing
        write_new(self.dir / "frozen.json", rec)
        self.s["frozen"] = rec
        self.save()
        self.ledger.append("frozen", candidate=name, tree_sha256=c["tree_sha256"])
        if self.binding["splits"]["unseen_check"]["count"] and "evaluate" in self.tools:
            try:
                res = self.execute("evaluate", {"candidate": name, "starts": "dev"}, host={"partition": "unseen"})
                self.s["unseen"] = res
            except CampaignError as e:
                self.s["unseen"] = {"error": str(e)}
            self.save()
            self.ledger.append("unseen_check", result=self.s["unseen"])
        return rec

    def heldout(self) -> dict:
        """Coordinator only: the sealed evaluation of the frozen candidate, once, by the shared held-out tool."""
        if not self.s["frozen"]:
            raise CampaignError("nothing frozen")
        if (self.dir / "heldout.json").exists():
            raise CampaignError("the sealed evaluation already ran")
        meta = self.cfg["station_meta"]
        self.site.tt.cmd_heldout(argparse.Namespace(campaign=meta["campaign"], method="astra", task=meta["task"], run_id=meta["run_id"],
                                                    sandbox_gpu=self.site.cfg.get("sandbox_gpu", True)))
        res = read(self.dir / "heldout.json")
        self.ledger.append("heldout", result=res)
        return res

    def handoff(self) -> dict:
        """Operator only, after the sealed evaluation: hand the frozen snapshot, byte for byte, to the task's real-robot
        chain (task spec real.handoff): a deliverable registry its station worker reads (the registry and worker the
        baselines use), or a package builder that wraps it into a real-robot package.  Real validation stays pending."""
        h = (self.task.get("real") or {}).get("handoff") or self.site.cfg.get("real_handoff")
        if not h:
            raise CampaignError("the task spec names no real.handoff (registry or builder)")
        if not (self.dir / "heldout.json").exists():
            raise CampaignError("hand off only after the sealed evaluation")
        meta, fr, ho = self.cfg["station_meta"], self.s["frozen"], read(self.dir / "heldout.json")
        name = f"astra_{meta['task']}_{meta['campaign']}_{meta['run_id']}".lower()
        facts = {"kind": "program" if fr["program"] else "policy", "task": meta["task"], "selected": fr["candidate"],
                 "campaign": f"astra station harness {meta['campaign']}/{meta['run_id']}",
                 "heldout_sim": f"{ho.get('success')}/{ho.get('heldout_n', ho.get('n'))}",
                 "dev_sim": [f"{e['success']}/{e['n']}" for e in fr["dev"]],
                 "note": f"frozen candidate {fr['candidate']} of the Astra station harness; real validation pending"}
        if h.get("builder"):
            out = self.site.task_path(self.task, h["out_dir"]) / name
            if out.exists():
                raise CampaignError(f"{out} already exists")
            argv = [self.site.worker_python, str(self.site.task_path(self.task, h["builder"])), "--frozen", str(self.dir / "frozen"),
                    "--entry", fr["policy_file"], "--kind", facts["kind"], "--out", str(out), "--facts", json.dumps(facts)]
            p = subprocess.run(argv, capture_output=True, text=True, timeout=1800)
            if p.returncode != 0:
                raise CampaignError(f"the package builder failed: {p.stderr[-1500:]}")
            if tree_hash(out / "policy") != fr["tree_sha256"]:
                raise CampaignError("the packaged deliverable differs from the frozen snapshot")
            self.ledger.append("handoff", name=name, package=str(out))
            return {"packaged": str(out), **facts}
        reg_p = self.site.task_path(self.task, h["registry"])
        reg = read(reg_p)
        if name in reg:
            raise CampaignError(f"{name} is already registered")
        dst = self.site.task_path(self.task, h["deliverables_dir"]) / name
        shutil.copytree(self.dir / "frozen", dst)
        if tree_hash(dst) != fr["tree_sha256"]:
            raise CampaignError("the copied deliverable differs from the frozen snapshot")
        reg[name] = {"path": str(dst / fr["policy_file"]), "sha256": self.site.tt.tree_sha(dst), **facts}
        write(reg_p, reg)
        self.ledger.append("handoff", name=name, registry=str(reg_p))
        return {"registered": name, **reg[name]}

    # ------------------------------------------------------------------ summaries for the deciding agent
    def leaderboard(self) -> list:
        rows = []
        for name, c in self.s["candidates"].items():
            ev = {}
            for e in c["evals"]:
                k = e["starts"]
                agg = ev.setdefault(k, {"success": 0, "n": 0, "terms": {}})
                agg["success"] += e["success"]
                agg["n"] += e["n"]
                for t, v in (e.get("terms") or {}).items():
                    agg["terms"][t] = agg["terms"].get(t, 0) + v
            rows.append({"candidate": name, "kind": c["kind"], "method": c["method"], "parent": c.get("parent"),
                         "note": c["note"], "evaluations": ev,
                         "dev_diagnostics": next((e.get("diagnostics") for e in reversed(c["evals"]) if e["starts"] == "dev"), None)})
        rows.sort(key=lambda r: -(r["evaluations"].get("dev", {}).get("success", -1) / max(r["evaluations"].get("dev", {}).get("n", 1), 1)))
        return rows

    def paired(self, a: str, b: str) -> dict | None:
        """Per-start comparison on the development starts (same ids for every candidate)."""
        def per(name):
            out = {}
            for e in self.s["candidates"][name]["evals"]:
                if e["starts"] == "dev":
                    for row in e.get("per_episode") or []:
                        out[row["id"]] = bool(row["success"])
            return out
        pa, pb = per(a), per(b)
        ids = sorted(set(pa) & set(pb))
        if not ids:
            return None
        return {"starts": len(ids), "both": sum(pa[i] and pb[i] for i in ids), f"only_{a}": sum(pa[i] and not pb[i] for i in ids),
                f"only_{b}": sum(pb[i] and not pa[i] for i in ids), "neither": sum(not pa[i] and not pb[i] for i in ids)}
