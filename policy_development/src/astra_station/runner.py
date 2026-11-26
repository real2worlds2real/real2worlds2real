"""The Astra decision loop over a station campaign (agent-neutral).

Each decision is one fresh session of the configured agent with the whole campaign context in the prompt: role and rules,
task, real-robot constraints, the registered method tools and the host tools, budgets, retrieved experience and borrowable
reward terms, datasets, the candidate leaderboard with paired comparisons, and the history.  The agent answers one JSON
object naming a tool and its arguments; the controller validates and executes it and records the result.  The loop ends
when the agent freezes a candidate or stops, or when a budget ends (then the host freezes the best development
candidate).  The sealed evaluation is never run here.
"""
from __future__ import annotations

import json
import time

from . import binding as binding_mod
from . import catalog, transport
from .campaign import Campaign, CampaignError, IneligibleCandidate
from .common import digest, now, tree_hash, validate, write
from .memory import CARD_SCHEMA, Library, WorkLibrary
from .site import REPO

ROLE = """You are Astra, the policy-development agent of a robot-learning harness. You decide; the harness executes and
measures. Your job: deliver the policy (a station policy module or a program) with the highest verified success on this task,
and one that will also work on the real robot. You may use any registered method: write a controller (code as policy), learn
from demonstrations (real, simulated or mixed), record demonstrations from any candidate or teacher, post-train a candidate with
residual reinforcement learning, train with reinforcement learning, or compose stages. Compare candidates on the same
development starts and keep the best. Retrieve experience from similar tasks (same scene, objects or phases) and borrow what
transfers (recipes, reward terms, failure modes, sim-to-real lessons); treat it as a hypothesis to re-validate here."""

RULES = """RULES
- One action per decision. State one falsifiable hypothesis and what result would change your plan.
- Never claim a result before it is measured. Evaluate before training more; an evaluation is cheap next to a training run.
- A candidate is better only if it beats the current best on the same development starts (see the paired comparisons);
  with 15 starts a difference of 1-2 successes is noise.
- The deliverable runs on the real robot: it may use only the observations the real robot provides at run time; simulator-only
  information may be used by teachers, data generation and rewards only. Respect the real-robot constraints below.
- Freeze when further work is unlikely to beat the best candidate within the remaining budget, or when the budget nearly ends.
- The development starts are open; the unseen-check starts are run once by the harness when you freeze; the sealed starts are
  never available to you."""

HOST_TOOLS = [
    {"tool": "reward_search", "family": "memory",
     "description": "Search the work library of reviewed RL rewards (~200: RoboGen, Text2Reward, HIL-SERL, BEHAVIOR, IndustReal, ResFiT, "
                    "AutoMATE, DrEureka, Eureka, RoboCasa and the lab's native rewards) and recipes by skill tags (e.g. skill:grasp, "
                    "skill:lift, skill:place, skill:insertion, form:dense, family:manipulation) and/or text. Results appear in the next "
                    "prompt. To use one, transcribe it into an `expression` reward term over station features. Counts toward the 3 "
                    "consecutive retrievals.",
     "arguments": {"type": "object", "properties": {"tags": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
                                                    "text": {"type": "string", "maxLength": 300},
                                                    "kinds": {"type": "array", "items": {"enum": ["reward", "recipe", "lesson", "failure", "task_experience"]}},
                                                    "k": {"type": "integer", "minimum": 1, "maximum": 20}}, "additionalProperties": False}},
    {"tool": "retrieve", "family": "memory",
     "description": "Search the experience library beyond what is shown (filters on card tags: scene, objects, phases, methods, "
                    "domain=sim|real, robot, or text). At most 3 consecutive retrievals.",
     "arguments": {"type": "object", "properties": {"filters": {"type": "object"}, "k": {"type": "integer", "minimum": 1, "maximum": 20}},
                   "additionalProperties": False}},
    {"tool": "remember", "family": "memory",
     "description": "Write one experience card (a verified finding of this campaign, with evidence = job ids / numbers) for "
                    "future tasks. Fields: id, title, tags{scene, objects, phases, methods, domain}, finding, evidence[], "
                    "applicability, limitations.",
     "arguments": {"type": "object", "required": ["card"], "properties": {"card": {"type": "object"}}, "additionalProperties": False}},
    {"tool": "freeze", "family": "selection",
     "description": "Hand over one candidate that has a development evaluation; ends the campaign. The harness then runs the "
                    "unseen-check starts once on it.",
     "arguments": {"type": "object", "required": ["candidate", "reason"],
                   "properties": {"candidate": {"type": "string"}, "reason": {"type": "string"}}, "additionalProperties": False}},
    {"tool": "stop", "family": "selection", "description": "End without freezing (only if no candidate can be delivered).",
     "arguments": {"type": "object", "required": ["reason"], "properties": {"reason": {"type": "string"}}, "additionalProperties": False}},
]


def _compact(v, limit=1600):
    s = json.dumps(v, ensure_ascii=False, allow_nan=False)
    return s if len(s) <= limit else s[:limit] + "...(truncated)"


TRANSPORT_PAUSE = 3          # consecutive transport failures before the loop pauses


class Runner:
    def __init__(self, campaign: Campaign, agent=None):
        self.c = campaign
        self.agent = agent or transport.make(campaign.cfg["decision"])
        self.lib = Library(campaign.site.library_dir)
        wl = campaign.site.cfg.get("work_library")
        self.wl = WorkLibrary({**wl, "module": str(REPO / wl["module"]) if not wl["module"].startswith("/") else wl["module"]}) if wl else None

    # ------------------------------------------------------------------ prompt
    def prompt(self) -> str:
        c = self.c
        tags = c.binding["task"]["tags"]
        task_md = (c.ws / "docs" / "task.md").read_text()
        station_md = (c.ws / "docs" / "station.md").read_text()
        auto = self.lib.retrieve(tags, k=12)
        extra = c.s["retrieved"][-3:]
        rewards = self.lib.rewards_for(tags.get("phases", []))
        board = c.leaderboard()
        best = next((r["candidate"] for r in board if r["evaluations"].get("dev")), None)
        pairs = {r["candidate"]: c.paired(r["candidate"], best) for r in board[1:6] if best and r["evaluations"].get("dev")} if best else {}
        menu = catalog.menu(c.ready_tools()) + HOST_TOOLS
        budget = {k: {"total": c.cfg["budget"][k], "left": round(c.left(k), 2)} for k in c.cfg["budget"]}
        caps = {k: {"cap": v, "left": c.cap_left(k)} for k, v in c.cfg["caps"].items()}
        hist = []
        for st in c.s["steps"][-14:]:
            hist.append({"step": st["step"], "decision": {k: st["decision"].get(k) for k in ("tool", "arguments", "hypothesis")},
                         "result": _compact(st["result"], 1400)})
        parts = [
            ROLE, RULES + self._real_time_rule(),
            ("SITE NOTE\n" + c.site.cfg["campaign_note"]) if c.site.cfg.get("campaign_note") else "",
            "TASK DOCUMENT (as given to every method)\n" + task_md,
            "STATION DOCUMENT (the simulation twin's interface; deliverable formats)\n" + station_md,
            "BINDING (public facts)\n" + _compact(binding_mod.public(c.binding), 4000),
            "REAL-ROBOT CONSTRAINTS (verified on this set-up)\n" + "\n".join("- " + x for x in c.binding["task"].get("real_constraints", [])),
            "TOOLS (answer with exactly one)\n" + json.dumps(menu, ensure_ascii=False, indent=0),
            "BUDGET\n" + json.dumps({"budget": budget, "per_method_caps": caps, "model_calls_used": c.s["used"]["model_calls"]}),
            "EXPERIENCE (retrieved by tag overlap with this task; hypotheses to re-validate)\n" + json.dumps(auto, ensure_ascii=False, indent=0),
            ("EXPERIENCE YOU RETRIEVED\n" + json.dumps(extra, ensure_ascii=False, indent=0)) if extra else "",
            ("LAB REWARD TERMS (implemented over station observations; usable directly in train_residual)\n" + json.dumps(rewards, ensure_ascii=False, indent=0))
            if "train_residual" in c.ready_tools() else
            ("LAB REWARD TERMS (references for research sessions, which receive them in rewards.md)\n" + json.dumps(rewards, ensure_ascii=False, indent=0)),
            self._work_library_section(tags),
            ("REWARD SEARCHES YOU RAN\n" + json.dumps(c.s.get("reward_searches", [])[-2:], ensure_ascii=False, indent=0)) if c.s.get("reward_searches") else "",
            "DATASETS\n" + (json.dumps(c.s["datasets"], ensure_ascii=False, indent=0) if c.s["datasets"] else "none yet (the real demonstrations "
                                                                                                         "in the workspace can be imported with a data tool)"),
            "CANDIDATES (best development rate first)\n" + (json.dumps(board, ensure_ascii=False, indent=0) if board else "none yet"),
            ("PAIRED COMPARISONS AGAINST THE CURRENT BEST (" + str(best) + ")\n" + json.dumps(pairs)) if pairs else "",
            "HISTORY OF THIS CAMPAIGN\n" + (json.dumps(hist, ensure_ascii=False, indent=0) if hist else "none yet"),
            'ANSWER with one JSON object only: {"hypothesis": "...", "tool": "<tool>", "arguments": {...}, "rationale": "<= 150 words"}',
        ]
        return "\n\n".join(p for p in parts if p)

    def _real_time_rule(self) -> str:
        """The real worker's per-step budget (task spec real.worker_cost); programs are measured against it in evaluate."""
        cost = (self.c.task.get("real") or {}).get("worker_cost")
        if not cost:
            return ""
        kin = (f" and solves fk / ik on a CPU (~{cost['ik_ms']:g} ms per ik point)" if cost.get("ik_ms") else "")
        return (f"\n- The real robot runs the deliverable on its station worker with a ~{cost['step_budget_ms']:g} ms per-step budget{kin}. "
                "Before freezing a program, evaluate it on training starts and check real_time: steps_over_budget must be 0 "
                "(a step that plans a whole path at once stalls the real robot).")

    def _work_library_section(self, tags) -> str:
        if self.wl is None:
            return ""
        from_task = self.wl.for_task(tags, k=10)
        import importlib.util
        spec = importlib.util.spec_from_file_location("astra_rewards_doc", REPO / "policy_development/runtime/rewards.py")
        feats = ""
        try:
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
            feats = ", ".join(m.FEATURES)
        except Exception:  # noqa: BLE001 - numpy missing on the controller host: list without the module
            pass
        head = ("WORK LIBRARY REWARDS (reviewed literature + lab rewards matched on this task's skills " + json.dumps(from_task["query_tags"])
                + f"; {from_task['total_matches']} matches, top {len(from_task['items'])} shown; use reward_search for more). References: they "
                "use simulator signals; ")
        if "train_residual" in self.c.ready_tools():
            how = ("to use one in train_residual, transcribe it as {\"id\": \"expression\", \"weight\": w, \"params\": "
                   "{\"expr\": \"...\", \"source\": \"<work library id>\"}} over station features: " + (feats or "see train_residual")
                   + "; math: tanh, exp, abs, min, max, sqrt, clip, ge, le.")
        else:
            how = "research sessions receive the matches in rewards.md and implement a reward over this task's observations themselves."
        return head + how + "\n" + json.dumps(from_task["items"], ensure_ascii=False, indent=0)

    # ------------------------------------------------------------------ host tools
    def _host(self, tool: str, args: dict) -> dict:
        c = self.c
        if tool == "reward_search":
            if self.wl is None:
                raise CampaignError("no work library configured for this site")
            if c.s["consecutive_retrievals"] >= 3:
                raise CampaignError("three consecutive retrievals; choose another tool")
            res = self.wl.search(tags=args.get("tags"), text=args.get("text", ""), kinds=args.get("kinds") or ["reward"], k=args.get("k", 8))
            c.s.setdefault("reward_searches", []).append({"query": args, **res})
            c.s["consecutive_retrievals"] += 1
            return {"total_matches": res["total_matches"], "titles": [i["title"] for i in res["items"]]}
        if tool == "retrieve":
            if c.s["consecutive_retrievals"] >= 3:
                raise CampaignError("three consecutive retrievals; choose another tool")
            res = self.lib.retrieve(c.binding["task"]["tags"], args.get("filters"), k=args.get("k", 8))
            c.s["retrieved"].append({"filters": args.get("filters"), "cards": res})
            c.s["consecutive_retrievals"] += 1
            return {"cards": [r["id"] for r in res]}
        c.s["consecutive_retrievals"] = 0
        if tool == "remember":
            if not any(e["starts"] == "dev" for v in c.s["candidates"].values() for e in v["evals"]):
                raise CampaignError("remember needs at least one development evaluation in this campaign")
            card = {**args["card"], "schema": "astra_experience_card_v1", "status": "candidate",
                    "source": f"campaign {c.cfg['campaign']}/{c.binding['task_id']}_{c.cfg['run_id']}", "date": now()[:10]}
            card["id"] = f"{c.cfg['campaign']}_{card.get('id', 'card')}".lower()
            validate(card, CARD_SCHEMA)
            c.s["memory"].append(self.lib.record(card))
            return {"recorded": card["id"]}
        if tool == "freeze":
            return {"frozen": c.freeze(args["candidate"], args["reason"]), "unseen_check": c.s.get("unseen")}
        if tool == "stop":
            c.s["ended"] = {"by": "agent", "reason": args["reason"], "at": now()}
            return {"stopped": True}
        raise CampaignError(f"unknown tool {tool}")

    # ------------------------------------------------------------------ loop
    def step(self) -> dict:
        c = self.c
        n = len(c.s["steps"]) + 1
        prompt = self.prompt()
        c.s["used"]["model_calls"] += 1
        c.save()
        t0 = time.time()
        try:
            decision, audit = self.agent.decide(prompt)
        except Exception as e:  # noqa: BLE001
            decision, audit = {}, {"error": f"{type(e).__name__}: {e}"[:800]}
        audit["seconds"] = round(time.time() - t0, 1)
        audit["prompt_sha256"] = digest(prompt)
        (c.dir / "decisions").mkdir(exist_ok=True)
        write(c.dir / "decisions" / f"{n:03d}.json", {"decision": decision, "audit": audit, "prompt": prompt})
        tool, args = decision.get("tool"), decision.get("arguments")
        c.ledger.append("decision", step=n, tool=tool, arguments=args, hypothesis=decision.get("hypothesis"), audit_transport=audit.get("transport"),
                        prompt_sha256=audit["prompt_sha256"])
        try:
            if not isinstance(args, dict) or not tool:
                raise CampaignError("no valid decision (needs tool and arguments)" + (f": {audit['error']}" if "error" in audit else ""))
            host = {h["tool"]: h for h in HOST_TOOLS}
            if tool in host:
                validate(args, host[tool]["arguments"])
                result = self._host(tool, args)
            else:
                c.s["consecutive_retrievals"] = 0
                result = c.execute(tool, args)
        except (CampaignError, ValueError) as e:
            result = {"error": str(e)[:1500]}
        c.s["steps"].append({"step": n, "decision": decision, "result": result})
        c.save()
        return {"step": n, "tool": tool, "result": result, "transport_error": not decision and "error" in audit}

    def run(self, log=print) -> dict:
        c = self.c
        failed = 0                                  # consecutive decisions the transport itself could not deliver
        while not c.s["frozen"] and not c.s["ended"]:
            if c.left("model_calls") < 1 or c.left("wall_hours") <= 0 or c.left("gpu_hours") <= 0:
                self._budget_end(log)
                break
            out = self.step()
            log(f"[astra] step {out['step']}: {out['tool']} -> {_compact(out['result'], 700)}")
            failed = failed + 1 if out["transport_error"] else 0
            if failed >= TRANSPORT_PAUSE:
                # not the agent's decisions (a logged-out CLI, no network): give the calls back and pause unended, so
                # the same `run --state` resumes once the transport works again
                c.s["used"]["model_calls"] -= failed
                c.save()
                c.ledger.append("paused", reason=out["result"].get("error", "")[:300], refunded_model_calls=failed)
                log(f"[astra] paused: the decision transport failed {failed} times in a row; fix it, then run again")
                return {"frozen": None, "unseen": None, "ended": None, "paused": out["result"].get("error", "")[:300]}
        c.s["ended"] = c.s["ended"] or {"by": "frozen", "at": now()}
        c.save()
        c.ledger.append("ended", ended=c.s["ended"], frozen=(c.s["frozen"] or {}).get("candidate"))
        try:
            c.s["memory"].append(self.summary_card())
            c.save()
        except (ValueError, OSError) as e:          # a duplicate id or an empty campaign: nothing to write back
            log(f"[astra] no summary card: {e}")
        return {"frozen": c.s["frozen"], "unseen": c.s["unseen"], "ended": c.s["ended"]}

    def summary_card(self) -> str:
        """Factual development record of this campaign for future retrieval (status candidate: not transfer-validated)."""
        c = self.c
        board = c.leaderboard()
        if not board:
            raise ValueError("no candidates")
        lines = []
        for r in board:
            ev = "; ".join(f"{k} {v['success']}/{v['n']}" for k, v in r["evaluations"].items()) or "not evaluated"
            lines.append(f"{r['candidate']} [{r['method']}]: {ev} - {r['note'][:160]}")
        fr = (c.s["frozen"] or {}).get("candidate")
        tags = dict(c.binding["task"]["tags"])
        card = {"schema": "astra_experience_card_v1",
                "id": f"campaign_{c.cfg['campaign']}_{c.binding['task_id']}_{c.cfg['run_id']}".lower(),
                "title": f"{c.binding['task_id']}: what this campaign tried and measured (frozen: {fr})",
                "tags": {"scene": tags.get("scene"), "objects": tags.get("objects", []), "phases": tags.get("phases", []),
                         "robot": tags.get("robot"), "methods": sorted({r["method"] for r in board}), "domain": ["sim"]},
                "finding": "Development results in the station twin (15 open starts unless noted): " + " | ".join(lines)
                           + (f" | unseen check of the frozen candidate: {c.s['unseen'].get('facts', {}).get('success')}" if c.s.get("unseen") else ""),
                "evidence": [str(c.dir / "state.json"), str(c.dir / "astra_ledger.jsonl")],
                "applicability": "Same scene and phases; a starting point, not a result.",
                "limitations": "Simulation development starts only; no sealed or real-robot result in this card.",
                "source": f"campaign {c.cfg['campaign']}", "status": "candidate", "date": now()[:10]}
        return self.lib.record(card)

    def _budget_end(self, log):
        c = self.c
        for row in c.leaderboard():
            if not row["evaluations"].get("dev"):
                continue
            name = row["candidate"]
            try:
                c.freeze(name, "host: budget ended; best eligible development rate")
            except IneligibleCandidate as exc:
                log(f"[astra] skipped {name}: {exc}")
                continue
            c.s["ended"] = {"by": "budget", "frozen": name, "at": now()}
            log(f"[astra] budget ended; host froze {name}")
            return
        c.s["ended"] = {"by": "budget", "reason": "budget ended without an eligible evaluated candidate", "at": now()}
        log("[astra] budget ended without an eligible evaluated candidate")
