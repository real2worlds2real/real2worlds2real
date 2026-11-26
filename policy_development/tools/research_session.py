"""Tool `research_session`: a confined auto-research session under the ENPIRE contract (reinforcement learning on the
station, learning from demonstrations, improving given candidates, or code as policy - the worker chooses), driven by
Astra's brief, the selected experience and reward terms.  It runs on its own station with its own training-episode cap;
its frozen (or best) evaluation becomes a campaign candidate.  Its evaluations count as development evaluations and its
finished training episodes as training episodes of the campaign."""
import json
import shutil
import time
from pathlib import Path

from _adapter import HERE, main

import diagnostics
import subsession


def run(job):
    a = job.args
    left_ev = int(job.request["budget_left"]["dev_evaluations"])
    left_tr = int(job.request["budget_left"]["train_episodes"])
    evals = min(int(a.get("evaluations", 6)), left_ev)
    train = min(int(a.get("train_episodes", 600)), left_tr)
    if evals < 1:
        raise RuntimeError("no development evaluation left for a research session")
    hours = float(a.get("hours", 4.0))
    limits = {"evals": evals, "train_episodes": train, "deadline": time.time() + hours * 3600, "brief": a["brief"]}
    ws, meta = subsession.build(job, "research", a["name"], limits)
    for nm, c in job.request["resolved"]["candidates"].items():           # candidates handed to the worker (read-only copies)
        shutil.copytree(c["path"], ws / "candidates" / nm)
    prompt = (HERE / "prompts" / (job.site.cfg.get("prompts") or {}).get("research", "research.md")).read_text().format(brief=a["brief"], train_episodes=train, evals=evals, wall_h=hours)
    audit = subsession.launch(job, ws, meta, prompt, "research", timeout_s=int(hours * 3600) + 1800)
    L = subsession.ledger(job, meta)
    used_train = subsession.station_train_episodes(job, meta)
    ev_rows = [r for r in L if r.get("kind") == "eval_result" and r.get("trials")]
    job.charge("dev_evaluations", sum(r.get("kind") == "eval" for r in L))
    job.charge("train_episodes", used_train)
    job.charge("gpu_hours", audit.get("wall_hours", 0.0))
    facts = {"evaluations": [{k: r.get(k) for k in ("name", "success", "trials", "terms")} for r in ev_rows], "train_episodes": used_train,
             "session": {k: audit.get(k) for k in ("exit_code", "num_turns", "total_cost_usd", "answered_models", "wall_hours")},
             "workspace": str(ws)}
    it = sorted((ws / "iterations").glob("*.md"))
    facts["iterations"] = [p.name for p in it][-8:]
    if not ev_rows:
        job.finish(facts=facts, status="failed", error="the session made no completed evaluation")
        return
    fz = job.tt.cdir(meta) / "frozen.json"
    if fz.exists():
        fr = json.loads(fz.read_text())
        pick = next(r for r in ev_rows if r["name"] == fr["evaluation"])
        facts["frozen_by_worker"] = fr["evaluation"]
    else:
        pick = max(ev_rows, key=lambda r: r["success"])
        facts["frozen_by_worker"] = None
    e0 = next(r for r in L if r.get("kind") == "eval" and r["name"] == pick["name"])
    snap = ws / "runs" / pick["name"] / "deliverable"
    d = job.ws / "candidates" / a["name"]
    shutil.copytree(snap, d, ignore=shutil.ignore_patterns("__pycache__"))
    entry = Path(e0["policy"]).name
    summ = json.loads((ws / "runs" / pick["name"] / f"{pick['name']}_summary.json").read_text())
    rows = [{"id": r["id"], "success": bool(r["success"]), "term": r["term"], "steps": int(r["steps"]), "program_error": r.get("program_error")}
            for r in summ["episodes"]]
    rows = diagnostics.from_artifacts(ws / "runs" / pick["name"] / "trials", job.binding["task"].get("labels", {}), rows)
    ev = {"candidate": a["name"], "starts": "dev", "n": summ["n"], "success": summ["success"], "terms": summ["terms"],
          "per_episode": [{k: r.get(k) for k in ("id", "success", "term", "steps", "program_error")} for r in rows],
          "diagnostics": diagnostics.aggregate(rows)}
    kind = "program" if e0.get("program") else "policy"
    facts["selected"] = {"evaluation": pick["name"], "development": f"{summ['success']}/{summ['n']}", "kind": kind}
    job.finish(facts=facts, created={"candidates": [{"name": a["name"], "kind": kind, "path": str(d), "entry": entry, "method": "research",
                                                     "note": f"research session deliverable '{pick['name']}' ({summ['success']}/{summ['n']} dev)",
                                                     "params": {"brief": a["brief"][:400], "train_episodes": train, "evaluations": evals}}],
                                     "evaluations": [ev]})


if __name__ == "__main__":
    main(run)
