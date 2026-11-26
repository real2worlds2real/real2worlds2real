"""Tool `synthesize_program`: a confined program-synthesis session (code as policy, ASPIRE fix-loop discipline) driven by
Astra's brief and the selected experience; its best numbered version becomes a campaign candidate together with the
development evaluation it was selected on.  Each numbered version run counts as one development evaluation."""
import json
import re
import shutil
import time
from pathlib import Path

from _adapter import HERE, main

import diagnostics
import subsession


def run(job):
    a = job.args
    left = int(job.request["budget_left"]["dev_evaluations"])
    versions = min(int(a.get("versions", 6)), left)
    if versions < 2:
        raise RuntimeError(f"a program session needs at least 2 development evaluations; {left} left")
    hours = float(a.get("hours", 3.0))
    limits = {"versions": versions, "probes": 20, "train_episodes": 0, "deadline": time.time() + hours * 3600, "brief": a["brief"]}
    ws, meta = subsession.build(job, "program", a["name"], limits)
    note = ""
    if a.get("start_from"):
        c = job.request["resolved"]["candidates"][a["start_from"]]
        if c["kind"] != "program":
            raise RuntimeError("start_from must be a program candidate")
        shutil.copy2(Path(c["path"]) / c["entry"], ws / "work" / "code" / f"start_{a['start_from']}.py")
        note = f"- `work/code/start_{a['start_from']}.py` - an earlier program of this campaign to start from (its development result is in the brief)."
    dev = list(job.site.cfg["splits"]["development"])
    cost = job.real_cost()
    real_note = (f" Per-step compute is measured: on the real robot `fk` / `ik` run on a CPU (~{cost['ik_ms']:g} ms per ik point) "
                 f"with ~{cost['step_budget_ms']:g} ms per step, so call `ik` for one or a few points per step (never a whole path "
                 "at once).") if cost and cost.get("ik_ms") else (
                 f" On the real robot each step must finish within ~{cost['step_budget_ms']:g} ms." if cost else "")
    prompt = (HERE / "prompts" / (job.site.cfg.get("prompts") or {}).get("program", "program.md")).read_text().format(brief=a["brief"], versions=versions, probes=limits["probes"],
                                                                  start_from_note=note, first_seed=dev[0],
                                                                  dev_seeds=" ".join(map(str, dev)), real_compute_note=real_note)
    audit = subsession.launch(job, ws, meta, prompt, "program", timeout_s=int(hours * 3600) + 1800)
    L = subsession.ledger(job, meta)
    progs = {r["job"]: r["program"] for r in L if r.get("kind") == "numbered"}
    results = [{"program": progs[r["job"]], "job": r["job"], "passed": r["passed"], "of": r["of"], "terms": r.get("terms")}
               for r in L if r.get("kind") == "result" and r.get("job") in progs and "passed" in r]
    job.charge("dev_evaluations", len({r["program"] for r in results}))
    facts = {"versions": [{k: r[k] for k in ("program", "passed", "of")} for r in results], "session": {k: audit.get(k) for k in (
        "exit_code", "num_turns", "total_cost_usd", "answered_models", "wall_hours")}, "workspace": str(ws)}
    rmd = ws / "work" / "RESULT.md"
    facts["result_md"] = rmd.read_text()[:1500] if rmd.exists() else None
    if not results:
        job.finish(facts=facts, status="failed", error="the session ran no numbered version on the development seeds")
        return
    best = max(r["passed"] for r in results)
    tied = [r for r in results if r["passed"] == best]
    named = re.findall(r"v\d\d_\w+\.py", facts["result_md"] or "")
    pick = next((r for r in tied if r["program"] in named), tied[0])
    src = ws / "work" / "code" / pick["program"]
    d = job.ws / "candidates" / a["name"]
    d.mkdir(parents=True)
    shutil.copy2(src, d / "program.py")
    (d / "CANDIDATE.md").write_text(f"program {pick['program']} of session {ws.name}; development {pick['passed']}/{pick['of']}\n")
    if (ws / "work" / "SKILL.md").is_file():                    # the session's usage note for the skill library
        shutil.copy2(ws / "work" / "SKILL.md", d / "SKILL.md")
    # the evaluation it was selected on: per-start outcomes from its cap-harness run directories
    dirs = {int(p.parent.parent.name): p.parent for p in ws.glob(f"work/debug/**/{pick['job']}/outcome.json")}
    rows = []
    for sid, rd in sorted(dirs.items()):
        o = json.loads((rd / "outcome.json").read_text())
        rows.append({"id": sid, "success": bool(o["task_success"]), "term": o["termination_reason"], "steps": o["steps_executed"],
                     "program_error": o.get("error")})
    rows = diagnostics.from_artifacts(None, job.binding["task"].get("labels", {}), rows, dirs={r["id"]: dirs[r["id"]] for r in rows})
    ev = {"candidate": a["name"], "starts": "dev", "n": pick["of"], "success": pick["passed"], "terms": pick["terms"] or {},
          "per_episode": [{k: r.get(k) for k in ("id", "success", "term", "steps", "program_error")} for r in rows],
          "diagnostics": diagnostics.aggregate(rows)}
    facts["selected"] = {"program": pick["program"], "development": f"{pick['passed']}/{pick['of']}"}
    job.finish(facts=facts, created={"candidates": [{"name": a["name"], "kind": "program", "path": str(d), "entry": "program.py",
                                                     "method": "program", "note": f"program synthesis: {pick['program']} ({pick['passed']}/{pick['of']} dev)",
                                                     "params": {"brief": a["brief"][:400], "versions": versions}}],
                                     "evaluations": [ev]})


if __name__ == "__main__":
    main(run)
