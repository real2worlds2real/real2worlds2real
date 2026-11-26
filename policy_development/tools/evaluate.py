"""Tool `evaluate`: one candidate on the open development starts (the baselines' runner, partition dev) or on N fresh
training starts; the harness alone may request the unseen-check partition (host field).  Returns success, termination
reasons, per-start outcomes (for paired comparisons) and task-independent diagnostics."""
import json
from pathlib import Path

from _adapter import main

import diagnostics


def run(job):
    a, host = job.args, job.request.get("host") or {}
    name = a["candidate"]
    cand = job.request["resolved"]["candidates"][name]
    entry = f"{cand['path']}/{cand['entry']}"
    prog = cand["kind"] == "program"
    labels = job.binding["task"].get("labels", {})
    if a["starts"] == "train":
        job.ensure_station()
        rt = job.runtime()
        out = job.out / "rollout.json"
        argv = [str(rt / "rollout_runner.py"), entry, str(out), "--episodes", str(a.get("episodes", 32))] + (["--program"] if prog else [])
        if prog and job.real_cost():
            argv += ["--real-cost", json.dumps(job.real_cost())]
        rc = job.sandbox(argv, job.jd / "rollout.log", timeout=6 * 3600)
        if (rc != 0 or not out.exists()) and job.link_lost(job.jd / "rollout.log"):      # dropped station link: once more
            job.link_retries += 1
            job.ensure_station()
            rc = job.sandbox(argv, job.jd / "rollout.log", timeout=6 * 3600)
        if rc != 0 or not out.exists():
            raise RuntimeError(f"rollout failed (exit {rc}): {job.tt.tail(job.jd / 'rollout.log')}")
        res = json.loads(out.read_text())
        job.charge("train_episodes", res["n"])
        starts = "train"
        real_time = res.get("real_time")
    else:
        real_time = None
        part = host.get("partition", "dev")
        ids = list(job.site.cfg["splits"]["unseen_check" if part == "unseen" else "development"])
        art = job.ws / "runs" / f"{job.request['job_id']}_{part}"
        res = job.eval_partition(Path(entry), part, ids, job.out, prog, f"eval_{part}", art)
        if "error" in res:
            raise RuntimeError(res["error"] + "\n" + res.get("log_tail", ""))
        rows = diagnostics.from_artifacts(art, labels, [{"id": r["id"], "success": bool(r["success"]), "term": r["term"],
                                                         "steps": int(r["steps"]), "program_error": r.get("program_error")} for r in res["episodes"]])
        res = {"n": res["n"], "success": res["success"], "terms": res["terms"], "per_episode": rows, "diagnostics": diagnostics.aggregate(rows)}
        if part == "dev":
            job.charge("dev_evaluations", 1)
        starts = "dev" if part == "dev" else "unseen"
    per = [{k: r.get(k) for k in ("id", "success", "term", "steps", "program_error")} for r in res["per_episode"]]
    errors = sorted({r["program_error"] for r in per if r.get("program_error")})[:3]
    ev = {"candidate": name, "starts": starts, "n": res["n"], "success": res["success"], "terms": res["terms"],
          "per_episode": per, "diagnostics": res["diagnostics"]}
    facts = {"candidate": name, "starts": starts, "success": f"{res['success']}/{res['n']}", "terms": res["terms"],
             "diagnostics": res["diagnostics"]}
    if errors:
        facts["program_errors"] = errors
    if real_time:
        facts["real_time"] = real_time
        ev["real_time"] = real_time
    job.finish(facts=facts, created={"evaluations": [ev]})


if __name__ == "__main__":
    main(run)
