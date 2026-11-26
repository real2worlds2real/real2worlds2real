"""Tool `record_demos`: run any candidate (a program, a learned policy, a teacher) on fresh training starts and keep its
successful episodes as a dataset in the real demonstrations' format.  Every finished episode counts against the
campaign's training-episode budget."""
import json

from _adapter import main


def run(job):
    a = job.args
    cand = job.request["resolved"]["candidates"][a["source"]]
    out = job.ws / "datasets" / a["name"]
    job.ensure_station()
    rt = job.runtime()
    summ = job.out / "rollout.json"
    left = int(job.request["budget_left"]["train_episodes"])
    max_ep = min(int(a.get("max_episodes", 3 * a["n_success"])), left)
    argv = [str(rt / "rollout_runner.py"), f"{cand['path']}/{cand['entry']}", str(summ), "--record", str(out), "--success-only",
            "--n-success", str(a["n_success"]), "--episodes", str(max_ep)] + (["--program"] if cand["kind"] == "program" else [])
    rc = job.sandbox(argv, job.jd / "rollout.log", timeout=10 * 3600)
    if not summ.exists():
        raise RuntimeError(f"recording failed (exit {rc}): {job.tt.tail(job.jd / 'rollout.log')}")
    res = json.loads(summ.read_text())
    job.charge("train_episodes", res["n"])
    n = res["recorded"]
    facts = {"source": a["source"], "episodes_run": res["n"], "successes_recorded": n, "terms": res["terms"],
             "diagnostics": res["diagnostics"]}
    if n == 0:
        job.finish(facts=facts, status="failed", error="no successful episode to record")
        return
    (out / "manifest.json").write_text(json.dumps({"source": a["source"], "source_tree_sha256": cand["tree_sha256"], **facts}, indent=1))
    job.finish(facts=facts, created={"datasets": [{"name": a["name"], "path": str(out), "episodes": n, "domain": "sim",
                                                   "note": f"successful station episodes of {a['source']} on training starts"}]})


if __name__ == "__main__":
    main(run)
