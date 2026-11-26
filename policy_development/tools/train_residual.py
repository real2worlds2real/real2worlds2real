"""Tool `train_residual`: residual RL post-training of a frozen base candidate (policy or program) on the station's
training starts, lab ResFiT recipe (bounded incremental palm / joint offset, TD3 with a residual-size penalty), reward =
chosen library terms + success bonus.  Creates candidates <name>_at<episodes> at up to four checkpoints."""
import json
import shutil

from _adapter import RUNTIME, main

import materialize


def run(job):
    a = job.args
    base = job.request["resolved"]["candidates"][a["base"]]
    left = int(job.request["budget_left"]["train_episodes"])
    episodes = min(int(a["episodes"]), left)
    if episodes < 32:
        raise RuntimeError(f"only {left} training episodes left")
    import rewards
    rewards.make(a["reward_terms"], job.binding["task"].get("labels", {}))        # refuse unknown terms / bad expressions before any station time
    root = job.ws / "residual" / a["name"]
    tmpl = root / "template"
    materialize.as_policy(job, base, tmpl / "base")
    materialize.kin_dir(job, tmpl / "kin")
    shutil.copy2(RUNTIME / "residual_policy.py", tmpl / "policy.py")
    space = a["action_space"]
    scale = float(a.get("scale", {"palm_xy": 0.03, "palm_xyz": 0.03, "arm_joints": 0.05}[space]))
    cfg = {"action_space": space, "scale": scale, "step_frac": float(a.get("step_frac", 0.1)), "labels": job.binding["task"].get("labels", {}),
           "max_steps": int(job.binding["policy_io"].get("max_steps") or 600), "base": a["base"], "reward_terms": a["reward_terms"]}
    (tmpl / "config.json").write_text(json.dumps(cfg, indent=1))
    job.ensure_station()
    rt = job.runtime()
    for f in ("residual_learner.py", "rewards.py"):
        shutil.copy2(RUNTIME / f, rt / f)
    out = root / "train"
    sw = next((float(t.get("weight", 10.0)) for t in a["reward_terms"] if t["id"] == "success_sparse"), 10.0)
    argv = [str(rt / "residual_learner.py"), str(tmpl), str(out), "--episodes", str(episodes), "--sigma", str(a.get("sigma", 0.1)),
            "--reward-json", json.dumps(a["reward_terms"]), "--offset-reg", str(a.get("offset_reg", 0.5)), "--success-weight", str(sw), "--ckpt-every", str(max(16, episodes // 8))]
    import time
    t0 = time.time()
    rc = job.sandbox(argv, job.jd / "learner.log", timeout=14 * 3600, gpu=job.site.cfg.get("sandbox_gpu", True))
    job.charge("gpu_hours", (time.time() - t0) / 3600.0)
    summ = out / "summary.json"
    curve = [json.loads(x) for x in (out / "learner.jsonl").read_text().splitlines()] if (out / "learner.jsonl").exists() else []
    interrupted = None
    if not summ.exists():
        n = curve[-1]["episodes"] if curve else 0
        saved = sorted(int(p.name.split("_")[1]) for p in out.glob("ckpt_*") if (p / "actor.pt").exists())
        if not (job.link_lost(job.jd / "learner.log") and saved):
            job.charge("train_episodes", n)
            raise RuntimeError(f"residual learner failed (exit {rc}): {job.tt.tail(job.jd / 'learner.log')}")
        # the station link dropped mid-run: keep the checkpoints already written (episodes after the last one are lost)
        interrupted = f"station link lost after {n} logged episodes; checkpoints up to {saved[-1]} kept"
        res = {"episodes": n, "successes": None, "terms": {}, "checkpoints": saved}
    else:
        res = json.loads(summ.read_text())
    job.charge("train_episodes", res["episodes"])
    cks = res["checkpoints"]
    keep = sorted({cks[int(round(i * (len(cks) - 1) / 3))] for i in range(4)}) if len(cks) > 4 else cks
    created = []
    for k in keep:
        d = job.ws / "candidates" / f"{a['name']}_at{k}"
        shutil.copytree(tmpl, d, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2(out / f"ckpt_{k}" / "actor.pt", d / "actor.pt")
        created.append({"name": f"{a['name']}_at{k}", "kind": "policy", "path": str(d), "entry": "policy.py", "method": "residual_rl",
                        "parent": a["base"], "note": f"{space} residual (scale {scale}) on {a['base']} after {k} episodes, reward {[t['id'] for t in a['reward_terms']]}",
                        "params": a})
    facts = {"episodes": res["episodes"], "training_successes": res["successes"], "terms": res["terms"],
             "curve": [{k: c[k] for k in ("episodes", "last50_success", "terms", "end_offset_m", "actor_abs_action", "actor_updating") if k in c}
                       for c in curve][-10:], "checkpoints_kept": keep}
    if interrupted:
        facts["interrupted"] = interrupted
    job.finish(facts=facts, created={"candidates": created})


if __name__ == "__main__":
    main(run)
