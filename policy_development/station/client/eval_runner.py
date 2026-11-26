"""Coordinator-owned evaluation of a delivered policy on sealed starts (runs inside the sandbox, never edited by a method).

    python eval_runner.py <policy.py | program.py> <partition> <ids comma-separated> <out.json> [--program] [--artifacts DIR]

policy.py must define   class Policy:  __init__(self, info)  /  reset(self, env)  /  act(self, obs) -> (num_envs, action_dim)
program.py is a CaP-style script run once per episode per env (the contract in program_runtime.py).
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from program_runtime import ProgramEnv, hold_action  # noqa: E402
from station import Station  # noqa: E402


def load_policy(path, info):
    spec = importlib.util.spec_from_file_location("delivered_policy", path)
    mod = importlib.util.module_from_spec(spec); sys.path.insert(0, str(Path(path).resolve().parent)); spec.loader.exec_module(mod)
    return mod.Policy(info)


def write_artifacts(root, rid, rec, src, trace, keyframes, result, error):
    """CaP-harness-style run directory: outcome.json, source/, episode/steps.jsonl, media/keyframes.npz (no success predicate)."""
    d = Path(root) / str(rid); (d / "source").mkdir(parents=True, exist_ok=True); (d / "episode").mkdir(exist_ok=True); (d / "media").mkdir(exist_ok=True)
    (d / "outcome.json").write_text(json.dumps({"id": rid, "program_ok": error is None, "task_success": bool(rec["success"]),
                                                "termination_reason": rec["term"], "steps_executed": int(rec["steps"]), "error": error}, indent=1))
    (d / "source" / Path(src).name).write_text(Path(src).read_text())
    try:
        json.dumps(result); res = result
    except (TypeError, ValueError):
        res = repr(result)[:2000]
    (d / "source" / "program-result.json").write_text(json.dumps({"result": res}, indent=1))
    with open(d / "episode" / "steps.jsonl", "w") as fh:
        for row in trace:
            fh.write(json.dumps(row) + "\n")
    if keyframes:
        np.savez_compressed(d / "media" / "keyframes.npz", step=np.asarray([k[0] for k in keyframes]), cloud=np.stack([k[1] for k in keyframes]),
                            labels=np.stack([k[2] for k in keyframes]))


def main():
    path, part, ids, out = sys.argv[1], sys.argv[2], [int(i) for i in sys.argv[3].split(",")], Path(sys.argv[4])
    is_prog = "--program" in sys.argv
    art = sys.argv[sys.argv.index("--artifacts") + 1] if "--artifacts" in sys.argv else None
    KEY_EVERY = 20
    st = Station(); info = st.info(); N = info["num_envs"]; rows = []; t0 = time.time()
    pol = None if is_prog else load_policy(path, info)
    for b0 in range(0, len(ids), N):
        batch = ids[b0:b0 + N]
        obs = st.reset_partition(part, batch)
        progs = []
        if is_prog:
            src = Path(path).read_text()
            for e in range(len(batch)):
                progs.append(ProgramEnv(src, info, st, obs, batch_index=e).start())
        else:
            for e in range(len(batch)):
                pol.reset(e)
        live = {e: True for e in range(len(batch))}
        traces = {e: [] for e in range(len(batch))}; keys = {e: [] for e in range(len(batch))}
        while True:
            hold = hold_action(info, obs)                   # stay still: measured joints (absolute targets) or a zero delta action
            if is_prog:
                acts = hold.copy()
                for e, p in enumerate(progs):
                    if live[e]:
                        acts[e] = p.next_action(hold[e])
            else:
                acts = np.asarray(pol.act(obs), np.float32)
            if art:
                for e in range(len(batch)):
                    if live[e]:
                        k = int(obs["step"][e])
                        row = {"step": k}
                        for key in info.get("proprio") or ("q29", "palm_pos", "palm_quat"):      # the station's proprioception keys
                            row[key] = [round(float(v), 5) for v in np.atleast_1d(obs[key][e])]
                        if "tactile_n" in obs:
                            row["tactile_n"] = [round(float(v), 3) for v in obs["tactile_n"][e]]
                        if "obs" in obs:                            # state-vector stations (e.g. nut): the observation vector of this step
                            row["obs"] = [round(float(v), 5) for v in obs["obs"][e]]
                        row["target" if ("obs" not in obs and "q29" in obs) else "action"] = [round(float(v), 5) for v in acts[e]]
                        traces[e].append(row)
                        if k % KEY_EVERY == 0 and "cloud" in obs:
                            keys[e].append((k, obs["cloud"][e].copy(), obs["labels"][e].copy()))
            obs, ended, all_done = st.step(acts)
            for rec in ended:
                rows.append({**rec, "id": batch[rec["env"]]}); live[rec["env"]] = False
                err = None; result = None
                if is_prog:
                    p = progs[rec["env"]]; p.stop(); time.sleep(0.05)
                    err, result = p.error, p.result
                    rows[-1].update(program_error=p.error)
                if art:
                    write_artifacts(art, batch[rec["env"]], rec, path, traces[rec["env"]], keys[rec["env"]], result, err)
            if is_prog:
                for e, p in enumerate(progs):
                    if live[e] and not p.done:
                        p.feed(obs)
            if all_done:
                break
    st.close()
    res = {"partition": part, "n": len(rows), "success": sum(r["success"] for r in rows),
           "terms": {t: sum(r["term"] == t for r in rows) for t in sorted({r["term"] for r in rows})}, "episodes": rows, "wall_s": round(time.time() - t0, 1)}
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(res, indent=1))
    print("[EVAL]", json.dumps({k: res[k] for k in ("partition", "n", "success", "terms", "wall_s")}))


if __name__ == "__main__":
    main()
