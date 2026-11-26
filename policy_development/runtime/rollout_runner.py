"""In-sandbox rollouts of a deliverable on fresh TRAINING starts of the station (never a sealed partition).

Used for evaluation on training starts and for recording demonstrations from any candidate (program or policy).  Every
finished episode counts against the campaign's training-episode budget (the station enforces it).  Recorded episodes use
the real demonstrations' format (cloud, labels, q29, target = the executed target after the station's arm / hand limits),
so real and simulated data mix without conversion.

    python rollout_runner.py <deliverable> <out.json> [--program] [--episodes N] [--record DIR [--success-only] [--n-success K]]
                             [--real-cost JSON]

--real-cost (programs): the real-robot worker's cost model from the site's robot profile, {"ik_ms": .., "fk_ms": ..,
"step_budget_ms": ..}; each program step is then measured as real compute = its own CPU time outside the station calls +
that model's cost of the fk / ik rows it requested (the station here solves them on a GPU, the robot's worker on a CPU).
"""
import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/station")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnostics import Tracker, aggregate  # noqa: E402
from program_runtime import ProgramEnv, hold_action  # noqa: E402
from station import Station  # noqa: E402

HSTEP_DEFAULT = 0.05              # hand-target rate limit per step (rad) when the station does not state its own


class TimedKinematics:
    """The station's fk / ik for programs; records per call (program serial, kind, rows, thread CPU time when it returned).
    The serial is set on each program thread at its start (thread idents are reused once a thread ends, so they would
    attribute one episode's calls to another)."""

    def __init__(self, st):
        self.st, self.calls = st, []

    def _rec(self, kind, fn, *args, **kw):
        out = fn(*args, **kw)
        rows = int(np.atleast_2d(np.asarray(args[0])).shape[0]) if args else 1
        self.calls.append((getattr(threading.current_thread(), "astra_serial", None), kind, rows, time.thread_time()))
        return out

    def fk(self, *a, **k):
        return self._rec("fk", self.st.fk, *a, **k)

    def ik(self, *a, **k):
        return self._rec("ik", self.st.ik, *a, **k)


class ThreadClockQueue:
    """queue wrapper used from the program thread: `get` stamps the program thread's CPU clock when an observation arrives,
    `put` stamps it when the program submits its next action (CPU time, so other envs' threads do not inflate it)."""

    def __init__(self, q, log):
        self.q, self.log = q, log

    def get(self, *a, **k):
        item = self.q.get(*a, **k)
        self.log.append(time.thread_time())
        return item

    def put(self, item, *a, **k):
        self.log.append(time.thread_time())
        return self.q.put(item, *a, **k)

    def __getattr__(self, name):
        return getattr(self.q, name)


ap = argparse.ArgumentParser()
ap.add_argument("deliverable"); ap.add_argument("out"); ap.add_argument("--program", action="store_true")
ap.add_argument("--episodes", type=int, default=0, help="stop after this many finished episodes")
ap.add_argument("--record", default=""); ap.add_argument("--success-only", action="store_true")
ap.add_argument("--n-success", type=int, default=0, help="stop after this many recorded successes")
ap.add_argument("--real-cost", default="", help="programs: the real worker's cost model (JSON: ik_ms, fk_ms, step_budget_ms)")
a = ap.parse_args()
COST = json.loads(a.real_cost) if a.real_cost else None

st = Station(timeout=900.0)
info = st.info()
N, ARM = info["num_envs"], float(info.get("arm_step_limit_rad", 0.08))
HSTEP = float(info.get("hand_step_limit_rad", HSTEP_DEFAULT))
labels = info.get("labels", {})
rec_dir = Path(a.record) if a.record else None
if rec_dir:
    rec_dir.mkdir(parents=True, exist_ok=True)
if not a.program:
    from eval_runner import load_policy  # noqa: E402  (the station's own loader: same import semantics as the sealed evaluation)
pol = None if a.program else load_policy(a.deliverable, info)
src = Path(a.deliverable).read_text() if a.program else None

obs = st.reset_train()
view = lambda o, e: {k: v[e] for k, v in o.items()}  # noqa: E731
progs = [None] * N
kin = TimedKinematics(st) if a.program else None
step_cost = []                    # per program step: estimated real-worker compute (ms), ik rows, fk rows
SERIAL = [0]


def start(e):
    if a.program:
        progs[e] = ProgramEnv(src, info, kin, obs, batch_index=e)
        progs[e].got, progs[e].acted = [0.0], []           # program-thread CPU clock: observation received / action put
        progs[e].to_prog = ThreadClockQueue(progs[e].to_prog, progs[e].got)
        progs[e].from_prog = ThreadClockQueue(progs[e].from_prog, progs[e].acted)
        SERIAL[0] += 1
        progs[e].thread.astra_serial = SERIAL[0]
        progs[e].start()
    else:
        pol.reset(e)


def account(p):
    """Per program step (observation received -> next action put): the program thread's CPU time, minus nothing for the
    station calls (they wait on a socket, little CPU), plus the real worker's CPU cost of the fk / ik rows in that step."""
    if COST is None:
        return
    sid = p.thread.astra_serial
    mine = [c for c in kin.calls if c[0] == sid]
    n = min(len(p.got), len(p.acted))
    for i in range(getattr(p, "accounted", 0), n):
        t0, t1 = p.got[i], p.acted[i]
        if t1 < t0:
            continue
        calls = [c for c in mine if t0 <= c[3] <= t1]
        ik_rows = sum(c[2] for c in calls if c[1] == "ik"); fk_rows = sum(c[2] for c in calls if c[1] == "fk")
        step_cost.append(((t1 - t0) * 1e3 + ik_rows * COST["ik_ms"] + fk_rows * COST["fk_ms"], ik_rows, fk_rows))
    p.accounted = n


for e in range(N):
    start(e)
track = [Tracker(labels) for _ in range(N)]
cmd_hand = obs["q29"][:, 7:].astype(np.float64).copy() if "q29" in obs and obs["q29"].shape[1] == 29 else None
GENERIC = [k for k in obs if k not in ("active", "step") and not k.startswith("priv")] if cmd_hand is None else None   # non-q29 stations: record the observation as is
buf = [dict(cloud=[], labels=[], q29=[], target=[]) if cmd_hand is not None else {} for _ in range(N)]
rows, n_rec, t0 = [], 0, time.time()


def done():
    if a.n_success and n_rec >= a.n_success:
        return True
    return bool(a.episodes) and len(rows) >= a.episodes


while not done():
    hold = hold_action(info, obs)                       # stay still exactly as the sealed evaluation does
    if a.program:
        acts = hold.copy()
        for e in range(N):
            acts[e] = progs[e].next_action(hold[e])
    else:
        acts = np.asarray(pol.act(obs), np.float32)
    if rec_dir is not None and cmd_hand is not None:
        q = obs["q29"].astype(np.float64)
        ex = acts.astype(np.float64).copy()
        ex[:, :7] = np.clip(ex[:, :7], q[:, :7] - ARM, q[:, :7] + ARM)
        cmd_hand = cmd_hand + np.clip(ex[:, 7:] - cmd_hand, -HSTEP, HSTEP)
        ex[:, 7:] = cmd_hand
        for e in range(N):
            b = buf[e]
            b["cloud"].append(np.asarray(obs["cloud"][e], np.float16)); b["labels"].append(np.asarray(obs["labels"][e], np.uint8))
            b["q29"].append(q[e].astype(np.float32)); b["target"].append(ex[e].astype(np.float32))
    if rec_dir is not None and cmd_hand is None:
        for e in range(N):
            b = buf[e]
            for k in GENERIC:
                b.setdefault(k, []).append(np.asarray(obs[k][e]))
            b.setdefault("action", []).append(acts[e].astype(np.float32))
    for e in range(N):
        track[e].add(view(obs, e))
    obs, ended, _ = st.step(acts)
    fresh = set()
    for r in ended:
        e = r["env"]
        row = {"id": f"train_{len(rows)}", "success": bool(r["success"]), "term": r["term"], "steps": int(r["steps"]),
               "diag": track[e].summary()}
        if a.program:
            p = progs[e]; account(p); p.stop(); time.sleep(0.02)
            row["program_error"] = p.error
        rows.append(row)
        if rec_dir is not None and (r["success"] or not a.success_only) and not (a.n_success and n_rec >= a.n_success):
            b = buf[e]
            if cmd_hand is None:
                np.savez_compressed(rec_dir / f"ep_{n_rec:05d}.npz", success=bool(r["success"]), term=r["term"],
                                    **{k: np.stack(v) for k, v in b.items() if len(v)})
            else:
                np.savez_compressed(rec_dir / f"ep_{n_rec:05d}.npz", cloud=np.stack(b["cloud"]), labels=np.stack(b["labels"]),
                                    q29=np.stack(b["q29"]), target=np.stack(b["target"]), success=bool(r["success"]), term=r["term"])
            n_rec += 1
        buf[e] = dict(cloud=[], labels=[], q29=[], target=[]) if cmd_hand is not None else {}
        track[e] = Tracker(labels)
        if cmd_hand is not None:
            cmd_hand[e] = obs["q29"][e, 7:]
        start(e)
        fresh.add(e)
    if a.program:
        for e in range(N):
            if e not in fresh and not progs[e].done:
                progs[e].feed(obs)
st.close()
if a.program:
    for p in progs:
        if p is not None:
            account(p)
real_time = None
if step_cost:
    c = np.array(step_cost)
    real_time = {"program_steps": int(len(c)), "est_real_step_ms_p50": round(float(np.median(c[:, 0])), 1),
                 "est_real_step_ms_p99": round(float(np.percentile(c[:, 0], 99)), 1), "est_real_step_ms_max": round(float(c[:, 0].max()), 1),
                 "steps_over_budget": int((c[:, 0] > COST["step_budget_ms"]).sum()), "budget_ms": COST["step_budget_ms"],
                 "max_ik_rows_in_one_step": int(c[:, 1].max()),
                 "note": "real station worker estimate (CPU fk / ik); steps over budget stall or stop the real robot"}
res = {"starts": "train", "n": len(rows), "real_time": real_time, "success": sum(r["success"] for r in rows),
       "terms": {t: sum(r["term"] == t for r in rows) for t in sorted({r["term"] for r in rows})},
       "recorded": n_rec, "per_episode": rows, "diagnostics": aggregate(rows), "wall_s": round(time.time() - t0, 1)}
Path(a.out).write_text(json.dumps(res, indent=1))
print("[ROLLOUT]", json.dumps({k: res[k] for k in ("n", "success", "terms", "recorded", "wall_s")}))
