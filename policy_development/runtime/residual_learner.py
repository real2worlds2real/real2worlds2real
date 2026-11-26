"""In-sandbox residual RL on the station (TD3 with a residual-size penalty; lab ResFiT recipe, incremental offsets).

    python residual_learner.py <template dir> <out dir> --episodes N [--sigma 0.1] [--utd 1] [--q-alpha 2.5]
           [--reward-json '[{"id": ..., "weight": ..., "params": {...}}]'] [--ckpt-every 50]

The template dir is a residual candidate without actor.pt (policy.py = residual_policy.py, config.json, base/, kin/).
Reward per step = the chosen library terms; at episode end + success_weight * station success.  Offsets are incremental, so
any constant actor output integrates to a saturated offset: the warm-up runs the base with small noise only (never uniform
actions), the critic gets `utd` updates per env transition, the actor starts only after the critic has seen
`actor_delay_episodes` finished episodes (their success bonuses) and `actor_delay` updates, and every step pays
`offset_reg * mean((offset / scale)^2)`, so a correction has to earn its size.  Every finished training
episode counts against the campaign's budget (the station enforces it).  Writes <out>/ckpt_<episodes>/actor.pt and
<out>/learner.jsonl."""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/station")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rewards  # noqa: E402
from station import Station  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("template"); ap.add_argument("out")
ap.add_argument("--episodes", type=int, required=True); ap.add_argument("--sigma", type=float, default=0.1)
ap.add_argument("--utd", type=float, default=1.0)
ap.add_argument("--q-alpha", type=float, default=2.5, help="TD3+BC weight of the normalised Q term against the pull to a zero correction")
ap.add_argument("--actor-delay", type=int, default=None, help="critic updates before the actor moves (default: one full wave of transitions)"); ap.add_argument("--actor-delay-episodes", type=int, default=-1)
ap.add_argument("--offset-reg", type=float, default=0.5)
ap.add_argument("--reward-json", default="[]"); ap.add_argument("--success-weight", type=float, default=10.0)
ap.add_argument("--ckpt-every", type=int, default=50); ap.add_argument("--prefill-steps", type=int, default=400)
ap.add_argument("--batch", type=int, default=256); ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
T, OUT = Path(a.template), Path(a.out)
OUT.mkdir(parents=True, exist_ok=True)
rp = __import__("importlib").util.spec_from_file_location("astra_residual_policy", T / "policy.py")
RP = __import__("importlib").util.module_from_spec(rp); rp.loader.exec_module(RP)
cfg = json.loads((T / "config.json").read_text())
core = RP.Core(cfg, T / "kin")
st = Station(timeout=900.0)
info = st.info()
N = info["num_envs"]
if a.actor_delay is None:
    a.actor_delay = int(round(a.utd * N * int(info.get("max_steps", 600))))
actor_delay_episodes = N if a.actor_delay_episodes < 0 else a.actor_delay_episodes
base = RP.load_module("astra_res_base_train", T / "base" / "policy.py").Policy(info)
reward_fn, _ = rewards.make(json.loads(a.reward_json), info.get("labels", {}))
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D, A = core.obs_dim, core.dim
actor = RP.mlp(D, A, out_act=torch.nn.Tanh()).to(dev)
with torch.no_grad():                                 # start near a zero correction (the base alone)
    actor[-2].weight.mul_(0.01); actor[-2].bias.zero_()
q1, q2 = RP.mlp(D + A, 1).to(dev), RP.mlp(D + A, 1).to(dev)
actor_t, q1_t, q2_t = copy.deepcopy(actor), copy.deepcopy(q1), copy.deepcopy(q2)
oa = torch.optim.Adam(actor.parameters(), 3e-4)
oq = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), 3e-4)
cap = a.episodes * int(info.get("max_steps", 600)) + 10
B = {"s": np.zeros((cap, D), np.float32), "a": np.zeros((cap, A), np.float32), "r": np.zeros(cap, np.float32),
     "s2": np.zeros((cap, D), np.float32), "d": np.zeros(cap, np.float32)}
nb = 0


def update():
    i = np.random.randint(0, nb, a.batch)
    s, act, r, s2, d = (torch.as_tensor(B[k][i], device=dev) for k in ("s", "a", "r", "s2", "d"))
    with torch.no_grad():
        noise = (torch.randn_like(act) * 0.2).clamp(-0.5, 0.5)
        a2 = (actor_t(s2) + noise).clamp(-1, 1)
        x2 = torch.cat([s2, a2], 1)
        y = r[:, None] + 0.99 * (1 - d[:, None]) * torch.min(q1_t(x2), q2_t(x2))
    x = torch.cat([s, act], 1)
    lq = F.mse_loss(q1(x), y) + F.mse_loss(q2(x), y)
    oq.zero_grad(); lq.backward(); oq.step()
    return lq.item()


def update_actor():
    i = np.random.randint(0, nb, a.batch)
    s = torch.as_tensor(B["s"][i], device=dev)
    pa = actor(s)
    q = q1(torch.cat([s, pa], 1))
    # TD3+BC with the zero correction as the behaviour action: the Q term is normalised by its own scale, so the pull to
    # "no correction" keeps a fixed weight whatever the reward scale (an unnormalised 0.01 penalty let the actor saturate)
    la = -(a.q_alpha / q.abs().mean().detach().clamp_min(1e-6)) * q.mean() + (pa ** 2).sum(1).mean()
    oa.zero_grad(); la.backward(); oa.step()
    with torch.no_grad():
        for net, tgt in ((actor, actor_t), (q1, q1_t), (q2, q2_t)):
            for p, pt in zip(net.parameters(), tgt.parameters()):
                pt.mul_(0.995).add_(0.005 * p)


obs = st.reset_train()
view = lambda o, e: {k: v[e] for k, v in o.items()}  # noqa: E731
envs = []
for e in range(N):
    base.reset(e)
    o = view(obs, e)
    envs.append({"offset": np.zeros(A, np.float32), "start": core.start(o), "step": 0, "ep": rewards.Episode(o, {}), "ret": 0.0})
    envs[e]["ep"].start = {v: core.start(o).get(k) for k, v in core.labels.items()}
finished, succ, steps_total, n_critic, t0 = [], 0, 0, 0, time.time()
log = open(OUT / "learner.jsonl", "a")
next_ckpt = a.ckpt_every
ulog = 0.0
while len(finished) < a.episodes:
    b = np.asarray(base.act(obs), np.float32)
    S = np.stack([core.features(view(obs, e), envs[e]["offset"], envs[e]["start"], envs[e]["step"]) for e in range(N)])
    with torch.no_grad():
        act = actor(torch.as_tensor(S, device=dev)).cpu().numpy()
    if steps_total * N < a.prefill_steps:              # warm-up: the base itself plus exploration noise
        act = np.zeros_like(act)
    act = np.clip(act + np.random.randn(*act.shape) * a.sigma, -1, 1)
    q = np.empty_like(b)
    for e in range(N):
        envs[e]["offset"] = core.step_offset(envs[e]["offset"], act[e])
        q[e] = core.apply(b[e], envs[e]["offset"])
    prev = obs
    obs, ended, _ = st.step(q)
    steps_total += 1
    done = {r["env"]: r for r in ended}
    for e in range(N):
        o0, o1 = view(prev, e), view(obs, e)
        if e in done:
            r = a.success_weight * float(done[e]["success"])
            s2, d = np.zeros(D, np.float32), 1.0
        else:
            r = reward_fn(o0, o1, envs[e]["ep"]) - a.offset_reg * float(np.mean((envs[e]["offset"] / cfg["scale"]) ** 2))
            s2, d = core.features(o1, envs[e]["offset"], envs[e]["start"], envs[e]["step"] + 1), 0.0
        if nb < cap:
            B["s"][nb], B["a"][nb], B["r"][nb], B["s2"][nb], B["d"][nb] = S[e], act[e], r, s2, d
            nb += 1
        envs[e]["ret"] += r
        envs[e]["step"] += 1
        if e in done:
            finished.append({"success": bool(done[e]["success"]), "term": done[e]["term"], "return": round(envs[e]["ret"], 2),
                             "offset_m": float(np.linalg.norm(envs[e]["offset"]))})
            succ += int(done[e]["success"])
            base.reset(e)
            envs[e] = {"offset": np.zeros(A, np.float32), "start": core.start(o1), "step": 0, "ep": rewards.Episode(o1, {}), "ret": 0.0}
            envs[e]["ep"].start = {v: envs[e]["start"].get(k) for k, v in core.labels.items()}
    if nb >= max(a.batch, a.prefill_steps):
        for k in range(max(1, int(round(a.utd * N)))):
            ulog = update()
            n_critic += 1
            if k % 2 == 1 and n_critic > a.actor_delay and len(finished) >= actor_delay_episodes:
                update_actor()
    while len(finished) >= next_ckpt:
        d = OUT / f"ckpt_{next_ckpt}"
        d.mkdir(exist_ok=True)
        torch.save(actor.state_dict(), d / "actor.pt")
        last = finished[-min(50, len(finished)):]
        with torch.no_grad():
            pa = actor(torch.as_tensor(B["s"][max(0, nb - 2000):nb], device=dev)).abs().mean().item()
        print(json.dumps({"episodes": next_ckpt, "last50_success": round(sum(x["success"] for x in last) / len(last), 3),
                          "terms": {t: sum(x["term"] == t for x in last) for t in sorted({x["term"] for x in last})},
                          "end_offset_m": round(float(np.mean([x["offset_m"] for x in last])), 4), "actor_abs_action": round(pa, 3),
                          "actor_updating": n_critic > a.actor_delay, "critic_updates": n_critic,
                          "critic_loss": round(ulog, 4), "transitions": nb, "wall_s": round(time.time() - t0)}), file=log, flush=True)
        next_ckpt += a.ckpt_every
d = OUT / f"ckpt_{len(finished)}"
d.mkdir(exist_ok=True)
torch.save(actor.state_dict(), d / "actor.pt")
st.close()
res = {"episodes": len(finished), "successes": succ, "terms": {t: sum(f["term"] == t for f in finished) for t in {f["term"] for f in finished}},
       "checkpoints": sorted(int(p.name.split("_")[1]) for p in OUT.glob("ckpt_*")), "wall_s": round(time.time() - t0)}
(OUT / "summary.json").write_text(json.dumps(res, indent=1))
print("[RESIDUAL]", json.dumps(res))
