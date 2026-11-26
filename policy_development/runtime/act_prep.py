"""Build the lab point-cloud ACT trainer's inputs (train.npz, dev.npz, normalization.pth) from datasets in the real
demonstrations' format (cloud (T,P,3), labels, q29, target).  Runs on the training host (numpy + torch).

Same contract as the lab point-cloud ACT recipe: labels = target - measured state
(delta_to_measured_state) or the absolute targets; optional hand-input masking (labels keep the measured hand); min-max
normalisation with floors.  The development split holds out a fixed fraction of each dataset's episodes.

    python act_prep.py <out_dir> --dataset DIR[:repeat] ... [--dev-fraction 0.1] [--mask-hand] [--pad-hold N] [--seed S]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--dataset", action="append", required=True)
ap.add_argument("--dev-fraction", type=float, default=0.1); ap.add_argument("--mask-hand", action="store_true")
ap.add_argument("--pad-hold", type=int, default=0); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--action-target", choices=("delta", "absolute"), default="delta")
ap.add_argument("--points", type=int, default=2048)
a = ap.parse_args()
OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(a.seed)


def episode(f):
    z = np.load(f)
    C, Q, A = z["cloud"].astype(np.float32), z["q29"].astype(np.float64), z["target"].astype(np.float64)
    if C.shape[1] != a.points:                         # fixed point count for the encoder
        idx = rng.choice(C.shape[1], a.points, replace=C.shape[1] < a.points)
        C = C[:, idx]
    if a.pad_hold:
        C = np.concatenate([C, np.repeat(C[-1:], a.pad_hold, 0)]); A = np.concatenate([A, np.repeat(Q[-1:], a.pad_hold, 0)])
        Q = np.concatenate([Q, np.repeat(Q[-1:], a.pad_hold, 0)])
    return Q, A, C.transpose(0, 2, 1)


train, dev, info = [], [], []
for spec in a.dataset:
    d, _, rep = spec.partition(":")
    rep = int(rep or 1)
    files = sorted(Path(d).glob("ep_*.npz"))
    if not files:
        raise SystemExit(f"no episodes in {d}")
    k = max(1, int(round(len(files) * a.dev_fraction))) if a.dev_fraction > 0 and len(files) > 1 else 0
    order = rng.permutation(len(files))
    dv, tr = [files[i] for i in order[:k]], [files[i] for i in order[k:]]
    train += [f for f in tr for _ in range(rep)]
    dev += dv
    info.append({"dataset": d, "repeat": rep, "train_episodes": len(tr), "dev_episodes": len(dv)})


def stack(files):
    S, A, P, L = [], [], [], []
    for f in files:
        q, t, c = episode(f)
        S.append(q); A.append(t); P.append(c); L.append(len(q))
    return np.concatenate(S), np.concatenate(A), np.concatenate(P).astype(np.float32), np.array(L)


st, at, pt, lt = stack(train)
sd, ad, pd, ld = stack(dev) if dev else stack(train[:1])
lab_t, lab_d = (at - st, ad - sd) if a.action_target == "delta" else (at.copy(), ad.copy())
if a.mask_hand:
    st, sd = st.copy(), sd.copy(); st[:, 7:] = 0; sd[:, 7:] = 0


def span(lo, hi, floor=0.05):
    mid, half = (lo + hi) / 2, np.maximum((hi - lo) / 2, floor / 2)
    return mid - half, mid + half


obs_min, obs_max = span(st.min(0), st.max(0))
act_min, act_max = span(np.percentile(lab_t, 0.1, 0), np.percentile(lab_t, 99.9, 0), floor=np.radians(1.0))
torch.save({"obs_min": torch.tensor(obs_min, dtype=torch.float32), "obs_max": torch.tensor(obs_max, dtype=torch.float32),
            "action_min": torch.tensor(act_min, dtype=torch.float32), "action_max": torch.tensor(act_max, dtype=torch.float32)},
           OUT / "normalization.pth")
for name, S, LAB, P, L in (("train", st, lab_t, pt, lt), ("dev", sd, lab_d, pd, ld)):
    states = np.clip(2 * (S - obs_min) / (obs_max - obs_min) - 1, -1, 1).astype(np.float32)
    acts = np.clip(2 * (LAB - act_min) / (act_max - act_min) - 1, -1, 1).astype(np.float32)
    np.savez(OUT / f"{name}.npz", traj_lengths=L, states=states, raw_states=S.astype(np.float32), actions=acts,
             raw_actions=LAB.astype(np.float32), joint_states=S.astype(np.float32), point_cloud=P)
    print(name, len(L), "episodes", int(L.sum()), "frames", P.shape, flush=True)
summary = {"datasets": info, "train_episodes": int(len(lt)), "train_frames": int(lt.sum()), "dev_episodes": int(len(ld)),
           "dev_frames": int(ld.sum()), "mask_hand": a.mask_hand, "pad_hold": a.pad_hold, "action_target": a.action_target,
           "action_semantics": "delta_to_measured_state" if a.action_target == "delta" else "absolute_joint_position_targets"}
(OUT / "prep_summary.json").write_text(json.dumps(summary, indent=1))
print("PREP", json.dumps(summary))
