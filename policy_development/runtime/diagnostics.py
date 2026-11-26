"""Task-independent episode diagnostics from what every cloud station provides (labelled cloud, joints, palm, tactile).

Per labelled object: how high it rose above its start and how far it moved in the table plane; per episode: palm travel,
final palm height, peak tactile force, length.  These describe *what happened* (grasped and lifted? pushed? never touched?)
without exposing the hidden judge.  numpy only; used inside the sandbox (live rollouts) and on the host (eval artifacts).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def centroid(cloud, labels, lab):
    m = labels == lab
    if m.sum() < 15:
        return None
    return np.median(np.asarray(cloud, np.float32)[m], axis=0)


class Tracker:
    """Accumulates one episode online."""

    def __init__(self, label_names: dict):
        self.names = {int(k): v for k, v in label_names.items() if v not in ("table", "hand", "robot")}
        self.start, self.max_rise, self.last = {}, {}, {}
        self.palm0 = self.palm = None
        self.palm_path = 0.0
        self.tact = 0.0
        self.steps = 0

    def add(self, obs_env: dict):
        self.steps += 1
        if "cloud" in obs_env and "labels" in obs_env:
            for lab, name in self.names.items():
                c = centroid(obs_env["cloud"], obs_env["labels"], lab)
                if c is None:
                    continue
                if name not in self.start:
                    self.start[name] = c
                self.max_rise[name] = max(self.max_rise.get(name, -1.0), float(c[2] - self.start[name][2]))
                self.last[name] = c
        pk = "palm_pos" if "palm_pos" in obs_env else "ee_pos" if "ee_pos" in obs_env else None
        if pk:
            p = np.asarray(obs_env[pk], float)
            if self.palm is not None:
                self.palm_path += float(np.linalg.norm(p - self.palm))
            self.palm0 = p if self.palm0 is None else self.palm0
            self.palm = p
        if "tactile_n" in obs_env:
            self.tact = max(self.tact, float(np.max(obs_env["tactile_n"])))

    def summary(self) -> dict:
        objs = {n: {"max_rise_m": round(self.max_rise[n], 3),
                    "xy_moved_m": round(float(np.linalg.norm(self.last[n][:2] - self.start[n][:2])), 3)} for n in self.start}
        out = {"steps": self.steps, "objects": objs, "peak_tactile_n": round(self.tact, 1), "palm_path_m": round(self.palm_path, 3)}
        if self.palm is not None:
            out["palm_final_z_rise_m"] = round(float(self.palm[2] - self.palm0[2]), 3)
        return out


def aggregate(rows: list) -> dict:
    """Campaign-facing summary over episodes (split by success)."""
    out = {}
    for tag, sel in (("success", [r for r in rows if r.get("success")]), ("failure", [r for r in rows if not r.get("success")])):
        if not sel:
            continue
        d = {"episodes": len(sel), "median_steps": int(np.median([r["diag"]["steps"] for r in sel if r.get("diag")] or [0]))}
        names = sorted({n for r in sel if r.get("diag") for n in r["diag"]["objects"]})
        for n in names:
            rise = [r["diag"]["objects"][n]["max_rise_m"] for r in sel if r.get("diag") and n in r["diag"]["objects"]]
            mv = [r["diag"]["objects"][n]["xy_moved_m"] for r in sel if r.get("diag") and n in r["diag"]["objects"]]
            d[n] = {"lifted_ge_5cm": int(sum(x >= 0.05 for x in rise)), "median_max_rise_m": round(float(np.median(rise)), 3),
                    "moved_ge_1cm": int(sum(x >= 0.01 for x in mv)), "median_xy_moved_m": round(float(np.median(mv)), 3)}
        tac = [r["diag"]["peak_tactile_n"] for r in sel if r.get("diag")]
        d["touched_ge_2n"] = int(sum(t >= 2.0 for t in tac))
        out[tag] = d
    return out


def from_artifacts(art_root, label_names: dict, rows: list, dirs: dict | None = None) -> list:
    """Attach diagnostics to eval_runner rows from its run directories (keyframes every 20 steps + steps.jsonl).
    `dirs` maps a row id to its run directory when the layout is not <art_root>/<id> (cap-harness runs)."""
    for r in rows:
        d = Path(dirs[r["id"]]) if dirs and r["id"] in dirs else Path(art_root) / str(r["id"])
        tr = Tracker(label_names)
        kf = d / "media" / "keyframes.npz"
        steps = d / "episode" / "steps.jsonl"
        if kf.exists():
            z = np.load(kf)
            for c, l in zip(z["cloud"], z["labels"]):
                tr.add({"cloud": c, "labels": l})
        tr.steps = 0
        if steps.exists():
            for line in steps.read_text().splitlines():
                row = json.loads(line)
                tr.add({k: np.asarray(v) for k, v in row.items() if k in ("palm_pos", "ee_pos", "tactile_n")})
        r["diag"] = tr.summary()
    return rows
