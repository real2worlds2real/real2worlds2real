"""A pick-and-place dataset (calibrated_scene_reconstruction.datagen.export.lerobot, LeRobot v2.1) as the joint-space LeRobot v2.1 dataset openpi's
pi0.5 fine-tuning reads (calibrated_scene_reconstruction.policy.openpi, as openpi's ALOHA data: joint positions in, joint targets out).

    envs/dp_official/bin/python tools/export_openpi.py <dataset> <out> [--held-out-tags cousin_04]

The successful episodes outside the held-out scenes, renumbered, at the dataset's rate:
* observation.state: the arm's and the hand's joints (`--arm`, `--hand` regexes over the state's names, in its order);
* action: the same joints' targets of the frame after (the baseline's labels: action_shift 1; the last repeats);
* the camera videos linked (`--link`: symbolic links to the source's; copy them over with tar -h), the task text.
openpi reads it at $HF_LEROBOT_HOME/<repo_id>."""
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def stats(x):
    x = np.asarray(x, np.float64)
    return {"min": x.min(0).tolist(), "max": x.max(0).tolist(), "mean": x.mean(0).tolist(), "std": x.std(0).tolist(),
            "count": [len(x)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("out")
    ap.add_argument("--held-out-tags", nargs="*", default=["cousin_04"], help="scenes left out (prefixes of the tags)")
    ap.add_argument("--arm", default="^R_arm_j", help="regex of the arm's joints in the state")
    ap.add_argument("--hand", default="^right_", help="regex of the hand's joints in the state")
    a = ap.parse_args()
    src, out = Path(a.dataset).resolve(), Path(a.out)
    info = json.loads((src / "meta" / "info.json").read_text())
    eps = sorted((json.loads(s) for s in (src / "meta" / "episodes.jsonl").read_text().splitlines() if s.strip()),
                 key=lambda e: e["episode_index"])
    old_stats = {json.loads(s)["episode_index"]: json.loads(s)["stats"]
                 for s in (src / "meta" / "episodes_stats.jsonl").read_text().splitlines() if s.strip()}
    keep = [e for e in eps if e["source"].get("success", True)
            and not any(e["source"]["tag"] == h or e["source"]["tag"].startswith(f"{h}__") for h in a.held_out_tags)]
    names = info["features"]["observation.state"]["names"]
    arm = [i for i, n in enumerate(names) if re.search(a.arm, n)]
    hand = [i for i, n in enumerate(names) if re.search(a.hand, n)]
    idx = arm + hand
    cams = [k for k, f in info["features"].items() if f["dtype"] == "video"]
    chunk = info["chunks_size"]
    print(f"{len(keep)} of {len(eps)} episodes, {len(arm)} arm + {len(hand)} hand joints, cameras {cams}", flush=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    new_eps, new_stats, index = [], [], 0
    for new, e in enumerate(keep):
        old = e["episode_index"]
        t = pq.read_table(src / info["data_path"].format(episode_chunk=old // chunk, episode_index=old))
        st = np.stack(t["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float32)[:, idx]
        ac = np.stack(t["action"].to_numpy(zero_copy_only=False)).astype(np.float32)[:, idx]
        n = len(st)
        ac = ac[np.minimum(np.arange(n) + 1, n - 1)]  # the frame after's command (action_shift 1)
        tab = pa.table({
            "observation.state": pa.array(list(st), type=pa.list_(pa.float32(), len(idx))),
            "action": pa.array(list(ac), type=pa.list_(pa.float32(), len(idx))),
            "timestamp": t["timestamp"], "frame_index": t["frame_index"],
            "episode_index": pa.array(np.full(n, new, np.int64)),
            "index": pa.array(np.arange(index, index + n, dtype=np.int64)),
            "task_index": t["task_index"],
        })
        f = out / info["data_path"].format(episode_chunk=new // chunk, episode_index=new)
        f.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(tab, f)
        for c in cams:
            v = out / info["video_path"].format(episode_chunk=new // chunk, video_key=c, episode_index=new)
            v.parent.mkdir(parents=True, exist_ok=True)
            if v.is_symlink() or v.exists():
                v.unlink()
            os.symlink(src / info["video_path"].format(episode_chunk=old // chunk, video_key=c, episode_index=old), v)
        s = {"observation.state": stats(st), "action": stats(ac),
             "timestamp": stats(t["timestamp"].to_numpy()[:, None]),
             "frame_index": stats(t["frame_index"].to_numpy()[:, None]),
             "episode_index": stats(np.full((n, 1), new)), "index": stats(np.arange(index, index + n)[:, None]),
             "task_index": stats(t["task_index"].to_numpy()[:, None])}
        s.update({c: old_stats[old][c] for c in cams})
        new_stats.append({"episode_index": new, "stats": s})
        new_eps.append({"episode_index": new, "tasks": e["tasks"], "length": n,
                        "source": {"dataset": a.dataset, "episode_index": old, "tag": e["source"]["tag"]}})
        index += n
    jn = [names[i] for i in idx]
    feats = {c: info["features"][c] for c in cams}
    feats.update({
        "observation.state": {"dtype": "float32", "shape": [len(idx)], "names": jn},
        "action": {"dtype": "float32", "shape": [len(idx)], "names": jn},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    })
    ninfo = {k: v for k, v in info.items() if k != "calibrated_scene_reconstruction"}
    ninfo.update(total_episodes=len(new_eps), total_frames=index, total_videos=len(new_eps) * len(cams),
                 total_chunks=(len(new_eps) - 1) // chunk + 1, splits={"train": f"0:{len(new_eps)}"}, features=feats)
    (out / "meta" / "info.json").write_text(json.dumps(ninfo, indent=2))
    (out / "meta" / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in new_eps))
    (out / "meta" / "episodes_stats.jsonl").write_text("".join(json.dumps(s) + "\n" for s in new_stats))
    (out / "meta" / "tasks.jsonl").write_text((src / "meta" / "tasks.jsonl").read_text())
    (out / "meta" / "r2s2r_openpi.json").write_text(json.dumps({
        "joints": jn, "n_arm": len(arm), "n_hand": len(hand), "state_index": idx, "action_shift": 1,
        "held_out_tags": a.held_out_tags, "fps": info["fps"]}, indent=1))
    print(f"done: {len(new_eps)} episodes, {index} frames, state / action {len(idx)} -> {out}")


if __name__ == "__main__":
    main()
