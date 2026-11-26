"""A pick-and-place dataset (calibrated_scene_reconstruction.datagen.export.lerobot, with its frame cache) as the official Diffusion Policy's replay
buffer (third_party/diffusion_policy: a zarr with data/<key> arrays and meta/episode_ends), for
calibrated_scene_reconstruction.policy.dp_official.ReplayImageDataset and configs/dp_official.

    envs/dp_official/bin/python tools/export_dp_official.py <dataset> <out.zarr.zip> [--every 3]
        [--held-out-tags cousin_04] [--camera head] [--frame-cache-quality 90]

Every `every`-th frame of each successful episode outside the held-out scenes (the official real-robot datasets are
10 Hz: 3 at our 30 Hz), the samples of a recording in order:
* <camera>: the frame cache's JPEG of that frame (calibrated_scene_reconstruction.policy.diffusion.frame_cache_jpeg, its size and quality),
  stored as is as the zarr chunk of the official Jpeg codec (one frame per chunk: no second encoding);
* robot_eef_pose (9): the hand's pose reached, in the robot's base frame (observation.tcp_base: xyz + 6D rotation);
* hand_qpos (22): the right hand's joints (the state's right_* joints, in its order);
* action (31): the frame after's command, as the baseline's labels (ee_label command, action_shift 1): the hand pose
  the joint command puts the hand at (apps/command_tcp.py) and the hand's joint targets.
The episodes and the layout go into the zarr's attributes (r2s2r_layout) for the server that runs the policy.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.core.paths import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="outputs/lerobot/<session>/<collection>/<robot> (with its frame cache)")
    ap.add_argument("out", help="the replay buffer, a .zarr.zip")
    ap.add_argument("--every", type=int, default=3, help="frames per sample (3: 10 Hz from 30 Hz)")
    ap.add_argument("--held-out-tags", nargs="*", default=["cousin_04"], help="scenes left out (prefixes of the tags)")
    ap.add_argument("--camera", default="head")
    ap.add_argument("--frame-cache-quality", type=int, default=90)
    ap.add_argument("--hand-joints", default="^right_", help="regex of the state's hand joints")
    ap.add_argument("--sources-from", help="another export of the same session: keep only its episodes (by source "
                    "episode), plus those of --also-tags (e.g. the earlier dataset plus new DAgger recoveries)")
    ap.add_argument("--also-tags", help="regex of the tags kept beside --sources-from's episodes")
    a = ap.parse_args()
    import re

    import pyarrow.parquet as pq
    import zarr
    from diffusion_policy.codecs.imagecodecs_numcodecs import Jpeg, register_codecs

    from calibrated_scene_reconstruction.policy.diffusion import command_tcp

    register_codecs()
    src = Path(a.dataset)
    info = json.loads((src / "meta" / "info.json").read_text())
    eps = sorted((json.loads(s) for s in (src / "meta" / "episodes.jsonl").read_text().splitlines() if s.strip()),
                 key=lambda e: e["episode_index"])
    keep = [e for e in eps if e["source"].get("success", True)
            and not any(e["source"]["tag"] == h or e["source"]["tag"].startswith(f"{h}__") for h in a.held_out_tags)]
    if a.sources_from:
        other = {json.loads(x)["source"]["episode"] for x in
                 (Path(a.sources_from) / "meta" / "episodes.jsonl").read_text().splitlines() if x.strip()}
        keep = [e for e in keep if e["source"]["episode"] in other
                or (a.also_tags and re.search(a.also_tags, e["source"]["tag"]))]
    names = info["features"]["observation.state"]["names"]
    hand = [i for i, n in enumerate(names) if re.search(a.hand_joints, n)]
    shape = info["features"][f"observation.images.{a.camera}"]["shape"]
    cache = src / "cache"
    jpg = sorted(cache.glob(f"{a.camera}_*_q{a.frame_cache_quality}.jpg"))
    if len(jpg) != 1:
        raise FileNotFoundError(f"one {a.camera} JPEG frame cache of quality {a.frame_cache_quality} in {cache}: {jpg}")
    jpg = jpg[0]
    h, w = (int(x) for x in jpg.stem.split("_")[1].split("x"))
    off = np.load(jpg.with_suffix(".idx.npy"))
    print(f"{len(keep)} of {len(eps)} episodes, the {a.camera} camera {h}x{w} (cache of {shape}), {len(hand)} hand joints, "
          f"every {a.every} frames", flush=True)

    eef, hq, act, ends, frames = [], [], [], [], []
    for e in keep:
        k = e["episode_index"]
        t = pq.read_table(src / info["data_path"].format(episode_chunk=k // info["chunks_size"], episode_index=k),
                          columns=["observation.state", "observation.tcp_base", "action", "index"])
        st = np.stack(t["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float32)
        tcp = np.stack(t["observation.tcp_base"].to_numpy(zero_copy_only=False)).astype(np.float32)
        cmd_joint = np.stack(t["action"].to_numpy(zero_copy_only=False)).astype(np.float32)
        gidx = t["index"].to_numpy()
        cmd = command_tcp([e])
        n = len(st)
        sub = np.arange(0, n, a.every)
        nxt = np.minimum(sub + 1, n - 1)  # the frame after's command (action_shift 1); the last repeats
        eef.append(tcp[sub])
        hq.append(st[sub][:, hand])
        act.append(np.concatenate([cmd[nxt], cmd_joint[nxt][:, hand]], 1))
        frames.append(gidx[sub])
        ends.append((ends[-1] if ends else 0) + len(sub))
    eef, hq, act, frames = (np.concatenate(x) for x in (eef, hq, act, frames))
    N = len(frames)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    store = zarr.ZipStore(str(out), mode="w")
    root = zarr.group(store)
    data, meta = root.create_group("data"), root.create_group("meta")
    meta.array("episode_ends", np.asarray(ends, np.int64), chunks=(len(ends),), compressor=None)
    for key, arr in (("robot_eef_pose", eef), ("hand_qpos", hq), ("action", act)):
        data.array(key, arr, chunks=arr.shape, compressor=None)
    data.create(a.camera, shape=(N, h, w, 3), chunks=(1, h, w, 3), dtype=np.uint8, compressor=Jpeg())
    with open(jpg, "rb") as f:
        for i, g in enumerate(frames):
            f.seek(int(off[g]))
            store[f"data/{a.camera}/{i}.0.0.0"] = f.read(int(off[g + 1] - off[g]))
            if i % 50000 == 0:
                print(f"  {i} / {N} frames", flush=True)
    root.attrs["r2s2r_layout"] = {
        "dataset": str(src), "every": a.every, "fps": info["fps"] / a.every, "camera": a.camera, "image_hw": [h, w],
        "hand_index": hand, "hand_names": [names[i] for i in hand], "held_out_tags": a.held_out_tags,
        "episodes": [e["episode_index"] for e in keep], "action": "command tcp_base (9) + hand joint targets",
        "action_shift": 1, "obs": {"robot_eef_pose": "tcp_base reached (9)", "hand_qpos": "hand joints"}}
    store.close()
    print(f"done: {len(keep)} episodes, {N} samples -> {out} ({out.stat().st_size / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()
