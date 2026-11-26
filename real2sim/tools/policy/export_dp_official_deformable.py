"""Deformable-task datasets as the official Diffusion Policy's replay buffer (third_party/diffusion_policy: a zarr with
data/<key> arrays and meta/episode_ends), for calibrated_scene_reconstruction.policy.dp_official.ReplayImageDataset and
configs/dp_official/{lehome_garment,rope_letter}_*.yaml.

    envs/dp_official/bin/python tools/export_dp_official_deformable.py <out.zarr.zip> <dataset>... [--every 3]
        [--held-out-tags seed60450 ...] [--size 240 320]

Two kinds of dataset, told apart by their layout (all of one kind per export):
* LeRobot v3.0 (meta/episodes/*.parquet): the LeHome Challenge garment folds (its teleoperated demonstrations, our
  replays of them in the realistic-look env, src/calibrated_scene_reconstruction/deformable/lehome_look.py): the cameras decoded from the
  episodes' AV1 videos (envs/bin/ffmpeg, libdav1d), resized to `size` and JPEG-encoded;
* LeRobot v2.1 with its frame cache (meta/episodes.jsonl, calibrated_scene_reconstruction.datagen.export.lerobot and calibrated_scene_reconstruction.policy.diffusion's
  frame_cache_jpeg): the Newton rope letters; the cache's JPEGs stored as they are (its size), the successful
  episodes outside `held_out_tags` only.
Every `every`-th frame of each episode (the official real-robot datasets are 10 Hz: 3 of our 30 frames a second):
* <camera>: the frame as the zarr chunk of the official Jpeg codec (one frame per chunk);
* robot_qpos: the joints (observation.state: LeHome's two SO-101s, 12; the rope's FR3, 9);
* action: the frame's joint targets (absolute; the joints reach them a few frames later).
An episode whose rows or video do not match its metadata (a replay whose saving failed) is left out. The sources and
the layout go into the zarr's attributes (r2s2r_layout) for the policy's client (calibrated_scene_reconstruction.policy.dp_official_client).
"""
import argparse
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.core.paths import ROOT
FFMPEG = ROOT / "envs" / "bin" / "ffmpeg"


def n_frames(video, cache={}):
    """A video's frame count (its packets, listed by the framecrc muxer without decoding)."""
    if video not in cache:
        r = subprocess.run([str(FFMPEG), "-v", "error", "-i", str(video), "-map", "0:v:0", "-c", "copy", "-f", "framecrc", "-"],
                           capture_output=True, text=True, check=True)
        cache[video] = sum(1 for line in r.stdout.splitlines() if line and not line.startswith("#"))
    return cache[video]


def decoded(video, h, w):
    """A video's frames, decoded and resized, (h, w, 3) uint8 one after another."""
    p = subprocess.Popen([str(FFMPEG), "-loglevel", "error", "-i", str(video), "-vf", f"scale={w}:{h}:flags=area",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    n = h * w * 3
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:  # done early: the rest of the file not decoded
        p.kill()
        p.wait()


class Export:
    """The samples gathered episode by episode, then the images written camera by camera."""

    def __init__(self, every):
        self.every, self.n = every, 0
        self.qpos, self.act, self.ends, self.sources = [], [], [], []
        self.images = {}  # camera -> [(source of the frame, sample index)] in the order they are read

    def episode(self, state, action, frames):
        """One episode: its (T, D) joints and joint targets, and per camera its frames' sources (one per kept frame)."""
        keep = np.arange(0, len(state), self.every)
        self.qpos.append(np.asarray(state, np.float32)[keep])
        self.act.append(np.asarray(action, np.float32)[keep])
        for c, src in frames.items():
            self.images.setdefault(c, []).extend((src(int(f)), self.n + j) for j, f in enumerate(keep))
        self.n += len(keep)
        self.ends.append(self.n)


def gather_lehome(ex, d, cameras, size):
    """A LeHome v3.0 dataset: its episodes' rows by their index ranges, their frames by (video, frame number)."""
    import pandas as pd

    info = json.loads((d / "meta" / "info.json").read_text())
    data = pd.concat([pd.read_parquet(f, columns=["observation.state", "action", "index"])
                      for f in sorted((d / "data").glob("*/*.parquet"))]).set_index("index").sort_index()
    eps = pd.concat([pd.read_parquet(f) for f in sorted((d / "meta" / "episodes").glob("*/*.parquet"))])
    used = 0
    for e in eps.sort_values("episode_index").to_dict("records"):
        rows = data.loc[int(e["dataset_from_index"]):int(e["dataset_to_index"]) - 1]
        where = {}
        for c in cameras:
            k = f"videos/observation.images.{c}"
            video = d / info["video_path"].format(video_key=f"observation.images.{c}", chunk_index=int(e[f"{k}/chunk_index"]),
                                                  file_index=int(e[f"{k}/file_index"]))
            where[c] = (str(video), int(round(float(e[f"{k}/from_timestamp"]) * info["fps"])))
        last = (len(rows) - 1) // ex.every * ex.every
        if len(rows) != int(e["length"]) or any(first + last >= n_frames(v) for v, first in where.values()):
            print(f"  {d} episode {e['episode_index']}: its rows or video do not match its length, left out", flush=True)
            continue
        ex.episode(np.stack(rows["observation.state"].to_numpy()), np.stack(rows["action"].to_numpy()),
                   {c: (lambda f, v=v, first=first: ("video", v, first + f)) for c, (v, first) in where.items()})
        used += 1
    ex.sources.append({"dataset": str(d), "episodes": used, "left_out": len(eps) - used})


def gather_cached(ex, d, cameras, held_out):
    """A v2.1 dataset with its frame cache: the successful episodes outside the held-out tags."""
    import pyarrow.parquet as pq

    info = json.loads((d / "meta" / "info.json").read_text())
    eps = [json.loads(s) for s in (d / "meta" / "episodes.jsonl").read_text().splitlines() if s.strip()]
    keep = [e for e in sorted(eps, key=lambda e: e["episode_index"]) if e["source"].get("success", True)
            and e["source"]["tag"] not in held_out]
    for e in keep:
        k = e["episode_index"]
        t = pq.read_table(d / info["data_path"].format(episode_chunk=k // info["chunks_size"], episode_index=k),
                          columns=["observation.state", "action", "index"])
        gidx = t["index"].to_numpy()
        ex.episode(np.stack(t["observation.state"].to_numpy(zero_copy_only=False)),
                   np.stack(t["action"].to_numpy(zero_copy_only=False)),
                   {c: (lambda f, g=gidx: ("cache", int(g[f]))) for c in cameras})
    ex.sources.append({"dataset": str(d), "episodes": len(keep), "held_out_tags": sorted(held_out)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="the replay buffer, a .zarr.zip")
    ap.add_argument("datasets", nargs="+", help="LeRobot dataset directories, all v3.0 (LeHome) or all v2.1 with a frame cache")
    ap.add_argument("--every", type=int, default=3, help="frames per sample (3: 10 Hz from 30)")
    ap.add_argument("--cameras", nargs="+", help="default: every camera of the first dataset")
    ap.add_argument("--size", type=int, nargs=2, default=[240, 320], metavar=("H", "W"), help="decoded videos (v3.0)")
    ap.add_argument("--held-out-tags", nargs="*", default=[], help="episodes' source tags left out (v2.1)")
    ap.add_argument("--cache-quality", type=int, default=90, help="the frame cache's JPEG quality (v2.1)")
    ap.add_argument("--quality", type=int, default=90, help="JPEG quality of decoded frames (v3.0)")
    a = ap.parse_args()
    import zarr
    from PIL import Image

    from diffusion_policy.codecs.imagecodecs_numcodecs import Jpeg, register_codecs

    register_codecs()
    srcs = [Path(d) for d in a.datasets]
    lehome = (srcs[0] / "meta" / "episodes").is_dir()
    if any((d / "meta" / "episodes").is_dir() != lehome for d in srcs):
        sys.exit("one kind of dataset per export (all LeRobot v3.0 or all v2.1 with a frame cache)")
    info = json.loads((srcs[0] / "meta" / "info.json").read_text())
    cams = a.cameras or [k.split(".")[-1] for k in info["features"] if k.startswith("observation.images.")]
    ex = Export(a.every)
    for d in srcs:
        gather_lehome(ex, d, cams, a.size) if lehome else gather_cached(ex, d, cams, set(a.held_out_tags))
        print(f"{d}: {ex.sources[-1]['episodes']} episodes, {ex.n} samples so far", flush=True)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    store = zarr.ZipStore(str(out), mode="w")
    root = zarr.group(store)
    meta, data = root.create_group("meta"), root.create_group("data")
    meta.array("episode_ends", np.asarray(ex.ends, np.int64), chunks=(len(ex.ends),), compressor=None)
    for key, arr in (("robot_qpos", np.concatenate(ex.qpos)), ("action", np.concatenate(ex.act))):
        data.array(key, arr, chunks=arr.shape, compressor=None)
    for c in cams:
        if lehome:  # each video decoded once, its kept frames encoded
            h, w = a.size
            by_video = {}
            for (_, v, f), i in ex.images[c]:
                by_video.setdefault(v, {})[f] = i
            data.create(c, shape=(ex.n, h, w, 3), chunks=(1, h, w, 3), dtype=np.uint8, compressor=Jpeg())
            for v, m in by_video.items():
                last = max(m)
                for f, im in enumerate(decoded(v, h, w)):
                    if f in m:
                        b = io.BytesIO()
                        Image.fromarray(im).save(b, format="JPEG", quality=a.quality)
                        store[f"data/{c}/{m[f]}.0.0.0"] = b.getvalue()
                    if f >= last:
                        break
        else:  # the frame cache's JPEGs as they are
            jpg = sorted((srcs[0] / "cache").glob(f"{c}_*_q{a.cache_quality}.jpg"))
            if len(jpg) != 1 or len(srcs) != 1:
                sys.exit(f"one dataset with one {c} frame cache of quality {a.cache_quality}: {jpg}")
            h, w = (int(x) for x in jpg[0].stem.split("_")[1].split("x"))
            off = np.load(jpg[0].with_suffix(".idx.npy"))
            data.create(c, shape=(ex.n, h, w, 3), chunks=(1, h, w, 3), dtype=np.uint8, compressor=Jpeg())
            with open(jpg[0], "rb") as f:
                for (_, g), i in ex.images[c]:
                    f.seek(int(off[g]))
                    store[f"data/{c}/{i}.0.0.0"] = f.read(int(off[g + 1] - off[g]))
        written = sum(1 for k in store.keys() if k.startswith(f"data/{c}/") and not k.endswith((".zarray", ".zattrs")))
        if written != ex.n:
            sys.exit(f"{c}: {written} frames written, {ex.n} wanted")
        print(f"  {c}: {written} frames", flush=True)
    root.attrs["r2s2r_layout"] = {
        "sources": ex.sources, "every": a.every, "fps": info["fps"] / a.every, "cameras": cams,
        "image_hw": list(data[cams[0]].shape[1:3]), "state_names": info["features"]["observation.state"].get("names"),
        "action": "joint targets (absolute)", "obs": {"robot_qpos": "the joints"}}
    store.close()
    print(f"done: {len(ex.ends)} episodes, {ex.n} samples -> {out} ({out.stat().st_size / 1e9:.1f} GB)")


if __name__ == "__main__":
    main()
