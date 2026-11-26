"""Export recorded episodes to a LeRobot v2.1 dataset (one dataset per robot: state / action sizes differ).

    python -m calibrated_scene_reconstruction.datagen.export.lerobot outputs/lerobot/<robot> sessions/<name> [sessions/<name2> ...] [--robot NAME]
                                   [--only-success] [--task-type pick|place] [--vcodec libx264]
                                   [--tags REGEX] [--cameras head ...] [--size WxH]

Input (written by apps/episode.py): sessions/<name>/episodes/<tag>/raw/
  meta.json   {"robot", "fps", "joint_names", "task", "success", "cameras": {cam: {"K", "size"}}, "scene", "tag", ...}
  steps.npz   t (T,), q (T, D), qd (T, D), action (T, D), tcp_pose (T, 7: xyz + quat wxyz), object_pose (T, 7),
              cam_<cam>_T_world (T, 4, 4) camera-to-world (OpenCV axes), segment (T,) trajgen.SEG,
              goal (T, 7) the place goal (pick-and-place episodes only: observation.goal; all or none of a dataset)
  frames/<cam>/00000.png ... (or, recorded as video, <cam>.mp4 at the render's size)
--base-frame adds the poses in the robot's base frame (result.json `base`: x, y, yaw deg; z 0): observation.tcp_base
(9: xyz + the first two columns of the rotation, the end-effector action of a policy), observation.object_base and
observation.goal_base (xyz), observation.base (x, y, yaw rad: the frame itself). `add_base_frame` writes them into an
existing dataset (parquet and stats only, the videos stay):
    python -m calibrated_scene_reconstruction.datagen.export.lerobot --add-base-frame outputs/lerobot/<...>
Output layout (LeRobot v2.1): meta/{info.json, episodes.jsonl, tasks.jsonl, episodes_stats.jsonl},
data/chunk-000/episode_000000.parquet, videos/chunk-000/observation.images.<cam>/episode_000000.mp4.
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from calibrated_scene_reconstruction.core.geometry.rotation6d import rot6d as rot6d, rot_from6d as rot_from6d

from calibrated_scene_reconstruction.core.paths import resolve

CHUNK = 1000
PHASES = "0 transit, 1 approach, 2 close, 3 squeeze/settle, 4 lift/hold, 5 transport, 6 place, 7 release, 8 retreat; " \
         "-1 unlabelled"
POSE7 = ["x", "y", "z", "qw", "qx", "qy", "qz"]


BASE_KEYS = ("observation.tcp_base", "observation.object_base", "observation.goal_base", "observation.base")
TCP9 = ["x", "y", "z", "r00", "r10", "r20", "r01", "r11", "r21"]






def base_frame(base, tcp_pose, object_pose, goal=None):
    """The base-frame columns (BASE_KEYS) of an episode: `base` (x, y, yaw deg), world poses (T, 7: xyz + quat wxyz)."""
    from scipy.spatial.transform import Rotation

    x, y, yaw = float(base[0]), float(base[1]), np.radians(float(base[2]))
    Rb = Rotation.from_euler("z", yaw).as_matrix()
    T = len(tcp_pose)
    to_b = lambda p: (np.asarray(p, float)[:, :3] - [x, y, 0.0]) @ Rb  # noqa: E731  (Rb.T @ (p - b), row-wise)
    R_tcp = Rb.T @ Rotation.from_quat(np.asarray(tcp_pose, float)[:, 3:7], scalar_first=True).as_matrix()
    cols = {"observation.tcp_base": np.concatenate([to_b(tcp_pose), rot6d(R_tcp)], 1),
            "observation.object_base": to_b(object_pose),
            "observation.goal_base": to_b(goal) if goal is not None else np.full((T, 3), np.nan),
            "observation.base": np.tile([x, y, yaw], (T, 1))}
    return {k: v.astype(np.float32) for k, v in cols.items()}


def base_features():
    vec = lambda n, nm: {"dtype": "float32", "shape": [n], "names": nm}  # noqa: E731
    return {"observation.tcp_base": vec(9, TCP9), "observation.object_base": vec(3, ["x", "y", "z"]),
            "observation.goal_base": vec(3, ["x", "y", "z"]), "observation.base": vec(3, ["x", "y", "yaw"])}


def pool_grasp(raw):
    """The BODex pool record (<round>/<grasp file>) behind an episode's executed grasp, or None."""
    res = json.loads((raw.parent / "result.json").read_text()) if (raw.parent / "result.json").exists() else {}
    ch = res.get("chosen")
    src = (ch.get("source") or "") if isinstance(ch, dict) else ""
    rec = resolve(src[len("bodex:"):]) if src.startswith("bodex:") else None
    if rec is None or not rec.exists():
        return None
    pf = json.loads(rec.read_text()).get("_pool_file")
    return "/".join(Path(pf).parts[-2:]) if pf else None


def find_episodes(sessions, robot=None, only_success=False, task_type=None, tags=None, pool_grasps=None):
    """[(raw dir, meta)]; `tags` a regex the episode's scene tag must match in full (one collection: .*__d\\d+);
    `pool_grasps` (["round_02/grasp_005.json", ...]): only the episodes whose grasp came from those pool records
    (one grasp, one mode of the motion to learn)."""
    eps = []
    for s in sessions:
        for raw in sorted(Path(s).glob("episodes/*/raw")):
            if not (raw / "meta.json").exists() or not (raw / "steps.npz").exists():  # a recording cut short
                continue
            meta = json.loads((raw / "meta.json").read_text())
            if robot and meta["robot"] not in robot.split(","):
                continue
            if only_success and not meta.get("success"):
                continue
            if task_type and meta.get("task_type", "pick") != task_type:
                continue
            if tags and not re.fullmatch(tags, (f"{Path(s).name}:" if ":" in tags else "") + str(meta.get("tag"))):
                continue  # a regex with ':' matches "<session>:<tag>" (per-session collections of one dataset)
            if pool_grasps and pool_grasp(raw) not in pool_grasps:
                continue
            eps.append((raw, meta))
    return eps


def _stats(x):
    x = np.asarray(x, np.float64).reshape(len(x), -1)
    return {"min": x.min(0).tolist(), "max": x.max(0).tolist(), "mean": x.mean(0).tolist(), "std": x.std(0).tolist(),
            "count": [len(x)]}


def _probe(path, *entries):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                          "stream=" + ",".join(entries), "-of", "csv=p=0", str(path)], check=True, capture_output=True,
                         text=True).stdout.strip().split(",")
    return [int(v) for v in out]


def video_length(path):
    return _probe(path, "nb_read_packets")[0]


def video_frames(path, idx=None):
    """(n, H, W, 3) uint8 frames of a video (`idx`: only those frame indices), decoded by ffmpeg."""
    w, h = _probe(path, "width", "height")
    vf = [] if idx is None else ["-vf", "select='" + "+".join(f"eq(n\\,{int(i)})" for i in idx) + "'"]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), *vf, "-vsync", "0", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)


def _image_stats(frames_dir, sample=16):
    from PIL import Image

    if Path(frames_dir).suffix == ".mp4":  # recorded as video
        n = video_length(frames_dir)
        files = range(n)
        px = video_frames(frames_dir, np.linspace(0, n - 1, min(sample, n)).astype(int)).astype(np.float32) / 255.0
    else:
        files = sorted(Path(frames_dir).glob("*.png")) or sorted(Path(frames_dir).glob("*.jpg"))
        pick = [files[i] for i in np.linspace(0, len(files) - 1, min(sample, len(files))).astype(int)]
        px = np.stack([np.asarray(Image.open(f).convert("RGB"), np.float32) / 255.0 for f in pick])  # (n, H, W, 3)
    c = px.reshape(-1, 3)
    shape = lambda v: [[[float(a)]] for a in v]  # noqa: E731  (3, 1, 1) per LeRobot
    return {"min": shape(c.min(0)), "max": shape(c.max(0)), "mean": shape(c.mean(0)), "std": shape(c.std(0)),
            "count": [len(files)]}


def make_features(names, cams, cam_meta, fps, vcodec="libx264", goal=False, base=False):
    """LeRobot v2.1 `features` of info.json: joint state / action in `names` order, poses, phase, one video per camera;
    `goal` adds observation.goal (pick-and-place), `base` the base-frame poses (BASE_KEYS)."""
    vec = lambda n, nm: {"dtype": "float32", "shape": [n], "names": nm}  # noqa: E731
    features = {
        "observation.state": vec(len(names), names), "observation.velocity": vec(len(names), names),
        "action": vec(len(names), names),
        "observation.tcp_pose": vec(7, POSE7), "observation.object_pose": vec(7, POSE7),
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
        "phase": {"dtype": "int64", "shape": [1], "names": None, "info": PHASES},
    }
    if goal:
        features["observation.goal"] = vec(7, POSE7)
    if base:
        features.update(base_features())
    for c in cams:
        w, h = cam_meta[c]["size"]
        features[f"observation.images.{c}"] = {
            "dtype": "video", "shape": [h, w, 3], "names": ["height", "width", "channels"],
            "info": {"video.height": h, "video.width": w, "video.codec": "h264" if vcodec == "libx264" else "av1",
                     "video.pix_fmt": "yuv420p", "video.is_depth_map": False, "video.fps": fps, "video.channels": 3,
                     "has_audio": False}}
    return features


def encode(frames_dir, out, fps, vcodec, size=None):
    """A camera's frames (a directory of images, or the video the recording wrote) -> the dataset's video."""
    out.parent.mkdir(parents=True, exist_ok=True)
    args = ["-c:v", vcodec, "-pix_fmt", "yuv420p"] + (["-crf", "19"] if vcodec == "libx264" else ["-crf", "30"])
    if Path(frames_dir).suffix == ".mp4":
        src = str(frames_dir)
    else:
        ext = "png" if any(Path(frames_dir).glob("*.png")) else "jpg"
        src = str(Path(frames_dir) / f"%05d.{ext}")
    if size:  # the policy's input resolution, not the render's: decoding it is the data loader's bottleneck
        args += ["-vf", f"scale={size[0]}:{size[1]}:flags=area"]
    inp = ["-i", src] if src.endswith(".mp4") else ["-framerate", str(fps), "-i", src]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", *inp, *args, "-g", "2", str(out)]
    subprocess.run(cmd, check=True)


def camera_source(raw, cam):
    """A recorded camera: raw/frames/<cam>.mp4 (recorded as video) or the directory raw/frames/<cam>/ of images."""
    v = raw / "frames" / f"{cam}.mp4"
    return v if v.exists() else raw / "frames" / cam


def _export_episode(job):
    """One episode of export(): its parquet, its videos, its stats (a worker process)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out, ei, raw, meta, ti, index, goal, base, cams, fps, vcodec, size = job
    d = np.load(raw / "steps.npz")
    T = len(d["t"])
    cols = {
        "observation.state": d["q"].astype(np.float32), "observation.velocity": d["qd"].astype(np.float32),
        "action": d["action"].astype(np.float32), "observation.tcp_pose": d["tcp_pose"].astype(np.float32),
        "observation.object_pose": d["object_pose"].astype(np.float32),
    }
    if goal:
        cols["observation.goal"] = d["goal"].astype(np.float32)
    if base:
        res = json.loads((raw.parent / "result.json").read_text())
        cols.update(base_frame(res["base"], d["tcp_pose"], d["object_pose"], d["goal"] if goal else None))
    table = {k: pa.array(list(v), type=pa.list_(pa.float32(), v.shape[1])) for k, v in cols.items()}
    table.update({"timestamp": pa.array(d["t"].astype(np.float32)), "frame_index": pa.array(np.arange(T)),
                  "episode_index": pa.array(np.full(T, ei)), "index": pa.array(np.arange(index, index + T)),
                  "task_index": pa.array(np.full(T, ti))})
    phase = d["segment"].astype(np.int64) if "segment" in d else np.full(T, -1)  # grasping.trajgen.SEG
    table["phase"] = pa.array(phase)
    p = out / "data" / f"chunk-{ei // CHUNK:03d}" / f"episode_{ei:06d}.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(table), p)
    st = {k: _stats(v) for k, v in cols.items()}
    st.update({"timestamp": _stats(d["t"]), "frame_index": _stats(np.arange(T)),
               "episode_index": _stats(np.full(T, ei)), "index": _stats(np.arange(index, index + T)),
               "task_index": _stats(np.full(T, ti)), "phase": _stats(phase)})
    for c in cams:
        fr = camera_source(raw, c)
        vid = out / "videos" / f"chunk-{ei // CHUNK:03d}" / f"observation.images.{c}" / f"episode_{ei:06d}.mp4"
        encode(fr, vid, fps, vcodec, size)
        st[f"observation.images.{c}"] = _image_stats(fr)
    ep_row = {"episode_index": ei, "tasks": [meta["task"]], "length": T,
              "source": {"session": str(raw.parents[2]), "tag": meta.get("tag"), "episode": raw.parent.name,
                         "success": meta.get("success"), "grasp": meta.get("grasp"),
                         "trajectory": meta.get("trajectory"), "task_type": meta.get("task_type", "pick"),
                         "goal": meta.get("goal"), "dr": meta.get("dr")}}
    print(f"episode {ei}: {raw} ({T} frames, success={meta.get('success')})", flush=True)
    return ep_row, {"episode_index": ei, "stats": st}


def export(out, sessions, robot=None, only_success=False, vcodec="libx264", task_type=None, tags=None, cameras=None,
           size=None, pool_grasps=None, base=False, workers=1):
    """`cameras` the cameras to export (default all), `size` (W, H) of their videos (default as rendered; the
    intrinsics in info.json calibrated_scene_reconstruction.cameras are scaled with it), `base` adds the base-frame poses (BASE_KEYS),
    `workers` episodes exported at once."""
    eps = find_episodes(sessions, robot, only_success, task_type, tags, pool_grasps)
    if base:  # a later failed run on the configuration can have rewritten result.json without the base
        nob = [raw for raw, _ in eps if "base" not in json.loads((raw.parent / "result.json").read_text())]
        for raw in nob:
            print(f"skipped (no base in result.json): {raw.parent.name}", flush=True)
        eps = [e for e in eps if e[0] not in nob]
    if not eps:
        raise SystemExit("no episodes found")
    robots = {m["robot"] for _, m in eps}
    if len(robots) > 1 and not robot:  # --robot a,b pools robots of one joint set (a torso held at another pose)
        raise SystemExit(f"several robots {sorted(robots)}: pass --robot")
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    (out / "meta").mkdir(parents=True)
    meta0 = eps[0][1]
    fps, names = int(meta0["fps"]), meta0["joint_names"]
    cams = sorted(cameras or meta0["cameras"])
    cam_meta = {c: dict(meta0["cameras"][c]) for c in cams}
    if size:
        for c in cams:
            w0, h0 = cam_meta[c]["size"]
            K = np.asarray(cam_meta[c]["K"], float) * np.array([[size[0] / w0], [size[1] / h0], [1.0]])
            cam_meta[c] = {**cam_meta[c], "size": list(size), "K": K.tolist()}
    goal = meta0.get("goal") is not None
    tasks, jobs, index = {}, [], 0
    for ei, (raw, meta) in enumerate(eps):
        if meta["joint_names"] != names or not set(cams) <= set(meta["cameras"]) or (meta.get("goal") is not None) != goal:
            raise SystemExit(f"{raw}: joint names / cameras / task type differ from the first episode")
        with np.load(raw / "steps.npz") as d:
            T = len(d["t"])
        jobs.append((out, ei, raw, meta, tasks.setdefault(meta["task"], len(tasks)), index, goal, base, cams, fps,
                     vcodec, size))
        index += T
    # episodes in parallel (their videos' decoding and encoding is the time), written and listed in order
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(max(1, workers)) as pool:
        rows = list(pool.map(_export_episode, jobs, chunksize=1))
    ep_rows, stats_rows = [r[0] for r in rows], [r[1] for r in rows]

    features = make_features(names, cams, cam_meta, fps, vcodec, goal, base)
    n = len(eps)
    info = {"codebase_version": "v2.1", "robot_type": meta0["robot"], "total_episodes": n, "total_frames": index,
            "total_tasks": len(tasks), "total_videos": n * len(cams), "total_chunks": (n - 1) // CHUNK + 1,
            "chunks_size": CHUNK, "fps": fps, "splits": {"train": f"0:{n}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features,
            "calibrated_scene_reconstruction": {"cameras": cam_meta}}
    (out / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    (out / "meta" / "tasks.jsonl").write_text("".join(json.dumps({"task_index": i, "task": t}) + "\n"
                                                      for t, i in sorted(tasks.items(), key=lambda kv: kv[1])))
    (out / "meta" / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ep_rows))
    (out / "meta" / "episodes_stats.jsonl").write_text("".join(json.dumps(r) + "\n" for r in stats_rows))
    print(f"-> {out}: {n} episodes, {index} frames, cameras {cams}")
    return out


def source_dir(src, length, tcp0):
    """The episode directory behind a dataset episode (episodes.jsonl `source`): its `episode` when recorded, else
    the one of its session and tag whose recording has this length and first TCP pose."""
    s = resolve(src["session"])
    if src.get("episode"):
        return s / "episodes" / src["episode"]
    for d in sorted((s / "episodes").glob(f"{src['tag']}__*")):
        m = d / "raw" / "meta.json"
        if not m.exists() or json.loads(m.read_text()).get("tag") != src["tag"]:
            continue
        z = np.load(d / "raw" / "steps.npz")
        if len(z["t"]) == length and np.allclose(z["tcp_pose"][0], tcp0, atol=1e-5):
            return d
    raise LookupError(f"no recording of {src['tag']} with {length} frames and that first TCP pose")


def add_base_frame(root):
    """Write the base-frame columns (BASE_KEYS) into an existing dataset: each episode's parquet, its stats and
    info.json; episodes.jsonl gains each source's episode directory. The videos are not touched."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    eps = [json.loads(line) for line in (root / "meta" / "episodes.jsonl").read_text().splitlines() if line]
    stats = {r["episode_index"]: r for r in (json.loads(line) for line in
                                             (root / "meta" / "episodes_stats.jsonl").read_text().splitlines() if line)}
    goal = "observation.goal" in info["features"]
    for e in eps:
        ei = e["episode_index"]
        p = root / info["data_path"].format(episode_chunk=ei // info["chunks_size"], episode_index=ei)
        tb = pq.read_table(p)
        tcp = np.stack(tb["observation.tcp_pose"].to_numpy(zero_copy_only=False))
        d = source_dir(e["source"], len(tcp), tcp[0])
        e["source"]["episode"] = d.name
        res = json.loads((d / "result.json").read_text())
        cols = base_frame(res["base"], tcp, np.stack(tb["observation.object_pose"].to_numpy(zero_copy_only=False)),
                          np.stack(tb["observation.goal"].to_numpy(zero_copy_only=False)) if goal else None)
        for k, v in cols.items():
            arr = pa.array(list(v), type=pa.list_(pa.float32(), v.shape[1]))
            tb = tb.set_column(tb.schema.get_field_index(k), k, arr) if k in tb.schema.names else tb.append_column(k, arr)
            stats[ei]["stats"][k] = _stats(v)
        pq.write_table(tb, p)
        print(f"episode {ei}: {d.name}", flush=True)
    info["features"].update(base_features())
    (root / "meta" / "info.json").write_text(json.dumps(info, indent=2))
    (root / "meta" / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in eps))
    (root / "meta" / "episodes_stats.jsonl").write_text("".join(json.dumps(stats[e["episode_index"]]) + "\n"
                                                                for e in eps))
    print(f"-> {root}: base-frame columns in {len(eps)} episodes")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--add-base-frame":
        for r in sys.argv[2:]:
            add_base_frame(r)
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--robot", help="the robot's episodes; several, comma-separated, pooled into one dataset when "
                                    "their joint names agree (robot_type: the first episode's)")
    ap.add_argument("--only-success", action="store_true")
    ap.add_argument("--task-type", choices=["pick", "place"], help="only episodes of this task (default: all)")
    ap.add_argument("--vcodec", default="libx264", choices=["libx264", "libsvtav1"])
    ap.add_argument("--tags", help="regex the episode's scene tag must match (a DR collection: '.*__d\\d+'); with a "
                                   "':' it matches '<session name>:<tag>'")
    ap.add_argument("--cameras", nargs="*", help="cameras to export (default: all)")
    ap.add_argument("--size", help="WxH of the videos (default: as rendered)")
    ap.add_argument("--base-frame", action="store_true", help="add the poses in the robot's base frame (BASE_KEYS)")
    ap.add_argument("--pool-grasps", nargs="*", help="only episodes whose grasp is one of these BODex pool records "
                                                     "(<round>/<grasp file>)")
    ap.add_argument("--workers", type=int, default=8, help="episodes exported at once")
    a = ap.parse_args()
    size = tuple(int(v) for v in a.size.lower().split("x")) if a.size else None
    export(a.out, a.sessions, a.robot, a.only_success, a.vcodec, a.task_type, a.tags, a.cameras, size, a.pool_grasps,
           a.base_frame, a.workers)


if __name__ == "__main__":
    main()
