"""Episode component consuming explicit environment state and injected services."""
import json
import shutil
import time
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from typing import Any, Protocol


class RecordingContext(Protocol):
    BATCH: Any
    BATCH_FURNITURE: Any
    BATCH_LOGS: Any
    ENCODER: Any
    ENCODES: Any
    H: Any
    QUAD_DONE: Any
    RECORDED: Any
    SINKS: Any
    SLIDE_HINTS: Any
    SWEEPS: Any
    TASK: Any
    W: Any
    annots: Any
    args: Any
    base_check: Any
    batch_trial: Any
    body: Any
    bx: Any
    by: Any
    byaw: Any
    etag0: Any
    ext: Any
    frame: Any
    furniture_state: Any
    grasp_source: Any
    log: Any
    n_phys: Any
    obj: Any
    out: Any
    p_obj: Any
    play: Any
    prepared: Any
    record_step: Any
    reset: Any
    restore_furniture: Any
    rob: Any
    robot_cams: Any
    s: Any
    services: Any
    side: Any
    spec: Any
    stage: Any
    substitute: Any
    succeeded: Any
    sweep: Any
    target: Any
    tf: Any
    third: Any
    trial: Any
    trials: Any
    ts: Any
    world: Any
    write_episode: Any


def record_step(ctx: RecordingContext, q_cmd):
    p, R = ctx.rob.tcp_pose(ctx.side)
    po, qo = ctx.obj.get_world_pose()
    ctx.log[0]["q"].append(ctx.rob.q())
    ctx.log[0]["qd"].append(np.asarray(ctx.rob.robot.get_joint_velocities(), float))
    ctx.log[0]["action"].append(np.asarray(q_cmd, float))
    ctx.log[0]["tcp_pose"].append(np.concatenate([p, Rotation.from_matrix(R).as_quat(scalar_first=True)]))
    ctx.log[0]["object_pose"].append(np.concatenate([np.asarray(po, float), np.asarray(qo, float)]))
    for name, (cam, *_) in ctx.robot_cams.items():
        T = np.asarray(ctx.services.omni.usd.get_world_transform_matrix(cam.GetPrim())).T  # USD camera (-z view) to world
        ctx.log[0][f"cam_{name}_T_world"].append(T @ np.diag([1, -1, -1, 1]))  # OpenCV axes


def record(ctx: RecordingContext, k, tgt, traj, t):
    k += ctx.args.first_k  # after a tag's existing episodes
    etag = ctx.etag0 if k == 0 else f"{ctx.etag0}__e{k}"
    ctx.out = ctx.s / "episodes" / etag
    if ctx.out.exists():
        shutil.rmtree(ctx.out)
    (ctx.out / "frames").mkdir(parents=True)
    if "furniture" not in t:  # a recorded failure too: furniture in the robot's way slides aside first
        ctx.services.support_collisions(ctx.stage, False)
        ctx.trial(traj, "furniture off")
        t["furniture"] = ctx.services.summarize(ctx.services.clear_robot(ctx.ts, ctx.sweep[0], ctx.target))
    ctx.services.support_collisions(ctx.stage, True)
    ctx.reset(traj.q[0], traj)
    if "third" not in ctx.annots:  # once per process: a render product per recording piled up (24 GB by the second)
        ctx.annots["third"] = ctx.services.rgb_annotator(ctx.third.GetPath(), ctx.W, ctx.H)
        ctx.annots.update({k_: ctx.services.rgb_annotator(cam.GetPath(), w, h) for k_, (cam, w, h, _) in ctx.robot_cams.items()})
    for k_ in ctx.annots:
        (ctx.out / "raw" / "frames" / k_).mkdir(parents=True, exist_ok=True)
    ctx.log[0] = {k_: [] for k_ in ("q", "qd", "action", "tcp_pose", "object_pose")}
    ctx.log[0].update({f"cam_{k_}_T_world": [] for k_ in ctx.robot_cams})
    ctx.frame[0] = 0
    for _ in range(10):
        ctx.world.render()
    ctx.sweep[0] = ctx.body.restart(ctx.rob)
    ctx.SINKS.clear()
    if ctx.args.video_only:
        ctx.SINKS["_on"] = None  # play() opens a writer per stream on its first frame
    o = ctx.play(traj, record=True)
    for k_, w_ in list(ctx.SINKS.items()):
        if w_ is not None:
            w_.close()
    ctx.SINKS.clear()
    final = ctx.services.clear_robot(ctx.ts, ctx.sweep[0], ctx.target, move=False)  # QA on the recorded rollout itself
    ctx.write_episode(ctx.out, etag, tgt, traj, t, o, ctx.log[0], ctx.frame[0], final)


def write_episode(ctx: RecordingContext, out, etag, tgt, traj, t, o, logs, n_frames, final):
    """result.json, raw/{steps.npz, meta.json} and episode.mp4 of a recorded rollout (outcome `o`, per-frame `logs`,
    robot-clearance QA `final`) whose frames are already in `out`."""
    print("recorded " + ctx.services.summarize(final), flush=True)
    result = {"robot": ctx.spec["name"], "target_substitute": ctx.substitute, "base": traj.info.get("base", [ctx.bx, ctx.by, ctx.byaw]),
              "object": ctx.p_obj.tolist(),
              "extent": ctx.ext.tolist(), "task_frame": ctx.tf.to_dict(), "grasp_source": ctx.grasp_source,
              "chosen": {"source": tgt.source, "params": tgt.params, "plan": t["plan"]},
              "lift_m": o[0], "tilt_deg": o[1], "tcp_dist_m": o[2], "inhand_rot_deg": traj.info.get("inhand_rot_deg"),
              "success": ctx.succeeded(o), "task_type": ctx.TASK,
              "goal": traj.info.get("goal"), "place": o[3] if len(o) > 3 else None, "n_trials": ctx.n_phys,
              "trials": ctx.trials, "base_check": ctx.base_check, "robot_clearance": {"final": final, "ok": final["ok"]}}
    ctx.s.write_json(f"episodes/{etag}/result.json", result)
    goal = ctx.services.PlaceGoal.from_dict(traj.info["goal"]) if "goal" in traj.info else None
    extra = {"goal": np.tile(goal.as_pose7(), (n_frames, 1))} if goal else {}  # per frame, the policy's goal input
    np.savez_compressed(out / "raw" / "steps.npz", t=np.arange(n_frames) / ctx.args.fps, segment=traj.seg, **extra,
                        **{k_: np.asarray(v, np.float32) for k_, v in logs.items()})
    cams = {"third": {"K": ctx.services.pinhole(ctx.W, ctx.H).tolist(), "size": [ctx.W, ctx.H]}}
    cams.update({k_: {"K": K.tolist(), "size": [w, h]} for k_, (_, w, h, K) in ctx.robot_cams.items()})
    (out / "raw" / "meta.json").write_text(json.dumps({
        "robot": ctx.spec["name"], "fps": ctx.args.fps, "joint_names": ctx.rob.names, "cameras": cams, "tag": ctx.args.tag,
        "variant": ctx.args.variant, "scene": ctx.s.name, "target": ctx.target, "success": result["success"], "lift_m": o[0],
        "grasp": {"source": tgt.source, "params": tgt.params}, "trajectory": t["plan"],
        "task_type": ctx.TASK, "goal": traj.info.get("goal"), "dr": ctx.prepared.get("dr"),
        "exec_noise_rad": traj.info.get("exec_noise_rad", 0.0),
        "task": ctx.args.task or ctx.services.task_string(ctx.target, goal)}, indent=1, default=float))
    if not (out / "episode.mp4").exists():  # --video-only recorded it already
        ctx.ENCODES.append(ctx.ENCODER.submit(ctx.services.encode_video, out / "frames", out / "episode.mp4", ctx.args.fps))
    print(f"[{etag}] object lift {o[0]:.3f} m, tilt {o[1]:.0f} deg, {o[2]:.2f} m from the TCP -> "
          f"{'success' if result['success'] else 'FAILED'}", flush=True)


def record_batches(ctx: RecordingContext, cands):
    """--batch: the batch-validated rollouts recorded in the scene copies, a batch at a time. Per rollout its copy's
    furniture slides out of the rollout's sweep (as finalize() does in the main scene) and collides again; one run
    of the batch is then both the final validation and the recording (the recorded physics is the validated one),
    every camera rendered per command in its own copy. Rollouts that fail it are not kept. Returns the recorded
    [(tgt, traj, t)]."""
    if not ctx.BATCH_FURNITURE:
        ctx.BATCH_FURNITURE.extend(ctx.furniture_state(i) for i in range(len(ctx.BATCH)))
    done, queue = [], list(cands)
    want = ctx.args.episodes - len(ctx.RECORDED)
    while queue and len(done) < want:
        group = queue[:min(len(ctx.BATCH), want - len(done) + 2)]  # two spares: the final run may fail one
        queue = queue[len(group):]
        dirs = []
        for j, (tgt, traj, t) in enumerate(group):
            root, off = f"/World/Batch/env{j}", ctx.BATCH[j][0]
            ctx.restore_furniture(j)
            sw, off_i = ctx.SWEEPS.pop(id(traj))
            t["furniture"] = ctx.services.summarize(ctx.services.clear_robot(ctx.ts, sw.shifted(off - off_i), ctx.target, root=root, offset=off,
                                                   hints=ctx.SLIDE_HINTS))
            ctx.services.support_collisions(ctx.stage, True, root=f"{root}/Support")
            d = ctx.s / "episodes" / "_batch" / f"{ctx.etag0}__rec{j}"  # per tag: configurations record side by side
            shutil.rmtree(d, ignore_errors=True)
            dirs.append(d)
        t0 = time.time()
        res, sws = ctx.batch_trial(group, rec=dirs, sweeps=True)
        print(f"[record] {len(group)} rollouts rendered and recorded in {time.time() - t0:.1f} s", flush=True)
        for j, ((tgt, traj, t), o) in enumerate(zip(group, res)):
            root, off = f"/World/Batch/env{j}", ctx.BATCH[j][0]
            qa = ctx.services.clear_robot(ctx.ts, sws[j], ctx.target, move=False, root=root, offset=off)  # QA on the recorded rollout
            ctx.services.support_collisions(ctx.stage, False, root=f"{root}/Support")
            ctx.restore_furniture(j)
            ok = ctx.succeeded(o)
            t["final"] = {"scene": "batch final", "lift_m": o[0], "tilt_deg": o[1], "tcp_dist_m": o[2],
                          "success": ok, **({"place": o[3]} if len(o) > 3 else {})}
            t["success"] = ok
            pl = {k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in (o[3] if len(o) > 3 else {}).items()
                  if k_ != "trace"}
            print(f"[final {j}] {t['furniture']} | lift {o[0]:.3f} tilt {o[1]:.0f} {pl} -> {ok}", flush=True)
            if ok and len(done) < want:
                k = ctx.args.first_k + len(ctx.RECORDED) + len(done)
                etag = ctx.etag0 if k == 0 else f"{ctx.etag0}__e{k}"
                out_k = ctx.s / "episodes" / etag
                shutil.rmtree(out_k, ignore_errors=True)
                dirs[j].rename(out_k)
                ctx.write_episode(out_k, etag, tgt, traj, t, o, ctx.BATCH_LOGS[j], len(traj.q), qa)
                done.append((tgt, traj, t))
                q_ = ((traj.info.get("goal") or {}).get("info") or {}).get("quadrant")
                if q_ in ctx.QUAD_DONE:
                    ctx.QUAD_DONE[q_] += 1
            else:
                shutil.rmtree(dirs[j], ignore_errors=True)
    return done


def capture_frame(ctx: RecordingContext, q_cmd):
    ctx.world.render()  # render only: the recorded rollout has exactly the validated physics
    imgs = {k_: ctx.services.grab(a_, ctx.world.render) for k_, a_ in ctx.annots.items()}
    first = next((k_ for k_ in imgs if k_ != "third"), None)
    preview = np.concatenate([imgs["third"]] + ([imgs[first]] if first else []), 1)
    if ctx.SINKS:  # one writer per stream, opened on its first frame (dict.setdefault would build, and so start
        # an ffmpeg on the same file, for every frame: the idle ones emptied the video when the process ended)
        if "_preview" not in ctx.SINKS:
            ctx.SINKS["_preview"] = ctx.services.VideoWriter(ctx.out / "episode.mp4", preview.shape[1], preview.shape[0], ctx.args.fps,
                                            crf=20)
        ctx.SINKS["_preview"].write(preview)
        for k_, im in imgs.items():
            if k_ not in ctx.SINKS:
                ctx.SINKS[k_] = ctx.services.VideoWriter(ctx.out / "raw" / "frames" / f"{k_}.mp4", im.shape[1], im.shape[0], ctx.args.fps)
            ctx.SINKS[k_].write(im)
    else:
        Image.fromarray(preview).save(ctx.out / "frames" / f"{ctx.frame[0]:05d}.jpg", quality=92)
        for k_, im in imgs.items():
            Image.fromarray(im).save(ctx.out / "raw" / "frames" / k_ / f"{ctx.frame[0]:05d}.png")
    ctx.record_step(q_cmd)
    ctx.frame[0] += 1
