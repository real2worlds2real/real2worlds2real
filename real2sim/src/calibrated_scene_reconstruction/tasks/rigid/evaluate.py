"""Episode component consuming explicit environment state and injected services."""
import shutil
import time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from calibrated_scene_reconstruction.core.paths import rel

def _phys_steps(ctx, n):
    """`n` physics steps at 120 Hz as steps of the scene's own rate (EpisodeRuntime.phys_steps); a context without
    that clock keeps n."""
    scaled = getattr(ctx, "phys_steps", None)
    return scaled(n) if scaled else n


def evaluate_policy(ctx, plans):
    """--policy: the checkpoint drives the robot from each plan's start pose (the plan's motion is not used: it gives
    a start pose and a goal the arm reaches) for --max-s seconds or until the place succeeds, observing what the
    dataset recorded (robot cameras, joint positions, the goal when the policy was trained with it); success as a
    recorded episode's. A policy with end-effector actions commands the hand pose through ee_command's IK.
    Writes eval/<checkpoint>/<tag>.json (+ <tag>_<k>.mp4 with --video)."""

    ck = Path(ctx.args.policy)
    runner, use_goal = ctx.services.make_runner(ctx.args)
    for k_, (cam, w, h, _) in ctx.robot_cams.items():
        ctx.annots[k_] = ctx.services.rgb_annotator(cam.GetPath(), w, h)
    if ctx.args.video:
        ctx.annots["third"] = ctx.services.rgb_annotator(ctx.third.GetPath(), ctx.W, ctx.H)
    res_dir = ctx.s / "eval" / (ctx.args.eval_name or f"{ck.parent.name}__{ck.name}")
    res_dir.mkdir(parents=True, exist_ok=True)
    res = []
    home = ctx.furniture_state(None, "/World/Support")
    for k, (tgt, traj, _) in enumerate(plans):
        b_ = traj.info.get("base")
        if b_ is not None and not np.allclose(b_, [ctx.bx, ctx.by, ctx.byaw]):
            ctx.move_base(*b_)
            ctx.bx, ctx.by, ctx.byaw = b_
        # the furniture as the recording had it: what the recorded motion sweeps through slides aside (clear_robot,
        # as the collection did), so the robot's own base is not under a stool the training images never showed
        ctx.restore_furniture(None, home)
        ctx.reset(traj.q[0], traj)
        sw = ctx.body.restart(ctx.rob)
        for q_ in traj.q[::3]:
            ctx.rob.teleport(q_)
            ctx.rob.sync()
            sw.sample()
        cleared = ctx.services.clear_robot(ctx.ts, sw, ctx.target, hints=ctx.SLIDE_HINTS)
        ctx.services.support_collisions(ctx.stage, True)
        ctx.reset(traj.q[0], traj)
        # the cup where the rollout starts (its recording's first frame, else where the scene put it): the sweep above
        # can knock it away (the hand passes through it) and the world reset does not always bring it back
        c0 = np.asarray(traj.info["cup0"], float) if "cup0" in traj.info else np.r_[ctx.p_obj, ctx.q_obj]
        cup_reset = float(np.linalg.norm(np.asarray(ctx.obj.get_world_pose()[0], float) - c0[:3]))
        if cup_reset > 0.003:
            ctx.obj.set_world_pose(c0[:3], c0[3:7])
            ctx.obj.set_linear_velocity(np.zeros(3))
            ctx.obj.set_angular_velocity(np.zeros(3))
            for _ in range(_phys_steps(ctx, 30)):
                ctx.rob.command(traj.q[0])
                ctx.world.step(render=False)
        runner.reset()
        Rb0 = Rotation.from_euler("z", np.radians(ctx.byaw)).as_matrix()
        for _ in range(10):
            ctx.world.render()
        # the first frames: a run that reads its starts (--policy-layouts) gets here without the search's minutes of
        # resets and rendering, while the renderer still loads the scene: its render products give no data for a while
        # (0 s in the runs that work; a renderer that gave none for 600 s never did -- 161 crashes in a day -- so a
        # stuck one is given up after 120 s and the run retried in a fresh Kit)
        t_warm = time.time()
        while k == 0:
            try:
                for a_ in ctx.annots.values():
                    ctx.services.grab(a_)
                print(f"[{ctx.etag0}] cameras ready after {time.time() - t_warm:.0f} s", flush=True)
                break
            except TypeError:
                if time.time() - t_warm > 120:
                    print(f"[{ctx.etag0}] cameras gave no data for {time.time() - t_warm:.0f} s", flush=True)
                    raise
                ctx.world.render()
                time.sleep(0.5)
        goal7 = ctx.services.PlaceGoal.from_dict(traj.info["goal"]).as_pose7() if use_goal and "goal" in traj.info else None
        vid = res_dir / f"{ctx.etag0}_{k}" if ctx.args.video else None
        if vid:
            vid.mkdir(exist_ok=True)
        t0, o, n = time.time(), None, int(ctx.args.max_s * ctx.args.fps)
        p_cup0 = np.asarray(ctx.obj.get_world_pose()[0], float)
        q_cmd, ik_err = np.asarray(traj.q[0], float).copy(), []
        trace_ = []  # per frame (world): TCP xyz, object xyz, joint state, TCP rotation (9), object quaternion wxyz
        environment = RigidPolicyEnvironment(ctx, traj, vid, goal7, Rb0, p_cup0, q_cmd, ik_err, trace_, n)
        rollout = ctx.services.run_rollout(runner, environment, n, ctx.args.fps)
        i, o = rollout.steps - 1, rollout.outcome
        trace_, ik_err = environment.trace, environment.ik_err
        ctx.services.support_collisions(ctx.stage, False)
        if trace_:
            np.save(res_dir / f"{ctx.etag0}_{k}_trace.npy", np.asarray(trace_, np.float32))
        ok = bool(ctx.succeeded(o))
        res.append({"k": k, "success": ok, "furniture": ctx.services.summarize(cleared), "steps": i + 1, "s": round(time.time() - t0, 1),
                    "base": [round(float(v), 4) for v in (ctx.bx, ctx.by, ctx.byaw)], "cup_reset_m": round(cup_reset, 4),
                    "cup0": np.asarray(trace_[0][3:6] if trace_ else ctx.obj.get_world_pose()[0], float).round(4).tolist(),
                    "lift_m": o[0],
                    "tilt_deg": o[1], "tcp_dist_m": o[2], **({"place": o[3]} if len(o) > 3 else {}),
                    "goal": traj.info.get("goal"), "start_from": traj.info.get("source"),
                    **({"ik_err_median": np.median(np.asarray(ik_err), 0).round(4).tolist(),
                        "ik_err_p95": np.percentile(np.asarray(ik_err), 95, 0).round(4).tolist()} if ik_err else {})})
        print(f"[{ctx.etag0}] policy rollout {k}: {'success' if ok else 'FAILED'} after {(i + 1) / ctx.args.fps:.1f} s",
              flush=True)
        if vid:
            ctx.services.encode_video(vid, res_dir / f"{ctx.etag0}_{k}.mp4", ctx.args.fps)
            shutil.rmtree(vid, ignore_errors=True)
    if ctx.VISITED:
        np.savez_compressed(res_dir / f"{ctx.etag0}_states.npz", q=np.asarray(ctx.VISITED, np.float32),
                            base=np.asarray(ctx.VISITED_BASE, np.float32), joint_names=np.asarray(ctx.rob.names))
    rate = float(np.mean([r["success"] for r in res])) if res else None
    ctx.s.write_json(f"eval/{res_dir.name}/{ctx.etag0}.json", {"checkpoint": rel(ck), "tag": ctx.args.tag, "robot": ctx.spec["name"],
                                                       "use_goal": use_goal, "success_rate": rate, "rollouts": res})
    print(f"[{ctx.etag0}] policy success {rate} over {len(res)} rollouts", flush=True)


class RigidPolicyEnvironment:
    """Task adapter for the generic policy loop; the rollout service owns inference order."""
    def __init__(self, ctx, traj, vid, goal7, Rb0, p_cup0, q_cmd, ik_err, trace, n):
        self.ctx, self.traj, self.vid = ctx, traj, vid
        self.goal7, self.Rb0, self.p_cup0 = goal7, Rb0, p_cup0
        self.q_cmd, self.ik_err, self.trace, self.n = q_cmd, ik_err, trace, n

    def observe(self, i, runner):
        if i % (self.ctx.args.fps // 2) == 0 and np.linalg.norm(np.asarray(self.ctx.obj.get_world_pose()[0], float) - self.p_cup0) < 0.005:
            self.ctx.VISITED.append(self.ctx.rob.q())  # a state the policy reached with the cup still where it stood
            self.ctx.VISITED_BASE.append([self.ctx.bx, self.ctx.by, self.ctx.byaw])
        self.ctx.world.render()
        imgs = {c: self.ctx.services.grab(self.ctx.annots[c], self.ctx.world.render) for c in runner.cams}
        if self.vid:
            extra = [np.asarray(Image.fromarray(im).resize((2 * round(im.shape[1] * self.ctx.H / im.shape[0] / 2), self.ctx.H)))  # even widths (x264)
                     for im in [imgs[c] for c in runner.cams[1:]]  # other camera, the policy's then the rest
                     + [self.ctx.services.grab(self.ctx.annots[c], self.ctx.world.render) for c in self.ctx.robot_cams if c not in runner.cams]]
            Image.fromarray(np.concatenate([self.ctx.services.grab(self.ctx.annots["third"], self.ctx.world.render), imgs[runner.cams[0]], *extra], 1)).save(
                self.vid / f"{i:05d}.jpg", quality=88)
        tcp_b = {}
        if getattr(runner, "wants_tcp", False):  # the hand's pose in the base frame (a runner that reads it)
            tp_, tR_ = self.ctx.rob.tcp_pose(self.ctx.side)
            tcp_b = {"tcp": (self.Rb0.T @ (np.asarray(tp_, float) - [self.ctx.bx, self.ctx.by, 0.0]), self.Rb0.T @ np.asarray(tR_, float))}
        return imgs, self.ctx.rob.q(), self.goal7, tcp_b

    def step(self, action):
        tp_, tR_ = self.ctx.rob.tcp_pose(self.ctx.side)
        cp_, cq_ = self.ctx.obj.get_world_pose()
        self.trace.append(np.r_[tp_, np.asarray(cp_, float), self.ctx.rob.q(), np.asarray(tR_, float).ravel(),
                            np.asarray(cq_, float)])
        if isinstance(action, dict):  # end-effector action
            self.q_cmd, e_ = self.ctx.ee_command(action, self.q_cmd)
            self.ik_err.append(e_)
        else:
            self.q_cmd = action
        self.ctx.rob.command(self.q_cmd)
        for _ in range(_phys_steps(self.ctx, 4)):
            self.ctx.world.step(render=False)

    def measure(self):
        p, q = self.ctx.obj.get_world_pose()
        o = self.ctx.outcome()
        o = (*o, self.ctx.place_meas(self.traj, p, q, self.ctx.rob.tcp_pose(self.ctx.side)[0], self.ctx.rob.q(), self.ctx.p_obj[2])) if "goal" in self.traj.info else o
        return o

    def succeeded(self, outcome):
        return self.ctx.succeeded(outcome)
