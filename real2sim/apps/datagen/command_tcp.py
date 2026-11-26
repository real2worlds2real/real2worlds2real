"""The commanded hand pose of recorded episodes: forward kinematics of each frame's joint command (steps.npz `action`)
in the robot's base frame, the end-effector action label a policy's IK can follow. The pose the hand reached (the
recorded tcp_pose) trails the command by the drives' lag and, while the fingers close on an object, moves with the
contact: commanded back through IK, those poses push the cup away (recorded episodes so replayed lifted 9 of 12 cups;
the commanded poses 12 of 12).

    tools/run_isaac.sh apps/command_tcp.py <session> --robot vega_1p_sharpa_arm [--episodes <glob>] [--force]

Writes <episode>/raw/command_tcp.npz: tcp_base (T, 9: xyz + the rotation's first two columns, as
observation.tcp_base), and the check: forward kinematics of the reached joints against the recorded tcp_pose.
"""
import argparse
import glob
import json
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("session")
ap.add_argument("--robot", default="vega_1p_sharpa_arm")
ap.add_argument("--episodes", default=None, help="a glob under <session>/episodes (default every place episode of the robot)")
ap.add_argument("--force", action="store_true", help="recompute episodes that have the file")
args = ap.parse_args()

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from calibrated_scene_reconstruction.datagen.export.lerobot import base_frame, rot6d  # noqa: E402
from calibrated_scene_reconstruction.sim.robot_articulation import Robot, add_robot  # noqa: E402
from calibrated_scene_reconstruction.robots.spec import load_spec  # noqa: E402


spec = load_spec(args.robot)
assert float(spec["base"].get("height", 0.0) or 0.0) == 0.0, "a raised base: place the robot at its height"
stage = omni.usd.get_context().get_stage()
world = World(stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=1 / 30)
add_robot(stage, spec, (0.0, 0.0), 0.0)  # the base frame is the world frame
world.reset()
rob = Robot(world, spec)
side = next(iter(spec["arms"]))

pattern = args.episodes or f"*__{args.robot}__place*"
eps = sorted(p for p in glob.glob(str(Path(args.session) / "episodes" / pattern)) if (Path(p) / "raw" / "steps.npz").exists())
t0, done, unchecked, worst = time.time(), 0, 0, 0.0


def fk(q):
    rob.teleport(q)
    rob.sync()
    return rob.tcp_pose(side)


for k, ep in enumerate(eps):
    out = Path(ep) / "raw" / "command_tcp.npz"
    if out.exists() and not args.force:
        continue
    # the poses are the base frame's: the base itself (result.json) only places the recorded poses for the check (a
    # later failed run on the configuration can have rewritten result.json without one)
    base = json.loads((Path(ep) / "result.json").read_text()).get("base")
    z = np.load(Path(ep) / "raw" / "steps.npz")
    P, R = zip(*(fk(q) for q in z["action"]))
    P, R = np.asarray(P), np.asarray(R)
    err_p = err_r = np.full(0, np.nan)
    if base is None:
        unchecked += 1
    else:  # the check: the reached joints' pose against the recorded one (both in the base frame)
        chk = np.linspace(0, len(z["q"]) - 1, 8).astype(int)
        rec = base_frame(base, z["tcp_pose"], z["object_pose"])["observation.tcp_base"]
        Pq, Rq = zip(*(fk(z["q"][i]) for i in chk))
        err_p = np.linalg.norm(np.asarray(Pq) - rec[chk, :3], axis=1)
        err_r = np.linalg.norm(rot6d(np.asarray(Rq)) - rec[chk, 3:], axis=1)
        worst = max(worst, float(err_p.max()))
    np.savez_compressed(out, tcp_base=np.concatenate([P, rot6d(R)], 1).astype(np.float32),
                        check_pos_m=err_p.astype(np.float32), check_rot6d=err_r.astype(np.float32))
    done += 1
    if done % 50 == 0 or k == len(eps) - 1:
        print(f"{k + 1}/{len(eps)} episodes ({done} computed, {time.time() - t0:.0f} s); worst check {1e3 * worst:.2f} mm",
              flush=True)
print(f"done: {done} episodes computed, {unchecked} unchecked (no base in result.json), worst check {1e3 * worst:.2f} mm", flush=True)
app.close()
