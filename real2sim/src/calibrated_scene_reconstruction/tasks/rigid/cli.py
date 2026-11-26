"""Argument parser shared by the current and compatibility entrypoints."""
import argparse

DEFAULT_ROBOT = "vega_1p_sharpa"
W, H = 960, 600

def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--tag", default="twin")
    ap.add_argument("--variant")
    ap.add_argument("--robot", default=DEFAULT_ROBOT)
    ap.add_argument("--robot-spec", help="the robot's spec from this JSON file instead of configs/robots/<robot>.json, under "
                    "the --robot name (a trial camera mount, say, rendered without touching the shared spec)")
    ap.add_argument("--base", default="random", help="random | best | x,y,yaw_deg")
    ap.add_argument("--episodes", type=int, default=1, help="successful episodes to record")
    ap.add_argument("--variations", type=int, default=3, help="random trajectories tried per grasp")
    ap.add_argument("--max-trials", type=int, default=30, help="physics trials before giving up")
    ap.add_argument("--max-grasps", type=int, default=24)
    ap.add_argument("--reach-check", type=int, default=8,
                    help="IK-check this many grasps from the drawn base first; a base reaching too few is swapped for "
                         "another placement that reaches more (0: off)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--task", default=None, help="language instruction stored with the episode")
    ap.add_argument("--task-type", choices=["pick", "place"],
                    help="pick (grasp + lift) or place (+ transport, place on a sampled goal, release, retreat); "
                         "default: scene_graph task.type, else pick")
    ap.add_argument("--receptacles", default="scene",
                    help="place targets: scene (the scene graph's task.place: room and weights), none (an invisible goal "
                         "point) or a comma-separated list of kinds (assets.receptacles.KINDS)")
    ap.add_argument("--transport-speed", default=None, help="lo,hi m/s cruise of the place task's transport "
                                                             "(grasping.place.PlaceCfg.transport_speed)")
    ap.add_argument("--goal-tries", type=int, default=None, help="draws of a place goal (PlaceCfg.tries): a cramped "
                                                                   "support with a large marker needs more")
    ap.add_argument("--plan-only", action="store_true", help="plan every grasp / variation and report, no physics or recording")
    ap.add_argument("--no-record", action="store_true",
                    help="statistics (tools/grasp_stats.py): validate, write every trial to "
                         "episodes/_stats/<tag>[__<robot>]__s<seed>.json, record nothing")
    ap.add_argument("--batch", type=int, default=0,
                    help="validate N planned trajectories at once, each in its own copy of the whole scene (every collider, "
                         "rigid body and material; not the lights) laid out beside the room in the same stage")
    ap.add_argument("--batch-video", action="store_true", help="render the first batch, one tile per copy, into episodes/_batch/")
    ap.add_argument("--policy", help="evaluate this LeRobot Diffusion Policy checkpoint instead of recording: --episodes "
                                     "rollouts, each from a plan's start pose to its goal (eval/<checkpoint>/<tag>.json)")
    ap.add_argument("--max-s", type=float, default=30.0, help="--policy: seconds per rollout")
    ap.add_argument("--eval-name", default=None, help="--policy: the results' directory in eval/ (default from the checkpoint)")
    ap.add_argument("--policy-action-steps", type=int, default=None,
                    help="--policy: actions executed per prediction (default the checkpoint's n_action_steps)")
    ap.add_argument("--policy-inference-steps", type=int, default=None,
                    help="--policy: denoising steps per prediction (default the checkpoint's num_inference_steps)")
    ap.add_argument("--video", action="store_true", help="--policy: write each rollout's video")
    ap.add_argument("--action-noise", type=float, default=0.0,
                    help="rad: executed arm commands carry a smooth random offset of up to this (the recorded action stays "
                         "the planned command, which pulls the arm back: corrective labels, DART-style); 0: none")
    ap.add_argument("--noise-p", type=float, default=0.7, help="--action-noise: the fraction of plans that carry it")
    ap.add_argument("--noise-near", type=float, default=0.2,
                    help="--action-noise: its weight on the approach, lift and place (NOISE_W; free space 1, the fingers "
                         "closing, holding or opening 0): more teaches the return to the grasp from near the target")
    ap.add_argument("--photo-base", type=float, default=0.0,
                    help="m: stand where the photo shows the real robot (scene graph photo_robot.base, its yaw +- the "
                         "--base-jitter degrees), every drawn base within this distance of it (0: the accepted placements)")
    ap.add_argument("--base-jitter", default="0,0",
                    help="m,deg: --batch: every planning round after the first on a new base, an accepted placement moved by "
                         "up to this (episodes recorded per round, on their own base)")
    ap.add_argument("--plan-passes", type=int, default=4,
                    help="--batch: passes over the grasps, each with new goals, starts and paths (a configuration with few "
                         "grasps ran out after one)")
    ap.add_argument("--render-cams", nargs="+", default=None,
                    help="instead of recording: render these robot cameras (--robot-spec: one the recordings did not have) "
                         "over this tag's recorded episodes, replayed kinematically (each frame's recorded joints and cup, "
                         "the recording's base and goal): raw/frames/<cam>.mp4, and the camera added to raw/meta.json")
    ap.add_argument("--render-out", help="--render-cams: write <episode>/<cam>.mp4 under this directory instead (a check "
                    "against the recorded videos: render head too and compare)")
    ap.add_argument("--render-overwrite", action="store_true", help="--render-cams: render episodes that have the videos")
    ap.add_argument("--render-cup-offset", nargs=2, type=float, metavar=("FWD", "LEFT"),
                    help="--render-cams --render-out: the cup drawn this far (m) from where it was, in the robot's base frame "
                         "(a counterfactual: does a policy follow the cup in its images?)")
    ap.add_argument("--render-joint-offset", nargs=2, metavar=("JOINT", "RAD"),
                    help="--render-cams --render-out: that joint drawn this far from its recorded angle (the hand moved in the "
                         "images while the recorded state says otherwise)")
    ap.add_argument("--render-max-episodes", type=int, default=0, help="--render-cams: at most this many episodes of the tag")
    ap.add_argument("--dp-server", help="--policy is an official Diffusion Policy checkpoint (configs/dp_official) served at "
                    "HOST:PORT by tools/dp_official_server.py (calibrated_scene_reconstruction.policy.runners.ServedDPRunner)")
    ap.add_argument("--openpi-server", help="--policy is a fine-tuned pi0.5 checkpoint (calibrated_scene_reconstruction.policy.openpi) served "
                    "at HOST:PORT by tools/openpi_serve.py (calibrated_scene_reconstruction.policy.runners.ServedOpenpiRunner)")
    ap.add_argument("--first-k", type=int, default=0,
                    help="index of the first episode recorded (episodes/<tag>__place__e<k>): add to a tag's episodes")
    ap.add_argument("--start-states", nargs="*", default=None,
                    help="npz files of joint states a policy reached (--policy writes <tag>_states.npz): plans start from "
                         "one of them with --start-states-p (DAgger-style: the expert's way on from the policy's states); "
                         "'{tag}' in a path stands for this run's episode tag (missing files skipped). States saved with "
                         "their base (--object-window): the copy that starts from one stands on that base")
    ap.add_argument("--start-states-p", type=float, default=0.5)
    ap.add_argument("--style", choices=["random", "fixed"], default="random",
                    help="fixed: one motion style (trajgen.fixed_style, place.fixed_style: the path's shape, timing and "
                         "pregrasp offset at their midpoints, no wrist roll at the standoff); the start pose stays random")
    ap.add_argument("--pre-release-s", default="0,0", help="lo,hi s: set down and still before the fingers open")
    ap.add_argument("--done-hold-s", default="0,0", help="lo,hi s: still after the retreat before the episode ends")
    ap.add_argument("--goal-direction", choices=["task", "base"], default="task",
                    help="base: the goal marker in a quadrant of the robot's base frame (front / left / back / right, "
                         "uniform per plan) at --goal-dist from the object (PlaceCfg.direction)")
    ap.add_argument("--goal-dist", default="0.15,0.40", help="lo,hi m: --goal-direction base")
    ap.add_argument("--goal-quadrants", default="front,left,back,right",
                    help="--object-window: the goal quadrants a copy may draw (a collection filling the rare directions)")
    ap.add_argument("--goal-max-reach", type=int, default=0,
                    help="give a goal draw up after this many failed reach checks (IK; 0: no limit)")
    ap.add_argument("--object-window", default=None,
                    help="f0,f1,l0,l1 m: --batch: every planning round draws its base so that the object lands uniformly "
                         "in this window of the base frame (forward f0..f1, left l0..l1), the yaw an accepted "
                         "placement's +- the --base-jitter degrees, clear of the room")
    ap.add_argument("--plans-per-copy", type=int, default=1,
                    help="--object-window: plans per copy's base and round (more: fewer distinct bases)")
    ap.add_argument("--fail-fast", type=int, default=0,
                    help="--batch: give a configuration up after this many physics trials without a success (0: never)")
    ap.add_argument("--max-rounds", type=int, default=30, help="--batch: planning rounds (bases) at most")
    ap.add_argument("--video-only", action="store_true",
                    help="record the cameras straight to video (raw/frames/<cam>.mp4, episode.mp4), no image files")
    ap.add_argument("--from-episodes", action="store_true",
                    help="--policy: start from this tag's recorded episodes (their base, first joint state and goal) "
                         "instead of new plans: a held-out scene's demonstrations are the test set")
    ap.add_argument("--resume-states", nargs="*", default=None,
                    help="npz files of states a policy failed in (tools/fail_states.py: joint state q, base, cup pose "
                         "xyz + wxyz; '{tag}' stands for this run's episode tag): each plan starts from one of them, the "
                         "robot on its base, the cup where it lay and the grasps re-aimed at it (DAgger from the first failed "
                         "grasp); with --batch 0, at most one episode per state")
    ap.add_argument("--resume-grasps", type=int, default=8, help="--resume-states: grasps tried per state")
    ap.add_argument("--policy-new-bases", action="store_true",
                    help="--policy without --from-episodes: every grasp's rollouts from a new base (--base-jitter, "
                         "--object-window, as the collection draws them), not the one placement: new layouts of a training "
                         "configuration, whose policy states (<tag>_states.npz) a next collection starts from (DAgger)")
    ap.add_argument("--policy-layouts", default=None,
                    help="--policy-new-bases: a file of the rollouts' starts ('{tag}' stands for this run's episode tag): "
                         "read in place of their search when it exists (the search takes minutes to an hour a "
                         "configuration), else written after it; the same configuration, seed and layout arguments draw "
                         "the same starts, so runs that share the file roll out from the same layouts")
    return ap.parse_args(argv)
