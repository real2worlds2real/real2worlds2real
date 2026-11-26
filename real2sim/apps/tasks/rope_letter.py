"""Rope writing: a rope laid into a capital letter drawn on the table in dashes (the visual goal), by a Franka holding
it by friction alone, in Newton (apps/newton_rope.py's scene: Newton's franka_cable_ik_pick_place, the FR3 in MuJoCo
on PD targets from GPU IK, the rope a VBD rod of capsules held by contact and friction).

  the letter (calibrated_scene_reconstruction.tasks.rope.letters: C J L M N S U V W Z, one stroke each, the corners rounded) is about
  --letter-height tall, the rope as long as its stroke (in 2 cm segments), set in front of the robot (shifted, turned a
  little, upright as the head camera sees it) and drawn on the table in dashes; the rope starts in a random smooth
  curve. Each pick-and-place is chosen by --planner: DeformableRavens' oracle as for the straight line
  (apps/newton_rope.py: the segment farthest from its place on the letter, the rope matched to the stroke end to end
  whichever way round is closer, put there) or model predictive control (calibrated_scene_reconstruction.tasks.rope.mpc: candidates rolled
  out in a model of the rope, a segment set down or traced along the letter), up to --max-actions. Success: every
  segment within --tol of its place.

    CUDA_VISIBLE_DEVICES=<index> tools/run_newton.sh apps/newton_rope_letter.py --seed 0 [--letters CJLMNSUVWZ]
        [--planner mpc] [--randomize] [--record <session> [--look random]] [--viewer null]

--record writes the episode in apps/episode.py's raw layout (calibrated_scene_reconstruction.sim.newton.recording: the head and front
cameras, the joints, the rope's segments); --look random draws the scene's look (calibrated_scene_reconstruction.tasks.rope.appearance: table,
floor, background, light, shadows, camera mounts, exposure) and the rope's and the dashes' colours and dash pattern;
--randomize draws the letter's height and the rope (thickness, bending stiffness, friction).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import newton.examples
from calibrated_scene_reconstruction.sim.newton import rope as newton_rope
from calibrated_scene_reconstruction.sim.newton.rope import SURFACE_Z, RopeStraighten

from calibrated_scene_reconstruction.tasks.rope.letters import LETTERS, dashes, goal_points, place, rope_length_for  # noqa: E402

JOINTS = [f"fr3_joint{j}" for j in range(1, 8)] + ["fr3_finger_joint1", "fr3_finger_joint2"]


FIXED_COLOURS = {"table": np.array([0.55, 0.57, 0.6]), "floor": np.full(3, 0.5), "background": np.full(3, 0.78),
                 "rope": np.array([0.85, 0.3, 0.15]), "ink": np.full(3, 0.1)}  # the fixed look's


class RopeLetter(RopeStraighten):
    def __init__(self, viewer, args):
        self.mpc = None
        super().__init__(viewer, args)  # the scene, the rope's random curve (and a straight goal, replaced here)
        rng = np.random.default_rng([args.seed, 5])
        self.letter = args.letter
        self.letter_centre = np.array([rng.uniform(0.47, 0.53), rng.uniform(-0.06, 0.06)])
        self.letter_up = float(np.radians(rng.uniform(-args.letter_turn, args.letter_turn)))  # 0: up along +x
        self.stroke = place(self.letter, args.length, self.letter_centre, self.letter_up, args.corner)
        self.goal = goal_points(self.stroke, self.n_seg)
        self.goal_angle = self.letter_up
        self.arms = [SimpleNamespace(k=0, dofs=slice(0, len(JOINTS)))]
        self.table_shape_idx = self.ground_shapes[0]
        self.hand = next(b for b in self.franka_bodies if self.model.body_label[b].endswith("fr3_hand"))
        self.recorder = None
        # waiting and done: the hand to the side, out of the head camera's view of the letter
        self.home_tcp = self.rest_tcp = (0.25, 0.35, SURFACE_Z + 0.3)
        self.keys[0][1:4] = self.home_tcp

    def choose(self, X, G, err):
        """--planner mpc: calibrated_scene_reconstruction.tasks.rope.mpc's pick-and-place; else the oracle's."""
        if self.args.planner != "mpc":
            return super().choose(X, G, err)
        if self.mpc is None:
            from calibrated_scene_reconstruction.tasks.rope.mpc import RopeMPC

            self.mpc = RopeMPC(self.goal, self.seg_len, newton_rope.TOL, seed=self.args.seed)
        if self.actions:  # what the model got wrong last time is not chosen again
            a = self.actions[-1]
            if not a.get("held_through_carry", True):
                self.mpc.went_wrong(a["pick"])
            elif a["max_error_after_m"] > a["max_error_before_m"] + 0.005:
                self.mpc.went_wrong(a["pick"], a["place"])
        k, to, info = self.mpc.plan(X)
        return k, to, info

    # --- what the recorder asks for
    def cameras(self):
        """{name: (eye, target, vertical fov deg)}, metres: the head over the letter, a little behind it, looking down
        (the letter reads upright there; the arm comes in below it), the front across the table."""
        return {"head": ((0.42, 0.0, SURFACE_Z + 0.9), (0.52, 0.0, SURFACE_Z), 40.0),
                "front": ((1.05, 0.25, SURFACE_Z + 0.5), (0.5, 0.0, SURFACE_Z + 0.05), 40.0)}

    def tcp(self, body_q=None, arm=None):
        """The TCP's position (the straightening's own use); with the arm, (position, the hand's quaternion xyzw)."""
        p = super().tcp()
        if arm is None:
            return p
        return p, np.asarray(body_q[self.hand, 3:7], float)

    def object_points(self):
        return self.segments().astype(np.float32)

    def look(self, kind):
        """The recorder's look: the scene's (appearance.scene_look's, or a fixed one), the rope's colour and the dashed
        letter on the table."""
        from calibrated_scene_reconstruction.deformable import appearance

        rng = np.random.default_rng([self.args.seed, 1])
        if kind == "random":
            look = appearance.scene_look(rng, self.cameras(), unit=0.01)
            rope = appearance.colour(rng, neutral_p=0.2)
            ink = appearance.contrasting(rng, look["table"], 0.3)
            dash, gap, width = rng.uniform(0.015, 0.03), rng.uniform(0.008, 0.015), rng.uniform(0.006, 0.01)
            if self.args.fixed_colours:  # the fixed look's colours over the drawn ones (the draws, and so the rest, kept)
                look.update({k: FIXED_COLOURS[k] for k in ("table", "floor", "background")})
                rope, ink = FIXED_COLOURS["rope"], FIXED_COLOURS["ink"]
        else:
            look = {"table": FIXED_COLOURS["table"], "floor": FIXED_COLOURS["floor"],
                    "background": FIXED_COLOURS["background"],
                    "light_direction": np.array([-0.3, -0.2, -0.93]) / np.linalg.norm([-0.3, -0.2, -0.93]),
                    "shadows": True, "gain": np.ones(3), "cameras": self.cameras()}
            rope, ink, dash, gap, width = FIXED_COLOURS["rope"], FIXED_COLOURS["ink"], 0.02, 0.012, 0.008
        look["shape_colours"] = {int(s): np.asarray(rope, float) for s in self.rope_shapes}
        look["rope_colour"] = np.asarray(rope, float).round(3)
        look["decal"] = {"lines": [d.round(5).tolist() for d in dashes(self.stroke, dash, gap)], "z": SURFACE_Z,
                         "width": width, "colour": np.asarray(ink, float).round(3).tolist(), "dash": dash, "gap": gap}
        return look

    def step(self, frame):
        super().step(frame)
        if self.recorder is not None and frame % 2 == 1:  # 30 frames a second
            self.recorder.capture(int(np.searchsorted(self.key_times, self.sim_time)))

    def episode_meta(self, res):
        """The episode's meta.json (apps/episode.py's fields, the letter's beside them; not `goal`, which the
        exporter takes for a pick-and-place pose)."""
        a = self.args
        return {"robot": "fr3", "arms": 1, "joint_names": [f"0:{n}" for n in JOINTS], "scene": "newton_rope_letter",
                "task": f"lay the rope along the dashed letter {self.letter}", "task_type": "rope_letter",
                "success": res["success"], "success_rule": f"every rope segment within {100 * a.tol:.1f} cm of its "
                "place on the letter (DeformableRavens' cable rule)", "seed": a.seed,
                "letter": {"letter": self.letter, "centre_m": self.letter_centre.tolist(), "height_m": a.letter_height,
                           "up_deg": float(np.degrees(self.letter_up)), "stroke_m": self.stroke.round(5).tolist(),
                           "segment_goals_m": self.goal.round(5).tolist()},
                "rope": {"length_m": a.length, "radius_m": a.radius, "segments": self.n_seg, "bend": a.bend, "mu": a.mu},
                "base": [0.0, 0.0, SURFACE_Z], "n_actions": res["n_actions"],
                "fraction_in_place": res["fraction_in_place"], "max_error_m": res["max_error_m"]}

    def report(self):
        res = super().report()
        res.update({"letter": self.letter, "letter_centre_m": self.letter_centre.tolist(),
                    "letter_up_deg": float(np.degrees(self.letter_up)), "tol_m": self.args.tol})
        return res


def main():
    parser = newton.examples.create_parser()
    newton.examples.add_coupled_view_args(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--letters", default="CJLMNSUVWZ", help="the letters drawn from (calibrated_scene_reconstruction.tasks.rope.letters)")
    parser.add_argument("--letter-turn", type=float, default=15.0, help="the letter turned up to this far (deg)")
    parser.add_argument("--letter-height", type=float, default=0.22, help="the letter about this tall (m); the rope "
                        "as long as its stroke, in segments about --segment-length long")
    parser.add_argument("--corner", type=float, default=0.03, help="the letter's corners rounded over this (m)")
    parser.add_argument("--tol", type=float, default=0.02, help="a segment is in place within this of its goal (m)")
    parser.add_argument("--segment-length", type=float, default=0.02, help="m")
    parser.add_argument("--radius", type=float, default=0.005, help="m (10 mm thick, Isaac Lab's Lift-Cable)")
    parser.add_argument("--density", type=float, default=800.0, help="kg/m^3: a cotton rope's (10 mm, about 60 g/m); "
                        "Isaac Lab's Lift-Cable's 100 is so light the table's friction does not hold the rope's bends, "
                        "which spring back when let go")
    parser.add_argument("--stretch", type=float, default=1.0e2)
    parser.add_argument("--bend", type=float, default=4.0e-4)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--grip-force", type=float, default=1500.0)
    parser.add_argument("--grip-offset", type=float, default=0.0, help="the TCP this far above the rope's centreline at the pinch (m)")
    parser.add_argument("--lift", type=float, default=0.05, help="carried this high above the pinch (m)")
    parser.add_argument("--drag-lift", type=float, default=0.01, help="a trace (--planner mpc) drags this high above "
                        "the pinch (m)")
    parser.add_argument("--drag-speed", type=float, default=0.06, help="a trace drags this fast (m/s)")
    parser.add_argument("--max-actions", type=int, default=40)
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--vbd-iterations", type=int, default=20)
    parser.add_argument("--mujoco-iterations", type=int, default=100)
    parser.add_argument("--mujoco-ls-iterations", type=int, default=20)
    parser.add_argument("--max-seconds", type=float, default=240.0, help="stop the simulation here even if not done")
    parser.add_argument("--randomize", action="store_true", help="draw the letter's height (0.18-0.25 m) and the "
                        "rope: 8-14 mm thick, bending stiffness x0.5-2 (log-uniform), friction 0.8-1.2")
    parser.add_argument("--planner", choices=("oracle", "mpc"), default="oracle", help="what chooses each "
                        "pick-and-place: DeformableRavens' oracle, or model predictive control (calibrated_scene_reconstruction.tasks.rope.mpc)")
    parser.add_argument("--record", default=None, help="a session directory: record the episode there")
    parser.add_argument("--tag", default=None, help="the episode's tag (default seed<seed>)")
    parser.add_argument("--look", choices=("fixed", "random"), default="fixed")
    parser.add_argument("--fixed-colours", action="store_true", help="with --look random: the fixed look's rope, "
                        "dash, table, floor and background colours (the light, cameras, exposure and dash pattern "
                        "still drawn)")
    parser.add_argument("--cam-size", default="384x240")
    parser.set_defaults(num_frames=0)
    viewer, args = newton.examples.init(parser)
    letters = [c for c in args.letters.upper() if c in LETTERS]
    rng = np.random.default_rng([args.seed, 3])
    args.letter = letters[rng.integers(len(letters))]
    if args.randomize:
        rng = np.random.default_rng([args.seed, 4])
        args.letter_height = float(rng.uniform(0.18, 0.25))
        args.radius = float(rng.uniform(0.004, 0.007))
        args.bend = float(args.bend * np.exp(rng.uniform(np.log(0.5), np.log(2.0))))
        args.mu = float(rng.uniform(0.8, 1.2))
    args.length = rope_length_for(args.letter, args.letter_height, args.corner)
    args.segments = max(10, int(round(args.length / args.segment_length)))
    newton_rope.TOL = args.tol  # the oracle's and the report's in-place distance
    ex = RopeLetter(viewer, args)
    if args.record:
        from calibrated_scene_reconstruction.sim.newton.recording import EpisodeRecorder

        W, H = (int(v) for v in args.cam_size.split("x"))
        ex.recorder = EpisodeRecorder(ex, args.record, args.tag or f"seed{args.seed}", ex.cameras(), size=(W, H),
                                      fps=ex.fps // 2, unit=1.0, look=ex.look(args.look))
    print(f"rope {args.length:.3f} m in {args.segments}, {2000 * args.radius:.1f} mm, seed {args.seed}: the letter {ex.letter} at "
          f"{ex.letter_centre.round(3).tolist()}, {np.degrees(ex.letter_up):.0f} deg", flush=True)
    frame = 0
    while frame < (args.num_frames or int(args.max_seconds * ex.fps)):
        ex.step(frame)
        ex.render()
        frame += 1
        if ex.done and ex.sim_time > ex.t_done + 2.0:
            break
    viewer.close()
    res = ex.report()
    out = Path(getattr(args, "output_path", "") or "outputs/deformable/newton/rope_letter/rope_letter.usd").with_suffix(".json")
    if args.record:
        out = Path(args.record) / "reports" / f"{args.tag or f'seed{args.seed}'}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    if ex.recorder is not None:
        ex.recorder.close(ex.episode_meta(res))
    print(json.dumps({k: v for k, v in res.items() if k not in ("log", "actions", "goal")}, indent=1, default=float), flush=True)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
