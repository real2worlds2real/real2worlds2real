"""Run the Newton rope environment and its task controller."""
import json
from pathlib import Path
import numpy as np
import newton.examples
from calibrated_scene_reconstruction.sim.newton.rope import RopeStraighten

def main():
    parser = newton.examples.create_parser()
    newton.examples.add_coupled_view_args(parser)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--length", type=float, default=0.5, help="m (the protocol's 0.5 m, SoftGym's)")
    parser.add_argument("--radius", type=float, default=0.005, help="m (10 mm thick, Isaac Lab's Lift-Cable)")
    parser.add_argument("--segments", type=int, default=25)
    parser.add_argument("--density", type=float, default=800.0, help="kg/m^3: a cotton rope's (10 mm, about 60 g/m); "
                        "Isaac Lab's Lift-Cable's 100 is so light the table's friction does not hold the rope's bends, "
                        "which spring back when let go")
    parser.add_argument("--stretch", type=float, default=1.0e2)
    parser.add_argument("--bend", type=float, default=4.0e-4)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--grip-force", type=float, default=1500.0)
    parser.add_argument("--grip-offset", type=float, default=0.0, help="the TCP this far above the rope's centreline at the pinch (m)")
    parser.add_argument("--lift", type=float, default=0.05, help="carried this high above the pinch (m)")
    parser.add_argument("--drag-lift", type=float, default=0.01, help="a trace drags this high above the pinch (m)")
    parser.add_argument("--drag-speed", type=float, default=0.06, help="a trace drags this fast (m/s)")
    parser.add_argument("--max-actions", type=int, default=20)
    parser.add_argument("--substeps", type=int, default=10)
    parser.add_argument("--vbd-iterations", type=int, default=20)
    parser.add_argument("--mujoco-iterations", type=int, default=100)
    parser.add_argument("--mujoco-ls-iterations", type=int, default=20)
    parser.add_argument("--max-seconds", type=float, default=120.0, help="stop the simulation here even if not done")
    parser.set_defaults(num_frames=0)
    viewer, args = newton.examples.init(parser)
    ex = RopeStraighten(viewer, args)
    print(f"rope {args.length} m, {args.segments} segments, seed {args.seed}; goal at {ex.goal.mean(0).round(3).tolist()}, "
          f"{np.degrees(ex.goal_angle):.0f} deg", flush=True)
    frame = 0
    while frame < (args.num_frames or int(args.max_seconds * ex.fps)):
        ex.step(frame)
        ex.render()
        frame += 1
        if ex.done and ex.sim_time > ex.t_done + 2.5:
            break
    viewer.close()
    res = ex.report()
    out = Path(getattr(args, "output_path", "") or "outputs/deformable/newton/rope/rope.usd").with_suffix(".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k not in ("log", "actions", "goal")}, indent=1, default=float), flush=True)
    print("saved", out, flush=True)

if __name__ == "__main__":
    main()
