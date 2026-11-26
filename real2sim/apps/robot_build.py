"""Build a robot spec's USD: assemble body (+ end effector) URDF, import it with the Isaac URDF importer, then load it
in a world and report the articulation (dofs, links, limits) as a check.

    tools/run_isaac.sh apps/robot_build.py <spec> [<spec> ...] [--force]

<spec> is a name under configs/robots/ (e.g. ur10e__robotiq_2f85). Specs with a ready USD (`usd`) are only checked.
Writes assets/robots/built/<name>/{<name>.urdf, <name>.usd, report.json} (git-ignored).
"""
import argparse
import json
import shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("specs", nargs="+")
ap.add_argument("--force", action="store_true")
args = ap.parse_args()

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.kit.commands  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from pxr import UsdGeom  # noqa: E402

from calibrated_scene_reconstruction.robots.spec import BUILT_DIR, load_spec  # noqa: E402
from calibrated_scene_reconstruction.robots.urdf import assemble  # noqa: E402



def import_urdf(urdf: Path, usd: Path, fixed_base: bool):
    _, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
    cfg.merge_fixed_joints = False  # keep flange / TCP / camera links addressable
    cfg.fix_base = fixed_base
    cfg.make_default_prim = True
    cfg.create_physics_scene = False
    cfg.import_inertia_tensor = True
    cfg.self_collision = False
    cfg.distance_scale = 1.0
    cfg.set_default_drive_type(1)  # position drives (UrdfJointTargetType.JOINT_DRIVE_POSITION)
    cfg.convex_decomp = False
    usd.parent.mkdir(parents=True, exist_ok=True)
    omni.kit.commands.execute("URDFParseAndImportFile", urdf_path=str(urdf), import_config=cfg, dest_path=str(usd))


def axis_name(v):
    i = int(np.argmax(np.abs(v)))
    return ("+" if v[i] > 0 else "-") + "xyz"[i]


def measure_axes(rob, side, q):
    """TCP-frame axes from the hand itself: link origins that move when the end effector closes are its fingers;
    fingers = their mean direction from the TCP link's origin when open; dexterous: approach (palm normal) = where the fingers curl
    to; parallel: closing = the main axis of the fingers' motion."""
    def links_in_tcp(qq):
        rob.teleport(qq)
        for _ in range(30):
            rob.world.step(render=False)
        p, R = rob.link_pose(rob.spec["arms"][side]["tcp_link"])  # the link origin: the TCP may sit past the fingers
        return {n: R.T @ (rob.link_pose(n)[0] - p) for n in rob.body_names}

    open_, closed = links_in_tcp(rob.with_ee(q, 0.0, side)), links_in_tcp(rob.with_ee(q, 1.0, side))
    moved = [n for n in rob.body_names if np.linalg.norm(closed[n] - open_[n]) > 0.004]
    if not moved:
        return {}
    Po, D = np.array([open_[n] for n in moved]), np.array([closed[n] - open_[n] for n in moved])
    fingers = Po.mean(0) / max(np.linalg.norm(Po.mean(0)), 1e-9)
    out = {"fingers": axis_name(fingers), "moving_links": len(moved)}
    curl = D.mean(0) - (D.mean(0) @ fingers) * fingers
    out["approach_dexterous"] = axis_name(curl)
    _, V = np.linalg.eigh(D.T @ D)
    out["closing_parallel"] = axis_name(V[:, -1]).lstrip("+-")
    return out


def check(spec):
    from calibrated_scene_reconstruction.sim.robot_articulation import Robot, add_robot

    omni.usd.get_context().new_stage()
    world = World(stage_units_in_meters=1.0, physics_dt=1 / 120)
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    add_robot(stage, spec, (0.0, 0.0), 0.0)
    world.reset()
    rob = Robot(world, spec)
    q = rob.home()
    rob.teleport(q)
    for _ in range(60):
        world.step(render=False)
    drift = float(np.abs(rob.q() - q).max())
    rep = {"dofs": len(rob.names), "links": len(rob.body_names), "fixed_base": rob.fixed_base, "home_drift": drift,
           "names": rob.names, "lower": rob.lower.tolist(), "upper": rob.upper.tolist()}
    for side, arm in spec["arms"].items():
        p, R = rob.tcp_pose(side)
        target = p + np.array([0.05, 0.0, 0.0])
        q2, err = rob.solve_ik(side, target, R, q)
        rep[f"tcp_{side}"] = np.round(p, 4).tolist()
        rep[f"ik_err_{side}"] = [round(e, 4) for e in err]
        rep[f"ee_axes_{side}"] = {"spec": arm.get("tcp_axes"), "measured": measure_axes(rob, side, q)}
        rob.teleport(rob.with_ee(q, 1.0, side))
        for _ in range(60):
            world.step(render=False)
        rep[f"close_ok_{side}"] = bool(np.all(np.isfinite(rob.q())))
    return rep


for name in args.specs:
    spec = load_spec(name)
    out = BUILT_DIR / spec["name"]
    if spec.get("urdf"):
        u = spec["urdf"]
        urdf = out / f"{spec['name']}.urdf"
        usd = out / f"{spec['name']}.usd"
        if args.force or not usd.exists():
            shutil.rmtree(out, ignore_errors=True)  # the importer will not rewrite an existing (e.g. empty) stage
            assemble(u["body"], urdf, u.get("ee"), u.get("flange"), u.get("prefix", "ee_"), u.get("xyz", (0, 0, 0)),
                     u.get("rpy", (0, 0, 0)), u.get("drop", ()))
            import_urdf(urdf, usd, spec["base"]["type"] != "floating")
            print(f"[{name}] imported -> {usd}", flush=True)
    rep = check(load_spec(name))
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(f"[{name}] dofs {rep['dofs']} links {rep['links']} fixed_base {rep['fixed_base']} home drift "
          f"{rep['home_drift']:.4f} | " +
          " ".join(f"{k}={v}" for k, v in rep.items() if k.startswith(("tcp_", "ik_err", "close_ok"))), flush=True)
app.close()
