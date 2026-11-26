"""BODex grasps for a session's task target, per robot hand (runs in envs/grasp).

    envs/grasp/bin/python -m calibrated_scene_reconstruction.grasping.synth sessions/<name> --robot vega_1p_sharpa [--variants twin cousin_00]
                                                  [--seeds 96] [--top-k 16] [--opt-iters 500] [--force]

For every scene (the twin and each cousin variant, or --variants) the target is rebuilt exactly as the simulator
instantiates it (grasping.targets), its cavities filled (targets.solid_mesh: grasps hold the outside of a cup, not its
inside), seated on its support plane (z = 0) and handed to the tabletop BODex synthesis (grasping.bodex.tabletop):
staged contact cost + force-closure QP + self-collision + a penalty on the support plane AND the environment around
the target (walls, furniture, other objects: boxes exported by apps/scene_obstacles.py into grasps/_env/<tag>.json),
then a pregrasp and a squeeze posture per record that clear the same environment. Seeds face the executing arm, sit on
the target's outside (--max-hull-depth) and are dropped where the palm would sit inside an obstacle; their roll about
the approach axis favours a wrist turned toward the arm, and a converged grasp whose wrist still faces away from it is
vetoed (the arm cannot present a forearm from the object's far side: on a Vega placement such grasps were out of IK
reach by 0.24-0.37 m). Each record notes how deep its contact spheres sit inside the target's convex hull
(--contact-depth gates on it) and whether they rest on the filling; those contact checks and the approach corridor the
trajectory planner flies in on (ContactGate) run inside the ranked walk, so a failing grasp does not take one of the
--top-k slots, and records whose grasp_error_max exceeds
--max-grasp-error are marked failed afterwards. Identical target meshes share one synthesis (a cousin keeping the
twin's cup, two cousins with the same YCB cup); a changed option (--seed among them) re-synthesizes. The same --seed
reproduces most records bit for bit; a few of 96 seeds can converge differently between runs (nondeterministic CUDA
reductions in the optimizer; PyTorch's deterministic mode trips an internal assert in its index_put), so success
rates come from tools/grasp_stats.py, not from one run.

Writes grasps/_scenes/<key>__solid/ (the seated solid and its CoACD parts, which the simulator also collides with) and
grasps/<hand>/<key>/{grasp_NNN.json, summary.json}; a record's stages hold the hand base link's pose (position, wxyz)
in the target's prim frame and the hand joints by name.
"""
from __future__ import annotations

import json
from collections import Counter
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh

from ..assets.physprops import mass_props  # noqa: F401 -- legacy public attribute
from ..robots.spec import load_spec  # noqa: F401 -- legacy public attribute
from calibrated_scene_reconstruction.core.contracts.scene import SceneInputs as Session
from . import require_bodex
from .targets import (GRASP_DIR,  # noqa: F401 -- legacy public helpers
                      arm_direction, grasp_dir, grasp_key, hand_name, local_mesh, mesh_key, object_label,
                      robot_axes, scene_env, scene_key, solid_mesh, target_spec)

ROOT = Path(__file__).resolve().parents[3]

#: Tabletop knobs left to the hand asset (CLI > asset `tabletop` section > shared default); None = not overridden.
TABLETOP_KEYS = (
    "final_table_clearance_m", "min_pregrasp_clearance_m", "seed_min_height_m", "seed_max_downward_cos",
    "seed_max_upward_cos", "require_pregrasp_feasible", "forbid_backhand", "backhand_margin_m",
    "pregrasp_posture_seed", "force_closure_weight", "penetration_weight", "selfcollision_weight", "table_margin_m",
    "table_weight", "friction_coefficient", "k_lower", "stage0_contact_distance_m", "force_closure_max_stage",
    "contact_surface_distance_limit_m", "contact_surface_distance_gate_m", "contact_surface_distance_limit_weight",
    "contact_surface_distance_role_weights", "contact_surface_distance_start_progress", "approach_alignment_weight",
    "approach_alignment_cone_deg", "approach_alignment_gate_deg", "palm_facing_weight", "palm_facing_cone_deg",
    "palm_facing_gate_deg", "palm_facing_start_progress", "contact_sphere_facing_weight",
    "contact_sphere_facing_cone_deg", "contact_sphere_facing_gate_deg", "contact_sphere_facing_start_progress",
    "enclosure_weight", "enclosure_depth_m", "enclosure_min_pairs", "enclosure_gate_m")




def variants(s: Session, names=None):
    """[(tag, variant)] for the twin and the cousins, or the named tags (cousins and the domain-randomized
    configurations of calibrated_scene_reconstruction.dr too); scene/twin.json overlaid as sim.scene.prepare_variant does."""
    from calibrated_scene_reconstruction.core.artifacts import tagged_json_path

    base = s.read_json("scene/twin.json")
    names = names or ["twin"] + sorted(f.stem for f in (s / "cousins").glob("cousin_*.json"))
    out = []
    for tag in names:
        f = tagged_json_path(s, tag, ("cousins", "dr"), identity_tag="twin", label="variant")
        out.append((tag, {**base, **json.loads(f.read_text())} if f else dict(base)))
    return out


def scene_dir(s: Session, key: str, V, F, com=None, solid=True) -> Path:
    """Tabletop scene directory of one target mesh (already seated: bottom centre at the origin): its solid (or, with
    solid=False, the mesh as reconstructed: a demo held it by a wall, a finger inside its opening), and its semantic
    centre of mass (the wrench space of the synthesis is taken about it)."""
    require_bodex()
    from .bodex.tabletop.scene import RestingPose, build_scene

    k = scene_key(key) if solid else f"{key}__real"
    d = s / GRASP_DIR / "_scenes" / k
    if not (d / "sequence.json").exists():
        with tempfile.TemporaryDirectory() as tmp:  # build_scene names the seated mesh after its source: <k>.obj
            src = Path(tmp) / f"{k}.obj"
            trimesh.Trimesh(*(solid_mesh(V, F) if solid else (V, F)), process=False).export(src)
            build_scene(src, d, resting_pose=RestingPose(0, np.eye(4), 1.0, "mesh_frame"),
                        metadata={"center_of_mass": com} if com is not None else None)
    meta = json.loads((d / "sequence.json").read_text())
    if meta.get("center_of_mass") != com:  # built before the centre of mass was known, or an A/B run without it
        meta = {k: v for k, v in meta.items() if k != "center_of_mass"}
        (d / "sequence.json").write_text(json.dumps({**meta, **({"center_of_mass": com} if com else {})}, indent=2)
                                         + "\n")
    if not (d / "real.obj").exists():  # the unfilled target: seed sites must lie on its surface
        trimesh.Trimesh(V, F, process=False).export(d / "real.obj")
    return d


class ContactGate:
    """Per-grasp checks of one target: how deep the record's contact spheres (the asset's contact roles) sit inside the
    target's convex hull (a handle hooked through, a body gripped beside its handle; failed beyond `max_depth`), and,
    with `real` (the target's own mesh (V, F), where V, F is its solid), how many touching spheres are more than
    `real_gap` from that real surface -- resting on the filling, a finger pressing on a cup's opening as if it were a
    lid (failed when any). With `env` (obstacle boxes in the target frame, grasping.targets.scene_env; the support's
    own box left out), the approach corridor the trajectory planner (grasping.trajgen) flies in on: the hand backed
    off along its approach axis (levelled when it dives toward the support) by `corridor` must miss every box, and at
    the last distance the target too -- otherwise every planned trajectory of that grasp is rejected ("final approach
    leg blocked", "the pregrasp posture does not clear the target"). Runs inside the synthesis' ranked walk (a failing
    grasp does not take a top-k slot) and again on the written records (quality_gate)."""

    def __init__(self, hand_asset: Path, V, F, max_depth=None, real=None, real_gap=0.006, env=None,
                 corridor=(0.04, 0.08, 0.12), margin=0.003):
        require_bodex()
        from curobo.types import DeviceCfg

        from .bodex.assets import load_end_effector
        from .bodex.clearance import ClearanceChecker

        asset = load_end_effector(hand_asset, validate_contacts=False)
        self.cc = ClearanceChecker(asset, DeviceCfg())
        sel = [v for r in asset.contact_roles.values() for v in ((r,) if isinstance(r, str) else r)]
        self.idx = self.cc.resolve_sphere_selectors(sel)
        self.solid = trimesh.Trimesh(V, F, process=False)
        hull = self.solid.convex_hull
        self.real_mesh = trimesh.Trimesh(*real, process=False) if real is not None else None
        self.radii = self.cc.radii.cpu().numpy()
        self.N = np.asarray(hull.face_normals)
        self.d = np.einsum("ij,ij->i", self.N, np.asarray(hull.triangles)[:, 0])
        self.max_depth, self.real_gap = max_depth, real_gap
        self.approach = np.asarray(asset.config.get("retarget", {}).get("approach_axis", (1.0, 0.0, 0.0)), float)
        self.env = [(np.asarray(o["center"], float), np.asarray(o["half"], float), np.asarray(o["R"], float),
                     o.get("name", "")) for o in (env or [])]
        self.corridor, self.margin = corridor, margin

    def approach_clear(self, action, P_all):
        """{"ok", "blocked_by", "at_m"} of the approach corridor (see the class docstring)."""
        from scipy.spatial.transform import Rotation

        R = Rotation.from_quat(np.asarray(action[3:7], float), scalar_first=True).as_matrix()
        back = -(R @ self.approach)
        back /= np.linalg.norm(back)
        if back[2] < 0.0:  # the planner levels a back-off that would dive into the support
            back[2] = 0.0
            if np.linalg.norm(back) < 0.5:
                return {"ok": False, "blocked_by": "approach from under the target", "at_m": 0.0}
            back /= np.linalg.norm(back)
        r = self.radii + self.margin
        for d in self.corridor:
            Q = P_all + d * back
            for c, h, Rb, name in self.env:
                if (np.abs((Q - c) @ Rb) <= h[None] + r[:, None]).all(axis=1).any():
                    return {"ok": False, "blocked_by": name, "at_m": d}
        Q = P_all + self.corridor[-1] * back
        if (trimesh.proximity.signed_distance(self.solid, Q) > -r).any():  # positive inside: still in the target
            return {"ok": False, "blocked_by": "target", "at_m": self.corridor[-1]}
        return {"ok": True, "blocked_by": None, "at_m": None}

    def env_hit(self, action):
        """Name of the first environment box the hand's spheres (at `action`, no back-off) touch, else None."""
        import torch

        a = torch.tensor([list(map(float, action))], dtype=torch.float32, device="cuda")
        P = self.cc.sphere_world_positions(a)[0].cpu().numpy()
        r = self.radii + self.margin
        for c, h, Rb, name in self.env:
            if (np.abs((P - c) @ Rb) <= h[None] + r[:, None]).all(axis=1).any():
                return name
        return None

    def __call__(self, action):
        """(ok, {"real_contact": .., "outer_contact": ..}) of one hand action (base pose + joints, the seated frame)."""
        import torch

        a = torch.tensor([list(map(float, action))], dtype=torch.float32, device="cuda")
        P_all = self.cc.sphere_world_positions(a)[0].cpu().numpy()
        P = P_all[self.idx]
        depth = float(-(P @ self.N.T - self.d).max(axis=1).min())  # deepest contact sphere centre inside the hull
        fake = 0
        if self.real_mesh is not None:  # spheres touching the solid but far from the real surface: on the filling
            gap_solid = -trimesh.proximity.signed_distance(self.solid, P_all) - self.radii
            touching = gap_solid < 0.003
            if touching.any():
                _, dist_real, _ = trimesh.proximity.closest_point(self.real_mesh, P_all[touching])
                fake = int(((dist_real - self.radii[touching]) > self.real_gap).sum())
        ok = self.max_depth is None or depth <= self.max_depth
        info = {"real_contact": {"spheres_on_filling": fake, "ok": fake == 0},
                "outer_contact": {"max_hull_depth_m": depth, "limit_m": self.max_depth, "ok": ok}}
        if self.env:
            info["approach_corridor"] = self.approach_clear(action, P_all)
            ok = ok and info["approach_corridor"]["ok"]
        return ok and fake == 0, info


def quality_gate(out: Path, gate: ContactGate, max_error=None):
    """Mark records failed whose `grasp_error_max` (BODex's worst contact error: how far the contacts are from the
    surface / the force closure from balance) exceeds `max_error` -- the tail of a crowded scene's seeds, which a
    simulator squeeze does not hold -- or whose contacts fail `gate` (ContactGate). Returns the number of records
    kept."""
    kept = 0
    for f in sorted(out.glob("grasp_*.json")):
        r = json.loads(f.read_text())
        if not r.get("success"):
            continue
        ok, info = gate(r["action"])
        r.update(info)
        q_ok = max_error is None or float(r.get("grasp_error_max", 0.0)) <= max_error
        r["quality_gate"] = {"grasp_error_max": r.get("grasp_error_max"), "limit": max_error, "ok": q_ok}
        if not (ok and q_ok):
            r["success"] = False
        else:
            kept += 1
        f.write_text(json.dumps(r, indent=1))
    return kept


def synthesize(s: Session, hand_asset: Path, key: str, V, F, seeds, top_k, opt_iters, seed, facing=None,
               min_facing_cos=0.2, env=None, force=False, max_hull_depth=0.008, contact_depth=None,
               max_error=0.05, com=None, region=None, solid=True, min_kept=None, max_seeds=None):
    """`com`: the target's semantic centre of mass (wrench space). `region` (calibrated_scene_reconstruction.datagen.demo.functional.seed_region):
    {"center", "radius"} in the seated mesh's frame -- seeds only there (a functional grasp: a mug by its handle); the
    grasp set gets its own key. solid=False: on the mesh as reconstructed, cavities open (a demo straddling a wall).
    With `min_kept`: fewer grasps pass the quality gate -> again with twice the seeds (the seed's first Halton draws
    plus as many more: a superset, not a reroll), up to `max_seeds`; summary["seeds_used"]."""
    require_bodex()
    from .bodex.tabletop.synthesize import solve_tabletop_grasps

    tag = grasp_key(key, facing, env) + ("" if solid else "__real")
    if region:
        tag += "__region" + "_".join(f"{v:+.2f}" for v in (*region["center"], region["radius"]))
    out = grasp_dir(s, hand_asset.stem, tag)
    options = {"target": "solid" if solid else "real", "seeds": seeds, "top_k": top_k,
               "seed_max_hull_depth": max_hull_depth,
               "outer_contact_depth": contact_depth, "max_grasp_error": max_error, "real_contacts": True,
               "seeds_on_real_surface": True, "center_of_mass": com, "wrist_facing_arm": facing is not None,
               "gated_walk": True, "approach_corridor": bool(env), "seed": seed, "min_kept": min_kept,
               **({"region": region} if region else {})}
    if (out / "summary.json").exists() and not force:
        summary = json.loads((out / "summary.json").read_text())
        if summary.get("r2s2r_options") == options:
            return tag, summary, True
    tabletop = {k: None for k in TABLETOP_KEYS}
    if facing is not None:
        tabletop.update(seed_facing_dir=[float(x) for x in facing], seed_min_facing_cos=float(min_facing_cos))
    if env:
        tabletop["scene_obstacles"] = env
    tabletop["seed_max_hull_depth"] = max_hull_depth
    if region:  # a handle's inside is the grasp: the hull-depth filter would drop it
        tabletop.update(seed_region_center=[float(v) for v in region["center"]], seed_region_radius=region["radius"],
                        seed_max_hull_depth=region.get("max_hull_depth"))
    sd = scene_dir(s, key, V, F, com, solid)
    tabletop["seed_real_mesh"] = str(sd / "real.obj")
    sup = f"room/{s.scene_graph['task']['support']}"  # the support's top is the planner's floor, not an obstacle
    gate = ContactGate(hand_asset, *(solid_mesh(V, F) if solid else (V, F)), contact_depth, real=(V, F),
                       env=[o for o in env or [] if o.get("name") != sup])
    n = seeds
    while True:
        for f in out.glob("grasp_*.json"):  # a rerun with more seeds replaces the records
            f.unlink()
        summary = solve_tabletop_grasps(
            scene_dir=sd, out_dir=out, asset_config=hand_asset, seeds=n, top_k=top_k, opt_iters=opt_iters, seed=seed,
            synthesis_overrides={"distance_threshold": None, "pregrasp_clearance_m": None},
            tabletop_overrides=tabletop, record_veto=gate)
        summary["quality_kept"] = quality_gate(out, gate, max_error)
        if not min_kept or summary["quality_kept"] >= min_kept or 2 * n > (max_seeds or n):
            break
        print(f"[synth] {tag}: {summary['quality_kept']} grasps pass the quality gate with {n} seeds, again with "
              f"{2 * n}", flush=True)
        n *= 2
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
    summary["seeds_used"] = n
    summary["r2s2r_options"] = options
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return tag, summary, False


def pool_synthesize(s: Session, hand_asset: Path, key: str, V, F, seeds, rounds, opt_iters, seed, com=None,
                    max_hull_depth=0.008, contact_depth=None, max_error=0.05, force=False, max_upward_cos=0.5):
    """The target mesh's grasp pool, shared by every scene and randomized configuration holding that mesh:
    `rounds` BODex solves of `seeds` seeds each (one GPU batch per solve, a new seed draw per round), seeds all around
    the target's sides (surface normals at most `max_upward_cos` up: over the whole surface, ~150 of ~190 converged
    seeds of 256 rested on a cup's filled opening), the support plane only, and every converged grasp that passes the contact checks and the quality gate
    kept (top-k = seeds). A configuration then takes the grasps that face its arm and clear its environment
    (select_from_pool) -- no synthesis per configuration. Writes grasps/<hand>/<key>__pool/round_NN/; a finished
    round with the same options is kept."""
    require_bodex()
    from .bodex.tabletop.synthesize import solve_tabletop_grasps

    out = grasp_dir(s, hand_asset.stem, f"{key}__pool")
    tabletop = {k: None for k in TABLETOP_KEYS}
    tabletop["seed_max_hull_depth"] = max_hull_depth
    tabletop["seed_max_upward_cos"] = max_upward_cos
    sd = scene_dir(s, key, V, F, com)
    tabletop["seed_real_mesh"] = str(sd / "real.obj")
    gate = ContactGate(hand_asset, *solid_mesh(V, F), contact_depth, real=(V, F))
    kept = 0
    for k in range(rounds):
        d = out / f"round_{k:02d}"
        options = {"pool": True, "seeds": seeds, "seed": seed + k, "opt_iters": opt_iters,
                   "seed_max_hull_depth": max_hull_depth, "outer_contact_depth": contact_depth,
                   "max_grasp_error": max_error, "center_of_mass": com, "seed_max_upward_cos": max_upward_cos}
        if (d / "summary.json").exists() and not force:
            summary = json.loads((d / "summary.json").read_text())
            if summary.get("r2s2r_options") == options:
                kept += summary.get("quality_kept", 0)
                continue
        for f in d.glob("grasp_*.json"):
            f.unlink()
        summary = solve_tabletop_grasps(
            scene_dir=sd, out_dir=d, asset_config=hand_asset, seeds=seeds, top_k=seeds, opt_iters=opt_iters,
            seed=seed + k, synthesis_overrides={"distance_threshold": None, "pregrasp_clearance_m": None},
            tabletop_overrides=tabletop, record_veto=gate)
        summary["quality_kept"] = quality_gate(d, gate, max_error)
        summary["r2s2r_options"] = options
        (d / "summary.json").write_text(json.dumps(summary, indent=1))
        kept += summary["quality_kept"]
        print(f"[pool] {key} round {k}: {summary['quality_kept']} grasps of {seeds} seeds", flush=True)
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
    return out, kept


def stage_action(r, stage):
    """A record's stage (pregrasp / grasp / squeeze) as a hand action: base position, wxyz, joints in record order."""
    st = r["stages"][stage]
    return [*st["position"], *st["orientation"], *(st["joints"][n] for n in r["joint_names"])]


# : A right hand grasps the way a right hand does: in the robot's base frame its palm does not face right (the hand
# : on the object's left, the forearm reaching across) and its thumb points neither down nor right (the wrist turned
# : over). Cosines; a left hand is mirrored.
NATURAL = {"palm_right_max": 0.3, "thumb_up_min": -0.3, "thumb_right_max": 0.5}


def hand_axes(hand_asset: Path):
    """(approach, thumb, fingers) unit axes of the hand's base link: the asset's approach and finger axes, the thumb
    their right-handed complement (fingers x approach; Sharpa: +x, +y, +z)."""
    import yaml

    rt = yaml.safe_load(hand_asset.read_text()).get("retarget") or {}
    a = np.asarray(rt.get("approach_axis", (1.0, 0.0, 0.0)), float)
    f = np.asarray(rt.get("finger_axis", (0.0, 0.0, 1.0)), float)
    return a, np.cross(f, a), f


def natural_grasp(R, axes, robot, side="right"):
    """(ok, {palm_right, thumb_up, thumb_right}) of a hand base-link orientation R (target frame) against the robot's
    (forward, left, up) in that frame (targets.robot_axes); NATURAL's limits."""
    a, t, _ = (R @ v for v in axes)
    _, left, up = robot
    right = -np.asarray(left) if side == "right" else np.asarray(left)
    m = {"palm_right": float(a @ right), "thumb_up": float(t @ up), "thumb_right": float(t @ right)}
    ok = (m["palm_right"] <= NATURAL["palm_right_max"] and m["thumb_up"] >= NATURAL["thumb_up_min"]
          and m["thumb_right"] <= NATURAL["thumb_right_max"])
    return ok, m


def select_from_pool(s: Session, hand_asset: Path, pool: Path, key: str, V, F, facing, env, top_k=32,
                     min_facing_cos=0.25, contact_depth=None, robot=None, side="right"):
    """One configuration's grasp set from the mesh's pool: the grasps whose wrist faces the arm (cosine between the
    hand's -finger axis and `facing`, both in the target frame, at least `min_facing_cos`: the synthesis' own
    final_wrist_facing_min_cos), whose pregrasp, grasp and squeeze hands miss every environment box and whose
    approach corridor is clear (ContactGate), and which a right hand takes the right-handed way (natural_grasp
    against `robot`, the base's axes in the target frame), ranked by grasp_error_max. Written where the per-configuration
    synthesis writes (grasps/<hand>/<grasp_key(key, facing, env)>), so apps/episode.py reads them unchanged."""
    import yaml
    from scipy.spatial.transform import Rotation

    asset = yaml.safe_load(hand_asset.read_text())
    finger = np.asarray((asset.get("retarget") or {}).get("finger_axis", (0.0, 0.0, 1.0)), float)
    sup = f"room/{s.scene_graph['task']['support']}"
    env_ = [o for o in env or [] if o.get("name") != sup]
    gate = ContactGate(hand_asset, *solid_mesh(V, F), contact_depth, real=(V, F), env=env_)
    recs = [json.loads(f.read_text()) | {"_pool_file": str(f.relative_to(s.dir))}
            for f in sorted(pool.glob("round_*/grasp_*.json"))]
    recs = [r for r in recs if r.get("success")]
    axes = hand_axes(hand_asset)
    why = {"pool": len(recs), "facing": 0, "unnatural": 0, "environment": 0, "corridor": 0, "unusable": 0}
    # pool grasps whose pregrasp hand reached into the target and neither opening nor backing off cleared it, in two
    # episodes or more (apps/episode.py lists them beside the pool): the grasp's own geometry, out wherever it is
    f = pool / "unusable.txt"
    seen = Counter(f.read_text().split()) if f.exists() else Counter()
    unusable = {str(Path(r["_pool_file"]).relative_to(pool.relative_to(s.dir))) for r in recs} & \
        {k for k, n in seen.items() if n >= 2}
    keep = []
    for r in recs:
        if str(Path(r["_pool_file"]).relative_to(pool.relative_to(s.dir))) in unusable:
            why["unusable"] += 1
            continue
        if facing is not None:
            R = Rotation.from_quat(np.asarray(r["action"][3:7], float), scalar_first=True).as_matrix()
            c = float(-(R @ finger) @ (np.asarray(facing, float) / np.linalg.norm(facing)))
            r["wrist_facing_arm_cos"] = c
            if c < min_facing_cos:
                why["facing"] += 1
                continue
        if robot is not None:
            ok_hand, r["hand_frame"] = natural_grasp(
                Rotation.from_quat(np.asarray(r["action"][3:7], float), scalar_first=True).as_matrix(), axes, robot,
                side)
            if not ok_hand:
                why["unnatural"] += 1
                continue
        hit = next((h for h in (gate.env_hit(stage_action(r, st)) for st in ("pregrasp", "grasp", "squeeze")) if h),
                   None) if env_ else None
        if hit:
            why["environment"] += 1
            continue
        ok, info = gate(r["action"])
        r.update(info)
        if not ok:
            why["corridor"] += 1
            continue
        keep.append(r)
    # ranked: a pregrasp reaching > 15 mm into the target last (the planner opens and backs off a shallower one; the
    # synthesis' pregrasps sit 6 mm in at the median), then by grasp error; and spread: a grasp within 1.5 cm and
    # 15 deg of one already kept waits until every distinct one is in (the error ranking put six near-copies first)
    keep.sort(key=lambda r: ((r.get("stage_report") or {}).get("pregrasp_clearance_m", 0.0) < -0.015,
                             float(r.get("grasp_error_max", 0.0))))
    picked, later = [], []
    for r in keep:
        a = np.asarray(r["action"][:7], float)
        Ra = Rotation.from_quat(a[3:7], scalar_first=True)
        near = any(np.linalg.norm(a[:3] - b[:3]) < 0.015 and
                   (Ra.inv() * Rotation.from_quat(b[3:7], scalar_first=True)).magnitude() < np.radians(15)
                   for b in (np.asarray(p_["action"][:7], float) for p_ in picked))
        (later if near else picked).append(r)
    keep = (picked + later)[:top_k]
    out = grasp_dir(s, hand_asset.stem, grasp_key(key, facing, env))
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("grasp_*.json"):
        f.unlink()
    for i, r in enumerate(keep):
        r["rank"] = i
        (out / f"grasp_{i:03d}.json").write_text(json.dumps(r, indent=1))
    summary = {"ok": bool(keep), "hand": hand_asset.stem, "top_k": len(keep), "quality_kept": len(keep),
               "pipeline": "pool_selection", "pool": str(pool.relative_to(s.dir)), "rejected": why,
               "r2s2r_options": {"pool": str(pool.relative_to(s.dir)), "top_k": top_k, "natural": NATURAL if robot is not None else None,
                                 "min_facing_cos": min_facing_cos, "pool_records": len(recs)}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return out.name, summary


def synth_demo(s: Session, hand_asset: Path, args, *, seated_mesh, seed_region):
    """Grasps for a demo's holds (demo.json of calibrated_scene_reconstruction.demo.generate): per held twin object, on its seated mesh; a
    functional hold restricts the seeds to its region, and one straddling a thin wall (a mug's rim) keeps the mesh's
    opening (solid=False: the filled solid has no wall to straddle). Table plane only, no arm-facing filter (the
    region already says where the hand goes)."""
    d = json.loads((Path(args.demo) / "demo.json").read_text())
    for k, h in d.get("holds", {}).items():
        V, F = seated_mesh(s, h["object"])
        region = seed_region(h["region"]) if h["functional"] else None
        solid = "thin" not in h["measure"]["why"]
        tag, summary, cached = synthesize(
            s, hand_asset, mesh_key({"name": h["object"]}, V), V, F, args.seeds, args.top_k, args.opt_iters, args.seed,
            None, args.min_facing_cos, None, args.force, None if args.max_hull_depth < 0 else args.max_hull_depth,
            None if args.contact_depth < 0 else args.contact_depth,
            None if args.max_grasp_error < 0 else args.max_grasp_error, region=region, solid=solid)
        print(f"[hold {k}] {h['object']} ({'functional' if h['functional'] else 'generic'}, "
              f"{'solid' if solid else 'as reconstructed'}) -> {tag}: {summary.get('top_k', 0)} grasps of "
              f"{summary.get('numeric_success_seed_count', 0)} converged seeds, {summary.get('quality_kept')} pass the "
              f"quality gate" + (" (cached)" if cached else ""), flush=True)
    return 0


def main(argv=None):
    """Compatibility CLI; cross-domain composition lives in tools/grasping."""
    import runpy
    from calibrated_scene_reconstruction.core.paths import ROOT as project_root

    entry = runpy.run_path(str(project_root / "tools/grasping/synthesize.py"))
    return entry["main"](argv)


if __name__ == "__main__":
    sys.exit(main())
