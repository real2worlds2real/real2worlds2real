"""Compose scene and demonstration inputs for BODex grasp synthesis."""
import argparse
import sys

from calibrated_scene_reconstruction.core.paths import robot_asset_path
from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.assets.physprops import mass_props
from calibrated_scene_reconstruction.robots.spec import load_spec
from calibrated_scene_reconstruction.datagen.demo.generate import seated_mesh
from calibrated_scene_reconstruction.datagen.demo.functional import seed_region
from calibrated_scene_reconstruction.grasping import require_bodex
from calibrated_scene_reconstruction.grasping.synth import (
    synth_demo, pool_synthesize, select_from_pool, synthesize, variants,
)
from calibrated_scene_reconstruction.grasping.targets import (
    hand_name, target_spec, local_mesh, arm_direction, scene_env, grasp_key,
    mesh_key, object_label, robot_axes,
)




def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--robot", required=True)
    ap.add_argument("--variants", nargs="*")
    ap.add_argument("--seeds", type=int, default=96, help="a crowded scene leaves few seeds a good optimum")
    ap.add_argument("--min-kept", type=int, default=3,
                    help="fewer grasps than this pass the quality gate: synthesize again with twice the seeds, up to "
                         "--max-seeds (one or two grasps leave an episode no choice when their plans fail)")
    ap.add_argument("--max-seeds", type=int, default=384)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--opt-iters", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--photo-base", action="store_true",
                    help="grasps facing the arm from where the photo shows the real robot (apps/episode.py --photo-base)")
    ap.add_argument("--min-facing-cos", type=float, default=0.2,
                    help="seed sites must face the arm's shoulder by this cosine (robot spec arms.<side>.shoulder)")
    ap.add_argument("--no-env", action="store_true", help="ignore grasps/_env (table plane only)")
    ap.add_argument("--max-hull-depth", type=float, default=0.008,
                    help="seed sites within this depth of the target's convex hull (its outside), -1: no filter")
    ap.add_argument("--contact-depth", type=float, default=-1,
                    help="drop grasps whose contact spheres sit deeper inside the target's hull (around a handle); "
                         "-1 (default): keep them, the depth is recorded")
    ap.add_argument("--max-grasp-error", type=float, default=0.05,
                    help="drop grasps whose grasp_error_max exceeds this (clean scenes reach < 0.015), -1: keep all")
    ap.add_argument("--pool-rounds", type=int, default=0,
                    help="> 0: one grasp pool per target mesh (pool_synthesize: this many solves of --seeds seeds, "
                         "all around the target) and each scene's grasps selected from it (select_from_pool); 0: a "
                         "synthesis per scene against its environment")
    ap.add_argument("--wrist-min-cos", type=float, default=0.25,
                    help="--pool-rounds: a selected grasp's wrist faces the arm by this cosine (the synthesis' own "
                         "final_wrist_facing_min_cos)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--geometric-com", action="store_true",
                    help="wrench space about the surface centroid, not the semantic centre of mass (A/B comparisons)")
    ap.add_argument("--demo", default=None,
                    help="sessions/<name>/demo/<demo> (calibrated_scene_reconstruction.demo.generate): synthesize for each of the demo's holds "
                         "on the twin object, seeds in its region when the hold is functional (table plane only)")
    args = ap.parse_args(argv)
    s = Session(args.session)
    spec = load_spec(args.robot)
    side = next(iter(spec["arms"]))
    if not hand_name(spec, side):
        print(f"{args.robot}: no BODex hand (ee.bodex) -- its grasps come from robots.grasps")
        return 0
    try:
        require_bodex()
    except ModuleNotFoundError as exc:
        ap.error(str(exc))
    hand_asset = robot_asset_path(spec["arms"][side]["ee"]["bodex"]["asset"], required=True)
    if args.demo:
        return synth_demo(s, hand_asset, args, seated_mesh=seated_mesh, seed_region=seed_region)
    done = {}
    for tag, v in variants(s, args.variants):
        tspec = target_spec(s, v)
        mesh = local_mesh(s, tspec)
        if mesh is None:
            print(f"[{tag}] target is a library model ({tspec.get('ph_model')}): no synthesis")
            continue
        V, F = mesh
        facing = arm_direction(s, spec, side, tspec, tag, photo_base=args.photo_base)
        _, env = scene_env(s, tag) if not args.no_env else (None, None)
        if env is None and not args.no_env:
            print(f"[{tag}] no grasps/_env/{tag}.json (apps/scene_obstacles.py): synthesizing against the table only")
        key = grasp_key(mesh_key(tspec, V), facing, env)
        com = None if args.geometric_com else \
            [round(c, 4) for c in mass_props(object_label(s, tspec), V)["com"]]  # semantic (assets.physprops)
        if args.pool_rounds > 0:
            mk = mesh_key(tspec, V)
            if mk not in done:
                done[mk] = pool_synthesize(s, hand_asset, mk, V, F, args.seeds, args.pool_rounds, args.opt_iters,
                                           args.seed, com, None if args.max_hull_depth < 0 else args.max_hull_depth,
                                           None if args.contact_depth < 0 else args.contact_depth,
                                           None if args.max_grasp_error < 0 else args.max_grasp_error, args.force)
            pool, n_pool = done[mk]
            key, summary = select_from_pool(s, hand_asset, pool, mk, V, F, facing, env, args.top_k,
                                            args.wrist_min_cos,
                                            None if args.contact_depth < 0 else args.contact_depth,
                                            robot=robot_axes(s, spec, tag, photo_base=args.photo_base), side=side)
            print(f"[{tag}] {hand_asset.stem} on {key}: {summary['top_k']} grasps selected from a pool of {n_pool} "
                  f"(rejected {summary['rejected']})", flush=True)
            continue
        if key not in done:
            key, summary, cached = synthesize(s, hand_asset, mesh_key(tspec, V), V, F, args.seeds, args.top_k,
                                              args.opt_iters, args.seed, facing, args.min_facing_cos, env, args.force,
                                              None if args.max_hull_depth < 0 else args.max_hull_depth,
                                              None if args.contact_depth < 0 else args.contact_depth,
                                              None if args.max_grasp_error < 0 else args.max_grasp_error, com=com,
                                              min_kept=args.min_kept, max_seeds=args.max_seeds)
            done[key] = summary
            import gc

            import torch

            gc.collect()  # one target's solver state is not needed for the next (a session has a few)
            torch.cuda.empty_cache()
            outer = summary.get("quality_kept")
            print(f"[{tag}] {hand_asset.stem} on {key}: {summary.get('top_k', 0)} grasps kept of "
                  f"{summary.get('numeric_success_seed_count', 0)} converged seeds"
                  + ("" if outer is None else f", {outer} pass the quality gate") + (" (cached)" if cached else ""),
                  flush=True)
        else:
            print(f"[{tag}] {key}: shared with an earlier scene")
    return 0


if __name__ == "__main__":
    sys.exit(main())
