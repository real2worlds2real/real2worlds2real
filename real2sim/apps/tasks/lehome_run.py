"""A LeHome Challenge module (lehome-official/lehome-challenge: Isaac Lab garment folding with two SO-101 arms) run
by the Isaac Sim 5.1 python, the way tools/run_isaac.sh runs ours: the challenge's checkout, its Isaac Lab fork and the
packages the image lacks live under envs/lehome (a venv made with --system-site-packages from the image's python, the
image's own packages pinned), added here after the image's site-packages so nothing of the image is shadowed.

    cd envs/lehome/lehome-challenge
    CUDA_VISIBLE_DEVICES=<index> ../../../tools/run_isaac.sh ../../../apps/tasks/lehome_run.py -m scripts.dataset_sim replay \
        --dataset_root Datasets/example/top_long_merged --num_replays 1 --disable_depth --device cpu --enable_cameras

The working directory is the challenge's checkout (its scripts read Assets/ and Datasets/ relative to it).
--merged-garments (before -m) fixes the challenge's replay of a merged dataset (merged_replay_fix): a run then covers
one garment's episodes.
"""
from __future__ import annotations

import os
import runpy
import site
import sys
from pathlib import Path

if __name__ == "__main__" and sys.argv[1:] == ["--help"]:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-garments", action="store_true")
    parser.add_argument("-m", metavar="MODULE", help="challenge module; subsequent arguments are forwarded")
    parser.parse_args()

from calibrated_scene_reconstruction.core.paths import ROOT
LEHOME = ROOT / "envs" / "lehome"
SITE = LEHOME / "venv" / "lib" / "python3.11" / "site-packages"
site.addsitedir(str(SITE))
sys.path.insert(0, os.getcwd())  # the challenge's `scripts` package
# worker processes (LeRobot's video encoding pool, spawned) get sys.path but not the editable installs' import hooks:
# the packages' own directories on sys.path as well (the app launch rewrites PYTHONPATH, so not through that)
SRC = LEHOME / "lehome-challenge"
sys.path[1:1] = [*(str(SRC / "third_party" / "IsaacLab" / "source" / p) for p in ("isaaclab", "isaaclab_assets", "isaaclab_tasks")),
                 str(SRC / "source" / "lehome")]
os.environ.setdefault("PYNPUT_BACKEND", "dummy")  # its teleop devices import pynput, which wants an X display
try:  # open3d (the garment's point cloud) links libusb, which the image lacks: the libusb-package wheel's, loaded first
    import ctypes

    import libusb_package

    ctypes.CDLL(str(libusb_package.get_library_path()), mode=ctypes.RTLD_GLOBAL)
except (ImportError, OSError):
    pass

# our variants of the challenge's env, registered by name (gym imports them only on make, once the app runs)
import gymnasium as gym  # noqa: E402

gym.register(id="R2S2R-LeHome-Garment-Look-v0", entry_point="calibrated_scene_reconstruction.tasks.lehome.appearance:LookGarmentEnv",
             disable_env_checker=True,
             kwargs={"env_cfg_entry_point": "lehome.tasks.bedroom.garment_bi_cfg_v2:GarmentEnvCfg"})



def merged_replay_fix(argv):
    """The challenge's replay (scripts.utils.dataset_replay) on a merged dataset (Datasets/example/<category>_merged:
    ten garments of 25 episodes each, in meta/garment_info.json's order, each garment's episodes numbered from 0)
    loads the first garment for every episode and that garment's initial poses by the episode's global index. Patched
    as the module loads: the garment is the one --start_episode falls in (a run covers one garment's episodes) and an
    episode's pose that garment's entry for its own number."""
    import importlib.abc
    import importlib.util
    import json

    arg = lambda k, d=None: argv[argv.index(k) + 1] if k in argv else d  # noqa: E731
    info = json.loads((Path(arg("--dataset_root")) / "meta" / "garment_info.json").read_text())
    names = list(info)
    n = len(info[names[0]])
    start = int(arg("--start_episode", 0))
    end = int(arg("--end_episode", start + 1))
    if start // n != (end - 1) // n:
        sys.exit(f"--start_episode / --end_episode: one garment's episodes ({n} each) per run")
    garment = names[start // n]

    def patch(m):
        m.get_garment_name_from_json = lambda root: garment
        m.load_initial_pose = lambda root, ep: {"Garment": info[names[ep // n]][str(ep % n)]["object_initial_pose"]}

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != "scripts.utils.dataset_replay":
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            run = spec.loader.exec_module

            def exec_module(m):
                run(m)
                patch(m)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, Finder())


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--merged-garments":
        del sys.argv[1]
        merged_replay_fix(sys.argv)
    if len(sys.argv) < 3 or sys.argv[1] != "-m":
        sys.exit("usage: lehome_run.py [--merged-garments] -m <module> [args...]")
    if "dp_official" in sys.argv:  # our policies in the challenge's registry (scripts.eval --policy_type dp_official)
        from calibrated_scene_reconstruction.tasks.lehome.policy import register_policy
        from calibrated_scene_reconstruction.policy.dp_official_client import DPOfficialClient
        register_policy(DPOfficialClient)
    mod, sys.argv = sys.argv[2], [sys.argv[2], *sys.argv[3:]]
    runpy.run_module(mod, run_name="__main__", alter_sys=True)
