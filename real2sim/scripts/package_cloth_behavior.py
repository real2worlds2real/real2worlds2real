"""Normalize an actual Newton cloth trajectory for the common Blender renderer."""

import argparse
import json
from pathlib import Path

import numpy as np

from real2sim_harness.artifacts import atomic_json, sha256_file

parser = argparse.ArgumentParser()
parser.add_argument("asset", type=Path)
parser.add_argument("trajectory", type=Path)
parser.add_argument("validation", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
manifest = json.loads((args.asset / "manifest.json").read_text())
validation = json.loads(args.validation.read_text())
if not validation["ok"]:
    raise ValueError("The Newton motion must pass before presentation")
data = np.load(args.trajectory, allow_pickle=False)
times = data["times"]
if len(times) < 2 or not np.allclose(np.diff(times), np.diff(times)[0], atol=1e-6):
    raise ValueError("Use uniformly sampled physical motion")
fps = round(1/float(times[1]-times[0]))
poses = np.zeros((len(times),1,7),dtype=np.float32)
poses[:,:,6] = 1
args.output.mkdir(parents=True,exist_ok=True)
np.savez_compressed(args.output / "behavior_motion.npz", bodies=poses, particles=data["positions"], times=times)
mesh = json.loads((args.asset / "sim_mesh.json").read_text())
atomic_json(args.output / "behavior_report.json", {
    "ok": True, "backend": "newton_vbd", "asset_kind": "deformable", "fps": fps, "frames":len(times),
    "duration_seconds":len(times)/fps, "body_ids":{"__base__":0}, "rest_matrices":{"__base__":np.eye(4).tolist()},
    "control_object":mesh["object_name"], "visual_followers":manifest["deformable_visual_objects"],
    "source_blend":manifest["source_blend"], "source_scene_sha256":manifest["source_blend_sha256"],
    "source_trajectory_sha256":sha256_file(args.trajectory), "validation":validation,
    "real_physics_calibrated":False, "robot_task_implemented":False,
})
print("CLOTH_BEHAVIOR_PACKAGED", len(times), fps, flush=True)
