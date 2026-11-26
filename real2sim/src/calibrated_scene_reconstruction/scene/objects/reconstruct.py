"""S2 object reconstruction with SAM 3D Objects, conditioned on the MoGe point map.

    python -m calibrated_scene_reconstruction.recon.objects sessions/<name> [--only cup bowl]

Reconstructs every non-template static/dynamic object of the scene graph. Each object runs in its own
subprocess: SAM 3D peaks at ~12.5 GB host RAM and does not release it between objects.
Writes objects/<obj>/{raw.glb, splat.ply, pose.json}; alignment to the photo happens in recon.align.
Other stages may pass their own inputs (`inputs`: npz with rgb, mask, points in that image's camera, optionally
out_dir relative to the session), e.g. a photo extended below its lower edge for an amodally completed piece of
furniture (recon.furniture) or a completed view of the room (recon.room_objects).
"""
from calibrated_scene_reconstruction.core.perception.sam3d import gpu_only_during as gpu_only_during, lean_decode as lean_decode, cap_sparse_structure as cap_sparse_structure, reconstruct as reconstruct, run_worker as run_worker  # noqa: F401
import argparse
import json
import os
import subprocess
import sys

import numpy as np

from calibrated_scene_reconstruction.core.paths import ROOT, SAM3D_DIR
from calibrated_scene_reconstruction.scene.session import Session


def object_specs(sg):
    specs = {}
    for group in ("dynamic_objects", "static_objects"):
        for name, spec in sg.get(group, {}).items():
            if not spec.get("template"):
                specs[name] = spec["ids"]
    return specs










def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--worker", nargs=2, metavar=("NAME", "IDS"), help=argparse.SUPPRESS)
    ap.add_argument("--inputs", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        reconstruct(args.session, args.worker[0], [int(i) for i in args.worker[1].split("+")], args.inputs)
        return
    specs = object_specs(Session(args.session).scene_graph)
    for name, ids in specs.items():
        if args.only and name not in args.only:
            continue
        if run_worker(args.session, name, ids) != 0:
            sys.exit(f"[{name}] reconstruction failed")




if __name__ == "__main__":
    main()
