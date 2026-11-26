"""Retain an actual Newton hybrid frame as a renderable USD snapshot."""

import argparse
import json
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom

from real2sim_harness.hybrid_visuals import HybridVisuals

parser = argparse.ArgumentParser()
parser.add_argument("folder", type=Path)
parser.add_argument("--frame", type=int, default=-1)
args = parser.parse_args()
stage = Usd.Stage.CreateNew(str(args.folder / "simulation_final.usda"))
root = UsdGeom.Xform.Define(stage, "/Asset").GetPrim()
root.GetReferences().AddReference("./visual.usdc")
stage.SetDefaultPrim(root)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
report = json.loads((args.folder / "hybrid_validation.json").read_text())
with np.load(args.folder / "hybrid_motion.npz", allow_pickle=False) as motion:
    HybridVisuals(args.folder, stage, report).update(motion["bodies"][args.frame], motion["particles"][args.frame])
stage.GetRootLayer().Save()
print("HYBRID_SNAPSHOT", args.frame, args.folder, flush=True)
