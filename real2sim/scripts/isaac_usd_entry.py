"""Run USD authoring/audits in Sim's bundled USD rather than mixing USD wheels."""
import argparse
import os
import runpy
import sys

from isaac_physx_compat import runtime

if len(sys.argv) < 2:
    raise SystemExit("usage: isaac_usd_entry.py SCRIPT [ARGS ...]")
target = sys.argv[1]
sys.argv = sys.argv[1:]
device = os.environ.get("REAL2SIM_RENDER_DEVICE", os.environ.get("REAL2SIM_PHYSICS_DEVICE", "cpu"))
args = argparse.Namespace(viz="none", headless=True, device=device, enable_cameras=False)
runtime.prepare(args)
with runtime.launch(args):
    runpy.run_path(target, run_name="__main__")
