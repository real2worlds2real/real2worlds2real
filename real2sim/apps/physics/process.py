"""Internal native worker for calibrated_scene_reconstruction.physics.process; use the agent process tools."""
import argparse
from importlib.metadata import version
from pathlib import Path

ap = argparse.ArgumentParser(__doc__)
ap.add_argument('round_directory')
args = ap.parse_args()

from calibrated_scene_reconstruction.core.paths import ROOT  # noqa: E402
from calibrated_scene_reconstruction.physics.process.contract import read, require, validate  # noqa: E402
from calibrated_scene_reconstruction.physics.process.workflow import current, intact  # noqa: E402

directory = Path(args.round_directory).resolve()
require(directory == current(directory.parent.parent).resolve() and intact(directory),
        'Native execution requires the current, intact process round')
plan=read(directory / 'plan.json')
validate(plan, ROOT)

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402
from calibrated_scene_reconstruction.physics.process.resources import watch  # noqa: E402

monitor = watch(directory)
experience=str(ROOT/'apps/physics/physics_only.kit') if plan['simulation'].get('runtime')=='physics_only' else ''
app = SimulationApp({'headless': True, 'width':64, 'height':64, 'window_width':64, 'window_height':64,
                     'disable_viewport_updates':True,'multi_gpu':False,'limit_cpu_threads':8},experience=experience)
try:
    from calibrated_scene_reconstruction.physics.process.native import execute
    execute(args.round_directory, version('isaacsim'))
finally:
    monitor.set()
    app.close()
