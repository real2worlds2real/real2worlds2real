"""Existing native liquid candidate composition."""
from pathlib import Path
from calibrated_scene_reconstruction.core.paths import resolve


def compose(q, stage, bind):
    if q.get('fluid_candidate'):
        fluid = bind(resolve(q['fluid_candidate']))
        from calibrated_scene_reconstruction.sim.fluid_reference import compose
        fluid_record = compose(stage, '/World/FluidCandidate', fluid)
        bind(Path(__file__).resolve().parents[2] / 'sim/fluid_reference.py')
    else:
        fluid_record = None
    return {'fluids': fluid_record}
