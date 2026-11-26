"""Bind a hand's particle-friction intervention to current native controls.

This capability proof grants neither a grasp nor asset acceptance. Runtime
separation from every other rigid collider is independently required.
"""
import json
from pathlib import Path

from .particle_friction_ablation import verify
from calibrated_scene_reconstruction.assets.particle_binding import compare_profile
from .contact_response import sha


def bind(path, runtime, *, physics_hz, solver):
    proof = verify(path)
    checks = {}
    dependencies = {str(Path(path).resolve()): sha(path), **proof['producer_sha256']}
    for name, file in proof['trials'].items():
        file = Path(file)
        trial = json.loads(file.read_text())
        dependencies.update(trial['input_sha256'])
        for item in [file, file.with_name('states.npz'), file.with_name('source.py')]:
            dependencies[str(item.resolve())] = sha(item)
        if trial['physics_hz'] != physics_hz or trial['backend']['solver'] != solver:
            raise ValueError('Particle friction controls must match the running clock and solver')
        if name in {'rigid_nominal', 'self_nominal'}:
            checks[name] = compare_profile(trial, runtime)
    return {'pass': True, 'qualification': str(Path(path).resolve()),
            'input_sha256': dependencies, 'profile_checks': checks,
            'intervention_phase': 'hold_after_lift', 'task_approved': False,
            'scope': 'Particle-wide rigid-friction intervention; whole-cloth isolation '
                     'from all non-hand colliders is required throughout the causal interval.'}
