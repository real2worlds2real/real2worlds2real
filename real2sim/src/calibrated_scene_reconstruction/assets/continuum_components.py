"""Explicit non-mesh task components, kept separate from rigid/cloth LINKS.

A liquid is neither a rigid puddle nor a cloth link. Its declaration participates
in the same complete task coverage and acceptance checks as mesh mechanisms.
"""
import re


FLUID_CHECKS = {'source_initial_state', 'native_dynamics', 'contact', 'mass_balance', 'passive_stability'}


def validate(contract):
    components = contract.get('continuum_components', {})
    if not isinstance(components, dict):
        raise ValueError('Continuum components must be a mapping')
    objects = contract['objects']
    refs = {f'{obj}/{link}' for obj, spec in objects.items() for link in spec['links']}
    task_owners = set()
    for name, spec in components.items():
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', name):
            raise ValueError('Continuum component requires a stable identifier')
        if name in {m['id'] for m in contract['mechanisms']}:
            raise ValueError('Continuum and mesh mechanism result identifiers must be distinct')
        if not isinstance(spec, dict) or spec.get('kind') != 'particle_fluid' or spec.get('backend') != 'physx_pbd_fluid':
            raise ValueError('Continuum component needs an explicit supported native fluid backend')
        owner = spec.get('task_object')
        container = spec.get('container_link')
        if bool(owner) == bool(container):
            raise ValueError('Fluid is either a separate task object or the contents of one actual link')
        if owner:
            if not isinstance(owner, str) or owner in objects or owner in task_owners:
                raise ValueError('A fluid task object must have exactly one non-mesh owner')
            task_owners.add(owner)
        elif container not in refs:
            raise ValueError('Fluid container must refer to an actual physical link')
        if not isinstance(spec.get('source'), str) or not spec['source'].strip():
            raise ValueError('Fluid geometry/material hypothesis requires source evidence')
        checks = spec.get('checks')
        if (not isinstance(checks, list) or any(not isinstance(c, str) or not c for c in checks)
                or len(set(checks)) != len(checks) or not FLUID_CHECKS <= set(checks)):
            raise ValueError('Fluid requires source state, native dynamics, contact, mass balance and stability checks')
        absorption = spec.get('absorption')
        if absorption is not None:
            if not isinstance(absorption, dict) or absorption.get('absorber_link') not in refs:
                raise ValueError('Fluid absorption must name an actual physical sheet or porous body')
            if not {'absorption', 'noncontact_control', 'capacity_limit'} <= set(checks):
                raise ValueError('Absorbable fluid requires contact, capacity and negative-control assays')
    return components


def task_objects(contract):
    return set(contract['objects']) | {s['task_object'] for s in validate(contract).values() if s.get('task_object')}


def result_failures(contract, results):
    failures = []
    for name, spec in validate(contract).items():
        result = results.get(name, {})
        checks = result.get('checks', {})
        if (result.get('pass') is not True or result.get('backend') != spec['backend']
                or result.get('native_states_read') is not True
                or result.get('solver_motion_observed') is not True
                or result.get('kinematic_playback') is not False
                or any(checks.get(c, {}).get('pass') is not True for c in spec['checks'])):
            failures.append(name + ': native fluid evidence incomplete or failed')
    return failures
