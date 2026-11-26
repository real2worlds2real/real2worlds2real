"""Commanded joints are uncertain pose priors, never exact metric observations.

Default angular uncertainty is an explicitly unmeasured modelling assumption.
Camera evidence, not task CAD or task dimensions, must support the resulting
articulation and scale. Fingers remain excluded from calibration by default.
"""
import math
import re

import numpy as np

from .visual_robot_fit import Settings


SEEDS = ('joint_nominal', 'joint_plus_sigma', 'joint_minus_sigma', 'joint_fixed')


def presentation_state(model, readout, estimated):
    """Keep observed non-calibration joints in the delivered robot pose.

    Fingers excluded from metric fitting must not silently revert to CAD home.
    Missing values remain explicitly unobserved. Hardware readout noise beyond
    CAD limits is projected for presentation and reported, never used to change
    the fitted body or treated as a new calibration observation.
    """
    if 'joints' in readout:
        rows = readout['joints']
        if len({row['joint'] for row in rows}) != len(rows):
            raise ValueError('Duplicate named joint values')
        readout = {row['joint']: row['position_rad'] for row in rows}
    names = {joint.name for joint in model.movable}
    if set(estimated) - names:
        raise ValueError('Estimated presentation joints must belong to this CAD')
    joints = {name: float(value) for name, value in estimated.items()}
    if not all(math.isfinite(value) for value in joints.values()):
        raise ValueError('Estimated presentation joints must be finite')
    measured, missing = {}, []
    for joint in model.movable:
        if joint.name in estimated:
            continue
        side = ('right' if joint.name.lower().startswith(('r_', 'right')) else
                'left' if joint.name.lower().startswith(('l_', 'left')) else '')
        aliases = [key for key in readout if side and key.lower().startswith(side) and
                   key.split('_', 1)[-1] == joint.name.split('_', 1)[-1]]
        if joint.name not in readout and len(aliases) > 1:
            raise ValueError('Ambiguous presentation readout: '+joint.name)
        key = joint.name if joint.name in readout else aliases[0] if aliases else None
        if key is None:
            missing.append(joint.name)
            continue
        value = float(readout[key])
        if not math.isfinite(value):
            raise ValueError('Presentation readout must be finite: '+joint.name)
        shown = float(np.clip(value, joint.lower, joint.upper))
        joints[joint.name] = shown
        measured[joint.name] = {'input_name': key, 'measured_value': value,
                               'presentation_value': shown, 'limit_projection': shown-value,
                               'unit': 'rad' if joint.kind == 'revolute' else 'm'}
    return {'joints_rad': joints, 'measured_noncalibration_joints': measured,
            'unobserved_joints': missing,
            'limitation': 'Readout used for presentation only; finite CAD-limit projections remain explicit.'}


def build(model, state, options=None):
    options = options or {}
    allowed = {'sigma_deg', 'max_delta_deg', 'weight', 'per_joint', 'floor_sigma_m', 'max_floor_offset_m'}
    if set(options) - allowed:
        raise ValueError('Unknown joint uncertainty settings: '+str(sorted(set(options)-allowed)))
    sigma = float(options.get('sigma_deg', 3.))
    delta = float(options.get('max_delta_deg', 12.))
    weight = float(options.get('weight', .002))
    floor_sigma=float(options.get('floor_sigma_m',.02))
    floor_limit=float(options.get('max_floor_offset_m',.06))
    if not (0 < sigma <= delta <= 45 and 0 < weight <= 1):
        raise ValueError('Require 0 < sigma_deg <= max_delta_deg <= 45 and positive prior weight <= 1')
    if not 0 < floor_sigma <= floor_limit <= .10:
        raise ValueError('Require 0 < floor_sigma_m <= max_floor_offset_m <= 0.10')
    if 'joints' in state:
        rows = state['joints']
        if len({row['joint'] for row in rows}) != len(rows):
            raise ValueError('Duplicate named joint values')
        state = {row['joint']: row['position_rad'] for row in rows}
    nominal, lower, upper, sigmas, matched = {}, {}, {}, {}, {}
    execution_lower,execution_upper={},{}
    active = [joint for joint in model.movable if joint.upper > joint.lower and not any(
        re.search(pattern, joint.name, re.I) for pattern in Settings().exclude_joints)]
    overrides = options.get('per_joint', {})
    if set(overrides)-{j.name for j in active}:
        raise ValueError('Uncertainty overrides must name estimated CAD joints')
    for joint in active:
        if joint.kind != 'revolute':
            raise ValueError('Angular joint prior cannot silently apply degree units to a prismatic joint')
        side = ('right' if joint.name.lower().startswith(('r_', 'right')) else
                'left' if joint.name.lower().startswith(('l_', 'left')) else '')
        aliases = [key for key in state if side and key.lower().startswith(side) and
                   key.split('_', 1)[-1] == joint.name.split('_', 1)[-1]]
        key = joint.name if joint.name in state else aliases[0] if len(aliases) == 1 else None
        if key is None:
            raise ValueError('Missing or ambiguous named command prior: '+joint.name)
        value = float(state[key])
        if not math.isfinite(value) or not joint.lower <= value <= joint.upper:
            raise ValueError('Joint command must be finite and inside CAD limits: '+joint.name)
        override = overrides.get(joint.name, {})
        if set(override)-{'sigma_deg', 'max_delta_deg'}:
            raise ValueError('Unknown per-joint uncertainty field')
        sd, md = float(override.get('sigma_deg', sigma)), float(override.get('max_delta_deg', delta))
        if not (0 < sd <= md <= 45):
            raise ValueError('Invalid per-joint angular uncertainty')
        nominal[joint.name] = value
        lower[joint.name] = max(joint.lower, value-np.radians(md))
        upper[joint.name] = min(joint.upper, value+np.radians(md))
        execution_lower[joint.name]=float(value-np.radians(md))
        execution_upper[joint.name]=float(value+np.radians(md))
        sigmas[joint.name] = float(np.radians(sd))
        matched[joint.name] = key
    if len(active) < 2:
        raise ValueError('At least two estimated articulated joints required')
    return {'nominal_rad': nominal, 'lower_rad': lower, 'upper_rad': upper,
            'sigma_rad': sigmas, 'weight': weight, 'matched_names': matched,
            'execution_lower_rad':execution_lower,'execution_upper_rad':execution_upper,
            'floor_sigma_m':floor_sigma,'max_floor_offset_m':floor_limit,
            'uncertainty_source': 'Configured modelling assumption; not measured servo accuracy',
            'unestimated_joints': [j.name for j in model.movable if j.name not in nominal]}


def seed(prior, name):
    if name not in SEEDS:
        raise ValueError('Unknown joint-prior hypothesis')
    sign = 1 if name == 'joint_plus_sigma' else -1 if name == 'joint_minus_sigma' else 0
    return {key: float(np.clip(value+sign*prior['sigma_rad'][key],
                               prior['lower_rad'][key], prior['upper_rad'][key]))
            for key, value in prior['nominal_rad'].items()}


def audit(prior, state, floor_offset_m=0.):
    if set(state) != set(prior['nominal_rad']):
        raise ValueError('Refined joint set differs from the prepared command prior')
    reasons, changes = [], {}
    if not math.isfinite(floor_offset_m) or abs(floor_offset_m) >= .99*prior.get('max_floor_offset_m',.06):
        reasons.append('Robot floor/root-height offset reaches or exceeds its uncertainty bound')
    for name, nominal in prior['nominal_rad'].items():
        value = float(state[name]); lo, hi = prior['lower_rad'][name], prior['upper_rad'][name]
        if not math.isfinite(value) or value < lo-1e-6 or value > hi+1e-6:
            reasons.append('Refined joint exceeds its configured execution uncertainty: '+name)
        # Contact with an artificial execution-error bound signals a failed
        # assumption. A legitimate hardware stop is not itself a failure.
        elo=prior.get('execution_lower_rad',prior['lower_rad'])[name]
        ehi=prior.get('execution_upper_rad',prior['upper_rad'])[name]
        if min(value-elo, ehi-value) < .01*(ehi-elo):
            reasons.append('Refined joint reaches a pose uncertainty bound: '+name)
        changes[name] = {'nominal_rad': nominal, 'refined_rad': value,
                         'delta_deg': float(np.degrees(value-nominal)),
                         'sigma_units': (value-nominal)/prior['sigma_rad'][name]}
    return {'ok': not reasons, 'reasons': reasons, 'joint_changes': changes,
            'floor_offset_m':floor_offset_m}
