"""Attached-lid preparation and opening evidence for hand interaction."""
import numpy as np


def preparation(protocol):
    return ['opening', *(['lid_hold'] if protocol.get('opening_hold_seconds', .5) > 0 else [])]


def initial(s, obj, pa, start, check):
    from calibrated_scene_reconstruction.assets.joint_state import select, measure as initial_joint_state
    hinge = select(s.config('agent/mechanisms.json', {}), obj+'/'+pa['body_link'], obj+'/'+pa['lid_link'])
    state = initial_joint_state(hinge, start['lid_angle_rad'], default=(-np.radians(5), np.radians(5)))
    check['pass'] &= state['pass']
    check.update(initial_lid_angle_rad=start['lid_angle_rad'], initial_joint_state=state)


def opened(a, start, state):
    selection = a.get('postopening_selection')
    return {'lid_open': {'pass': min(row['lid_angle_rad'] for row in a['rows'][start:]) >= 1.5,
                         'angle_at_opening_rad': state['lid_angle_rad']},
            'rim_material_after_opening': {'pass': bool(selection and selection.get('on_actual_material'))}}
