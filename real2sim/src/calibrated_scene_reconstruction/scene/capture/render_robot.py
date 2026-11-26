"""Use a verified capture joint readout for scene presentation when requested."""
import copy
import math


def capture_spec(session, spec):
    """Keep generic robot defaults unless the graph requests the captured state.

    Calibration is not run here. Only its existing matched-name audit and fitted
    root height are consumed. Partial readouts must not look like full captures.
    """
    photo = session.scene_graph.get('photo_robot', {})
    if photo.get('render_visual_state', False):
        state = session.read_json('parsing/robot_visual_state.json')
        if state.get('robot') != spec['name'] or state.get('calibration_ok') is not True:
            raise ValueError('Visual robot presentation requires a passing fit for this CAD spec')
        joints = {name: float(value) for name, value in state['joints_rad'].items()}
        if 'presentation' in state:
            shown = {name: float(value) for name, value in state['presentation']['joints_rad'].items()}
            if state.get('source') != 'refined command prior' or any(
                    name not in shown or shown[name] != value for name, value in joints.items()):
                raise ValueError('Presentation readout cannot override the calibrated robot joints')
            joints = shown
        height = float(state['base_height_m'])
        if not joints or not all(math.isfinite(value) for value in [height, *joints.values()]):
            raise ValueError('Visual robot hypotheses must be finite')
        prepared = copy.deepcopy(spec)
        prepared['home'] = {**prepared.get('home', {}), **joints}
        prepared['base']['height'] = height
        return prepared
    if not photo.get('render_capture_state', False):
        return spec
    meta = session.read_json('parsing/robot_cloud.json')
    state = session.config('robot_state.json')
    if meta['spec'] != spec['name'] or meta.get('unmatched_kept_home'):
        raise ValueError('Capture render requires this robot spec and a fully matched joint readout')
    if 'joints' in state:
        state = {j['joint']: j['position_rad'] for j in state['joints']}
    matched = meta['matched']
    home = {joint: float(state[key]) for joint, key in matched.items()}
    if not home or not all(math.isfinite(v) for v in home.values()):
        raise ValueError('Capture render requires finite matched joints')
    prepared = copy.deepcopy(spec)
    prepared['home'] = home
    fit = session.read_json('splat/robot_fit.json')
    height = float(fit['height_offset_m'])
    if not math.isfinite(height):
        raise ValueError('Capture robot root height is not finite')
    prepared['base']['height'] = height
    return prepared
