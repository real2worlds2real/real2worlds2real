"""Declared scene actions and camera views, independent of any simulator release."""

from __future__ import annotations

import numpy as np


def view_spec(camera):
    if isinstance(camera, dict) and 'calibration' in camera:
        from .scene_camera import calibrated_view
        view = calibrated_view(camera['calibration'])
        if set(camera)-set(view):
            raise ValueError('Unknown calibrated view fields')
        for key in set(camera)-{'calibration'}:
            if not np.allclose(camera[key], view[key]):
                raise ValueError('Derived view disagrees with calibrated pose')
        return view
    if not isinstance(camera, dict) or set(camera)-{"eye_m", "target_m", "up", "horizontal_fov_deg"}:
        raise ValueError("Invalid review camera fields")
    view = {}
    for key, default in (("eye_m", None), ("target_m", None), ("up", [0, 0, 1])):
        vector = np.asarray(camera.get(key, default), dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError("Camera vectors must contain three finite values")
        view[key] = vector.tolist()
    direction = np.subtract(view["target_m"], view["eye_m"])
    if np.linalg.norm(direction) < 1e-6 or np.linalg.norm(np.cross(direction, view["up"])) < 1e-6:
        raise ValueError("Camera eye, target and up must define an unambiguous view")
    fov = camera.get("horizontal_fov_deg", 55)
    if isinstance(fov, bool) or not isinstance(fov, (int, float)) or not np.isfinite(fov) or not 5 <= fov <= 150:
        raise ValueError("Use a camera field of view from 5 to 150 degrees")
    view["horizontal_fov_deg"] = float(fov)
    return view


def camera_path(keys):
    if not isinstance(keys, list) or not 2 <= len(keys) <= 128:
        raise ValueError("Use 2..128 camera path keys")
    result = []
    for key in keys:
        if not isinstance(key, dict) or "seconds" not in key:
            raise ValueError("A camera key needs seconds and a view")
        t = key["seconds"]
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not np.isfinite(t) or t < 0:
            raise ValueError("Camera key time must be finite and nonnegative")
        if 'calibration' in key:
            raise ValueError('Calibrated camera paths require a pose-sampling adapter; use a fixed calibrated view')
        view = view_spec({k:v for k,v in key.items() if k != "seconds"})
        if result and (t <= result[-1]["seconds"] or view["horizontal_fov_deg"] != result[0]["horizontal_fov_deg"]):
            raise ValueError("Camera times must increase and field of view must stay constant")
        result.append(dict(seconds=float(t), **view))
    if result[0]["seconds"] != 0:
        raise ValueError("A camera path must start at zero")
    return result


def camera_at(spec, seconds):
    keys = spec.get("review_camera_path")
    if not keys:
        return spec.get("review_camera")
    times = [key["seconds"] for key in keys]
    i = int(np.clip(np.searchsorted(times, seconds, side="right")-1, 0, len(keys)-2))
    a,b = keys[i:i+2]
    u = float(np.clip((seconds-a["seconds"])/(b["seconds"]-a["seconds"]), 0, 1))
    u = u**3*(u*(6*u-15)+10)
    result = {k:((1-u)*np.asarray(a[k])+u*np.asarray(b[k])).tolist() for k in ("eye_m", "target_m", "up")}
    result["horizontal_fov_deg"] = a["horizontal_fov_deg"]
    return view_spec(result)


def actions(spec, object_ids):
    if not isinstance(spec, dict) or set(spec)-set(object_ids):
        raise ValueError("Actions must name existing scene objects")
    result = {}
    for name, action in spec.items():
        if not isinstance(action, dict) or set(action)-{"type", "start_seconds", "hold_last_controls", "joints", "controls"}:
            raise ValueError("Invalid scene action fields")
        t = action.get("start_seconds", 0)
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not np.isfinite(t) or t < 0:
            raise ValueError("Action start must be finite and nonnegative")
        hold = action.get("hold_last_controls", False)
        if type(hold) is not bool or action.get("type") not in {"native_test", "joint_targets"}:
            raise ValueError("Declare native_test or joint_targets and a boolean hold policy")
        out = dict(type=action["type"], start_seconds=float(t), hold_last_controls=hold)
        if action["type"] == "joint_targets":
            joints = action.get("joints")
            if not isinstance(joints, list) or not 1 <= len(joints) <= 64:
                raise ValueError("Joint actions need a bounded target list")
            seen = set()
            for joint in joints:
                if not isinstance(joint, dict) or set(joint)!={"joint", "keyframes"} or not isinstance(joint["joint"],str) or not joint["joint"] or joint["joint"] in seen:
                    raise ValueError("Joint targets need unique joint names and keyframes")
                seen.add(joint["joint"])
                keys = np.asarray(joint["keyframes"],dtype=float)
                if keys.ndim!=2 or keys.shape[1]!=2 or not 2<=len(keys)<=128 or not np.isfinite(keys).all() or keys[0,0]!=0 or (np.diff(keys[:,0])<=0).any():
                    raise ValueError("Joint keys must be ordered [seconds, native-coordinate target] from zero")
            out["joints"] = joints
        elif "joints" in action:
            raise ValueError("Native tests do not take joint targets")
        if "controls" in action:
            if action["type"] != "native_test" or not isinstance(action["controls"], list) or len(action["controls"]) > 128:
                raise ValueError("Native rigid control overrides must be a bounded list")
            out["controls"] = action["controls"]
        result[name] = out
    return result
