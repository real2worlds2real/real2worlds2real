"""Object-relative camera observations for agent model fitting in fixed-camera videos.

Tracking supplies a pose initialization, not an accepted object reconstruction.
The camera is transformed into each tracked object's frame so a moving rigid
object can use the same multi-view silhouette evidence contract as a static one.
"""
from calibrated_scene_reconstruction.physics.common.schema import read_document
import json
from pathlib import Path

import numpy as np


def calibration(path, mode='video'):
    path = Path(path)
    if mode == 'video':
        twin = read_document(path / 'twin.json')
        return {'K': twin['K'], 'T_world_cam': twin['T_world_cam'], 'support_z': 0.,
                'scale': twin.get('scale')}
    camera = read_document(path / 'camera.json')
    twin = read_document(path / 'model_initialization.json')
    return {'K': camera['K'], 'T_world_cam': camera['T_world_cam'], 'support_z': twin['support']['z']}


def object_cameras(K, T, poses):
    """World camera -> object-frame camera; poses are [x,y,z,yaw_deg]."""
    poses = np.asarray(poses, float)
    rotations, centres = [], []
    for x, y, z, yaw in poses:
        a = np.radians(yaw)
        R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.]])
        rotations.append(R.T @ np.asarray(T)[:3, :3])
        centres.append((np.asarray(T)[:3, 3] - [x, y, z]) @ R)
    return {'K': np.repeat(np.asarray(K, float)[None], len(poses), axis=0),
            'R_world_cam': np.asarray(rotations), 'centre': np.asarray(centres)}


def require_model(path, mode='video', applied=False):
    """Check artifact provenance and optionally the mesh already applied to twin.json."""
    from calibrated_scene_reconstruction.assets.model_artifact import require_reviewed
    from calibrated_scene_reconstruction.core.paths import ROOT, resolve
    path = Path(path)
    proof, V, F = require_reviewed(path, ROOT / 'configs' / path.name / 'agent/objects/object/model.py')
    if proof.get('mode') != mode or proof.get('calibration') != calibration(path, mode):
        raise ValueError('Approved model belongs to different camera calibration or physics route')
    if applied:
        obj = read_document(path / 'twin.json')['object']
        if obj.get('route') != 'agent_blender' or obj.get('model_round') != proof['round']:
            raise ValueError('Apply the reviewed Blender round before simulation')
        mesh_path = path / obj['mesh'] if mode == 'video' else resolve(obj['mesh'])
        with np.load(mesh_path) as mesh:
            if not np.array_equal(mesh['V'], V) or not np.array_equal(mesh['F'], F):
                raise ValueError('Applied mesh differs from its approved model')
        if not np.allclose(obj['pose'], proof['world_pose'], rtol=0, atol=1e-10):
            raise ValueError('Applied object pose differs from its approved model')
    return proof, V, F


def observations(path, mode='video'):
    path = Path(path)
    context = calibration(path, mode)
    with np.load(path / 'track.npz') as data:
        track = dict(data)
    if mode == 'video':
        with np.load(path / 'perception.npz') as data:
            per = dict(data)
        masks = np.asarray(per['object'], bool)
        poses = np.c_[track['xy'], np.zeros(len(masks)), track['yaw']]
        valid = np.asarray(track['valid'], bool)
        ref = int(read_document(path / 'twin.json')['ref_frame'])
    else:
        from .scene import ray_plane
        twin = read_document(path / 'model_initialization.json')
        height = twin['object']['dims']['height']
        centres = ray_plane(context['K'], context['T_world_cam'], track['centroid'],
                            context['support_z'] + height / 2)
        poses = np.c_[centres[:, :2], np.full(len(centres), context['support_z']), np.zeros(len(centres))]
        stride = int(track['mask_stride'])
        masks = np.repeat(np.repeat(track['masks'], stride, axis=1), stride, axis=2).astype(bool)
        camera = read_document(path / 'camera.json')
        masks = masks[:, :camera['height'], :camera['width']]
        valid = np.asarray(track['valid'], bool)
        ref = int(np.argmax(valid))
    keep = np.flatnonzero(valid & np.isfinite(poses).all(1) & masks.any(axis=(1, 2)))
    if len(keep) < 8:
        raise ValueError('At least eight valid calibrated silhouettes required for Blender iteration')
    if ref not in keep:
        raise ValueError('The reference object pose is not observable')
    z = object_cameras(context['K'], context['T_world_cam'], poses[keep])
    H, W = masks.shape[1:]
    windows = np.repeat([[0., 0., W, H, 1.]], len(keep), axis=0)
    return (z, np.arange(len(keep)), list(masks[keep]), windows), poses[ref], context, keep
