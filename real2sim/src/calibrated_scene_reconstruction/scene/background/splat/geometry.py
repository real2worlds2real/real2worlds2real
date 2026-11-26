"""Coordinate conversion shared by captured backgrounds and code-room measurement."""
import numpy as np

from .config import sim3


def world_of_model(sp, s):
    """model frame -> world: the inverse of the training's normalization, then splat/align.json's similarity."""
    from calibrated_scene_reconstruction.scene.background.splat.fix import model_transform

    T, _ = model_transform(sp)
    c, R, t = sim3(s)
    A = np.eye(4)
    A[:3, :3], A[:3, 3] = c * R, t
    return A @ np.linalg.inv(T), T
