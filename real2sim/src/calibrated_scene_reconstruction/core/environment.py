"""Resolve operator settings across the reconstruction package rename."""
import os


def setting(name, default=None):
    key = "SCENE_RECONSTRUCTION_" + name
    return os.environ[key] if key in os.environ else os.environ.get("R2S2R_" + name, default)
