"""Session directory layout shared by every stage.

sessions/<name>/
  capture/rgb.png                      input photo
  capture/views/<stem>.png             more photos of the scene (optional; recon.multiview)
  capture/keyframes.{json,jpg}         a video's chosen frames (calibrated_scene_reconstruction.keyframes): the photo's index, the views'
  parsing/{geometry.npz, masks.npz, instances.json, overlay.png, world_frame.json}
  parsing/{geometry.moge.npz, multiview.json, multiview/}   recon.multiview (MoGe-2's own scale kept)
  objects/<obj>/{raw.glb, splat.ply, pose.json, mesh_cam.glb, align.json, qa_align.png, albedo.png, retrieved.json}
  objects/<group>/{mesh_local.glb, albedo.png, furniture.json, qa_furniture.png}     shared furniture
  scene/{plate.png, plate_albedo.png, irradiance_photo.npy, parts/*, manifest.json, twin.json}
  cousins/cousin_XX.json, qa/clearance_<tag>.json, renders/, placements/, episodes/<tag>/

configs/<name>/ holds the per-photo decisions (scene_graph.json, lights.json, views.json).
"""
import json
from pathlib import Path

import numpy as np

from .paths import CONFIGS


class Session:
    """Shared session files and configuration access, without reconstruction policy."""
    def __init__(self, path):
        self.dir = Path(path)
        self.name = self.dir.name
        self.config_dir = CONFIGS / self.name

    def __truediv__(self, rel):
        return self.dir / rel

    # --- configs ----------------------------------------------------------------------------------------
    def config(self, name, default=None):
        p = self.config_dir / name
        if p.exists():
            return json.loads(p.read_text())
        if default is not None:
            return default
        raise FileNotFoundError(f"missing {p} (per-photo decision file; see docs/pipeline.md)")


    # --- products -----------------------------------------------------------------------------------------
    def read_json(self, rel):
        return json.loads((self.dir / rel).read_text())

    def write_json(self, rel, obj):
        p = self.dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
        return p

    @property
    def manifest(self):
        return self.read_json("scene/manifest.json")

    def geometry(self):
        return np.load(self.dir / "parsing" / "geometry.npz")

    def masks(self):
        return np.load(self.dir / "parsing" / "masks.npz")["masks"]
