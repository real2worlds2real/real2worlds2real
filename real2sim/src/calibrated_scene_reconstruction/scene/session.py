"""Legacy scene-session adapter; storage is shared, reconstruction policy migrates with scene."""
import json
from pathlib import Path as Path  # noqa: F401 -- legacy module attribute

import numpy as np

from calibrated_scene_reconstruction.core.paths import CONFIGS
from calibrated_scene_reconstruction.core.session import Session as FileSession
from calibrated_scene_reconstruction.scene.graph import close_room


class Session(FileSession):
    """Preserve the original scene API while core remains free of scene policy."""

    def __init__(self, path, config_dir=None):
        super().__init__(path)
        self.config_dir = CONFIGS / self.name
        if config_dir is not None:
            snapshot=Path(config_dir).resolve()
            if not snapshot.is_relative_to(CONFIGS.resolve()) or not snapshot.is_dir():
                raise ValueError('Configuration snapshot must be an existing workspace configs directory')
            self.config_dir=snapshot

    @property
    def scene_graph(self):
        """The reviewed scene graph, its walls closed around the floor (scenegraph.close_room: pieces the photo or
        the panorama left open are derived here, so every stage builds the same closed room)."""
        sg = self.config("scene_graph.json")
        if "structure" in sg:
            sg["structure"] = close_room(sg["structure"])
        return sg

    def world_frame(self):
        """World frame from the floor plane: z up, x = camera right on the floor, origin under the camera.

        Returns (R_wc, o) with p_world = R_wc @ (p_cam - o) and T_world_cam (OpenCV camera axes in world).
        """
        p = self.dir / "parsing" / "world_frame.json"
        if p.exists():
            d = json.loads(p.read_text())
            return np.asarray(d["R_wc"]), np.asarray(d["o"]), np.asarray(d["T_world_cam_cv"])
        geo, masks = self.geometry(), self.masks()
        floor_ids = self.scene_graph["structure"]["floor"]["ids"]
        m = np.zeros(masks.shape[1:], bool)
        for i in floor_ids:
            m |= masks[i]
        fl = geo["points"][m & geo["mask"]]
        c = fl.mean(0)
        _, V = np.linalg.eigh(np.cov((fl - c).T))  # 3x3 eig: never run a full SVD on an N x 3 matrix (N x N U!)
        n = V[:, 0]
        if n[1] > 0:  # OpenCV camera: +y points down, the floor normal must point up
            n = -n
        d = -n @ c
        x = np.array([1.0, 0, 0]) - n[0] * n
        x /= np.linalg.norm(x)
        R_wc = np.stack([x, np.cross(n, x), n])
        o = -d * n
        T = np.eye(4)
        T[:3, :3] = R_wc
        T[:3, 3] = -R_wc @ o
        p.write_text(json.dumps({"R_wc": R_wc.tolist(), "o": o.tolist(), "T_world_cam_cv": T.tolist(),
                                 "camera_height_m": float(d)}, indent=1))
        return R_wc, o, T
