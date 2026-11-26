"""Mesh coordinates and frames for scene assets and demonstration inputs."""
import numpy as np
import trimesh


def twin_world(s, name):
    m = trimesh.load(s / "objects" / name / "mesh_cam.glb", force="mesh")
    # before recon.layout wrote the manifest (a demo twin built only through the object meshes): the same frame
    T = np.asarray(s.manifest["photo_camera"]["T_world_cam_cv"] if (s / "scene" / "manifest.json").exists()
                   else s.world_frame()[2])
    return np.asarray(m.vertices) @ T[:3, :3].T + T[:3, 3]
