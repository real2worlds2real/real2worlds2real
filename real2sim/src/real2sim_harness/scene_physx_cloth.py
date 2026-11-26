"""Materialize passive particle cloth into the caller's PhysX scene.

Configuration and binding checks can run without Kit. Runtime imports stay in
the runtime class; no simulation, controller or application is started here.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np

from .artifacts import sha256_file
from .physx_cloth import resolve_cloth_settings


def read_configuration(folder, bundle=None):
    folder = Path(folder).resolve()
    bundle = bundle or json.loads((folder / "asset_bundle.json").read_text())
    if (bundle.get("backend") != "physx" or bundle.get("asset_kind") != "deformable"
            or bundle.get("native_representation") != "surface"):
        raise ValueError("A PhysX particle-cloth surface bundle is required")
    filename = bundle.get("cloth_configuration")
    if not isinstance(filename, str) or not filename:
        raise ValueError("PhysX cloth needs a retained cloth_configuration")
    required = {filename, "sim_mesh.json", "surface_binding.json", "surface_binding.npz", "visual.usdc"}
    for name in required:
        path = (folder / name).resolve()
        if (Path(name).is_absolute() or not path.is_relative_to(folder)
                or name not in bundle.get("files", {}) or not path.is_file()
                or sha256_file(path) != bundle["files"][name]):
            raise ValueError("Missing or changed PhysX cloth configuration/binding: " + name)
    c = json.loads((folder / filename).read_text())
    resolve_cloth_settings(c)
    if c.get("grasps") or c.get("probes"):
        raise ValueError("Scene cloth must be passive; express fixtures and controls in the scene")
    mesh = json.loads((folder / "sim_mesh.json").read_text())
    if mesh["vertices"] != c["geometry"]["vertices"] or mesh["faces"] != c["geometry"]["triangles"]:
        raise ValueError("Cloth binding topology differs from the physical control mesh")
    meta = json.loads((folder / "surface_binding.json").read_text())
    for key, name in (("visual_usd_sha256", "visual.usdc"), ("sim_mesh_sha256", "sim_mesh.json"),
                      ("binding_sha256", "surface_binding.npz")):
        if meta.get(key) != sha256_file(folder / name):
            raise ValueError("Stale PhysX cloth visual binding")
    if not meta.get("ok") or not meta.get("meshes"):
        raise ValueError("PhysX cloth needs a successful source visual binding")
    return c


def with_scene_timestep(configuration, physics_hz):
    """Explicitly resolve a shared scene clock without changing the source bundle."""
    c = copy.deepcopy(configuration)
    resolved = resolve_cloth_settings(c)
    fps = c.get("fps", 30)
    if type(physics_hz) is not int or physics_hz % fps or not 1 <= physics_hz // fps <= 255:
        raise ValueError("Scene physics frequency must be an integer multiple of cloth FPS (1..255 substeps)")
    c["physx"] = dict(resolved["settings"], substeps=physics_hz // fps)
    return c, {"source_dt_s": resolved["dt_s"], "scene_dt_s": 1 / physics_hz,
               "overridden": not math.isclose(resolved["dt_s"], 1 / physics_hz)}


def particle_mapping(authored, cooked, *, tolerance=1e-5):
    """Require an unambiguous bijection before assigning controls or visuals."""
    from scipy.spatial import cKDTree

    authored, cooked = np.asarray(authored), np.asarray(cooked)
    if authored.shape != cooked.shape or authored.ndim != 2 or authored.shape[1] != 3:
        raise ValueError("PhysX changed cloth particle count; re-export a compatible control mesh")
    if not np.isfinite(authored).all() or not np.isfinite(cooked).all():
        raise ValueError("Nonfinite cloth initialization")
    distance, indices = cKDTree(cooked).query(authored, k=2)
    if (np.max(distance[:, 0]) > tolerance or np.any(distance[:, 1] <= tolerance)
            or len(np.unique(indices[:, 0])) != len(authored)):
        raise ValueError("Cooked cloth particles do not map uniquely to the source; inspect initialization")
    return indices[:, 0], float(np.max(distance[:, 0]))


def audit_supports(positions, triangles, supports, *, thickness, tolerance):
    """Use full triangle/box SAT in each authored support's local frame."""
    from .deformable_geometry import audit_box_contacts

    results = []
    positions = np.asarray(positions)
    poses = np.zeros((len(positions), 1, 7))
    poses[:, :, 6] = 1
    for support in supports:
        transform = np.asarray(support["transform"])
        local = (positions - transform[:3, 3]) @ transform[:3, :3]
        result = audit_box_contacts(local, triangles, poses,
            [{"id": support["id"], "shape": "box", "size_m": support["size_m"]}],
            surface_thickness_m=thickness, tolerance_m=tolerance)
        results.append({"id": support["id"], **result})
    return {"ok": all(r["ok"] for r in results), "supports": results, "tolerance_m": tolerance}


class ScenePhysxCloth:
    def __init__(self, folder, stage, item, *, physics_hz, simulation_owner,
                 max_stretch=2.0, disable_rigid_collision=False):
        from pxr import UsdGeom, UsdPhysics

        from .physx_cloth import create_cloth
        from .usd_surface_binding import UsdSurfaceBinding

        self.name = item["id"]
        self.folder = Path(folder)
        self.stage = stage
        self.placement = np.asarray(item["transform"], dtype=float)
        source = read_configuration(folder)
        self.configuration, self.timing = with_scene_timestep(source, physics_hz)
        self.faces = np.asarray(source["geometry"]["triangles"], dtype=np.int32)
        self.local_rest = np.asarray(source["geometry"]["vertices"], dtype=float)
        self.world_rest = self.local_rest @ self.placement[:3, :3].T + self.placement[:3, 3]
        world_config = copy.deepcopy(self.configuration)
        world_config["geometry"]["vertices"] = self.world_rest.tolist()
        self.path = f"/World/ClothPhysics/{self.name}"
        self.cloth, self.system, self.material, self.resolved = create_cloth(
            stage, world_config, prim_path=self.path, simulation_owner=simulation_owner)
        # Control geometry is separate from source-material visual followers.
        UsdGeom.Imageable(stage.GetPrimAtPath(self.path)).MakeInvisible()
        self.material.set_gravity_scale(0.0)
        # The 5.1 nonParticleCollisionEnabled attribute is not a reliable
        # negative control. Filter explicit collider pairs on the particle system.
        self.filtered_colliders = []
        if disable_rigid_collision:
            targets = [p.GetPath() for p in stage.Traverse()
                       if p.HasAPI(UsdPhysics.CollisionAPI)
                       and str(p.GetPath()).startswith("/World/Scene/")]
            UsdPhysics.FilteredPairsAPI.Apply(self.system.prim).CreateFilteredPairsRel().SetTargets(targets)
            self.filtered_colliders = list(map(str, targets))
        self.rigid_collision_enabled = not disable_rigid_collision
        self.binding = UsdSurfaceBinding(folder, stage, root=f"/World/Scene/Objects/{self.name}/Asset")
        self.edges = np.unique(np.sort(self.faces[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0)
        self.rest_lengths = np.linalg.norm(self.local_rest[self.edges[:, 1]] - self.local_rest[self.edges[:, 0]], axis=1)
        self.max_stretch = max_stretch
        self.max_edge_ratio = 1.0
        self.finite = True
        self.positions, self.velocities, self.times = [], [], []
        self.contact_samples = []

    def initialize(self):
        import omni.physics.tensors as tensors

        self.simulation_view = tensors.create_simulation_view("torch")
        self.simulation_view.set_subspace_roots("/")
        self.view = self.simulation_view.create_particle_cloth_view(self.path)
        if self.view.count != 1:
            raise ValueError("PhysX cloth was not created as one native cloth body")
        cooked = self.view.get_positions().reshape(-1, 3).cpu().numpy()
        self.mapping, self.mapping_error = particle_mapping(self.world_rest, cooked)
        self.material.set_gravity_scale(1.0)
        self.sample(0.0)

    def sample(self, seconds):
        points = self.view.get_positions().reshape(-1, 3).cpu().numpy()[self.mapping].copy()
        velocities = self.view.get_velocities().reshape(-1, 3).cpu().numpy()[self.mapping].copy()
        self.finite &= bool(np.isfinite(points).all() and np.isfinite(velocities).all())
        if not self.finite:
            raise ValueError("Nonfinite PhysX cloth state")
        ratio = np.linalg.norm(points[self.edges[:, 1]] - points[self.edges[:, 0]], axis=1) / self.rest_lengths
        self.max_edge_ratio = max(self.max_edge_ratio, float(ratio.max()))
        self.positions.append(points)
        self.velocities.append(velocities)
        self.times.append(seconds)

    def update_visuals(self):
        local = (self.positions[-1] - self.placement[:3, 3]) @ self.placement[:3, :3]
        self.binding.update(local, refresh_transforms=True, world_from_binding=self.placement)

    def inspect_contacts(self):
        """Sample live collider proximity, explicitly distinct from force sensing."""
        from omni.physx import get_physx_scene_query_interface

        query = get_physx_scene_query_interface()
        radius = self.resolved["settings"]["contact_offset"] + self.resolved["settings"]["solid_rest_offset"] + 0.002
        hits = {}
        for index, point in enumerate(self.positions[-1]):
            def on_hit(hit, index=index):
                collider = str(hit.collision)
                if collider.startswith(("/World/Scene/Environment/", "/World/Scene/Objects/")):
                    hits.setdefault(collider, []).append(index)
                return True
            query.overlap_sphere(radius, tuple(map(float, point)), on_hit, False)
        self.contact_samples.append({"time_s": self.times[-1], "colliders": {
            path: {"node_count": len(set(nodes)), "example_node": nodes[0]} for path, nodes in hits.items()}})

    def finish(self, output, *, supports, penetration_tolerance):
        output = Path(output)
        path = output / f"cloth_{self.name}_states.npz"
        np.savez_compressed(path, times_s=np.asarray(self.times), positions=np.asarray(self.positions),
                            velocities=np.asarray(self.velocities), triangles=self.faces,
                            source_to_runtime=self.mapping)
        collider_paths = sorted({p for s in self.contact_samples for p in s["colliders"]})
        late = self.contact_samples[-3:]
        environment = any("/Environment/" in p for s in late for p in s["colliders"])
        rigid = sorted({p.split("/Objects/", 1)[1].split("/", 1)[0] for p in collider_paths if "/Objects/" in p})
        late_rigid = sorted({p.split("/Objects/", 1)[1].split("/", 1)[0] for s in late for p in s["colliders"] if "/Objects/" in p})
        support_audit = audit_supports(self.positions, self.faces, supports,
            thickness=self.configuration["material"]["thickness_m"], tolerance=penetration_tolerance)
        return {"id": self.name, "finite": self.finite,
                "deformation_ok": self.max_edge_ratio <= self.max_stretch and support_audit["ok"],
                "max_edge_ratio": self.max_edge_ratio, "max_allowed_edge_ratio": self.max_stretch,
                "max_motion_m": float(np.linalg.norm(np.asarray(self.positions) - self.world_rest, axis=-1).max()),
                "mapping_error_m": self.mapping_error, "particles": len(self.mapping),
                "timing": self.timing, "resolved_settings": self.resolved,
                "rigid_collision_enabled": self.rigid_collision_enabled,
                "diagnostic_filtered_colliders": self.filtered_colliders,
                "environment_proximity_late": environment, "rigid_proximity_ids": rigid,
                "rigid_proximity_late_ids": late_rigid, "support_surface_audit": support_audit,
                "contact_samples": self.contact_samples,
                "contact_scope": "Live PhysX overlap queries at actual simulated cloth nodes; geometric contact proximity, not measured forces or complete contact coverage.",
                "state_file": {"path": path.name, "sha256": sha256_file(path)}}
