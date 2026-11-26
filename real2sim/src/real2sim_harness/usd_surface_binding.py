"""Preserve Blender's render meshes and materials while Newton drives the cloth."""

import json
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom, Vt

from real2sim_harness.artifacts import atomic_json, sha256_file
from real2sim_harness.surface_binding import build_binding, deform_surface


def object_name(prim):
    while prim and not prim.IsPseudoRoot():
        attr = prim.GetAttribute("userProperties:blender:object_name")
        if attr and attr.Get():
            return attr.Get()
        prim = prim.GetParent()
    return None


def build(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    followers = set(manifest.get("deformable_visual_objects", []))
    if not followers:
        raise ValueError("No deformable visual followers recorded; re-export the Blender scene")
    control = json.loads((folder / "sim_mesh.json").read_text())
    points, faces = np.asarray(control["vertices"]), np.asarray(control["faces"])
    stage = Usd.Stage.Open(str(folder / "visual.usdc"))
    cache = UsdGeom.XformCache()
    records, arrays, found = [], {}, set()
    for prim in stage.Traverse():
        name = object_name(prim)
        if not prim.IsA(UsdGeom.Mesh) or name not in followers:
            continue
        mesh = UsdGeom.Mesh(prim)
        local = np.asarray(mesh.GetPointsAttr().Get())
        transform = np.asarray(cache.GetLocalToWorldTransform(prim))
        world = local @ transform[:3, :3] + transform[3, :3]
        mapping = build_binding(points, faces, world)
        reconstructed = deform_surface(points, faces, mapping)
        error = float(np.max(np.linalg.norm(reconstructed - world, axis=-1)))
        if error > 1e-5:
            raise ValueError(f"Visual binding reconstruction failed for {name}: {error}")
        key = f"mesh_{len(records):03d}"
        for field, value in mapping.items():
            arrays[f"{key}_{field}"] = value
        arrays[f"{key}_inverse"] = np.linalg.inv(transform)
        records.append({"key": key, "source_object": name, "usd_path": str(prim.GetPath()),
                        "vertices": len(local), "rest_error_max_m": error,
                        "offset_max_m": float(np.linalg.norm(mapping["offsets"], axis=-1).max())})
        found.add(name)
    if followers - found:
        raise ValueError(f"Visual follower meshes missing from USD: {sorted(followers - found)}")
    np.savez_compressed(folder / "surface_binding.npz", **arrays)
    report = {
        "schema_version": 1, "ok": True, "meshes": records,
        "visual_usd_sha256": sha256_file(folder / "visual.usdc"),
        "sim_mesh_sha256": sha256_file(folder / "sim_mesh.json"),
        "binding_sha256": sha256_file(folder / "surface_binding.npz"),
        "mapped_vertices": sum(row["vertices"] for row in records),
        "control_nodes": len(points),
    }
    atomic_json(folder / "surface_binding.json", report)
    print(f"SURFACE_BINDING_BUILT meshes={len(records)} vertices={report['mapped_vertices']}", flush=True)
    return report


class UsdSurfaceBinding:
    def __init__(self, folder, stage, root="/World/Visuals"):
        self.folder = Path(folder)
        self.report = json.loads((self.folder / "surface_binding.json").read_text())
        for field, filename in (("visual_usd_sha256", "visual.usdc"),
                                ("sim_mesh_sha256", "sim_mesh.json"),
                                ("binding_sha256", "surface_binding.npz")):
            if self.report[field] != sha256_file(self.folder / filename):
                raise ValueError(f"Stale cloth visual binding: {filename} changed")
        data = json.loads((self.folder / "sim_mesh.json").read_text())
        self.faces = np.asarray(data["faces"], dtype=np.int32)
        self.arrays = np.load(self.folder / "surface_binding.npz", allow_pickle=False)
        self.meshes = []
        for record in self.report["meshes"]:
            relative = record["usd_path"].removeprefix("/Asset")
            mesh = UsdGeom.Mesh.Get(stage, root + relative)
            if not mesh or len(mesh.GetPointsAttr().Get()) != record["vertices"]:
                raise ValueError(f"Visual topology does not match binding: {root + relative}")
            mesh.GetNormalsAttr().Block()
            self.meshes.append((record, mesh))

    def update(self, positions, *, refresh_transforms=False, world_from_binding=None):
        positions = np.asarray(positions)
        if positions.shape != (self.report["control_nodes"], 3):
            raise ValueError("Newton node count changed after visual binding")
        transform_cache = UsdGeom.XformCache() if refresh_transforms else None
        for record, mesh in self.meshes:
            key = record["key"]
            mapping = {field: self.arrays[f"{key}_{field}"]
                       for field in ("triangle_ids", "weights", "offsets")}
            world = deform_surface(positions, self.faces, mapping)
            if world_from_binding is not None:
                placement = np.asarray(world_from_binding)
                world = world @ placement[:3, :3].T + placement[:3, 3]
            inverse = (np.linalg.inv(np.asarray(transform_cache.GetLocalToWorldTransform(mesh.GetPrim())))
                       if transform_cache is not None else self.arrays[f"{key}_inverse"])
            local = world @ inverse[:3, :3] + inverse[3, :3]
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(local.astype(np.float32)))
        return {"ok": True, "meshes": len(self.meshes),
                "mapped_vertices": self.report["mapped_vertices"]}
