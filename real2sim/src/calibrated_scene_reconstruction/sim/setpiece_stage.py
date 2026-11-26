"""The agent's object and surface fixes (apps/scene/review_objects.py -> configs/<name>/scene_graph.json), applied by
sim.scene.build_scene in the twin and every cousin:

  static_objects.<name>.fix   {"drop": true}                 removed (an object the scene is better without)
                              {"snap": true}                 lowered / raised onto the support under it (a
                                                             reconstruction left floating or sunk)
                              {"swap": {model, height, yaw}} a library model in its place (as a retrieval swap)
                              {"kit": kind, ...params}       a procedural set piece in its place (assets.kits):
                                                             placed and sized from the photo (recon.setpieces) or the
                                                             reconstruction's footprint, params override
  set_pieces.<name>           {"kit": kind, "center": [x, y], "yaw_deg", "on": furniture, "inset", ...params}: one
                              the photo shows but the scene graph has no instance for (a cooktop in the counter
                              texture), `inset` m into the furniture's top
  synthetic_furniture.<name>.fix {"drop": true} | {"center": [x, y]}: a piece the completion invented, removed or
                              moved (out of another piece's base)
  furniture.<name>.look       {"pbr": texture, "tile_m", "tint": "match" | [r, g, b], "parts": ["top", ...]}: the
                              furniture's faces get a Poly Haven material colour-matched to the photo's albedo
                              (recon.materials), instead of a projected photo texture smeared by what stood on it
"""
import numpy as np
from pxr import UsdGeom, UsdPhysics

from ..assets import kits
from calibrated_scene_reconstruction.assets.support_geometry import support_top
from .clearance import shift, world_points
from .receptacle import look_material
from .usd import add_mesh, set_pose


def components(F, n_vertices):
    """The faces of a mesh grouped by connected pieces (sharing vertices): [face index arrays]."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    F = np.asarray(F)
    rows = np.r_[F[:, 0], F[:, 1], F[:, 2]]
    cols = np.r_[F[:, 1], F[:, 2], F[:, 0]]
    g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_vertices, n_vertices))
    _, lab = connected_components(g, directed=False)
    fl = lab[F[:, 0]]
    return [np.flatnonzero(fl == c) for c in np.unique(fl)]


def add_set_piece(stage, path, kind, params, origin, yaw, seed=0):
    """Build a kit (assets.kits) under `path` at origin (its bottom centre) turned by yaw; static colliders (none for a
    kit without, `collide` False: a decal)."""
    kit = kits.build(kind, params, seed)
    root = UsdGeom.Xform.Define(stage, path)
    root.GetPrim().SetCustomDataByKey("calibrated_scene_reconstruction:kit", kind)
    for k, part in enumerate(kit["parts"]):
        mat = look_material(stage, f"/World/Looks/{path.strip('/').replace('/', '_')}_{k}", part["material"])
        geom = add_mesh(stage, f"{path}/{part['name']}", part["V"], part["F"], part["UV"], mat)
        if not kit.get("collide", True):
            continue
        if kit.get("collider") == "mesh":  # a static triangle-mesh collider (a measured structure: kits.mesh)
            UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim()).CreateApproximationAttr("none")
            continue
        comps = components(part["F"], len(part["V"]))
        if len(comps) == 1:
            UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim()).CreateApproximationAttr("convexHull")
            continue
        # a part of separate pieces (a desk's legs and beams, a chair's five legs): a hull each -- one hull over
        # them all filled the space under a desk and between a board's legs
        for j, fi in enumerate(comps):
            vi = np.unique(part["F"][fi])
            remap = np.full(len(part["V"]), -1)
            remap[vi] = np.arange(len(vi))
            col = UsdGeom.Mesh.Define(stage, f"{path}/{part['name']}_col{j}")
            col.CreatePointsAttr([tuple(map(float, v)) for v in part["V"][vi]])
            col.CreateFaceVertexCountsAttr([3] * len(fi))
            col.CreateFaceVertexIndicesAttr(remap[part["F"][fi]].ravel().tolist())
            col.CreatePurposeAttr(UsdGeom.Tokens.guide)
            UsdPhysics.CollisionAPI.Apply(col.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(col.GetPrim()).CreateApproximationAttr("convexHull")
    set_pose(root.GetPrim(), [float(v) for v in origin], (np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)))
    return root.GetPrim(), kit


def support_boxes(ts):
    """What an item can stand on: the room's collision boxes and the room objects reconstructed from other views
."""
    return {**ts.manifest["collision_boxes"], **getattr(ts, "synthetic_boxes", {})}


def snap(ts, prim):
    """Rest an item (no xform of its own needed) on the support under it; returns the vertical shift."""
    P = world_points(prim, 20000)
    lo, hi = P.min(0), P.max(0)
    bottom = float(np.percentile(P[:, 2], 0.5))
    z = support_top(support_boxes(ts), lo[:2], hi[:2], bottom + 0.02)
    dz = z + 0.001 - bottom
    shift(prim, [0.0, 0.0, dz])
    return dz
