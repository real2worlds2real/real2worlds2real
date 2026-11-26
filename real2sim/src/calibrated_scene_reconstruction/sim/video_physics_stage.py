"""The twin of a video recording as a USD stage (Isaac python): a support slab and the object, a primitive or the
SAM 3D mesh (vidphys.mesh), as one rigid body with a convex-hull collider; batch copies beside it; the video's camera
and lights for rendering.

/World/PhysicsScene          gravity 9.81, CCD
/World/Support/top           static box collider, its top the support plane z = 0; /World/Support/look its visible quad
/World/Support/disc          instead of top, a support that turns (a turntable): a heavy body gravity leaves alone,
                             its top at z = 0, turned by the angular velocity apps/video_physics.py sets on every step
/World/Objects/object        rigid body; /geom the mesh (origin: the middle of the footprint, on the plane)
/World/Batch/env<i>/...      copies of Support and Objects (apps/video_physics.py --batch)
/World/Camera, /World/Lights the video's camera (twin.json K, T_world_cam) and a dome + a sun from above
"""
import json
from pathlib import Path

import numpy as np
from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdLux, UsdPhysics

from calibrated_scene_reconstruction.assets.primitive_mesh import primitive_mesh
from calibrated_scene_reconstruction.sim.usd import (add_box_collider, add_camera, add_mesh, color_material, quat_from_matrix, set_pose,
                       textured_material)

HALF = 3.0  # m: the slab reaches this far around the camera's foot


def flat_base(V, frac=0.04, least=0.004):
    """V with everything within max(least, frac x height) of the plane put on it. A mesh reconstructed from one view
    has a slightly round underside; its convex hull then stands on a point and rocks (a mustard bottle leant 12 deg
    and fell over in a slide the real one took upright). What stands on a support has a flat base."""
    V = np.array(V, float)
    V[V[:, 2] < max(least, frac * float(V[:, 2].max())), 2] = 0.0
    return V


def sphere_mesh(r, n=24):
    """A sphere of radius r standing on the plane: (V, F)."""
    th, ph = np.meshgrid(np.linspace(0, np.pi, n // 2 + 1), np.linspace(0, 2 * np.pi, n, endpoint=False), indexing="ij")
    V = np.stack([r * np.sin(th) * np.cos(ph), r * np.sin(th) * np.sin(ph), r + r * np.cos(th)], -1).reshape(-1, 3)
    F = []
    for i in range(n // 2):
        for j in range(n):
            a, b, c, d = i * n + j, i * n + (j + 1) % n, (i + 1) * n + j, (i + 1) * n + (j + 1) % n
            F += [[a, c, b], [b, c, d]]
    return V, np.array(F)


def round_collider(V):
    """What a rolling or bouncing object is to PhysX, from its mesh V (standing on the plane): a sphere, or a
    cylinder lying along the mesh's long horizontal axis when that is a third longer than it is thick or tall.
    {"type", "radius", "centre" (3,), and for a cylinder "length", "axis_deg" (its axis in the mesh's frame)}.
    An exact sphere or cylinder rolls; the convex hull PhysX makes of a mesh (64 vertices) hobbles."""
    lo, hi = V.min(0), V.max(0)
    c, h = (lo + hi) / 2, float(hi[2])
    xy = V[:, :2] - c[:2]
    U = np.linalg.eigh(np.cov(xy.T))[1]
    a, b = float(np.ptp(xy @ U[:, -1])), float(np.ptp(xy @ U[:, 0]))
    if a > 1.35 * max(b, h):
        return {"type": "cylinder", "radius": h / 2, "length": a, "centre": [float(c[0]), float(c[1]), h / 2],
                "axis_deg": float(np.degrees(np.arctan2(U[1, -1], U[0, -1])))}
    r = (a + b + h) / 6
    return {"type": "sphere", "radius": r, "centre": [float(c[0]), float(c[1]), r]}


def object_mesh(path, obj, flat=True, radius=None):
    """twin.json's object in its own frame: {"V", "F", "UV" or None, "albedo" path or None, "Vs", "Fs" the coarse
    outline mesh for silhouettes}. flat: the base flattened (flat_base; not for what rolls). radius: an object
    without a mesh is a sphere of this radius instead of its primitive."""
    if obj["shape"] == "mesh":
        d = np.load(Path(path) / obj["mesh"])
        fb = flat_base if flat and obj.get('route') != 'agent_blender' else np.asarray
        return {"V": fb(d["V"]), "F": d["F"], "UV": d["UV"] if "UV" in d else None,
                "Vs": fb(d["Vs"]), "Fs": d["Fs"],
                "albedo": Path(path) / obj["albedo"] if obj.get("albedo") else None}
    V, F = sphere_mesh(radius) if radius else primitive_mesh(obj["shape"], obj["dims"])
    return {"V": V, "F": F, "UV": None, "albedo": None, "Vs": V, "Fs": F}


def build(stage, path, twin, mass, n_env=1, mesh=None, collider=None, disc=None, slab=True):
    """Populate an empty stage; returns [(offset (3,), root path)] per scene copy (copy 0 is /World). mesh: the
    object_mesh dict (default: the twin's object, flat-based); collider: a round_collider dict instead of the mesh's
    convex hull; disc: {"centre" (2,), "radius"} of a support that turns about its middle, instead of the slab;
    slab False: nothing to stand on (what hangs from a joint)."""
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.DefinePrim("/World", "Xform")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
    scene.CreateGravityMagnitudeAttr(9.81)
    PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim()).CreateEnableCCDAttr(True)
    for scope in ("/World/Looks", "/World/Support", "/World/Objects", "/World/Lights"):
        UsdGeom.Scope.Define(stage, scope)
    top = add_box_collider(stage, "/World/Support/top", [-HALF, -HALF, -0.05], [HALF, HALF, 0.0])
    if disc is not None or not slab:
        UsdPhysics.CollisionAPI(top.GetPrim()).CreateCollisionEnabledAttr(False)
    if disc is not None:
        dx = UsdGeom.Xform.Define(stage, "/World/Support/disc")
        set_pose(dx.GetPrim(), [disc["centre"][0], disc["centre"][1], 0.0])
        dc = UsdGeom.Cylinder.Define(stage, "/World/Support/disc/collider")
        dc.CreateAxisAttr("Z")
        dc.CreateHeightAttr(0.05)
        dc.CreateRadiusAttr(float(disc["radius"]))
        dc.CreatePurposeAttr(UsdGeom.Tokens.guide)
        set_pose(dc.GetPrim(), [0.0, 0.0, -0.025])
        UsdPhysics.CollisionAPI.Apply(dc.GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(dx.GetPrim())
        UsdPhysics.MassAPI.Apply(dx.GetPrim()).CreateMassAttr(1000.0)  # against an object of a few hundred grams
        PhysxSchema.PhysxRigidBodyAPI.Apply(dx.GetPrim()).CreateDisableGravityAttr(True)
    quad = np.array([[-HALF, -HALF, 0], [HALF, -HALF, 0], [HALF, HALF, 0], [-HALF, HALF, 0]], float)
    add_mesh(stage, "/World/Support/look", quad, np.array([[0, 1, 2], [0, 2, 3]]),
             material=color_material(stage, "/World/Looks/support", (0.6, 0.6, 0.6), roughness=0.8))
    m = mesh or object_mesh(path, twin["object"])
    V, F, UV, tex = m["V"], m["F"], m["UV"], m["albedo"]
    xf = UsdGeom.Xform.Define(stage, "/World/Objects/object")
    mat = (textured_material(stage, "/World/Looks/object", str(tex), roughness=0.6) if tex is not None and UV is not None
           else color_material(stage, "/World/Looks/object", tuple(m.get("color", (0.8, 0.3, 0.2)))))
    geom = add_mesh(stage, "/World/Objects/object/geom", V, F, UV=UV if tex is not None else None, material=mat)
    agent_model = twin['object'].get('route') == 'agent_blender'
    if collider is None or agent_model:
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(geom.GetPrim()).CreateApproximationAttr('sdf' if agent_model else 'convexHull')
        if agent_model:
            PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(geom.GetPrim()).CreateSdfResolutionAttr(256)
    else:
        if collider["type"] == "sphere":
            col = UsdGeom.Sphere.Define(stage, "/World/Objects/object/collider")
        else:
            col = UsdGeom.Cylinder.Define(stage, "/World/Objects/object/collider")
            col.CreateAxisAttr("X")
            col.CreateHeightAttr(float(collider["length"]))
        col.CreateRadiusAttr(float(collider["radius"]))
        col.CreatePurposeAttr(UsdGeom.Tokens.guide)
        h = np.radians(collider.get("axis_deg", 0.0)) / 2
        set_pose(col.GetPrim(), collider["centre"], (np.cos(h), 0, 0, np.sin(h)))
        UsdPhysics.CollisionAPI.Apply(col.GetPrim())
        PhysxSchema.PhysxRigidBodyAPI.Apply(xf.GetPrim()).CreateEnableCCDAttr(True)  # a small ball at 3 m/s
    UsdPhysics.RigidBodyAPI.Apply(xf.GetPrim())
    UsdPhysics.MassAPI.Apply(xf.GetPrim()).CreateMassAttr(float(mass))
    envs = [(np.zeros(3), "/World")]
    layer = stage.GetRootLayer()
    cols = int(np.ceil(np.sqrt(n_env)))
    for i in range(1, n_env):
        off = np.array([(i % cols) * (2 * HALF + 2.0), (i // cols) * (2 * HALF + 2.0), 0.0])
        root = f"/World/Batch/env{i}"
        set_pose(UsdGeom.Xform.Define(stage, root).GetPrim(), off)
        for part in ("Objects", "Support"):
            Sdf.CopySpec(layer, f"/World/{part}", layer, f"{root}/{part}")
        envs.append((off, root))
    return envs


def place(stage, root, xy, yaw_deg, z=0.001):
    """The object of a scene copy at (x, y) with yaw, its base z above the plane (default: standing, a millimetre
    above it)."""
    h = np.radians(yaw_deg) / 2
    set_pose(stage.GetPrimAtPath(f"{root}/Objects/object"), [xy[0], xy[1], z], (np.cos(h), 0, 0, np.sin(h)))


def pose(stage, root, pos, quat_wxyz):
    """The object of a scene copy with its origin at pos (3,) and the orientation quat (w, x, y, z)."""
    set_pose(stage.GetPrimAtPath(f"{root}/Objects/object"), pos, quat_wxyz)


def view_pose(cam, T_world_cam):
    """Move add_view's camera (a camera that moves against the twin's frame: a support that carries the frame)."""
    T = np.asarray(T_world_cam, float)
    set_pose(cam.GetPrim(), T[:3, 3], quat_from_matrix(T[:3, :3] @ np.diag([1, -1, -1])))


def add_view(stage, twin, width, height):
    """The video's camera and neutral lights: a dome and a sun from straight above (the real light is unknown; its
    shadow is only laid over the video's own picture as a darkening, see apps/video_physics.py --render)."""
    K = np.asarray(twin["K"], float)
    cam = add_camera(stage, "/World/Camera", K, width, height, T_world_cam_cv=np.asarray(twin["T_world_cam"], float))
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/sky")
    dome.CreateIntensityAttr(700.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/Lights/sun")
    sun.CreateIntensityAttr(1500.0)
    sun.CreateAngleAttr(8.0)
    return cam


def load(path):
    path = Path(path)
    return (json.loads((path / "meta.json").read_text()), json.loads((path / "twin.json").read_text()),
            json.loads((path / "physics.json").read_text()))
