"""Calibrated physical parameters (configs/<name>/physics.json, written by apps/real2sim_replay.py --calibrate; see
calibrated_scene_reconstruction.physics.replay.params) authored onto a built stage, before the physics scene is parsed (World.reset()).

Pair coefficients: the object's material has friction 1 with combine mode `multiply` and its restitution with mode
`max` (PhysX uses the higher-priority mode of the two: average < min < multiply < max), so the object-gripper and
object-support pairs get exactly the gripper pads' and the support's own friction, and the object's restitution.
"""
import numpy as np
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

SUPPORT_ROOTS = ("/World/Room", "/World/Support", "/World/Static")
_GAINS = {}  # attribute path -> the gain add_robot authored: repeated calibration calls scale from it, not compound


def physics_material(stage, path, static, dynamic, restitution=0.0, friction_combine=None, restitution_combine=None):
    mat = UsdShade.Material.Define(stage, path)
    pm = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    pm.CreateStaticFrictionAttr().Set(float(static))
    pm.CreateDynamicFrictionAttr().Set(float(dynamic))
    pm.CreateRestitutionAttr().Set(float(restitution))
    if friction_combine or restitution_combine:
        px = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
        if friction_combine:
            px.CreateFrictionCombineModeAttr().Set(friction_combine)
        if restitution_combine:
            px.CreateRestitutionCombineModeAttr().Set(restitution_combine)
    return mat


def bind_colliders(stage, roots, mat):
    """Bind `mat` (physics purpose) to every collider under the given prim paths; returns how many."""
    n = 0
    for root in roots:
        prim = stage.GetPrimAtPath(root)
        if not prim:
            continue
        for p in Usd.PrimRange(prim):
            if p.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(p).Bind(mat, UsdShade.Tokens.strongerThanDescendants, "physics")
                n += 1
    return n


def mesh_centre(prim):
    """Bounding-box centre of the render meshes under an object prim, in its frame (the prim origin is its bottom)."""
    pts = np.concatenate([np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get()) for p in Usd.PrimRange(prim)
                          if p.IsA(UsdGeom.Mesh) and UsdGeom.Imageable(p).ComputePurpose() != UsdGeom.Tokens.guide])
    return (pts.max(0) + pts.min(0)) / 2


def apply_scene(stage, cal, obj_path, supports=SUPPORT_ROOTS, tag="calibrated"):
    """Gravity, the object's mass / centre of mass / material and the support material from a physics.json dict.
    `tag` keeps the materials of batch copies apart (/World/Looks/<tag>_*)."""
    if "gravity" in cal:  # every physics scene: World() adds its own (/physicsScene) beside the twin's
        for p in stage.Traverse():
            if p.IsA(UsdPhysics.Scene):
                UsdPhysics.Scene(p).CreateGravityMagnitudeAttr().Set(float(cal["gravity"]))
    o = cal.get("object", {})
    prim = stage.GetPrimAtPath(obj_path)
    if "mass" in o:
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr().Set(float(o["mass"]))
        if "com_offset" in o:  # PhysX keeps the inertia of the collision shapes, shifted to this centre
            mass.CreateCenterOfMassAttr().Set(Gf.Vec3f(*(mesh_centre(prim) + np.asarray(o["com_offset"], float))))
    mat = physics_material(stage, f"/World/Looks/{tag}_object", 1.0, 1.0, o.get("restitution", 0.0),
                           friction_combine="multiply", restitution_combine="max")
    bind_colliders(stage, [obj_path], mat)
    s = cal.get("support")
    if s:
        sm = physics_material(stage, f"/World/Looks/{tag}_support", s["static_friction"],
                              s.get("dynamic_friction", s["static_friction"]))
        bind_colliders(stage, supports, sm)


def apply_robot(stage, cal, hand_roots, robot_path="/World/Robot", tag="calibrated"):
    """Gripper pad material on the hand's colliders and the drive gain multipliers on every joint drive of the robot
    (call after robots.articulation.add_robot, which authors the spec's gains)."""
    g = cal.get("gripper")
    if g:
        gm = physics_material(stage, f"/World/Looks/{tag}_gripper", g["static_friction"],
                              g.get("dynamic_friction", g["static_friction"]))
        bind_colliders(stage, hand_roots, gm)
    d = cal.get("drive", {})
    ks, kd = float(d.get("stiffness_scale", 1.0)), float(d.get("damping_scale", 1.0))
    if ks == 1.0 and kd == 1.0 and not _GAINS:
        return
    for p in Usd.PrimRange(stage.GetPrimAtPath(robot_path)):
        for kind in ("angular", "linear"):
            if p.HasAPI(UsdPhysics.DriveAPI, kind):
                drive = UsdPhysics.DriveAPI(p, kind)
                for attr, k in ((drive.GetStiffnessAttr(), ks), (drive.GetDampingAttr(), kd)):
                    if attr and attr.Get() is not None:
                        base = _GAINS.setdefault(str(attr.GetPath()), float(attr.Get()))
                        attr.Set(base * k)


def hand_prims(stage, links, robot_path="/World/Robot"):
    """Prim paths of the named links (robots.articulation.hand_links) under a robot."""
    return [str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(robot_path)) if p.GetName() in links]
