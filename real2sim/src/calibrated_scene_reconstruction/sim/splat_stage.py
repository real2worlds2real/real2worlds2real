"""The twin's optional Gaussian-splat background in an Isaac Sim stage (recon.splat_export): the NuRec volume
referenced under /World/SplatBackground with its COLMAP -> world transform, rendered by RTX beside the meshes; the
mesh visuals it stands for (splat.json `hide`: walls, ceiling, the room's static objects and room objects outside
the task zone) made invisible, their colliders kept. The task zone's meshes (the support, what stands on it) and the
robot stay as they are.

Apps take it with `--background splat` (default `mesh`: the twin as built, nothing of this loaded).
"""
import json

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom
from .render_profile import PROFILE_KEY, photometric_profile


PATH = "/World/SplatBackground"
NUREC_SCHEMA = "omni.usd.schema.omni_nurec_types"  # Kit 107.3 (Isaac Sim 5.1) loads it on demand


def override_background(stage, field):
    """Replace a complete USD field reference or its raw .nurec payload.

    A USDZ is a scene package, not a density/colour payload. Writing it into
    filePath silently renders an empty field in RTX. Keep the root's calibrated
    transform while replacing the complete wrapper and its field bounds.
    """
    from pathlib import Path as FilePath
    field=FilePath(field).resolve();root=stage.GetPrimAtPath(PATH)
    if not field.is_file() or not root:raise ValueError('Existing GS root and field asset required')
    if field.suffix.lower() in ('.usd','.usda','.usdc','.usdz'):
        source=Usd.Stage.Open(str(field))
        if not source or not source.GetDefaultPrim() or len([p for p in source.Traverse() if p.GetTypeName()=='OmniNuRecFieldAsset'])!=2:
            raise ValueError('USD context asset needs a default prim and density/colour fields')
        root.GetReferences().SetReferences([Sdf.Reference(str(field))])
        kind='usd_reference'
    else:
        if field.suffix.lower()!='.nurec':raise ValueError('Use a complete USD context asset or a raw .nurec payload')
        for prim in stage.Traverse():
            if str(prim.GetPath()).startswith(PATH+'/') and prim.GetTypeName()=='OmniNuRecFieldAsset':
                prim.GetAttribute('filePath').Set(Sdf.AssetPath(str(field)))
        kind='nurec_payload'
    fields=[p for p in stage.Traverse() if str(p.GetPath()).startswith(PATH+'/') and p.GetTypeName()=='OmniNuRecFieldAsset']
    if len(fields)!=2:raise ValueError('Expected density and colour NuRec fields after override')
    if any(FilePath(p.GetAttribute('filePath').Get().path).suffix.lower() in ('.usd','.usda','.usdc','.usdz') for p in fields):
        raise ValueError('A scene wrapper cannot be a NuRec filePath')
    return kind


def fixed_exposure(profile=None):
    """No auto exposure over a splat: the splat holds the frames' own light, and the histogram, set by the meshes in
    view, darkened them against it."""
    import carb

    carb.settings.get_settings().set("/rtx/post/histogram/enabled", False)
    # linear tone mapping: the splat's colours are the frames' own, Kit's default curve lifted them
    carb.settings.get_settings().set("/rtx/post/tonemap/op", 1)
    for path, value in (profile or {}).items():
        carb.settings.get_settings().set(path, value)


def add_background(stage, session, kind="mesh", cousin=""):
    """`kind` "splat": the splat background, the duplicated mesh visuals hidden; "mesh": nothing; "zone": the task
    zone alone on a plain floor (sim.zone); `cousin`: a splat
    cousin's background (recon.splat_cousin, scene/splat/cousins/<cousin>/) in place of the twin's. Returns the hidden
    prim paths."""
    if kind == "mesh":
        return []
    if kind == 'blender':
        from .environment_stage import add_environment
        return add_environment(stage, session, cousin or 'twin')
    if kind == "zone":  # the task zone alone (sim.zone)
        from .zone_stage import add_zone_background

        return add_zone_background(stage, session)
    if kind != "splat":
        raise ValueError(f"background {kind!r}: mesh, splat, blender or zone")
    fixed_exposure()
    s = session.session
    d = s / "scene" / "splat" / ("cousins/" + cousin if cousin else "")
    if not (d / "background.usdz").exists():
        raise SystemExit(f"no splat background in {d}: run_session.sh {s.dir.name} --splat (recon.splat_export)")
    import omni.kit.app

    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(NUREC_SCHEMA, True)
    asset = Usd.Stage.Open(str((d/'background.usdz').resolve()))
    profile = photometric_profile(dict(asset.GetRootLayer().customLayerData))
    M = np.asarray(json.loads((d / "transform.json").read_text())["matrix"], float)
    xf = UsdGeom.Xform.Define(stage, PATH)
    UsdGeom.Xformable(xf).AddTransformOp().Set(Gf.Matrix4d(M.T.tolist()))  # USD: row vectors
    xf.GetPrim().GetReferences().AddReference(str((d / "background.usdz").resolve()))
    if not session.meshed_structures:  # only what the task touches is a mesh: the rest of the room is the splat's alone --
        # the task zone's meshes as background zone builds them, its floor a collider the camera does not see, its
        # dome light and fixed exposure (the user, 2026-10-27)
        from .zone_stage import add_zone_background

        hidden = add_zone_background(stage, session, floor_visible=False)
        fixed_exposure(profile)
        metadata = dict(stage.GetRootLayer().customLayerData)
        metadata[PROFILE_KEY] = profile
        stage.GetRootLayer().customLayerData = metadata
        return hidden
    fixed_exposure(profile)
    metadata = dict(stage.GetRootLayer().customLayerData)
    metadata[PROFILE_KEY] = profile
    stage.GetRootLayer().customLayerData = metadata
    hide = set(session.settings["hide"])
    inside = session.inside
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    hidden = []

    def away(prim):
        r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        return r.IsEmpty() or not inside(np.asarray([(np.asarray(r.GetMin()) + np.asarray(r.GetMax())) / 2]))[0]

    for name, part in s.manifest["parts"].items():
        n = np.asarray(part.get("normal", [0, 0, 0]), float)
        role = "ceiling" if n[2] < -0.7 else "walls" if part.get("kind") == "structure" and abs(n[2]) < 0.3 else None
        prim = stage.GetPrimAtPath(f"/World/Room/{name}")
        if role in hide and prim.IsValid():
            hidden.append(prim)
    for root, key in (("/World/Static", "static"), ("/World/Synthetic", "synthetic")):
        top = stage.GetPrimAtPath(root)
        if key in hide and top.IsValid():
            hidden += [p for p in top.GetChildren() if p.IsA(UsdGeom.Imageable) and "_collider" not in p.GetName()
                       and away(p)]
    for p in hidden:  # hidden from the camera alone: they still bounce the lights onto the task zone
        UsdGeom.PrimvarsAPI(p).CreatePrimvar("hideForCamera", Sdf.ValueTypeNames.Bool,
                                             UsdGeom.Tokens.constant).Set(True)
    return [str(p.GetPath()) for p in hidden]
