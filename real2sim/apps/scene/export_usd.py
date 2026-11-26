"""The built scene (the twin, or a cousin: --tag, --variant) as one self-contained USD, with the robot (--robot) at the
base the views use (the real robot's when it is accepted, standpoints.view_bases) in its home posture.

    tools/run_isaac.sh apps/scene/export_usd.py sessions/<name> [--robot R | --no-robot] [--tag cousin_00 --variant ...]

Writes export/usd/<tag>/<name>_<tag>.usd (binary crate, flattened: the library models and the robot inlined) with its
textures beside it in textures/, referenced by relative paths, and export/<name>_<tag>_usd.zip of that folder. The
scene keeps its physics (colliders, rigid bodies, masses, the robot's drives and joint states), so it plays in Isaac
Sim as built; the robot's links are also written in the posture PhysX settles them in, for viewers without physics.
"""
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("session")
ap.add_argument('--configuration', help='Explicit configuration snapshot for a separately tagged export')
ap.add_argument("--robot", default=None, help="robot spec (default: the one the session's placements are for)")
ap.add_argument("--no-robot", action="store_true")
ap.add_argument("--tag", default="twin")
ap.add_argument("--variant", default=None, help="a cousin's json (default: the twin)")
ap.add_argument("--background", default="mesh", choices=["mesh", "splat", "zone", "blender"],
                help="splat: the twin's Gaussian-splat background (recon.splat_export, sim.splat)")
ap.add_argument("--environment-variant", default="twin", help="Background-only Blender variant from environment.json")
args = ap.parse_args()
if args.background == "blender" and args.variant:
    ap.error("Use --environment-variant for background cousins; task-changing --variant is incompatible")

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "renderer": "PathTracing" if args.background == 'blender' else "RaytracedLighting", "width": 1280, "height": 720})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom  # noqa: E402

from calibrated_scene_reconstruction.core.paths import resolve as repo_path  # noqa: E402
from calibrated_scene_reconstruction.scene.session import Session  # noqa: E402
from calibrated_scene_reconstruction.scene.capture.render_robot import capture_spec  # noqa: E402
from calibrated_scene_reconstruction.robots.viewpoints import coverage_score, is_photo_base, view_bases  # noqa: E402
from calibrated_scene_reconstruction.sim.robot_articulation import ROBOT_PATH, Robot, add_robot  # noqa: E402
from calibrated_scene_reconstruction.robots.spec import load_spec  # noqa: E402
from calibrated_scene_reconstruction.sim.runtime import grab, rgb_annotator  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.variants import photo_goal, prepare_variant  # noqa: E402
from calibrated_scene_reconstruction.scene.twin.runtime_inputs import prepare_background, prepare_scene  # noqa: E402
from calibrated_scene_reconstruction.sim.scene_stage import build_scene, configure_render  # noqa: E402
from calibrated_scene_reconstruction.sim.splat_stage import PATH as SPLAT_PATH, add_background  # noqa: E402
from calibrated_scene_reconstruction.sim.sweep import RobotSweep, clear_robot, summarize, support_collisions  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import add_camera  # noqa: E402
from calibrated_scene_reconstruction.scene.objects.mechanisms import declaration as mechanism_declaration, sampling_tolerances  # noqa: E402
from calibrated_scene_reconstruction.sim.mechanism_stage import passive_audit  # noqa: E402
from calibrated_scene_reconstruction.scene.simulation import settings as simulation_settings  # noqa: E402

configure_render()
s = Session(args.session,args.configuration)
mechanisms = mechanism_declaration(s)
if mechanisms and s.config('task.json', {}).get('mechanism_contract') and simulation_settings(s) is None:
    raise ValueError('Declare simulation.json before mechanism delivery; checking and USD/Z must share a timestep/contact model')
tag = args.tag + ({"splat": "_splat", "zone": "_zone", "blender": "_blender"}.get(args.background, ""))
out = (s / "export" / "usd" / tag).resolve()
if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)

world = World(stage_units_in_meters=1.0, physics_dt=1 / 120, rendering_dt=1 / 30)
stage = omni.usd.get_context().get_stage()
variant = json.loads(Path(args.variant).read_text()) if args.variant else {}
ts = build_scene(stage, prepare_scene(s), photo_goal(s.dir, prepare_variant(s.dir, variant)))
if mechanisms:
    s.write_json(f'qa/passive_export_{tag}.json', passive_audit(stage, mechanisms,sampling_tolerances=sampling_tolerances(s)))
add_background(stage, prepare_background(s.dir), args.background,
               args.environment_variant if args.background == "blender" else "")  # the duplicated mesh visuals hidden; the volume itself below
robot = None
if not args.no_robot:
    name = args.robot or next((p.stem for p in sorted((s / "placements").glob("*.json")) if p.stem != "placements"),
                              "vega_1p_sharpa")
    spec = capture_spec(s, load_spec(name))
    pl = s.read_json("placements/placements.json" if name == "vega_1p_sharpa" else f"placements/{name}.json")
    photo_base = (s.scene_graph.get("photo_robot") or {}).get("base")
    cands, warn = view_bases(pl["candidates"], photo_base)
    if warn:
        print(f"[warn] {warn}")
    bx, by, byaw = (cands[0] if is_photo_base(cands[0], photo_base) else max(cands, key=coverage_score))["base"]
    zt = float(s.scene_graph["furniture"][s.scene_graph["task"]["support"]]["box"][1][2])
    add_robot(stage, spec, (bx, by), np.radians(byaw), support_top=zt)
    robot = {"spec": name, "base": [bx, by, byaw], "real_base": bool(is_photo_base(cands[0], photo_base))}
    support_collisions(stage, False)
from calibrated_scene_reconstruction.sim.asset_composition import audit as composition_audit  # noqa: E402
composition = composition_audit(stage)
s.write_json(f'qa/export_composition_{tag}.json', composition)
if not composition['pass']:
    raise ValueError('Source asset has unresolved USD references; repair the asset before complete-scene export')
task_source = None
if stage.GetPrimAtPath('/World/Objects'):
    from calibrated_scene_reconstruction.sim.authored_task_state import AuthoredTaskState  # noqa: E402
    task_source = AuthoredTaskState(stage)
world.reset()
if robot:
    vega = Robot(world, spec)
    vega.teleport(vega.home())
    for _ in range(round(.75/world.get_physics_dt())):
        vega.command(vega.home())
        world.step(render=False)
    if pl.get('scope') != 'presentation_only':
        sweep = RobotSweep(stage, vega)
        sweep.sample()
        print(summarize(clear_robot(ts, sweep)), flush=True)
    support_collisions(stage, True)
    # the links as PhysX holds them (the USD's own link transforms are the zero pose; a viewer without physics shows those)
    tf = vega.link_transforms()
    body_names = list(vega.body_names)
    # Stop before authoring the exported pose: stopping can reset native state.
    world.stop()
    links = {p.GetName(): p for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT_PATH)) if p.IsA(UsdGeom.Xformable)}
    cache = UsdGeom.XformCache()
    for k, b in enumerate(body_names):  # parents before children: a link's parent is posed before it is read
        prim = links.get(b)
        if prim is None:
            continue
        x, y, z, qx, qy, qz, qw = (float(v) for v in tf[k])
        W = Gf.Matrix4d().SetRotate(Gf.Quatd(qw, qx, qy, qz)) * Gf.Matrix4d().SetTranslate(Gf.Vec3d(x, y, z))
        P = cache.GetLocalToWorldTransform(prim.GetParent())
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTransformOp().Set(W * P.GetInverse())
        cache.Clear()
elif task_source is not None:
    world.stop()
if task_source is not None:
    restored = task_source.restore(stage)
    s.write_json(f'qa/authored_task_export_{tag}.json', restored)
    # Runtime settling must not silently become the initial state of a new
    # package, especially for flexible surfaces with material rest geometry.
    if mechanisms:
        s.write_json(f'qa/passive_export_{tag}.json', passive_audit(
            stage, mechanisms, sampling_tolerances=sampling_tolerances(s)))

# one layer: references (library models, the robot) composed in, asset paths made relative to copies beside it.
# Flatten resolves a search-path asset to this machine's file: an MDL module of Kit's library (the robot's
# OmniPBR.mdl, a converted model's gltf/pbr.mdl) goes back to the name it was authored with, which every Kit finds
# (a copy beside the USD does not load: MDL takes "./materials/OmniPBR.mdl" for a module "::.::materials::OmniPBR")
searched = {}
for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
    for a in prim.GetAttributes():
        v = a.Get() if a.GetTypeName() == Sdf.ValueTypeNames.Asset else None
        if v and v.resolvedPath and v.path.endswith(".mdl") and not v.path.startswith(("/", "./", "../")):
            searched[v.resolvedPath] = v.path
context=dict(stage.GetRootLayer().customLayerData).get('r2s2r_environment_context')
has_splat=args.background=='splat' or bool(context)
if has_splat:  # a NuRec USDZ flattened would name files inside its package by this machine's path
    stage.RemovePrim(SPLAT_PATH)
layer = stage.Flatten()
copied = {}


def localize(path):
    """A texture an asset path names copied into textures/<hash>/ under its own name, its new relative path (None for
    an MDL module outside Kit's library: a copy beside the USD does not load, it is reported instead)."""
    src = Path(path)
    if not src.is_file() or src.suffix == ".mdl":
        return None
    if src not in copied:
        h = hashlib.sha1(str(src).encode()).hexdigest()[:8]
        copied[src] = f"./textures/{h}/{src.name}"
        (out / "textures" / h).mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out / copied[src][2:])
    return copied[src]


def anchored(ap):
    """The file an asset path names (absolute, or relative to the repository root as the scene writes its own
    textures), None for a search-path asset (Kit's OmniGlass.mdl)."""
    p = ap.resolvedPath or ap.path
    if not p:
        return None
    if Path(p).is_absolute():
        return p
    return str(repo_path(p)) if repo_path(p).is_file() else None


missing = set()


def visit(path):
    spec = layer.GetObjectAtPath(path)
    if not isinstance(spec, Sdf.AttributeSpec) or spec.typeName not in (Sdf.ValueTypeNames.Asset,
                                                                        Sdf.ValueTypeNames.AssetArray):
        return
    v = spec.default
    if isinstance(v, Sdf.AssetPath) and (v.resolvedPath or v.path) in searched:
        spec.default = Sdf.AssetPath(searched[v.resolvedPath or v.path])
    elif isinstance(v, Sdf.AssetPath):
        p = anchored(v)
        if p:
            new = localize(p)
            if new:
                spec.default = Sdf.AssetPath(new)
            else:
                missing.add(p)
    elif v is not None:
        spec.default = Sdf.AssetPathArray([Sdf.AssetPath(localize(anchored(a)) or a.path) if anchored(a) else a
                                           for a in v])


layer.Traverse(Sdf.Path.absoluteRootPath, visit)
if has_splat:  # the volume as a reference to its package beside the USD, placed by its transform
    field=Path(context['asset']) if context else s/'scene/splat/background.usdz'
    shutil.copy2(field, out / "background.usdz")
    spec = Sdf.CreatePrimInLayer(layer, SPLAT_PATH)
    spec.specifier, spec.typeName = Sdf.SpecifierDef, "Xform"
    spec.referenceList.Prepend(Sdf.Reference("./background.usdz"))
    M = np.asarray(json.loads(context['matrix_json']) if context else json.loads((s / "scene" / "splat" / "transform.json").read_text())["matrix"], float)
    op = Sdf.AttributeSpec(spec, "xformOp:transform", Sdf.ValueTypeNames.Matrix4d)
    op.default = Gf.Matrix4d(M.T.tolist())
    Sdf.AttributeSpec(spec, "xformOpOrder", Sdf.ValueTypeNames.TokenArray).default = ["xformOp:transform"]
usd = out / f"{s.dir.name}_{tag}.usd"
layer.Export(str(usd))
# opened on its own: every file asset is a copy beside it that resolves, or a module of Kit's MDL library
check = Usd.Stage.Open(str(usd))
if has_splat:
    field=check.GetPrimAtPath(SPLAT_PATH)
    if not field or not field.GetChildren():
        raise ValueError('Exported hybrid/GS scene cannot resolve its NuRec field')
values = [a.Get() for p in check.TraverseAll() if not str(p.GetPath()).startswith(SPLAT_PATH)  # its package's own
          for a in p.GetAttributes() if a.GetTypeName() == Sdf.ValueTypeNames.Asset and a.Get() and a.Get().path]
dangling = sorted({v.path for v in values if v.path not in searched.values() and "/" in v.path
                   and not (v.path.startswith("./textures/") and v.resolvedPath)})
n_prims = sum(1 for _ in check.TraverseAll())
meta = {"session": s.dir.name, "tag": tag, "background": args.background, "robot": robot, "up_axis": "Z", "meters_per_unit": 1.0,
        "prims": n_prims, "textures": len(copied), "unresolved_assets": len(missing), "dangling": len(dangling)}
(out / "export.json").write_text(json.dumps(meta, indent=1) + "\n")

zp = s / "export" / f"{s.dir.name}_{tag}_usd.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(out.rglob("*")):
        if f.is_file():
            z.write(f, Path(f"{s.dir.name}_{tag}") / f.relative_to(out))


# the file on its own, without physics, as a viewer shows it (the robot in the link transforms written above): from
# the photo camera and from above behind it, into the folder as preview_*.jpg
pc = s.manifest["photo_camera"]
T = np.asarray(pc["T_world_cam_cv"])
w = 1280
h = int(round(w * pc["height"] / pc["width"]))
K = np.asarray(pc["K"]) * np.array([[w / pc["width"]], [h / pc["height"]], [1]])
K[0, 0] = K[1, 1] = min(K[0, 0], K[1, 1])
sup = s.scene_graph["furniture"][s.scene_graph["task"]["support"]]["box"]
centre = (np.asarray(sup[0]) + np.asarray(sup[1])) / 2
fwd = T[:3, 2] * np.array([1, 1, 0]) / max(np.linalg.norm(T[:3, 2][:2]), 1e-6)
omni.usd.get_context().open_stage(str(usd.resolve()))  # by a relative path RTX found no ./textures (all black)
view = omni.usd.get_context().get_stage()
room = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(
    view.GetPrimAtPath("/World/Room")).ComputeAlignedRange()
eye = T[:3, 3] - 0.8 * fwd + np.array([0, 0, 0.5])
# A workspace-only scene has no room ceiling. Its floor bounding box must
# never lower an overview camera below the modeled table.
if args.background != 'zone' and room.GetMax()[2] > sup[1][2] + .5:
    eye[2] = min(eye[2], room.GetMax()[2] - 0.25)
view.SetEditTarget(view.GetSessionLayer())  # the cameras stay out of the exported file
fixed = args.background in ("splat", "zone")
configure_render(auto_exposure=not fixed)  # opening a stage resets renderer settings
if args.background == 'blender':
    from calibrated_scene_reconstruction.sim.environment_stage import apply_environment_render  # noqa: E402
    apply_environment_render(view)
if fixed:
    from calibrated_scene_reconstruction.sim.splat_stage import fixed_exposure  # noqa: E402

    fixed_exposure(dict(view.GetRootLayer().customLayerData).get('r2s2r_background_render'))
cams = {"photo": add_camera(view, "/Preview/photo", K, w, h, T_world_cam_cv=T),
        "above": add_camera(view, "/Preview/above", K, w, h, eye=eye, target=centre)}
anns = {k: rgb_annotator(c.GetPath(), w, h) for k, c in cams.items()}
for _ in range(200):  # the textures stream in after the stage opens: earlier frames showed them black or blank
    app.update()
rep.orchestrator.step(rt_subframes=24)  # the timeline is stopped: app updates alone deliver no annotator data
with zipfile.ZipFile(zp, "a") as z:
    for k, a in anns.items():
        img = grab(a)
        if img.size == 0:
            print(f"[warn] preview {k}: no image")
            continue
        Image.fromarray(img).save(out / f"preview_{k}.jpg", quality=90)
        z.write(out / f"preview_{k}.jpg", Path(f"{s.dir.name}_{tag}") / f"preview_{k}.jpg")
print(f"saved {usd} ({usd.stat().st_size / 1e6:.0f} MB, {len(copied)} textures, robot {robot}); {zp} "
      f"({zp.stat().st_size / 1e6:.0f} MB)", flush=True)
for p in sorted(missing)[:10]:
    print(f"[warn] asset not found, left as is: {Path(p).name}")
for p in dangling[:10]:
    print(f"[warn] does not resolve beside the USD: {p}")
app.close()
