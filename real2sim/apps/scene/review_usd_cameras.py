"""Render an existing USD at fixed captured and novel review cameras.

Uses the cameras.json made by the session review kit. Edits only the USD session
layer. By default no physics steps are allowed; registered startup hooks permit
one bounded step and records it. Original images are references only for
offset0; novel views have no corresponding ground-truth image.
"""
import argparse
import hashlib
import json
from pathlib import Path

from calibrated_scene_reconstruction.core.capabilities import entries, invoke

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('usd')
parser.add_argument('cameras')
parser.add_argument('--output', required=True)
parser.add_argument('--renderer', choices=['rt', 'pt'], default='rt', help='Use pt for native Blender environments; GS needs rt')
parser.add_argument('--background-field', help='Optional complete USD context asset or raw .nurec payload in the same COLMAP frame')
parser.add_argument('--render-settings', help='Optional JSON of explicit /rtx scalar settings for a controlled render ablation')
parser.add_argument('--material-proposal', help='Optional explicit numeric shader-input ablation in the anonymous layer; never an appearance approval')
parser.add_argument('--background-transform', help='Optional JSON matrix for a payload with an explicitly changed coordinate frame')
parser.add_argument('--context-only', action='store_true', help='Diagnostic: hide background meshes to audit GS preservation; never an approval render')
parser.add_argument('--context-empty', action='store_true', help='With --context-only, hide GS too, to identify pixels with actual field support')
parser.add_argument('--source-context', help='With --context-only, render the original same-capture GS as a preservation reference')
parser.add_argument('--context-background',nargs=3,type=float,help='Diagnostic-only GS background RGB for opacity/invariance measurement')
parser.add_argument('--context-disabled',action='store_true',help='Diagnostic: keep native meshes, disable GS to measure its actual visible contribution')
for flag, options in entries('render_options').items():
    parser.add_argument(flag, **options)
args = parser.parse_args()
usd = Path(args.usd).resolve()
cameras = Path(args.cameras).resolve()
manifest = json.loads(cameras.read_text())
out = Path(args.output).resolve()
out.mkdir(parents=True, exist_ok=False)

from calibrated_scene_reconstruction.sim.application import SimulationApp  # noqa: E402

W, H = manifest['width'], manifest['height']
app = SimulationApp({'headless': True, 'renderer': 'PathTracing' if args.renderer == 'pt' else 'RaytracedLighting', 'width': W, 'height': H})

import numpy as np  # noqa: E402
import carb  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image, ImageDraw, ImageOps  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdPhysics  # noqa: E402

from calibrated_scene_reconstruction.sim.runtime import grab, rgb_annotator  # noqa: E402
from calibrated_scene_reconstruction.sim.scene_stage import configure_render  # noqa: E402
from calibrated_scene_reconstruction.sim.splat_stage import fixed_exposure, override_background  # noqa: E402
from calibrated_scene_reconstruction.sim.usd import add_camera  # noqa: E402
from calibrated_scene_reconstruction.scene.review.cameras import view_name  # noqa: E402
from calibrated_scene_reconstruction.sim import material_proposal  # noqa: E402


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


inputs = {str(usd): digest(usd), str(cameras): digest(cameras),
          str(Path(__file__).resolve()): digest(__file__),
          str(Path(material_proposal.__file__).resolve()): digest(material_proposal.__file__)}
from calibrated_scene_reconstruction.core.paths import ROOT
for relative in entries('render_fingerprint').values():
    path = ROOT/relative
    inputs[str(path)] = digest(path)
inputs.update(manifest.get('input_sha256', {}))
for path, expected in inputs.items():
    if digest(path) != expected:
        raise ValueError('Review preparation input changed: '+path)
omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate('omni.usd.schema.omni_nurec_types', True)
omni.usd.get_context().open_stage(str(usd))
stage = omni.usd.get_context().get_stage()
stage.SetEditTarget(stage.GetSessionLayer())
material_ablation = None
if args.material_proposal:
    proposal_path = Path(args.material_proposal).resolve()
    inputs[str(proposal_path)] = digest(proposal_path)
    material_ablation = material_proposal.apply(stage, json.loads(proposal_path.read_text()))
omni.timeline.get_timeline_interface().stop()
review_hooks = [invoke('render_prepare', name, stage, args, app, rep, carb)
                for name in entries('render_prepare')]
# Replicator otherwise initializes/advances the physics timeline even in a
# camera-only review. Keep protected bodies fixed in an anonymous layer.
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        prim.CreateAttribute('physxArticulation:articulationEnabled',Sdf.ValueTypeNames.Bool).Set(False)
    if prim.IsA(UsdPhysics.Joint):
        UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
for hook in review_hooks:
    hook.start()


def world_transforms():
    cache=UsdGeom.XformCache()
    return {str(p.GetPath()):np.asarray(cache.GetLocalToWorldTransform(p),float) for p in stage.Traverse()
            if str(p.GetPath()).startswith('/World/') and UsdGeom.Xformable(p)}


before_transforms=world_transforms()
if args.context_empty and not args.context_only:raise ValueError('--context-empty requires --context-only')
if args.context_disabled:
    if args.context_only or args.source_context or args.background_field or args.background_transform or args.context_background:
        raise ValueError('GS-disabled composite cannot also change its field, context mode or background')
    root=stage.GetPrimAtPath('/World/SplatBackground')
    if not root:raise ValueError('GS-disabled diagnostic requires an existing composed field')
    UsdGeom.Imageable(root).MakeInvisible()
if args.context_background is not None and (not args.context_only or not np.isfinite(args.context_background).all() or not all(0<=v<=1 for v in args.context_background)):
    raise ValueError('Context background needs a context-only diagnostic and bounded linear RGB')
if args.source_context:
    if not args.context_only or args.context_empty or args.background_field or args.background_transform:
        raise ValueError('Source reference requires only --context-only, without other field overrides')
    source_context=Path(args.source_context).resolve()
    meta=dict(stage.GetRootLayer().customLayerData).get('r2s2r_environment_context') or {}
    if str(source_context)!=meta.get('source_asset') or digest(source_context)!=meta.get('source_asset_sha256'):
        raise ValueError('Source reference must be the original captured GS bound by this candidate')
    stage.GetPrimAtPath('/World/SplatBackground').GetReferences().SetReferences([Sdf.Reference(str(source_context))])
    inputs[str(source_context)]=digest(source_context)
if args.context_only:
    if not stage.GetPrimAtPath('/World/SplatBackground'):
        raise ValueError('Context diagnostic requires a composed GS background')
    for prim in stage.Traverse():
        if str(prim.GetPath()).startswith('/World/Environment/') and prim.IsA(UsdGeom.Mesh):
            UsdGeom.Imageable(prim).MakeInvisible()
    if args.context_empty:UsdGeom.Imageable(stage.GetPrimAtPath('/World/SplatBackground')).MakeInvisible()
background_field_kind=None
if args.background_field:
    field = Path(args.background_field).resolve()
    if not field.is_file():
        raise ValueError('Background field payload missing')
    background_field_kind=override_background(stage,field)
    inputs[str(field)] = digest(field)
if args.background_transform:
    transform_path = Path(args.background_transform).resolve()
    matrix = np.asarray(json.loads(transform_path.read_text())['matrix'], float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1]) or np.linalg.det(matrix[:3, :3]) <= 0):
        raise ValueError('Background transform must be a finite proper homogeneous matrix')
    root = UsdGeom.Xformable(stage.GetPrimAtPath('/World/SplatBackground'))
    root.ClearXformOpOrder()
    root.AddTransformOp(opSuffix='review').Set(Gf.Matrix4d(matrix.T.tolist()))
    inputs[str(transform_path)] = digest(transform_path)
for layer in stage.GetUsedLayers():
    if not layer.anonymous:
        path = Path(layer.realPath)
        if path.is_file():
            inputs[str(path.resolve())] = digest(path)
for prim in stage.Traverse():
    for attribute in prim.GetAttributes():
        value = attribute.Get()
        if isinstance(value, Sdf.AssetPath):
            resolved = value.resolvedPath
            if resolved:
                path = Path(resolved.split('[', 1)[0])
                if path.is_file():
                    inputs[str(path.resolve())] = digest(path)
configure_render(auto_exposure=False)
if stage.GetRootLayer().customLayerData.get('r2s2r_environment_render'):
    from calibrated_scene_reconstruction.sim.environment_stage import apply_environment_render  # noqa: E402
    apply_environment_render(stage)
else:
    fixed_exposure(dict(stage.GetRootLayer().customLayerData).get('r2s2r_background_render'))
render_settings = None
if args.context_background is not None:
    carb.settings.get_settings().set('/rtx/background/source/type',2)
    carb.settings.get_settings().set('/rtx/background/source/color',args.context_background)
if args.render_settings:
    settings_path = Path(args.render_settings).resolve()
    render_settings = json.loads(settings_path.read_text())
    if not isinstance(render_settings, dict) or any(not key.startswith('/rtx/') or
            not isinstance(value, (bool, int, float, str)) for key, value in render_settings.items()):
        raise ValueError('Render ablation settings must be explicit /rtx scalar values')
    for key, value in render_settings.items():
        carb.settings.get_settings().set(key, value)
    inputs[str(settings_path)] = digest(settings_path)
annotator = None
# Controlled field overrides may deliberately change field-internal transforms.
# The immutable task and native scene are checked before and after setup too.
setup_transforms=world_transforms()
if any(k not in setup_transforms or not np.allclose(v,setup_transforms[k],rtol=0,atol=1e-9)
       for k,v in before_transforms.items() if not k.startswith('/World/SplatBackground')):
    raise ValueError('Protected/native transforms changed while preparing review')
before_transforms=setup_transforms
rows = []
for anchor in manifest['anchors']:
    source = Path(anchor['source'])
    inputs[str(source)] = digest(source)
    if inputs[str(source)] != anchor['source_sha256']:
        raise ValueError('Original review image changed: '+str(source))
    K = np.asarray(anchor['K'], float).copy()
    K[0] *= W / anchor['source_size'][0]
    K[1] *= H / anchor['source_size'][1]
    if abs(K[0, 1]) > 1e-6 or abs(K[1, 0]) > 1e-6:
        raise ValueError('Rectify skewed capture intrinsics before USD camera review')
    paths = []
    for view in anchor['views']:
        T = np.eye(4)
        T[:3, :3] = view['R_world_cam']
        T[:3, 3] = view['centre']
        cam = add_camera(stage, '/Review/Camera', K, W, H, T_world_cam_cv=T)
        cam.CreateHorizontalApertureOffsetAttr(float((W/2-K[0, 2])/K[0, 0]*18.))
        cam.CreateVerticalApertureOffsetAttr(float((K[1, 2]-H/2)/K[1, 1]*18.))
        if annotator is None:
            annotator = rgb_annotator(cam.GetPath(), W, H)
        for _ in range(40):
            app.update()
        if not any(hook.capture_running for hook in review_hooks):
            rep.orchestrator.step(rt_subframes=24,delta_time=0.)
        for hook in review_hooks:
            hook.verify()
        path = out/(view_name(source.stem, view['offset_m'])+'.png')
        Image.fromarray(grab(annotator, render=app.update)).save(path)
        paths.append(path)
        print('[review.usd] '+path.name, flush=True)
    panel = Image.new('RGB', (640*(1+len(paths)), 390), (28, 28, 28))
    draw = ImageDraw.Draw(panel)
    panel.paste(ImageOps.pad(Image.open(source).convert('RGB'), (640, 360)), (0, 30))
    draw.text((8, 8), 'Original: '+source.name, fill='white')
    for i, (path, view) in enumerate(zip(paths, anchor['views']), 1):
        panel.paste(ImageOps.pad(Image.open(path), (640, 360)), (640*i, 30))
        label = view.get('label') or ('Matched captured camera' if view['offset_m'] == 0 else f'Novel +{view["offset_m"]:.2f}m (no GT)')
        draw.text((640*i+8, 8), label, fill='white')
    panel.save(out/(source.stem+'_comparison.jpg'), quality=95)
    rows.append({'source': str(source), 'renders': [str(p) for p in paths]})
for path, expected in inputs.items():
    if digest(path) != expected:
        raise ValueError('Review input changed: '+path)
after_transforms=world_transforms()
extra_report = {}
for hook in review_hooks:
    extra_report.update(hook.finish(out))
if any(k not in after_transforms or not np.allclose(v,after_transforms[k],rtol=0,atol=1e-9) for k,v in before_transforms.items()):
    raise ValueError('Scene transforms changed during camera-only review')
(out/'review.json').write_text(json.dumps({'scope': 'Existing USD, read only; physics not stepped; novel views unpaired.',
    'usd':str(usd),
    'camera_manifest': str(cameras),
    'diagnostic':'context_disabled' if args.context_disabled else ('context_source' if args.source_context else 'context_empty' if args.context_empty else 'context_only') if args.context_only else dict(stage.GetRootLayer().customLayerData).get('r2s2r_diagnostic'),
    'source_context_override':str(Path(args.source_context).resolve()) if args.source_context else None,
    'effective_photometry':{key:carb.settings.get_settings().get(key) for key in [
        '/rtx/post/histogram/enabled','/rtx/post/tonemap/op','/rtx/post/tonemap/filmIso',
        '/rtx/post/tonemap/exposureTime','/rtx/post/tonemap/fNumber','/rtx/post/tonemap/cm2Factor',
        '/rtx/post/tonemap/enableSrgbToGamma','/rtx/post/tonemap/colorMode']},
    'runtime_lighting':{'indirect_diffuse':carb.settings.get_settings().get('/rtx/indirectDiffuse/enabled')},
    'environment': dict(stage.GetRootLayer().customLayerData).get('r2s2r_environment'),
    'environment_context': dict(stage.GetRootLayer().customLayerData).get('r2s2r_environment_context'),
    'static_scene_preserved':True,
    **extra_report,
    'background_field_override': str(Path(args.background_field).resolve()) if args.background_field else None,
    'background_field_kind':background_field_kind,
    'background_world_matrix':np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath('/World/SplatBackground'))).T.tolist() if stage.GetPrimAtPath('/World/SplatBackground') else None,
    'context_background_override':args.context_background,
    'render_settings': render_settings,
    'material_proposal': material_ablation,
    'background_transform_override': str(Path(args.background_transform).resolve()) if args.background_transform else None,
    'input_sha256': inputs, 'views': rows}, indent=2)+'\n')
app.close()
