"""Versioned agent-authored whole environments, separate from task-object loops.

python -m calibrated_scene_reconstruction.scene.background.blender_env SESSION {request,build,verify} [--variant NAME]
"""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

from calibrated_scene_reconstruction.core.paths import resolve
from calibrated_scene_reconstruction.scene.review.environment import digest, protected_state
from calibrated_scene_reconstruction.scene.session import Session


def label(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,63}', value):
        raise ValueError('Environment variant/entity names must be simple identifiers')
    return value


def merge(base, overrides):
    result = deepcopy(base)
    for key, value in overrides.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else deepcopy(value)
    return result


def render_profile(overrides=None):
    """Explicit display-exposure hypothesis; never inherit Kit's camera preset."""
    defaults = {'mode': 'pt', 'spp': 128, 'bounces': 8, 'auto_exposure': False,
        'photometry': {'tonemap_op': 6, 'film_iso': 100., 'exposure_time_s': 1., 'f_number': 1., 'cm2_factor': 1.}}
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError('Environment render profile must be a dictionary')
    result = merge(defaults, overrides or {})
    if (set(result) != set(defaults) or result['mode'] not in ('pt', 'rt')
            or type(result['auto_exposure']) is not bool
            or type(result['spp']) is not int or not 1 <= result['spp'] <= 4096
            or type(result['bounces']) is not int or not 1 <= result['bounces'] <= 64):
        raise ValueError('Invalid environment rendering mode/samples/bounces/exposure policy')
    p = result['photometry']
    if not isinstance(p, dict) or set(p) != set(defaults['photometry']):
        raise ValueError('Explicit known environment photometry fields required')
    if type(p['tonemap_op']) is not int or not 0 <= p['tonemap_op'] <= 7:
        raise ValueError('Known tone mapping operator required')
    for key in ('film_iso', 'exposure_time_s', 'f_number', 'cm2_factor'):
        value = p[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Finite positive environment photometry required: '+key)
    return result


def specification(session, variant='twin'):
    label(variant)
    cfg = session.config('environment.json', {})
    if cfg.get('version') != 1:
        raise ValueError('environment.json version1 required; no automatic room template fallback')
    chosen = {key: deepcopy(cfg.get(key, default)) for key, default in
              [('model', 'agent/environment/model.py'), ('parameters', {}), ('assets', {}), ('render', {}), ('physics', {}), ('splat_context', None), ('surface_atlases', {}), ('display_response',None), ('captured_photos',[]), ('completion',None), ('texture_scale_checks',[]), ('front_view_checks',[]), ('shape_profile_checks',[])]}
    if variant != 'twin':
        override = cfg.get('variants', {}).get(variant)
        if not isinstance(override, dict) or not override:
            raise ValueError('Declare a nonempty background-only variant in environment.json')
        if set(override)-{'model', 'parameters', 'assets', 'render', 'physics', 'splat_context', 'surface_atlases', 'display_response','captured_photos','completion','description','texture_scale_checks','front_view_checks','shape_profile_checks'}:
            raise ValueError('Environment variants may only override model, parameters, assets, render, physics, splat_context, surface_atlases, display_response and description')
        chosen = merge(chosen, override)
    for key in ('parameters', 'assets', 'physics'):
        if not isinstance(chosen[key], dict):
            raise ValueError('Environment parameters and assets must be dictionaries')
    # Reject NaN/Inf before writing portable JSON or starting Blender.
    json.dumps(chosen, allow_nan=False)
    from calibrated_scene_reconstruction.scene.background.appearance_plan import assembly_policy
    assembly_policy(chosen['parameters'],session)
    chosen['render'] = render_profile(chosen['render'])
    if chosen['completion'] is not None or chosen['splat_context'] is not None:
        from calibrated_scene_reconstruction.scene.background.completion import specification as complete_spec
        graph=session.scene_graph
        complete_spec(chosen['completion'], {graph.get('task',{}).get('support'), *graph.get('dynamic_objects',{})})
    model = (session.config_dir/chosen['model']).resolve()
    root = (session.config_dir/'agent/environment').resolve()
    if not model.is_relative_to(root) or model.suffix != '.py':
        raise ValueError('Environment source belongs in configs/<session>/agent/environment/')
    chosen['model_path'] = model
    chosen['source_root'] = root
    return chosen


def protected_inputs(session):
    """Bind current task/support models and legal calibration without using actual-size evaluation."""
    graph = session.scene_graph
    support = graph.get('task', {}).get('support')
    specs = [*graph.get('dynamic_objects', {}).values(), graph.get('furniture', {}).get(support, {})]
    paths = [session.config_dir/name for name in ('twin.json', 'scene_graph.json', 'task.json', 'robot_state.json', 'robot_visual.json')]
    paths += [session/name for name in ('parsing/posed_frames/cameras.npz', 'splat/align.json')]
    for spec in specs:
        fix = spec.get('fix') or {}
        paths += [resolve(fix[k]) for k in ('mesh', 'visual_glb') if fix.get(k)]
        if fix.get('model'):
            paths.append(session.config_dir/fix['model'])
    return {str(p.resolve()): digest(p) for p in paths if p.is_file()}


def request(session):
    if (session/'qa/independent_freeze.json').exists():
        raise ValueError('Frozen delivery: create a derivative environment session')
    graph = session.scene_graph
    protected = sorted(set(graph.get('dynamic_objects', {})) | {graph.get('task', {}).get('support')} - {None})
    result = {'version': 1, 'route': 'agent_blender_environment',
              'source': str(session.config_dir/'agent/environment/model.py'),
              'config': str(session.config_dir/'environment.json'),
              'interface': 'build(parameters, context): assemble Blender scene in metric world coordinates; no return required.',
              'entity_tag': 'Every render mesh has object["r2s2r_entity"]; context["reserved_entities"] must not be rebuilt.',
              'reserved_entities': protected,
              'assets': 'Declare textures/HDRs as environment.json assets; context["assets"] maps IDs to immutable snapshot paths.',
              'quality': 'Whole-scene photorealism requires source/matched/novel comparisons, PBR texture scale, coherent lighting, contact shadows, occlusion and layout. Export success is not acceptance.',
              'appearance_policy': 'Default: blurry, ghosted, incomplete or unverified GS routes to agent-authored mesh, never coverage-only acceptance. Retain GS only with source and free-view reliability proof. Agent must complete unseen structural parts with explicit inference provenance.',
              'assembly_policy': 'Declare required structural subassemblies and actual mesh part names. Unobserved parts are completion work, not permission to omit them. Open structures need source evidence.',
              'completion': 'Hybrid environments require an agent-inferred footprint, floor, walls and ceiling, explicit observed/inferred provenance, visible native gap fillers, and a mandatory ray audit of actual exported enclosure triangles before presentation/delivery.',
              'scope': 'Background only. Preserve task, support, robot, cameras and calibration. No actual task/table dimensions or task CAD; no autonomous robot execution.',
              'cousins': 'Named parameter/asset/model overrides may change background shape, materials, physical properties and layout. Each needs a separate whole-scene review.'}
    dest = session/'qa/environment_request.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2)+'\n')
    return result


def bind_render_profile(asset, profile):
    """A standalone environment must carry the same exposure as scene assembly.

    Without this metadata the camera review used the GS/default camera preset,
    which can make an otherwise lit mesh asset appear black. This records the
    declared display hypothesis; it neither changes lights nor approves them.
    """
    from pxr import Usd
    stage=Usd.Stage.Open(str(asset))
    if not stage or not stage.GetDefaultPrim():
        raise ValueError('A standalone environment needs a valid default prim')
    data=dict(stage.GetRootLayer().customLayerData)
    data['r2s2r_environment_render']=render_profile(profile)
    stage.GetRootLayer().customLayerData=data
    stage.GetRootLayer().Save()


def constant_world_lights(asset):
    """Keep uniform world radiance without a tiny exporter-generated HDR texture.

    RTX rejected the4x4 constant HDR emitted by Blender in actual scene exports.
    A constant dome color is the same light and needs no GPU image upload.
    Nonuniform captured HDR illumination is retained unchanged.
    """
    import cv2
    import numpy as np
    from pxr import Gf,Usd,UsdLux
    stage=Usd.Stage.Open(str(asset));changed=[]
    for prim in stage.Traverse():
        if not prim.IsA(UsdLux.DomeLight):continue
        dome=UsdLux.DomeLight(prim);texture=dome.GetTextureFileAttr().Get()
        if not texture or not texture.resolvedPath:continue
        image=cv2.imread(texture.resolvedPath,cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim!=3 or image.shape[2]!=3 or not np.isfinite(image).all():continue
        if np.max(np.ptp(image.reshape(-1,3),axis=0))>1e-6:continue
        rgb=image[0,0,::-1]*np.asarray(dome.GetColorAttr().Get(),float)
        dome.GetColorAttr().Set(Gf.Vec3f(*map(float,rgb)))
        dome.GetTextureFileAttr().Set('')
        changed.append(str(prim.GetPath()))
    if changed:stage.GetRootLayer().Save()
    return changed


def zero_specular_emission(asset):
    """Retain an explicitly reflection-free captured-radiance material in USD.

    Blender writes a nonstandard PreviewSurface `specular` scalar that RTX does
    not use in its default metalness workflow. The remaining IOR=1.5 reflected
    native lights over baked appearance, bleaching dark captured surfaces.
    Only image-linked emission with black diffuse, zero authored specular and
    zero clearcoat/metalness qualifies; other PBR materials are retained.
    """
    from pxr import Gf,Sdf,Usd,UsdShade
    stage=Usd.Stage.Open(str(asset));changed=[]
    for prim in stage.Traverse():
        if not prim.IsA(UsdShade.Shader):continue
        shader=UsdShade.Shader(prim)
        if shader.GetIdAttr().Get()!='UsdPreviewSurface':continue
        emission=shader.GetInput('emissiveColor');base=shader.GetInput('diffuseColor');specular=shader.GetInput('specular')
        if not (emission and emission.HasConnectedSource() and base and tuple(base.Get() or ())==(0.,0.,0.)
                and specular and specular.Get()==0.):continue
        if any(shader.GetInput(k) and (shader.GetInput(k).Get() or 0.)!=0. for k in ['clearcoat','metallic']):continue
        shader.CreateInput('ior',Sdf.ValueTypeNames.Float).Set(1.)
        shader.CreateInput('useSpecularWorkflow',Sdf.ValueTypeNames.Int).Set(1)
        shader.CreateInput('specularColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.))
        changed.append(str(prim.GetPath()))
    if changed:stage.GetRootLayer().Save()
    return changed


def build(session, variant='twin'):
    from calibrated_scene_reconstruction.scene.capture.inputs import validate_session
    validate_session(session)
    request(session)
    spec = specification(session, variant)
    if not spec['model_path'].is_file():
        raise ValueError('Agent environment source missing; see qa/environment_request.json')
    root = session/'scene/environment'/label(variant)
    root.mkdir(parents=True, exist_ok=True)
    rounds = [int(p.name[6:]) for p in root.glob('round_*') if p.name[6:].isdigit()]
    out = root/f'round_{max(rounds, default=0)+1}'
    out.mkdir()
    inputs = protected_inputs(session)
    inputs[str(Path(__file__).resolve())]=digest(Path(__file__))
    # Agent-authored rectangular members may use the shared explicit section
    # frame. Keep its implementation bound along with the authoring source.
    from calibrated_scene_reconstruction.scene.background import structural_axes, appearance_plan
    for module in (structural_axes,appearance_plan):
        inputs[str(Path(module.__file__).resolve())]=digest(module.__file__)
    config = session.config_dir/'environment.json'
    inputs[str(config.resolve())] = digest(config)
    source = out/'source'
    source.mkdir()
    for path in sorted(spec['source_root'].rglob('*.py')):
        if path.is_symlink():
            raise ValueError('Environment source snapshot must not contain symlinks')
        inputs[str(path.resolve())] = digest(path)
        target = source/path.relative_to(spec['source_root'])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    assets = {}
    for name, value in spec['assets'].items():
        label(name)
        path = resolve(value).resolve()
        inputs[str(path)] = digest(path)
        # Blender's USD exporter keys images by basename. Two logical assets
        # called radiance.hdr in different directories otherwise overwrite one
        # another in textures/, even though their source paths are distinct.
        filename=(name+'_'+path.name) if path.suffix.lower() in {'.png','.jpg','.jpeg','.hdr','.exr','.tif','.tiff','.webp'} else path.name
        target = out/'assets'/name/filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        assets[name] = str(target.resolve())
    protected = protected_state(session)
    from calibrated_scene_reconstruction.scene.background.observed_context import validate as validate_context
    observed_context = validate_context(session, spec, assets)
    if observed_context:
        proof=json.loads(Path(assets[spec['splat_context']['evidence']]).read_text())
        observed_context['coverage_rebuild_required']=proof.get('geometry_rebuild_required',False)
    display_response=None;response=None
    from calibrated_scene_reconstruction.scene.background.display_response import validate_profile
    if spec['display_response'] is not None:
        if spec['display_response'] not in assets:raise ValueError('Declare the display response as a logical asset')
        path=assets[spec['display_response']];response=json.loads(Path(path).read_text())
        validate_profile(response,spec['render'],observed_context)
        display_response={'asset':path,'sha256':digest(path),'profile':response['profile']}
    from calibrated_scene_reconstruction.scene.background import photo_assets
    inputs[str(Path(photo_assets.__file__).resolve())]=digest(photo_assets.__file__)
    converted_photos=photo_assets.convert(spec['captured_photos'],assets,response,
                                        display_response['asset'] if display_response else None,out/'photo_assets')
    atlases={}; partial_atlases={}
    if not isinstance(spec['surface_atlases'],dict):raise ValueError('Surface atlases need an entity-to-assets dictionary')
    for entity,bindings in spec['surface_atlases'].items():
        label(entity)
        if bindings is None:continue  # a shape-changing cousin may disable a base atlas
        if isinstance(bindings,dict) and bindings.get('attachment') == 'usd_partial':
            from calibrated_scene_reconstruction.scene.background import partial_atlases as partial_source
            partial_atlases[entity], partial_inputs = partial_source.prepare(
                session,entity,bindings,spec,assets,protected,observed_context)
            inputs.update(partial_inputs)
            continue
        if not isinstance(bindings,dict) or set(bindings)!={'mesh','radiance','evidence'} or any(key not in assets for key in bindings.values()):
            raise ValueError('Each surface atlas must declare mesh/radiance/evidence asset IDs')
        atlases[entity]={key:assets[value] for key,value in bindings.items()}
        proof=json.loads(Path(atlases[entity]['evidence']).read_text())
        fraction=proof.get('observed_fraction',0)
        if isinstance(fraction,bool) or not isinstance(fraction,(int,float)) or not .25<=fraction<=1:
            raise ValueError('Whole-mesh captured appearance requires at least25% observed surface; use PBR completion or refine the geometry first')
        if proof.get('display_response'):validate_profile(proof['display_response'],spec['render'],observed_context)
        camera=session/'parsing/posed_frames/cameras.npz'
        if (proof.get('protected_task_state')!=protected
                or proof.get('input_sha256',{}).get(str(camera.resolve()))!=digest(camera)):
            raise ValueError('Surface atlas belongs to different capture calibration/task state')
    context = {'assets': assets, 'units': 'meters', 'up_axis': 'Z',
               'reserved_entities': request(session)['reserved_entities'], 'surface_atlases':atlases}
    worker = Path(__file__).with_name('blender_environment_worker.py')
    shutil.copyfile(worker, out/'worker.py')
    inputs[str(worker.resolve())] = digest(worker)
    # Blender's isolated Python does not inherit the caller's PYTHONPATH.
    # Provide the pure geometry helper from this immutable round, not live code.
    geometry_helpers = out/'engine/calibrated_scene_reconstruction/scene/background'
    geometry_helpers.mkdir(parents=True)
    shutil.copyfile(structural_axes.__file__, geometry_helpers/'structural_axes.py')
    job = {'model': str((source/spec['model_path'].relative_to(spec['source_root'])).resolve()),
           'engine_root': str((out/'engine').resolve()),
           'source_root': str(source.resolve()), 'parameters': spec['parameters'], 'context': context,
           'output': str(out.resolve())}
    (out/'job.json').write_text(json.dumps(job, indent=2, allow_nan=False)+'\n')
    command = ['tools/run_blender.sh', '--python', str((out/'worker.py').resolve()), '--', str((out/'job.json').resolve())]
    with (out/'build.log').open('w') as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    for path, expected in inputs.items():
        if digest(path) != expected:
            raise ValueError('Build input changed: '+path)
    if protected != protected_state(session):
        raise ValueError('Protected task state changed during environment build')
    from calibrated_scene_reconstruction.scene.background import appearance_plan
    exported_entities=json.loads((out/'entities.json').read_text())
    appearance_plan.entity_scope(spec['parameters'],exported_entities)
    appearance_plan.material_continuity(spec['parameters'],exported_entities)
    appearance_plan.required_parts(spec['parameters'].get('assembly_requirements',{}),exported_entities)
    native = out/'environment.usdc'
    if not native.is_file():
        raise ValueError('Blender did not produce a native USD environment')
    # Native USD conversion is executed at the backend boundary; scene policy
    # fingerprints its source without importing simulator authoring services.
    from calibrated_scene_reconstruction.core.paths import ROOT
    import sys
    for relative in ('sim/environment_optics.py','sim/object_materials.py','sim/usd.py'):
        path=ROOT/'src/calibrated_scene_reconstruction'/relative
        inputs[str(path.resolve())]=digest(path)
    audit_path=out/'material_optics_audit.json'
    subprocess.run([sys.executable,'-m','calibrated_scene_reconstruction.sim.environment_optics',str(native),
                    str(out/'material_optics.json'),'--output',str(audit_path)],check=True)
    optics_audit=json.loads(audit_path.read_text())
    partial_audit={}
    if partial_atlases:
        partial_audit,partial_inputs=partial_source.apply(native,partial_atlases,out/'partial_photos')
        inputs.update(partial_inputs)
        (out/'partial_photos.json').write_text(json.dumps(partial_audit,indent=2)+'\n')
    from calibrated_scene_reconstruction.scene.background import texture_scale
    inputs[str(Path(texture_scale.__file__).resolve())]=digest(texture_scale.__file__)
    texture_audit=texture_scale.audit(native,spec['texture_scale_checks'],assets)
    (out/'texture_scale.json').write_text(json.dumps(texture_audit,indent=2)+'\n')
    if digest(texture_scale.__file__)!=inputs[str(Path(texture_scale.__file__).resolve())]:raise ValueError('Texture audit code changed during build')
    if texture_audit['pass'] is not True:raise ValueError('Exported texture UV scale failed; inspect '+str(out/'texture_scale.json'))
    from calibrated_scene_reconstruction.scene.background import front_views
    inputs[str(Path(front_views.__file__).resolve())]=digest(front_views.__file__)
    front_audit=front_views.audit(native,spec['front_view_checks'],session/'parsing/posed_frames/cameras.npz',session/'splat/data/images')
    inputs.update(front_audit['input_sha256'])
    (out/'front_views.json').write_text(json.dumps(front_audit,indent=2)+'\n')
    if front_audit['pass'] is not True:raise ValueError('Exported furniture front faces away from annotated source views; inspect '+str(out/'front_views.json'))
    constant_world=constant_world_lights(native)
    zero_specular=zero_specular_emission(native)
    bind_render_profile(native,spec['render'])
    from calibrated_scene_reconstruction.scene.background import shape_profiles
    shape_audit=shape_profiles.audit(native,spec['shape_profile_checks'])
    inputs.update(shape_audit['input_sha256'])
    (out/'shape_profiles.json').write_text(json.dumps(shape_audit,indent=2)+'\n')
    if shape_audit['pass'] is not True:
        raise ValueError('Exported background violates its declared part profiles; inspect '+str(out/'shape_profiles.json'))
    if any(digest(path) != sha for path,sha in inputs.items()):
        raise ValueError('Environment input changed during exported-surface audits')
    files = {str(p.resolve()): digest(p) for p in out.rglob('*') if p.is_file() and p.name != 'build.log'}
    manifest = {'version': 1, 'variant': variant, 'asset': str(native.resolve()),
                'render': spec['render'],
                'physics': spec['physics'], 'physics_required': True,'assembly_requirements':spec['parameters'].get('assembly_requirements',{}),
                'splat_context': observed_context,
                'completion': spec['completion'],
                'display_response':display_response,
                'captured_photos':converted_photos,
                'partial_photos':partial_audit,
                'material_optics_audit':optics_audit,
                'texture_scale_audit':texture_audit,
                'front_view_audit':front_audit,
                'shape_profile_audit':shape_audit,
                'gs_visibility_required':observed_context is not None,
                'constant_world_lights':constant_world,
                'zero_specular_emission':zero_specular,
                'input_sha256': inputs, 'output_sha256': files, 'protected_task_state': protected,
                'entities': json.loads((out/'entities.json').read_text()), 'photorealism_approved': False,
                'scope': 'Unapproved native USD environment; object geometry/export success cannot pass whole-scene review.'}
    path = out/'manifest.json'
    path.write_text(json.dumps(manifest, indent=2)+'\n')
    (root/'current.json').write_text(json.dumps({'path': str(path.resolve()), 'sha256': digest(path)}, indent=2)+'\n')
    return manifest


def verify(session, variant='twin'):
    root = session/'scene/environment'/label(variant)
    pointer = json.loads((root/'current.json').read_text())
    path = Path(pointer['path']).resolve()
    if not path.is_relative_to(root.resolve()) or digest(path) != pointer['sha256']:
        raise ValueError('Environment manifest changed')
    result = json.loads(path.read_text())
    if result['protected_task_state'] != protected_state(session):
        raise ValueError('Protected task state changed since environment build')
    for key in ('input_sha256', 'output_sha256'):
        for source, expected in result[key].items():
            if digest(source) != expected:
                raise ValueError('Environment source/asset changed: '+source)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session')
    parser.add_argument('command', choices=['request', 'build', 'verify'])
    parser.add_argument('--variant', default='twin')
    args = parser.parse_args()
    session = Session(args.session)
    result = request(session) if args.command == 'request' else globals()[args.command](session, args.variant)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
