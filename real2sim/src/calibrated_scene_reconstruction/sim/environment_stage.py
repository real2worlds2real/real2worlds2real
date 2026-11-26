"""Compose native Blender background assets without augmenting the task scene."""
import json

from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


def native_face_culling(stage, context):
    required=any(p.get('one_sided_closed') is True
                 for group in context.get('gap_masks',{}).values()
                 for p in (group.get('meshes') or {'mesh':group}).values())
    if required:
        for prim in stage.Traverse():
            if (prim.IsA(UsdGeom.Mesh) and not str(prim.GetPath()).startswith('/World/Environment/')
                    and prim.GetAttribute('singleSided').Get() is True):
                raise ValueError('Background culling would change an external singleSided mesh: '+str(prim.GetPath()))
    return required


def apply_environment_render(stage):
    profile = dict(stage.GetRootLayer().customLayerData).get('r2s2r_environment_render')
    if not profile:
        return False
    import carb
    from .scene_stage import configure_render
    carb.settings.get_settings().set('/rtx/rendermode', 'PathTracing' if profile['mode'] == 'pt' else 'RaytracedLighting')
    configure_render(mode=profile['mode'], spp=int(profile['spp']), bounces=int(profile['bounces']),
                     auto_exposure=bool(profile['auto_exposure']))
    carb.settings.get_settings().set('/rtx/post/histogram/enabled', bool(profile['auto_exposure']))
    names = {'tonemap_op': 'op', 'film_iso': 'filmIso', 'exposure_time_s': 'exposureTime',
             'f_number': 'fNumber', 'cm2_factor': 'cm2Factor'}
    for key, value in profile.get('photometry', {}).items():
        carb.settings.get_settings().set('/rtx/post/tonemap/'+names[key], value)
    context = dict(stage.GetRootLayer().customLayerData).get('r2s2r_environment_context')
    if context:
        from .render_profile import photometric_profile
        from .splat_stage import fixed_exposure
        fixed_exposure(photometric_profile(json.loads(context['source_layer_data_json'])))
        carb.settings.get_settings().set('/rtx/background/source/type',2)
        carb.settings.get_settings().set('/rtx/background/source/color',json.loads(context['background_color_linear_json']))
        carb.settings.get_settings().set('/rtx/hydra/faceCulling/enabled',bool(context.get('native_face_culling',False)))
    return True


def add_environment(stage, inputs, variant='twin'):
    from .zone_stage import keep_task_zone

    result = inputs.environment(variant)
    asset = Usd.Stage.Open(result['asset'])
    if (not asset or not asset.GetDefaultPrim() or UsdGeom.GetStageUpAxis(asset) != 'Z'
            or abs(UsdGeom.GetStageMetersPerUnit(asset)-1.) > 1e-9):
        raise ValueError('Environment USD needs a default prim and metric Z-up coordinates')
    hidden = keep_task_zone(stage, inputs)
    # Lighting comes from the agent-authored environment, including its HDR/world.
    lights = stage.GetPrimAtPath('/World/Lights')
    if lights:
        lights.SetActive(False)
        hidden.append(str(lights.GetPath()))
    root = UsdGeom.Xform.Define(stage, '/World/Environment').GetPrim()
    if not root.GetReferences().AddReference(result['asset']):
        raise ValueError('Could not reference native environment asset')
    reserved = set(inputs.session.scene_graph.get('dynamic_objects', {}))
    reserved.add(inputs.session.scene_graph.get('task', {}).get('support'))
    for prim in Usd.PrimRange(root):
        entity = prim.GetAttribute('userProperties:r2s2r_entity').Get()
        if entity in reserved:
            raise ValueError('Environment duplicates protected task/support entity: '+str(entity))
        encoded = prim.GetAttribute('userProperties:r2s2r_physics').Get()
        if not encoded:
            continue
        physical = json.loads(encoded)
        for mesh in Usd.PrimRange(prim):
            if mesh.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(mesh)
                if physical['type'] == 'ACTIVE':
                    UsdPhysics.MeshCollisionAPI.Apply(mesh).CreateApproximationAttr('convexHull')
        if physical['type'] == 'ACTIVE':
            body = UsdPhysics.RigidBodyAPI.Apply(prim)
            body.CreateKinematicEnabledAttr(bool(physical['kinematic']))
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(physical['mass']))
        mat_path = Sdf.Path('/World/EnvironmentPhysics').AppendChild(prim.GetName())
        mat = UsdShade.Material.Define(stage, mat_path)
        properties = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
        properties.CreateStaticFrictionAttr(float(physical['friction']))
        properties.CreateDynamicFrictionAttr(float(physical['friction']))
        properties.CreateRestitutionAttr(float(physical['restitution']))
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, materialPurpose='physics')
    if result.get('physics_required'):
        # Surroundings must also collide in the delivered asset. Export them as
        # static scene geometry; a separate mandatory audit releases copies under
        # gravity, so a static flag cannot conceal floating/unstable assemblies.
        declared = result.get('physics', {}).get('entities', {})
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            ancestor, entity = prim, None
            while ancestor and ancestor != root and not entity:
                entity = ancestor.GetAttribute('userProperties:r2s2r_entity').Get()
                ancestor = ancestor.GetParent()
            if entity not in declared:
                raise ValueError('Background mesh needs a physical support declaration: '+str(entity))
            UsdPhysics.CollisionAPI.Apply(prim)
    metadata = dict(stage.GetRootLayer().customLayerData)
    metadata['r2s2r_environment'] = {'variant': variant, 'asset': result['asset'],
        'asset_sha256': result['output_sha256'][result['asset']], 'photorealism_approved': False}
    metadata['r2s2r_environment_render'] = result['render']
    if result.get('completion') or result.get('splat_context'):
        metadata['r2s2r_environment_completion'] = {'required':True,
            'session':str(inputs.session.dir.resolve()),'variant':variant}
    context=result.get('splat_context')
    if context:
        from .context_stage import add_observed_context
        hidden += add_observed_context(stage,root,context)
        metadata['r2s2r_environment_context'] = {
            'asset':context['asset'],'asset_sha256':context['asset_sha256'],
            'evidence_sha256':context['evidence_sha256'],
            'source_asset':context.get('source_asset') or '',
            'source_asset_sha256':context.get('source_asset_sha256') or '',
            'source_background_scope_json':json.dumps(context.get('source_background_scope',{})),
            'native_light_gain':float(context.get('native_light_gain',1.)),
            'native_face_culling':native_face_culling(stage,context),
            'background_color_linear_json':json.dumps(context.get('background_color_linear',[0.,0.,0.])),
            'matrix_json':json.dumps(context['matrix']),
            'native_entities_json':json.dumps(context['native_entities']),
            'source_layer_data_json':json.dumps(context['source_layer_data'])}
    stage.GetRootLayer().customLayerData = metadata
    apply_environment_render(stage)
    return hidden
