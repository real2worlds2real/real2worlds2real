"""Blender-only environment assembler. Keeps entities, PBR materials and lights."""
import importlib.util
import json
import math
from pathlib import Path
import sys

import bpy


def apply_surface_atlases(context):
    """Attach capture radiance without silently replacing the agent's shape."""
    import hashlib
    import numpy as np
    from mathutils.bvhtree import BVHTree
    from mathutils.kdtree import KDTree
    def sha(path):
        with open(path,'rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    def geometry(objects):
        vertices=[];faces=[];offset=0;bpy.context.view_layer.update();deps=bpy.context.evaluated_depsgraph_get()
        for obj in objects:
            evaluated=obj.evaluated_get(deps);mesh=evaluated.to_mesh();mesh.calc_loop_triangles()
            v=np.array([obj.matrix_world@p.co for p in mesh.vertices],float)
            f=np.array([tuple(t.vertices) for t in mesh.loop_triangles],int)
            vertices.append(v);faces.append(f+offset);offset+=len(v);evaluated.to_mesh_clear()
        return np.concatenate(vertices),np.concatenate(faces)
    def compare(first,second):
        vertex_delta=0.;surface_delta=0.
        for (V,F),(W,G) in [(first,second),(second,first)]:
            points=KDTree(len(W))
            for i,p in enumerate(W):points.insert(tuple(p),i)
            points.balance()
            for point in V:
                distance=points.find(tuple(point))[2];vertex_delta=max(vertex_delta,distance)
                if distance>2e-5:raise ValueError('Surface atlas vertices changed from the agent mesh')
            tree=BVHTree.FromPolygons(W.tolist(),G.tolist(),all_triangles=True)
            samples=V[F].mean(1)
            for point in samples:
                nearest=tree.find_nearest(tuple(point))
                surface_delta=max(surface_delta,nearest[3])
                # Native USD export and Blender can choose opposite diagonals
                # on slightly nonplanar bevel quads. Vertex positions must still
                # match; bound and report this surface-only triangulation delta.
                if nearest[0] is None or nearest[3]>.0005:
                    raise ValueError('Surface atlas geometry is stale or changed from the agent mesh: '
                                     +str({'sample':point.tolist(),'nearest':list(nearest[0]) if nearest[0] else None,
                                           'distance':nearest[3],'source_bounds':[V.min(0).tolist(),V.max(0).tolist()],
                                           'atlas_bounds':[W.min(0).tolist(),W.max(0).tolist()]}))
        return {'vertex_delta_m':vertex_delta,'surface_delta_m':surface_delta,
                'vertex_tolerance_m':2e-5,'triangulation_tolerance_m':.0005}
    audits={}
    for entity,assets in context.get('surface_atlases',{}).items():
        if entity in context['reserved_entities']:raise ValueError('Surface atlas cannot replace protected task/support')
        proof=json.loads(Path(assets['evidence']).read_text())
        if proof.get('version')!=1 or proof.get('entity')!=entity:raise ValueError('Surface atlas belongs to another entity')
        for key in ['mesh','radiance']:
            if sha(assets[key])!=proof[key+'_sha256']:raise ValueError('Surface atlas payload changed')
        original=[o for o in bpy.context.scene.objects if o.type=='MESH' and o.get('r2s2r_entity')==entity]
        if not original or any(o.rigid_body for o in original):raise ValueError('Capture atlas requires native static background geometry with an external physics contract')
        before=geometry(original);objects=set(bpy.context.scene.objects)
        bpy.ops.wm.usd_import(filepath=assets['mesh'])
        added=[o for o in bpy.context.scene.objects if o not in objects]
        meshes=[o for o in added if o.type=='MESH']
        if not meshes or any(o.type not in ('MESH','EMPTY') for o in added):raise ValueError('Surface atlas mesh asset contains unexpected objects')
        audits[entity]=compare(before,geometry(meshes))
        audits[entity].update(observed_fraction=proof['observed_fraction'],appearance='source-lit capture; not relightable reflectance')
        # Imported geometry is in metric world coordinates; preserve its exact
        # placement while grouping the UV-split members as the same native body.
        parent=bpy.data.objects.new(entity+'_captured_surface',None);bpy.context.collection.objects.link(parent)
        parent['r2s2r_body_entity']=entity
        mat=bpy.data.materials.new(entity+'_captured_radiance');mat.use_nodes=True
        shader=mat.node_tree.nodes.get('Principled BSDF');shader.inputs['Base Color'].default_value=(0,0,0,1)
        shader.inputs['Roughness'].default_value=.95;shader.inputs['Specular IOR Level'].default_value=0.
        shader.inputs['Emission Strength'].default_value=1.
        image=mat.node_tree.nodes.new('ShaderNodeTexImage');image.image=bpy.data.images.load(assets['radiance'],check_existing=True)
        mat.node_tree.links.new(image.outputs['Color'],shader.inputs['Emission Color'])
        for obj in meshes:
            world=obj.matrix_world.copy();obj.parent=parent;obj.matrix_world=world
            obj['r2s2r_entity']=entity;obj.data.materials.clear();obj.data.materials.append(mat)
        for obj in original:bpy.data.objects.remove(obj,do_unlink=True)
        for obj in added:
            if obj.type=='EMPTY':bpy.data.objects.remove(obj,do_unlink=True)
    return audits


def main():
    job = json.loads(Path(sys.argv[sys.argv.index('--')+1]).read_text())
    out = Path(job['output'])
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    scene = bpy.context.scene
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.scale_length = 1.
    if job.get('engine_root'):
        sys.path.insert(0, job['engine_root'])
    sys.path.insert(0, job['source_root'])
    spec = importlib.util.spec_from_file_location('environment_model', job['model'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.build(job['parameters'], job['context'])
    atlases=apply_surface_atlases(job['context'])
    (out/'surface_atlases.json').write_text(json.dumps(atlases,indent=2)+'\n')
    rows = []
    meshes = [o for o in scene.objects if o.type == 'MESH' and not o.hide_render]
    if not meshes:
        raise ValueError('Environment needs visible agent-authored meshes')
    # Procedural material networks must be baked by the agent, rather than silently
    # exporting flat colors that appear photoreal only in a Blender preview.
    supported = {'ShaderNodeOutputMaterial', 'ShaderNodeBsdfPrincipled', 'ShaderNodeTexImage',
                 'ShaderNodeNormalMap', 'ShaderNodeUVMap', 'ShaderNodeMapping', 'ShaderNodeTexCoord',
                 'ShaderNodeSeparateColor', 'ShaderNodeSeparateRGB', 'ShaderNodeValue', 'ShaderNodeRGB', 'NodeReroute'}
    for obj in meshes:
        entity = obj.get('r2s2r_entity')
        if not isinstance(entity, str) or not entity or entity in job['context']['reserved_entities']:
            raise ValueError('Every background mesh needs an entity tag distinct from task/support: '+obj.name)
        if not all(math.isfinite(v) for row in obj.matrix_world for v in row):
            raise ValueError('Invalid environment transform: '+obj.name)
        deps = bpy.context.evaluated_depsgraph_get()
        evaluated = obj.evaluated_get(deps)
        mesh = evaluated.to_mesh()
        if not mesh.vertices or not all(math.isfinite(v) for p in mesh.vertices for v in p.co):
            raise ValueError('Empty/nonfinite environment geometry: '+obj.name)
        evaluated.to_mesh_clear()
        for material in obj.data.materials:
            if material and material.use_nodes:
                unsupported = {n.bl_idname for n in material.node_tree.nodes}-supported
                if unsupported:
                    raise ValueError('Bake unsupported material nodes before USD export: '+str(sorted(unsupported)))
                for node in material.node_tree.nodes:
                    if node.bl_idname != 'ShaderNodeBsdfPrincipled':continue
                    emission=node.inputs.get('Emission Color');strength=node.inputs.get('Emission Strength')
                    # Native USD connects an emission texture directly and drops
                    # the Principled multiplier. Silently accepting it produced
                    # clipped, featureless displays in the actual capture review.
                    if emission and emission.is_linked and strength and (
                            strength.is_linked or abs(strength.default_value-1.)>1e-7):
                        raise ValueError('Bake image-linked emission strength into its texture before USD export: '+material.name)
        body = obj.rigid_body
        if body:
            obj['r2s2r_physics'] = json.dumps({'type': body.type, 'mass': body.mass,
                'friction': body.friction, 'restitution': body.restitution, 'kinematic': body.kinematic})
        rows.append({'object': obj.name, 'entity': entity,
                     'materials': [m.name if m else None for m in obj.data.materials],
                     'physics': {'type': body.type, 'mass': body.mass, 'friction': body.friction,
                                 'restitution': body.restitution, 'kinematic': body.kinematic} if body else None})
    # Declare every image as a copied/baked source, not an external machine path.
    roots = [Path(job['source_root']).parent.resolve()]
    for image in bpy.data.images:
        if image.source == 'FILE' and image.users:
            p = Path(bpy.path.abspath(image.filepath)).resolve()
            if not any(p.is_relative_to(root) for root in roots):
                raise ValueError('Texture/HDR must be declared in environment assets or baked into this round: '+str(p))
            if not p.is_file():
                raise ValueError('Missing texture/HDR: '+str(p))
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 128
    optical = {}
    for material in bpy.data.materials:
        if not material.users or not material.use_nodes:
            continue
        shaders = [n for n in material.node_tree.nodes if n.bl_idname == 'ShaderNodeBsdfPrincipled']
        for shader in shaders:
            transmission = shader.inputs.get('Transmission Weight')
            if transmission is None:
                continue
            if transmission.is_linked:
                raise ValueError('Bake or explicitly adapt linked optical transmission before environment export: '+material.name)
            if transmission.default_value <= 0:
                continue
            if len(shaders) != 1:
                raise ValueError('Optical environment export requires an unambiguous surface shader: '+material.name)
            sockets = {'color':'Base Color','roughness':'Roughness','metallic':'Metallic','ior':'IOR','opacity':'Alpha'}
            if any(shader.inputs[key].is_linked for key in sockets.values()):
                raise ValueError('Explicit optical texture adaptation required; do not silently discard linked channels: '+material.name)
            optical[material.name] = {key:(list(shader.inputs[socket].default_value[:3]) if key=='color'
                                          else float(shader.inputs[socket].default_value)) for key,socket in sockets.items()}
            optical[material.name]['transmission'] = float(transmission.default_value)
    (out/'material_optics.json').write_text(json.dumps(optical, indent=2)+'\n')
    bpy.ops.wm.save_as_mainfile(filepath=str(out/'environment.blend'))
    result = bpy.ops.wm.usd_export(filepath=str(out/'environment.usdc'), root_prim_path='/Environment',
        export_animation=False, export_materials=True, export_textures=True, relative_paths=True,
        export_lights=True, export_cameras=False, export_custom_properties=True,
        generate_preview_surface=True, convert_world_material=True, convert_scene_units='METERS',
        meters_per_unit=1., use_instancing=False, triangulate_meshes=True, evaluation_mode='RENDER')
    if 'FINISHED' not in result:
        raise ValueError('Native Blender USD export failed')
    (out/'entities.json').write_text(json.dumps(rows, indent=2)+'\n')


if __name__ == '__main__':
    main()
