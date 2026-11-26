"""Blender-only worker. Subprocess pattern adapted from real2sim blender_harness.py; no external imports.

Read JSON requests on stdin, write one atomic response per request. A worker evaluates many parameter sets.
"""
import importlib.util
import json
import sys
import traceback
from pathlib import Path

# Blender's bundled Python ignores PYTHONPATH unless explicitly enabled. Add
# only this checkout's source, never another environment's binary packages.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

import bpy
import numpy as np
from mathutils import Vector


def build(request):
    out = Path(request['output'])
    out.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    spec = importlib.util.spec_from_file_location('object_model', request['model'])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.build(request['params'])
    objects = sorted((o for o in bpy.context.scene.objects if o.type == 'MESH'), key=lambda o: o.name)
    if not objects:
        raise ValueError('build(params) made no meshes')
    source_parts = [o.name for o in objects]
    if request.get('artifacts'):
        # Preserve exact authored solids before their same-body Boolean union.
        # Runtime may use these only after checking closed convex ownership;
        # this is not permission to replace concave parts by bounding hulls.
        source_v=[];source_f=[];source_ranges=[];vi=fi=0
        deps=bpy.context.evaluated_depsgraph_get()
        for obj in objects:
            evaluated=obj.evaluated_get(deps);mesh=evaluated.to_mesh();mesh.calc_loop_triangles()
            v=np.array([tuple(evaluated.matrix_world@p.co) for p in mesh.vertices],float)
            f=np.array([tuple(t.vertices) for t in mesh.loop_triangles],np.int32)
            if np.linalg.det(np.array(evaluated.matrix_world)[:3,:3])<0:f=f[:,::-1]
            source_ranges.append({'name':obj.name,'vertex_start':vi,'vertex_count':len(v),
                                  'face_start':fi,'triangle_count':len(f)})
            source_v.append(v);source_f.append(f+vi);vi+=len(v);fi+=len(f)
            evaluated.to_mesh_clear()
        np.savez_compressed(out/'source_solids.npz',V=np.concatenate(source_v),F=np.concatenate(source_f),
                            parts_json=np.array(json.dumps(source_ranges)))
    declared = request.get('links')
    by_name = {o.name: o for o in objects}
    if declared:
        names = [n for link in declared.values() for n in link['meshes']]
        if set(names) != set(by_name) or len(names) != len(set(names)):
            raise ValueError('LINKS must own every built mesh exactly once; no unmodeled moving parts')
        groups = [(name, [by_name[n] for n in link['meshes']]) for name, link in declared.items()]
    else:
        groups = [('model', objects)]
    # Boolean transfers layers by name. Independently authored meshes often
    # use different names for their active UVs; without a common channel the
    # union silently fills one material's UVs with zeros. Preserve coordinates
    # before union, including explicit UV-map node references.
    for _, group in groups:
        if len(group) < 2:
            continue
        channel = 'r2s2r_surface_uv'
        for obj in group:
            layers = obj.data.uv_layers
            uv = next((layer for layer in layers if layer.active_render), layers.active)
            if uv is None:
                continue
            old = uv.name
            if old != channel and channel in layers:
                raise ValueError('Conflicting canonical UV channel; make source material mapping explicit')
            uv.name = channel
            layers.active = uv
            uv.active_render = True
            for slot in obj.material_slots:
                material = slot.material
                if material is None or not material.use_nodes:
                    continue
                explicit = [node for node in material.node_tree.nodes
                            if node.type == 'UVMAP' and node.uv_map == old]
                if explicit:
                    material = material.copy()
                    slot.material = material
                    for node in material.node_tree.nodes:
                        if node.type == 'UVMAP' and node.uv_map == old:
                            node.uv_map = channel
    objects, body_names = [], {}
    for link_name, group in groups:
        # A rigid assembly is one solid union: overlapping parts must not double its mass or internal SDF walls.
        # Union only within a physical link. Never erase a joint, mating gap or flexible-body boundary.
        body = group[0]
        for other in group[1:]:
            if declared and declared[link_name]['kind'] != 'rigid':
                raise ValueError('Flexible links must supply one simulation surface; rigid Boolean union refused')
            bpy.context.view_layer.objects.active = body
            modifier = body.modifiers.new('solid_union', 'BOOLEAN')
            modifier.operation, modifier.solver, modifier.object = 'UNION', 'EXACT', other
            modifier.material_mode = 'TRANSFER'
            bpy.ops.object.modifier_apply(modifier=modifier.name)
            bpy.data.objects.remove(other, do_unlink=True)
        objects.append(body)
        body_names[body.name] = link_name
    vertices, faces, uvs, normals, parts = [], [], [], [], []
    removed_degenerate = 0
    canonicalization = []
    face_materials, material_names = [], []
    offset, face_offset = 0, 0
    deps = bpy.context.evaluated_depsgraph_get()
    for obj in objects:
        evaluated = obj.evaluated_get(deps)
        mesh = evaluated.to_mesh()
        mesh.calc_loop_triangles()
        V = np.array([tuple(evaluated.matrix_world @ v.co) for v in mesh.vertices], dtype=float)
        F = np.array([tuple(t.vertices) for t in mesh.loop_triangles], dtype=np.int32)
        uv = mesh.uv_layers.active
        UV = np.array([[tuple(uv.data[j].uv) for j in t.loops] for t in mesh.loop_triangles]) if uv else np.zeros((len(F), 3, 2))
        normal_matrix = np.linalg.inv(np.array(evaluated.matrix_world)[:3, :3]).T
        N = np.array([[tuple(mesh.corner_normals[j].vector) for j in t.loops] for t in mesh.loop_triangles])
        N = N @ normal_matrix.T
        N /= np.maximum(np.linalg.norm(N, axis=2, keepdims=True), 1e-20)
        if np.linalg.det(np.array(evaluated.matrix_world)[:3, :3]) < 0:
            F, UV, N = F[:, ::-1], UV[:, ::-1], N[:, ::-1]
        # Exact Boolean intersections can leave zero-area triangle pairs. They
        # have no physical surface and become nonmanifold after vertex welding.
        # Remove only numerical zero area; the host still requires the result
        # to be closed, consistently wound and positive-volume (no hole repair).
        if not declared or declared[body_names[obj.name]]['kind'] == 'rigid':
            from calibrated_scene_reconstruction.assets.mesh_numeric import canonical_triangles
            F, keep, numerical = canonical_triangles(V, F)
            canonicalization.append({'object': obj.name, **numerical})
        else:
            keep = np.linalg.norm(np.cross(V[F[:, 1]]-V[F[:, 0]], V[F[:, 2]]-V[F[:, 0]]), axis=1) > 1e-18
        removed_degenerate += int((~keep).sum())
        F, UV, N = F[keep], UV[keep], N[keep]
        from calibrated_scene_reconstruction.assets.mesh_normals import complete as complete_normals
        N, normal_report = complete_normals(V,F,N)
        canonicalization.append({'object':obj.name,'shading_normals':normal_report})
        slots=[material.name if material else '' for material in mesh.materials]
        names=[slots[t.material_index] if t.material_index<len(slots) else '' for t,k in zip(mesh.loop_triangles,keep) if k]
        for name in dict.fromkeys(names):
            if name not in material_names:material_names.append(name)
        face_materials.append(np.array([material_names.index(name) for name in names],dtype=np.int32))
        parts.append({'name': body_names[obj.name] if declared else obj.name,
                      'kind': declared[body_names[obj.name]]['kind'] if declared else 'rigid',
                      **({'backend':declared[body_names[obj.name]].get('backend','particle')}
                         if declared and declared[body_names[obj.name]]['kind']=='cloth' else {}),
                      **({'backend':declared[body_names[obj.name]]['backend']}
                         if declared and declared[body_names[obj.name]]['kind'] in {'deformable','rope'} and 'backend' in declared[body_names[obj.name]] else {}),
                      'vertex_start': offset, 'vertex_count': len(V), 'face_start': face_offset,
                      'triangle_count': len(F)})
        vertices.append(V)
        faces.append(F + offset)
        uvs.append(UV)
        normals.append(N)
        offset += len(V)
        face_offset += len(F)
        evaluated.to_mesh_clear()
    V, F, UV, N = np.concatenate(vertices), np.concatenate(faces), np.concatenate(uvs), np.concatenate(normals)
    materials = {}
    for name in material_names:
        material = bpy.data.materials.get(name)
        if material is None:
            continue
        node = material.node_tree.nodes.get('Principled BSDF') if material.use_nodes else None
        materials[name] = {'color': list(node.inputs['Base Color'].default_value[:3]) if node else list(material.diffuse_color[:3]),
                           'roughness': float(node.inputs['Roughness'].default_value) if node else float(material.roughness),
                           'metallic': float(node.inputs['Metallic'].default_value) if node else float(material.metallic),
                           'opacity': float(node.inputs['Alpha'].default_value) if node else float(material.diffuse_color[3]),
                           'transmission': float((node.inputs.get('Transmission Weight') or node.inputs.get('Transmission')).default_value) if node else 0.,
                           'ior': float(node.inputs['IOR'].default_value) if node else 1.5}
        if node is not None:
            from calibrated_scene_reconstruction.assets.subsurface_optics import from_blender
            scattering = from_blender(node.inputs)
            if scattering is not None:
                materials[name]['subsurface'] = scattering
    joints = mod.JOINTS(request['params']) if callable(getattr(mod, 'JOINTS', None)) else request.get('joints', [])
    has_cloth_rest = callable(getattr(mod, 'CLOTH_REST', None))
    cloth_rest = mod.CLOTH_REST(request['params']) if has_cloth_rest else {}
    if has_cloth_rest:
        expected = {p['name'] for p in parts if p['kind'] == 'cloth'}
        if not isinstance(cloth_rest, dict) or not declared or not expected or set(cloth_rest) != expected:
            raise ValueError('CLOTH_REST must cover exactly the declared cloth links')
        from calibrated_scene_reconstruction.assets.cloth_rest import validate as validate_rest
        for part in parts:
            if part['name'] in cloth_rest:
                a, n, b, count = (part[k] for k in ('vertex_start', 'vertex_count', 'face_start', 'triangle_count'))
                validate_rest(cloth_rest[part['name']], V[a:a+n], F[b:b+count]-a)
    volume_initial=mod.VOLUME_INITIAL(request['params']) if callable(getattr(mod,'VOLUME_INITIAL',None)) else {}
    expected_volume={p['name'] for p in parts if p.get('backend')=='volume_fem'}
    if not isinstance(volume_initial,dict) or set(volume_initial)!=expected_volume:
        raise ValueError('VOLUME_INITIAL must cover exactly the selected native volume FEM links')
    if volume_initial:
        from calibrated_scene_reconstruction.assets.volume_initial_state import skin_mapping
        for part in parts:
            if part['name'] not in volume_initial:continue
            a,n,b,count=(part[k] for k in ('vertex_start','vertex_count','face_start','triangle_count'))
            skin_mapping(volume_initial[part['name']],V[a:a+n],F[b:b+count]-a)
    from calibrated_scene_reconstruction.assets.surface_bindings import validate as validate_bindings
    bindings = mod.SURFACE_BINDINGS(request['params']) if callable(getattr(mod,'SURFACE_BINDINGS',None)) else []
    validate_bindings(bindings,parts,V,F)
    from calibrated_scene_reconstruction.assets.surface_bindings import validate_islands
    validate_islands(cloth_rest,bindings,parts)
    from calibrated_scene_reconstruction.scene.objects.blender.root_geometry import verify as verify_roots
    root_geometry = verify_roots(V,F,parts,bindings)
    np.savez_compressed(out / 'mesh.npz', V=V, F=F, UV=UV, normals=N,
                        face_material=np.concatenate(face_materials),material_names=np.array(material_names,dtype=str),
                        materials_json=np.array(json.dumps(materials)),
                        links_json=np.array(json.dumps(parts if declared else [])),
                        surface_bindings_json=np.array(json.dumps(bindings)),
                        joints_json=np.array(json.dumps(joints)), cloth_rest_json=np.array(json.dumps(cloth_rest)),
                        **({'volume_initial_json':np.array(json.dumps(volume_initial))} if volume_initial else {}))
    opening = mod.OPENING(request['params']) if callable(getattr(mod, 'OPENING', None)) else None
    report = {'ok': True, 'parts': parts, 'opening': opening, 'mesh': 'mesh.npz', 'previews': [], 'source_parts': source_parts,
              'numerical_canonicalization': canonicalization,
              'links': declared, 'joints': joints, 'surface_bindings':bindings, 'root_geometry':root_geometry,
              'removed_numerical_zero_area_triangles': removed_degenerate}
    if request.get('artifacts'):
        bpy.ops.wm.save_as_mainfile(filepath=str(out / 'scene.blend'))
        bpy.ops.object.select_all(action='DESELECT')
        for obj in objects:
            obj.select_set(True)
        bpy.ops.export_scene.gltf(filepath=str(out / 'object.glb'), export_format='GLB', use_selection=True,
                                  export_yup=False, export_apply=True)
        centre = (V.min(0) + V.max(0)) / 2
        radius = float(np.linalg.norm(np.ptp(V, axis=0)))
        scene = bpy.context.scene
        scene.render.engine = 'CYCLES'
        scene.cycles.device = 'CPU'
        scene.cycles.samples = 12
        scene.render.resolution_x = scene.render.resolution_y = 384
        scene.render.resolution_percentage = 100
        scene.world = bpy.data.worlds.new('World')
        scene.world.use_nodes = True
        scene.world.node_tree.nodes['Background'].inputs[0].default_value = (0.15, 0.15, 0.15, 1)
        for index, direction in enumerate(((2, -3, 2), (-2, 3, 1.5))):
            bpy.ops.object.camera_add(location=centre + radius * np.array(direction) * 0.65)
            cam = bpy.context.object
            cam.rotation_euler = (Vector(centre) - cam.location).to_track_quat('-Z', 'Y').to_euler()
            cam.data.type = 'ORTHO'
            cam.data.ortho_scale = radius * 1.3
            # Blender's default 10cm near plane cuts through centimetre-scale
            # task objects at these size-relative preview camera distances.
            cam.data.clip_start = max(1e-5, radius * 0.01)
            cam.data.clip_end = max(1.0, radius * 20)
            scene.camera = cam
            bpy.ops.object.light_add(type='AREA', location=centre + radius * np.array((1, -2, 3)))
            bpy.context.object.data.energy = max(5, 300 * radius ** 2)
            bpy.context.object.data.shape = 'DISK'
            bpy.context.object.data.size = radius * 2
            scene.render.filepath = str(out / f'preview_{index}.png')
            bpy.ops.render.render(write_still=True)
            report['previews'].append(f'preview_{index}.png')
    return report


def main():
    for line in sys.stdin:
        request = json.loads(line)
        if request.get('stop'):
            return
        try:
            result = build(request)
        except Exception as exc:
            result = {'ok': False, 'error': str(exc), 'traceback': traceback.format_exc()}
        out = Path(request['output'])
        out.mkdir(parents=True, exist_ok=True)
        tmp = out / 'response.tmp'
        tmp.write_text(json.dumps(result))
        tmp.replace(out / 'response.json')


if __name__ == '__main__':
    main()
