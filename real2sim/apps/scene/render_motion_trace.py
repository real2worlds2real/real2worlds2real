"""Blender playback of recorded native mesh/rigid states; never a physics proof.

Run through tools/run_blender.sh --python this_file -- manifest.json. The agent
supplies the asset, trace field mapping, source appearance and camera. No poses
or flexible shapes are interpolated or synthesized by this renderer.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

# Keep --help usable before the optional Blender runtime is imported.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('manifest', type=Path, help='JSON native trace, asset, appearance and camera manifest')
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:])

import bpy
from mathutils import Vector, Matrix
import numpy as np

manifest_path = args.manifest
manifest = json.loads(manifest_path.read_text())
out = Path(manifest['output'])
out.mkdir(parents=True, exist_ok=manifest.get('resume', False))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


for path, expected in manifest['sha256'].items():
    if digest(path) != expected:
        raise ValueError('Recorded input changed: '+path)
raw = np.load(manifest['asset'])
# An NPZ member is decompressed on every __getitem__. Cache the immutable
# native arrays once; repeated whole-trajectory decompression per frame is
# prohibitively expensive for densely sampled flexible surfaces.
with np.load(manifest['trace']) as snapshot:
    trace = {name: snapshot[name] for name in snapshot.files}
links = json.loads(str(raw['links_json']))
rest = json.loads(str(raw['cloth_rest_json'])) if 'cloth_rest_json' in raw else {}
atlas = np.load(manifest['appearance_mesh'])
atlas_uv = atlas['UV']
if atlas_uv.shape == (len(atlas['F']), 3, 2):
    # Blender metric proposals retain per-corner UVs, including real seams.
    face_uv = atlas_uv
elif atlas_uv.shape == (len(atlas['V']), 2):
    face_uv = atlas_uv[atlas['F']]
else:
    raise ValueError('Appearance requires vertex or triangle-corner UV coordinates')
quality = json.loads(Path(manifest['quality']).read_text())
if len(trace['times']) != len(quality):
    raise ValueError('Every rendered state needs its corresponding quality record')
task_motion=None
if 'task_motion' in manifest:
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
    from calibrated_scene_reconstruction.scene.review.motion_scope import check as check_motion_scope
    task_motion=check_motion_scope(manifest,trace,quality)
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)


def material(name, color, roughness=.5, texture=None, *, transmission=0.,ior=1.5,opacity=1.,metallic=0.):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bs = mat.node_tree.nodes.get('Principled BSDF')
    bs.inputs['Base Color'].default_value = (*color, opacity)
    bs.inputs['Roughness'].default_value = roughness
    bs.inputs['Alpha'].default_value=opacity
    bs.inputs['Transmission Weight'].default_value=transmission
    bs.inputs['IOR'].default_value=ior
    bs.inputs['Metallic'].default_value=metallic
    if texture and transmission==0:
        tex = mat.node_tree.nodes.new('ShaderNodeTexImage')
        tex.image = bpy.data.images.load(texture)
        # A masked source photograph has unobserved pixels, not black cloth
        # or missing mesh. Preserve the authored material there; material
        # opacity/transmission remain independent physical appearance inputs.
        mix = mat.node_tree.nodes.new('ShaderNodeMixRGB')
        mix.inputs[1].default_value = (*color, 1.)
        mat.node_tree.links.new(tex.outputs['Alpha'], mix.inputs[0])
        mat.node_tree.links.new(tex.outputs['Color'], mix.inputs[2])
        mat.node_tree.links.new(mix.outputs['Color'], bs.inputs['Base Color'])
    return mat


objects = {}
source_materials=json.loads(str(raw['materials_json'])) if 'materials_json' in raw else {}
source_names=raw['material_names'].tolist() if 'material_names' in raw else []
for link in links:
    name = link['name']
    a, n, b, count = [link[k] for k in ('vertex_start', 'vertex_count', 'face_start', 'triangle_count')]
    vertices, faces = raw['V'][a:a+n], raw['F'][b:b+count]-a
    data = bpy.data.meshes.new(name)
    data.from_pydata(vertices.tolist(), [], faces.tolist())
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    objects[name] = obj
    flexible = 'points' in manifest['links'][name]
    groups=(raw['face_material'][b:b+count] if source_materials and 'face_material' in raw else np.zeros(count,int))
    palette={};plain_palette={}
    for group in sorted(set(groups.tolist())):
        props=source_materials.get(source_names[group],{}) if source_names else {}
        texture=manifest.get('cloth_textures',{}).get(name) if flexible else manifest['albedo']
        mat=material(name+'_'+str(group),props.get('color',(.88,.89,.86) if flexible else (.8,.8,.8)),
                     props.get('roughness',.95 if flexible else .45),texture,
                     transmission=props.get('transmission',0.),ior=props.get('ior',1.5),
                     opacity=props.get('opacity',1.),metallic=props.get('metallic',0.))
        palette[group]=len(data.materials);data.materials.append(mat)
        if manifest.get('untextured_faces',{}).get(name):
            plain_palette[group]=len(data.materials)
            data.materials.append(material(name+'_unobserved_'+str(group),props.get('color',(.8,.8,.8)),
                props.get('roughness',.5),transmission=props.get('transmission',0.),
                ior=props.get('ior',1.5),opacity=props.get('opacity',1.),metallic=props.get('metallic',0.)))
    untextured=set(manifest.get('untextured_faces',{}).get(name,[]))
    if any(type(i)is not int or not 0<=i<count for i in untextured):
        raise ValueError('Untextured appearance faces must belong to the actual source part')
    for polygon,group in zip(data.polygons,groups):
        polygon.material_index=plain_palette[group] if polygon.index in untextured else palette[group]
    uv = data.uv_layers.new(name='UVMap')
    if flexible and name in rest:
        chart = np.asarray(rest[name]['points'])
        coordinates = (chart[:, :2]-chart[:, :2].min(0))/np.ptp(chart[:, :2], axis=0)
        mapped = coordinates[faces]
        for mat in data.materials:
            noise = mat.node_tree.nodes.new('ShaderNodeTexNoise')
            noise.inputs['Scale'].default_value = 650.
            bump = mat.node_tree.nodes.new('ShaderNodeBump')
            bump.inputs['Strength'].default_value = .06
            bump.inputs['Distance'].default_value = .00007
            texcoord = mat.node_tree.nodes.new('ShaderNodeTexCoord')
            mat.node_tree.links.new(texcoord.outputs['UV'], noise.inputs['Vector'])
            mat.node_tree.links.new(noise.outputs['Fac'], bump.inputs['Height'])
            mat.node_tree.links.new(bump.outputs['Normal'], mat.node_tree.nodes.get('Principled BSDF').inputs['Normal'])
    else:
        if b+count > len(face_uv):
            raise ValueError('Appearance does not cover rigid source faces')
        mapped = face_uv[b:b+count]
    for polygon in data.polygons:
        for k, loop in enumerate(polygon.loop_indices):
            uv.data[loop].uv = mapped[polygon.index, k]
    data.update()

# Optional independently moving rigid geometry, e.g. the actual fingers in a
# recorded grasp. These transforms are saved native states, never keyframes or
# inferred hand motion. Keep them separate from the reconstructed asset atlas.
rigid_context = []
for context in manifest.get('rigid_context', []):
    path = str(Path(context['mesh']))
    if path not in manifest['sha256']:
        raise ValueError('Recorded rigid context needs a hashed source mesh')
    with np.load(path) as mesh:
        vertices, faces = mesh['V'].copy(), mesh['F'].copy()
    positions = trace[context['position']]
    rotations = trace[context['rotation']]
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices)
            or faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces)
            or faces.dtype.kind not in 'iu' or faces.min() < 0 or faces.max() >= len(vertices)
            or positions.shape != (len(trace['times']), 3)
            or rotations.shape != (len(trace['times']), 3, 3)
            or not all(np.isfinite(x).all() for x in [vertices, positions, rotations])
            or not np.allclose(rotations.transpose(0, 2, 1)@rotations, np.eye(3), atol=1e-5)
            or not np.allclose(np.linalg.det(rotations), 1., atol=1e-5)):
        raise ValueError('Rigid context requires source triangles and every actual native pose')
    data = bpy.data.meshes.new(context['name'])
    data.from_pydata(vertices.tolist(), [], faces.tolist())
    obj = bpy.data.objects.new(context['name'], data)
    bpy.context.collection.objects.link(obj)
    data.materials.append(material(context['name'], context.get('color', (.18, .2, .23)),
                                   context.get('roughness', .6)))
    rigid_context.append((obj, vertices, positions, rotations))

# Optional material-grid overlay follows the actual recorded native vertices.
grids = []
for name, settings in manifest.get('material_grid', {}).items():
    rows, columns = settings['shape']
    if rows*columns != len(objects[name].data.vertices):
        raise ValueError('Material grid does not match native node topology')
    indices = np.arange(rows*columns).reshape(rows, columns)
    paths = [indices[j] for j in range(0, rows, settings['row_stride'])]
    paths += [indices[:, i] for i in range(0, columns, settings['column_stride'])]
    curve = bpy.data.curves.new(name+'_material_grid', 'CURVE')
    curve.dimensions = '3D'
    curve.bevel_depth = settings.get('radius', .000075)
    curve.bevel_resolution = 0
    for path in paths:
        spline = curve.splines.new('POLY')
        spline.points.add(len(path)-1)
    obj = bpy.data.objects.new(curve.name, curve)
    bpy.context.collection.objects.link(obj)
    curve.materials.append(material(curve.name, (.08, .24, .4), .8))
    grids.append((name, curve, paths))

support = manifest.get('support')
if support:
    z = np.load(support['mesh'])
    vertices = z['V']
    angle = np.radians(support['pose']['yaw_deg'])
    R = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    vertices = vertices@R.T+[*support['pose']['xy'], support['pose'].get('support_z', 0.)]
    pose = manifest['original_world_pose']
    angle = np.radians(pose['yaw_deg'])
    R = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    vertices = (vertices-[*pose['xy'], pose['support_z']])@R
    data = bpy.data.meshes.new('support')
    data.from_pydata(vertices.tolist(), [], z['F'].tolist())
    obj = bpy.data.objects.new('support', data)
    bpy.context.collection.objects.link(obj)
    support_names=z['material_names'].tolist() if 'material_names' in z else []
    support_materials=json.loads(str(z['materials_json'])) if 'materials_json' in z else {}
    if support_names:
        for name in support_names:
            properties=support_materials.get(name,{})
            data.materials.append(material('support_'+name,properties.get('color',(.5,.32,.16)),
                                            properties.get('roughness',.6),metallic=properties.get('metallic',0.)))
    else:
        data.materials.append(material('wood', (.5, .32, .16), .6))
        data.materials.append(material('frame', (.04, .045, .05), .5))
    if 'face_material' in z:
        for polygon, idx in zip(data.polygons, z['face_material']):
            if not 0<=int(idx)<len(data.materials):raise ValueError('Support material index is outside its source palette')
            polygon.material_index = int(idx)

scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.samples = manifest.get('samples', 12)
scene.cycles.use_denoising = True
render_device = manifest.get('device', 'GPU')
if render_device not in {'CPU', 'GPU'}:
    raise ValueError('Choose explicit CPU or GPU rendering')
if render_device == 'GPU':
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'OPTIX'
    prefs.get_devices()
    for device in prefs.devices:
        device.use = device.type != 'CPU'
else:
    threads = manifest.get('threads', 4)
    if type(threads) is not int or not 1 <= threads <= 16:
        raise ValueError('Bounded CPU render thread budget required')
    scene.render.threads_mode = 'FIXED'
    scene.render.threads = threads
scene.cycles.device = render_device
scene.world.use_nodes = True
scene.world.node_tree.nodes['Background'].inputs[0].default_value = (.75, .82, .92, 1.)
scene.world.node_tree.nodes['Background'].inputs[1].default_value = .45
for location, energy, size in [((0, -.6, 1.5), 45, 1.1), ((.6, .4, 1), 25, .7)]:
    bpy.ops.object.light_add(type='AREA', location=location)
    light = bpy.context.object
    light.data.energy, light.data.size = energy, size
    light.rotation_euler = (Vector((0, 0, .2))-light.location).to_track_quat('-Z', 'Y').to_euler()
bpy.ops.object.camera_add(location=manifest['camera']['eye'])
camera = bpy.context.object
camera.rotation_euler = (Vector(manifest['camera']['look'])-camera.location).to_track_quat('-Z', 'Y').to_euler()
camera.data.lens = manifest['camera'].get('lens_mm', 35)
camera.data.clip_start = .005
scene.camera = camera
scene.render.resolution_x, scene.render.resolution_y = manifest.get('size', [960, 720])
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = 'PNG'
scene.view_settings.view_transform = 'AgX'
scene.render.fps = manifest['fps']
for frame in range(len(trace['times'])):
    frame_path = out/f'frame_{frame:04d}.png'
    if frame_path.exists():
        if not manifest.get('resume') or str(frame_path) not in manifest['sha256']:
            raise ValueError('Existing frame has no verified resume fingerprint: '+str(frame_path))
        continue
    frame_bounds=[]
    for name, fields in manifest['links'].items():
        obj = objects[name]
        if 'points' in fields:
            vertices = trace[fields['points']][frame]
        else:
            link = next(link for link in links if link['name'] == name)
            a, n = link['vertex_start'], link['vertex_count']
            vertices = raw['V'][a:a+n]@trace[fields['rotation']][frame].T+trace[fields['position']][frame]
        obj.data.vertices.foreach_set('co', np.asarray(vertices, np.float32).ravel())
        obj.data.update()
        frame_bounds.extend([np.asarray(vertices).min(0),np.asarray(vertices).max(0)])
    for obj, vertices, positions, rotations in rigid_context:
        matrix = np.eye(4)
        matrix[:3, :3], matrix[:3, 3] = rotations[frame], positions[frame]
        obj.matrix_world = Matrix(matrix.tolist())
        world = vertices@rotations[frame].T+positions[frame]
        frame_bounds.extend([world.min(0), world.max(0)])
    orbit=manifest['camera'].get('orbit')
    if orbit:
        fraction=frame/max(1,len(trace['times'])-1)
        angle=np.radians((1-fraction)*orbit['yaw_start_deg']+fraction*orbit['yaw_end_deg'])
        radius=orbit['radius'];height=orbit['height'];look=np.asarray(manifest['camera']['look'])
        framing=manifest['camera'].get('auto_frame')
        if framing:
            bounds=np.asarray(frame_bounds);lo,hi=bounds.min(0),bounds.max(0);look=(lo+hi)/2
            radius=max(framing['min_radius'],framing['radius_scale']*float((hi-lo).max()))
            height=radius*framing['elevation_ratio']
        camera.location=look+np.array([radius*np.cos(angle),radius*np.sin(angle),height])
        camera.rotation_euler=(Vector(look)-camera.location).to_track_quat('-Z','Y').to_euler()
    for name, curve, paths in grids:
        vertices = trace[manifest['links'][name]['points']][frame]
        for spline, path in zip(curve.splines, paths):
            spline.points.foreach_set('co', np.c_[vertices[path], np.ones(len(path))].astype(np.float32).ravel())
    scene.render.filepath = str(frame_path)
    bpy.ops.render.render(write_still=True)
for path, expected in manifest['sha256'].items():
    if digest(path) != expected:
        raise ValueError('Recorded input changed during rendering: '+path)
(out/'scope.json').write_text(json.dumps({'manifest': manifest, 'native_states_only': True,
                                         'interpolated_frames': False, 'physics_proof': False,
                                         'task_motion':task_motion}, indent=2)+'\n')
