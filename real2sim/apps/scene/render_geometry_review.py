"""Blender neutral geometry views, without texture or physics approval.

tools/run_blender.sh --python apps/scene/render_geometry_review.py -- manifest.json
The manifest gives mesh, output, size and named views (eye, target, rotation,
orthographic_scale). Rotation changes only the presentation pose, never source
vertices. Use source, novel and operational orientations for structural review.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

# Parse before importing Blender so interface discovery also works in an
# ordinary Python environment. Blender passes script arguments after "--".
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('manifest', type=Path, help='JSON mesh, output and named neutral-view manifest')
args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:])

import bpy
from mathutils import Matrix, Vector
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


manifest_path = args.manifest.resolve()
manifest = json.loads(manifest_path.read_text())
source = Path(manifest['mesh']).resolve(); source_hash = digest(source)
out = Path(manifest['output']).resolve(); out.mkdir(parents=True, exist_ok=False)
with np.load(source) as data:
    vertices = data['V'].copy(); faces = data['F'].copy()
bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
mesh = bpy.data.meshes.new('candidate_surface')
mesh.from_pydata(vertices.tolist(), [], faces.tolist()); mesh.update()
body = bpy.data.objects.new('candidate', mesh); bpy.context.collection.objects.link(body)
material = bpy.data.materials.new('uniform_neutral'); material.use_nodes = True
shader = material.node_tree.nodes.get('Principled BSDF')
shader.inputs['Base Color'].default_value = (.42, .42, .42, 1.)
shader.inputs['Roughness'].default_value = .72
body.data.materials.append(material)
for polygon in mesh.polygons: polygon.use_smooth = True
# Preserve real creases. Smoothing every triangulated face across a thin panel
# invents diagonal folds in a geometry-only inspection of a planar surface.
mesh.set_sharp_from_angle(angle=.5235987755982988)
scene = bpy.context.scene; scene.render.engine = 'CYCLES'; scene.cycles.samples = 48
device = manifest.get('device', 'CPU')
if device not in {'CPU', 'CUDA', 'OPTIX'}:
    raise ValueError('Geometry renderer device must be CPU, CUDA or OPTIX')
if device != 'CPU':
    preferences = bpy.context.preferences.addons['cycles'].preferences
    preferences.compute_device_type = device
    preferences.get_devices()
    selected = [item for item in preferences.devices if item.type == device]
    if not selected:
        raise RuntimeError('Requested geometry rendering device is unavailable: '+device)
    for item in preferences.devices:
        item.use = item in selected
    scene.cycles.device = 'GPU'
scene.cycles.use_denoising = True
scene.render.resolution_x, scene.render.resolution_y = manifest.get('size', [1024, 768])
scene.render.resolution_percentage = 100
scene.world.use_nodes = True
scene.world.node_tree.nodes['Background'].inputs['Color'].default_value = (.16, .16, .16, 1)
scene.world.node_tree.nodes['Background'].inputs['Strength'].default_value = .45
bpy.ops.object.camera_add(); camera = bpy.context.object; scene.camera = camera
camera.data.type = 'ORTHO'; camera.data.clip_start = .0001; camera.data.clip_end = 100.
lights = []
for name in ('key', 'fill'):
    data = bpy.data.lights.new(name, 'AREA')
    obj = bpy.data.objects.new(name, data); bpy.context.collection.objects.link(obj); lights.append(obj)
records = {}
for name, view in manifest['views'].items():
    if not name.isidentifier(): raise ValueError('Safe view identifier required')
    rotation = np.asarray(view.get('rotation', np.eye(3)), float)
    if rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8) or not np.isclose(np.linalg.det(rotation), 1):
        raise ValueError('Presentation rotation must preserve exact geometry')
    body.matrix_world = Matrix(rotation.tolist()).to_4x4()
    target = Vector(view['target']); eye = Vector(view['eye'])
    scale = float(view['orthographic_scale'])
    if not .001 < scale < 10: raise ValueError('Bounded view scale required')
    camera.location = eye; camera.rotation_euler = (target - eye).to_track_quat('-Z', 'Y').to_euler()
    camera.data.ortho_scale = scale
    for i, light in enumerate(lights):
        light.location = target + (eye - target) + Vector(((-1) ** i * scale, -.3 * scale, scale))
        light.rotation_euler = (target - light.location).to_track_quat('-Z', 'Y').to_euler()
        light.data.energy = (30 if i == 0 else 10) * scale * scale
        light.data.shape = 'DISK'; light.data.size = scale * 1.2
    image = out / (name + '.png'); scene.render.filepath = str(image)
    bpy.ops.render.render(write_still=True)
    records[name] = {'file': image.name, 'sha256': digest(image), **view}
if digest(source) != source_hash: raise ValueError('Candidate changed while rendering')
(out / 'render.json').write_text(json.dumps({
    'task_approved': False, 'material_policy': 'uniform_diffuse_no_texture',
    'normal_policy': 'geometric_face_normals_no_smoothing',
    'mesh': str(source), 'mesh_sha256': source_hash,
    'manifest_sha256': digest(manifest_path), 'renderer_sha256': digest(__file__),
    'render_device': device,
    'views': records, 'scope': 'Geometry inspection only; presentation rotations are not physics.'
}, indent=2) + '\n')
