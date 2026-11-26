"""Blender meshes use the existing visibility-aware multi-frame object texture baker."""
import shutil
import hashlib

import numpy as np
from PIL import Image

from calibrated_scene_reconstruction.scene.objects.object_texture import bake, qa_sheet, atlas_coverage
from calibrated_scene_reconstruction.scene.objects.blender.contract import write_json


def uniform_albedo(image,coverage):
    selected=np.asarray(image)[np.asarray(coverage)>0]
    if not len(selected):raise ValueError('Uniform material needs observed texture samples')
    colour=np.median(selected,axis=0).astype(np.uint8)
    return np.broadcast_to(colour,np.asarray(image).shape).copy()


def run(s, obj, resolution=512):
    import json
    if (s/'qa/independent_freeze.json').exists():
        raise ValueError('Frozen reconstruction: create a new appearance iteration session')
    path = s.config_dir / 'scene_graph.json'
    g = json.loads(path.read_text())
    fix = g['dynamic_objects'][obj]['fix']
    request=g['dynamic_objects'][obj].get('fit') or {}
    mode=request.get('texture_mode','multiview')
    if mode not in ('multiview','uniform'):raise ValueError('texture_mode must be multiview or uniform')
    gain=float(request.get('texture_gain',(fix.get('look') or {}).get('gain',1.)))
    if not np.isfinite(gain) or not 0<gain<=4:raise ValueError('Texture gain must be finite within0..4')
    normalize = request.get('texture_normalize_exposure', True)
    if type(normalize) is not bool:
        raise ValueError('texture_normalize_exposure must be boolean')
    b = bake(s, obj, fix, resolution, fix['support_z'], normalize_exposure=normalize)
    if b is None or b['seen'] <= 0:
        raise ValueError('No visible texture samples')
    if mode=='uniform':b['albedo']=uniform_albedo(b['albedo'],b['coverage'])
    root=s/'objects'/obj/f'round_{fix["round"]}'
    out=root/'look';number=2
    while out.exists():out=root/f'look_{number}';number+=1
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'mesh.npz', V=b['V'], F=b['F'], UV=b['UV'])
    write_json(out/'atlas_geometry.json', atlas_coverage(b['V'], b['F'], b['UV']))
    from calibrated_scene_reconstruction.assets.metric_mesh import asset
    from calibrated_scene_reconstruction.core.paths import resolve
    with np.load(resolve(fix['mesh'])) as raw:
        preserve_surface = (('links_json' in raw and bool(json.loads(str(raw['links_json']))))
                            or ('materials_json' in raw and bool(json.loads(str(raw['materials_json'])))))
    original = asset(fix) if preserve_surface else {}
    if original:
        V, F, index = [], [], 0
        for part in original['parts']:
            V.append(part['V'])
            F.append(part['F']+index)
            index += len(part['V'])
        V, F = np.concatenate(V), np.concatenate(F)
        if b['F'].shape != F.shape or not np.allclose(b['V'][b['F']], V[F], atol=1e-7, rtol=0):
            raise ValueError('Texture atlas changed source triangle ordering; physical-link binding refused')
        np.savez_compressed(out/'mesh.npz', V=b['V'], F=b['F'], UV=b['UV'], face_uv=b['UV'][b['F']])
    policies = request.get('texture_materials')
    if policies:
        # Keep calibrated samples before any material prior. Later finish
        # iterations must not compound tint corrections or recover clipped
        # samples by trying to invert a previously applied policy.
        Image.fromarray(b['albedo']).save(out/'sampled_albedo.png')
        from .texture_materials import apply
        b['albedo'], material_report = apply(s, obj, fix, b, policies, out)
        write_json(out/'material_finishes.json', material_report)
    Image.fromarray(b['albedo']).save(out / 'albedo.png')
    Image.fromarray(b['coverage']).save(out / 'coverage.png')
    if b.get('photometric'):
        write_json(out/'photometric.json', b['photometric'])
    qa = s / 'qa/object_texture'
    qa.mkdir(parents=True, exist_ok=True)
    previous=(fix.get('look') or {}).get('mesh')
    if previous and (qa/f'{obj}.jpg').exists():
        saved=(s/previous).parent/f'{obj}.jpg'
        if not saved.exists():shutil.copyfile(qa/f'{obj}.jpg',saved)
    qa_sheet(out, obj, b, np.load(s / 'parsing/posed_frames/cameras.npz'))
    if not (out/f'{obj}.jpg').exists():raise ValueError('No current texture observation preview')
    shutil.copyfile(out/f'{obj}.jpg',qa/f'{obj}.jpg')
    fix['look'] = {**fix.get('look', {}), 'mesh': str((out / 'mesh.npz').relative_to(s.dir)),
                   'albedo': str((out / 'albedo.png').relative_to(s.dir)),
                   'gain':gain,'mode':mode,'appearance_source':'Reviewed current-capture image samples; hidden appearance inferred'}
    if b.get('photometric'):
        fix['look']['photometric'] = str((out/'photometric.json').relative_to(s.dir))
    if policies:
        fix['look'].update(material_finishes=str((out/'material_finishes.json').relative_to(s.dir)),
                           material_policy=policies,
                           sampled_source={'mesh':str((out/'mesh.npz').relative_to(s.dir)),
                                           'albedo':str((out/'sampled_albedo.png').relative_to(s.dir)),
                                           'coverage':str((out/'coverage.png').relative_to(s.dir))})
    if original:
        fix['look'].update(preserves_links=bool(original.get('links')), preserves_surface=True,
                           source_mesh_sha256=hashlib.sha256(resolve(fix['mesh']).read_bytes()).hexdigest())
    write_json(path, g)
    if request.get('optical_print_source'):
        from .optical_print import from_config
        from_config(s,obj)
        fix['look']=s.scene_graph['dynamic_objects'][obj]['fix']['look']
    review=s/'qa/review';review.mkdir(parents=True,exist_ok=True)
    (review/'S4.ran').touch()
    return {'ok': True, 'seen': b['seen'], 'sheet': f'qa/object_texture/{obj}.jpg', **fix['look']}
