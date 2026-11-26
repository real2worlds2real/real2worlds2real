"""Propose opaque source printing independently of transmitted background RGB.

The agent reviews source regions and a direct view; this tool projects only
front-owned dark ink or an explicitly outlined opaque label. It never approves
a proposal or replaces the unprinted transmitting material.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import trimesh

from calibrated_scene_reconstruction.scene.objects.object_hull import project
from calibrated_scene_reconstruction.scene.objects.object_texture import texels, front_buffer


def propose(s, obj, request, atlas_path, output):
    mode = request.get('mode', 'ink')
    if mode not in {'ink', 'opaque_patch'}:
        raise ValueError('Optical surface proposal mode must be ink or opaque_patch')
    frame = request['frame']
    boxes = np.asarray(request['regions_xyxy'], float)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not len(boxes) or not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError('Printing needs reviewed source image regions')
    if not request.get('basis'):
        raise ValueError('Explain why the source regions are surface print')
    resolution = int(request.get('resolution', 1536))
    kernel = int(request.get('contrast_kernel', 11))
    if not 128 <= resolution <= 4096 or kernel < 3 or kernel > 51 or kernel % 2 != 1:
        raise ValueError('Invalid print sampling or contrast kernel')
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    with np.load(atlas_path) as atlas:
        V, F, UV, corner_uv = (atlas[k].copy() for k in ('V', 'F', 'UV', 'face_uv'))
    from calibrated_scene_reconstruction.assets.metric_mesh import asset
    fix = s.scene_graph['dynamic_objects'][obj]['fix']
    parts = asset(fix)['parts']
    actual = np.concatenate([p['V'][p['F']] for p in parts])
    if actual.shape != V[F].shape or not np.allclose(actual, V[F], atol=1e-7, rtol=0):
        raise ValueError('Print atlas must retain actual physical triangles and order')
    r, c, triangles, P = texels(V, F, UV, resolution)
    N = trimesh.Trimesh(V, F, process=False).face_normals[triangles]
    from calibrated_scene_reconstruction.assets.object_pose import rotation as object_rotation
    R = object_rotation(fix)
    offset = np.array([*fix['xy'], fix['support_z']])
    P, N, world = P@R.T+offset, N@R.T, V@R.T+offset
    camera_file = s/'parsing/posed_frames/cameras.npz'
    with np.load(camera_file) as cameras:
        i = list(map(str, cameras['names'])).index(frame)
        K, rotation, centre = (cameras[k][i] for k in ('K', 'R_world_cam', 'centre'))
    source = s/'splat/data/images'/frame
    image = np.array(Image.open(source).convert('RGB'))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, np.ones((kernel, kernel), 'uint8'))
    low, span = request.get('contrast_low', 8), request.get('contrast_span', 12)
    if not 0 <= low <= 255 or not 0 < span <= 255:
        raise ValueError('Invalid source contrast interval')
    signal = np.clip((dark.astype(float)-low)/span, 0, 1)
    signal *= np.ptp(image.astype(float), axis=2) < request.get('chroma_max', 55)
    signal = cv2.dilate(signal.astype('float32'), np.ones((3, 3), 'uint8'))
    polygons = []
    if mode == 'opaque_patch':
        for raw in request.get('polygons_xy', []):
            polygon = np.asarray(raw, float)
            if (polygon.ndim != 2 or polygon.shape[1:] != (2,) or len(polygon) < 3
                    or not np.isfinite(polygon).all() or (polygon < 0).any()
                    or (polygon >= [image.shape[1], image.shape[0]]).any()
                    or abs(cv2.contourArea(polygon.astype('float32'))) < 1):
                raise ValueError('Opaque patches need finite nonempty source polygons inside the image')
            polygons.append(np.rint(polygon).astype('int32'))
        if not polygons:
            raise ValueError('Opaque patch requires reviewed source outlines; a bounding box cannot identify a label')
        signal = np.zeros(image.shape[:2], np.uint8)
        cv2.fillPoly(signal, polygons, 1)
    _, depths = front_buffer(world, F, K, rotation, centre, image.shape[:2])
    u, v, depth = project(P, K, rotation, centre)
    ui, vi = np.rint(u).astype(int), np.rint(v).astype(int)
    owner = (depth > .05)&(ui >= 0)&(ui < image.shape[1])&(vi >= 0)&(vi < image.shape[0])
    view = centre-P; view /= np.linalg.norm(view, axis=1)[:, None]
    owner &= (N*view).sum(1) > request.get('front_cosine_min', .15)
    # Only transmitting faces can acquire a printed-glass finish. This also
    # prevents a nearby shaft or an opaque cap from stealing a label pixel.
    transmitting = np.concatenate([np.array([
        p['materials'][p['material_names'][index]].get('transmission', 0.) > 0
        for index in p['face_material']], bool) for p in parts])
    owner &= transmitting[triangles]
    selected = np.flatnonzero(owner)
    owner[selected] &= abs(depths[vi[selected], ui[selected]]-depth[selected]) < request.get('depth_tolerance_m', .001)
    region = np.zeros(len(P), bool)
    for x0, y0, x1, y1 in boxes:
        region |= (u >= x0)&(u <= x1)&(v >= y0)&(v <= y1)
    values = np.zeros(len(P)); selected = np.flatnonzero(owner&region)
    values[selected] = signal[vi[selected], ui[selected]]
    mask = np.zeros((resolution, resolution), np.float32); mask[r, c] = values
    mask = (cv2.GaussianBlur(mask[::-1], (3, 3), .45).clip(0, 1)*255).astype('uint8')
    Image.fromarray(mask).save(out/'opaque_mask.png')
    Image.fromarray(255-mask).save(out/'transmission_mask.png')
    if mode == 'opaque_patch':
        colours = np.zeros((resolution, resolution, 3), np.uint8)
        colours[r[selected], c[selected]] = image[vi[selected], ui[selected]]
        # Pad the source colour beyond the opacity boundary to avoid black
        # fringes under bilinear filtering; it adds no opaque surface area.
        from scipy.ndimage import distance_transform_edt
        known = np.zeros((resolution, resolution), bool)
        known[r[selected], c[selected]] = values[selected] > 0
        if not known.any():raise ValueError('No front-owned opaque patch samples')
        nearest = distance_transform_edt(~known, return_distances=False, return_indices=True)
        colours[~known] = colours[nearest[0][~known], nearest[1][~known]]
        Image.fromarray(colours[::-1]).save(out/'ink_color.png')
    else:
        Image.new('RGB', (16, 16), tuple(request.get('ink_rgb', [38, 40, 44]))).save(out/'ink_color.png')
    np.savez_compressed(out/'mesh.npz', V=V, F=F, UV=UV, face_uv=corner_uv)
    marked = image.copy()
    for x0, y0, x1, y1 in boxes.astype(int):cv2.rectangle(marked, (x0, y0), (x1, y1), (20, 230, 60), 2)
    if polygons:cv2.polylines(marked, polygons, True, (240, 30, 30), 1)
    Image.fromarray(marked).save(out/'source_regions.png')
    report = {'approved': False, 'request': request, 'ink_pixels': int((mask > 30).sum()),
              'scope': 'Source appearance proposal only; no material reflectance, geometry or task approval',
              'inputs': {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                         for p in (source, camera_file, Path(atlas_path), Path(__file__))}}
    (out/'proposal.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('session', 'object', 'request', 'atlas', 'output'):p.add_argument(name)
    a = p.parse_args()
    from calibrated_scene_reconstruction.scene.session import Session
    print(json.dumps(propose(Session(a.session), a.object, json.loads(Path(a.request).read_text()), a.atlas, a.output)))


def from_config(s,obj):
    """Reproject declared print after each atlas revision; require source review."""
    from calibrated_scene_reconstruction.core.session_code import source_path
    graph=s.scene_graph;spec=graph['dynamic_objects'][obj]
    request_name=spec.get('fit',{}).get('optical_print_source')
    if not request_name:return None
    request_path=source_path(s.config_dir,request_name)
    request=json.loads(request_path.read_text());look=spec['fix']['look']
    out=(s.dir/look['mesh']).parent/'optical_print'
    report=propose(s,obj,request,s.dir/look['mesh'],out)
    look.pop('optical_print',None)
    look['pending_optical_print']={'proposal':str((out/'proposal.json').relative_to(s.dir)),
        'request':request_name,'request_sha256':hashlib.sha256(request_path.read_bytes()).hexdigest()}
    (s.config_dir/'scene_graph.json').write_text(json.dumps(graph,indent=2)+'\n')
    return report


def review_source(s,obj,basis):
    """Agent source-region review; appearance gate still needs a native render."""
    from calibrated_scene_reconstruction.core.session_code import source_path
    if not isinstance(basis,str) or not basis.strip():raise ValueError('Explain the source-region review')
    graph=s.scene_graph;fix=graph['dynamic_objects'][obj]['fix'];look=fix['look']
    pending=look['pending_optical_print'];path=s.dir/pending['proposal'];report=json.loads(path.read_text())
    request=source_path(s.config_dir,pending['request'])
    if hashlib.sha256(request.read_bytes()).hexdigest()!=pending['request_sha256']:raise ValueError('Source print request changed')
    evidence={}
    for name,expected in report['inputs'].items():
        file=Path(name).resolve()
        if hashlib.sha256(file.read_bytes()).hexdigest()!=expected:raise ValueError('Print proposal input changed')
        if file.is_relative_to(s.dir.resolve()):evidence[str(file.relative_to(s.dir.resolve()))]=expected
    paths={'opaque_mask':path.parent/'opaque_mask.png','transmission_mask':path.parent/'transmission_mask.png',
           'albedo':path.parent/'ink_color.png'}
    reviewed={'status':'resolved','basis':basis,'source_mesh_sha256':look['source_mesh_sha256'],
              'evidence_files':evidence,'request':report['request'],
              'sha256':{key:hashlib.sha256(file.read_bytes()).hexdigest() for key,file in paths.items()}}
    review=path.parent/'source_review.json';review.write_text(json.dumps(reviewed,indent=2)+'\n')
    look['optical_print']={key:str(file.relative_to(s.dir)) for key,file in {**paths,'review':review}.items()}
    look.pop('pending_optical_print')
    (s.config_dir/'scene_graph.json').write_text(json.dumps(graph,indent=2)+'\n')
    return reviewed


if __name__ == '__main__':main()
