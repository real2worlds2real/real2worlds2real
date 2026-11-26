"""Keep reviewed blurry facets native while preserving GS on other surfaces.

This changes appearance cutout masks only. Field rows, native geometry and
frozen foreground remain unchanged; fresh composite/free-view QA is required.
"""
import copy
import argparse
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from calibrated_scene_reconstruction.scene.review.environment import digest, protected_state


def complete_facets(image, payload, axis, cosine):
    axis = np.asarray(axis, float)
    if (axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-8
            or isinstance(cosine, bool) or not .8 <= cosine <= 1.):
        raise ValueError('Explicit outward direction and bounded facet angle required')
    axis = axis/np.linalg.norm(axis)
    world, counts, indices, uv = [np.asarray(payload[k]) for k in ('world', 'counts', 'indices', 'uv')]
    if (uv.shape != (len(indices), 2) or not np.isfinite(uv).all()
            or np.any(uv < 0) or np.any(uv > 1) or counts.sum() != len(indices)):
        raise ValueError('Finite original per-face-vertex atlas coordinates required')
    # crop_gaps saves PNG rows flipped relative to its USD UV coordinates.
    atlas = np.flipud(image).copy(); h, w = atlas.shape
    selected = np.zeros_like(atlas); other = np.zeros_like(atlas); offset = 0; faces = 0
    for count in counts:
        vertices = world[indices[offset:offset+count]]
        normal = np.cross(vertices[1]-vertices[0], vertices[2]-vertices[0])
        normal /= max(1e-12, np.linalg.norm(normal))
        chosen = float(normal@axis) >= cosine
        polygon = np.rint(uv[offset:offset+count]*[w, h]).astype('int32')
        cv2.fillPoly(selected if chosen else other, [polygon], 255)
        faces += int(chosen); offset += count
    if not faces or np.any((selected > 0)&(other > 0)):
        raise ValueError('Native facets must have distinct nonoverlapping atlas charts')
    paint = (cv2.dilate(selected, np.ones((3, 3), 'uint8')) > 0)&(other == 0)
    changed = int(np.count_nonzero(paint&(atlas < 255)))
    if not changed:
        raise ValueError('Selected native facets have no GS cutouts to close')
    atlas[paint] = 255
    return np.flipud(atlas), {'faces': faces, 'changed_pixels': changed}


def prepare(session, context_path, request_path, output):
    from calibrated_scene_reconstruction.scene.background.blender_env import verify
    from calibrated_scene_reconstruction.scene.background.environment_physics import current
    from calibrated_scene_reconstruction.scene.background.context_rebind import geometry_signature
    context_path, request_path, out = [Path(p).resolve() for p in (context_path, request_path, output)]
    if out.exists() or not out.is_relative_to(session.dir.resolve()) or (session/'qa/independent_freeze.json').exists():
        raise ValueError('New unfrozen native-facet proposal required')
    built = verify(session); physical, _ = current(session)
    context, request = [json.loads(p.read_text()) for p in (context_path, request_path)]
    if built['splat_context']['evidence_sha256'] != digest(context_path):
        raise ValueError('Use the currently bound retained field')
    review_path = Path(request['render_review']).resolve(); review = json.loads(review_path.read_text())
    if (review.get('diagnostic') or review['usd'] != physical['usd']
            or review['environment_context']['evidence_sha256'] != digest(context_path)
            or context['protected_task_state'] != protected_state(session)):
        raise ValueError('Actual current composed source/free views required')
    job_path = Path(json.loads((session/'qa/environment_physics/twin/current.json').read_text())['path']).parent/'job.json'
    job = json.loads(job_path.read_text())
    if geometry_signature(job) != context['selection_frozen']['geometry_signature']:
        raise ValueError('Native geometry changed since coverage measurement')
    inputs = {**context['input_sha256'], **review['input_sha256']}
    for p in (context_path, request_path, review_path, job_path, Path(job['geometry']), Path(__file__).resolve()):
        inputs[str(p)] = digest(p)
    for surface in request['surfaces']:
        evidence = set(surface.get('evidence', []))
        paired = [v for v in review['views'] if {v['source'], *v['renders']} <= evidence]
        if not surface.get('basis', '').strip() or len(paired) < 2:
            raise ValueError('Two reviewed captured/free-view pairs and an explicit facet repair basis required')
        for p in evidence: inputs[str(Path(p).resolve())] = digest(p)
    if any(digest(p) != sha for p, sha in inputs.items()):
        raise ValueError('Native-facet source evidence changed')
    out.mkdir(parents=True); result = copy.deepcopy(context); reports = []
    for index, surface in enumerate(request['surfaces']):
        owner, part = surface['owner'], surface['part']
        paths = [r['path'] for r in job['parts'] if r['entity'] == owner and r.get('part') == part and not r['external']]
        if (len(paths) != 1 or context['appearance_ownership'].get(owner, {}).get('preserve_gs') is not True
                or paths[0] not in result['gap_masks'].get(owner, {}).get('meshes', {})):
            raise ValueError('Explicit existing background part with GS cutouts required')
        payload = result['gap_masks'][owner]['meshes'][paths[0]]
        if digest(payload['path']) != payload['sha256'] or digest(payload['uv_path']) != payload['uv_sha256']:
            raise ValueError('Native cutout payload changed')
        with np.load(payload['uv_path']) as coordinates:
            image, report = complete_facets(cv2.imread(payload['path'], 0), coordinates,
                                           surface['outward_axis'], surface.get('min_cosine', .95))
        path = out/('native_facets_'+str(index)+'.png'); cv2.imwrite(str(path), image)
        payload.update(path=str(path), sha256=digest(path), visible_fraction=float((image > 0).mean()))
        inputs[str(path)] = digest(path); reports.append({**surface, **report, 'mesh': paths[0]})
    shutil.copyfile(context_path.parent/'context.ply', out/'context.ply')
    inputs[str(out/'context.ply')] = digest(out/'context.ply')
    result.update(input_sha256=inputs, photorealism_approved=False,
                  native_surface_completion={'surfaces': reports, 'field_rows_unchanged': True, 'appearance_pass': False})
    if any(digest(p) != sha for p, sha in inputs.items()):
        raise ValueError('Native-facet evidence changed during preparation')
    (out/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('session', 'context', 'request'): parser.add_argument(name)
    parser.add_argument('--output', required=True); args = parser.parse_args()
    result = prepare(Session(args.session), args.context, args.request, args.output)
    print('Unapproved native-facet proposal:', len(result['native_surface_completion']['surfaces']))


if __name__ == '__main__': main()
