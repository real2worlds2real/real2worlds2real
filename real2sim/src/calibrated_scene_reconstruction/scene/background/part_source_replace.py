"""Replace a named native part only where source ownership AND 3D proximity agree.

The intermediate distance proposal is never adopted. This conservative route
preserves nearby other contents and same-ray geometry at unrelated depths. It
keeps all surviving original kernel rows unchanged and requires fresh opacity,
coverage, physical, source/free-view and full-scene appearance checks afterward.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess

import numpy as np
from plyfile import PlyData, PlyElement

from calibrated_scene_reconstruction.core.paths import ROOT
from calibrated_scene_reconstruction.scene.review.environment import digest


def intersection(near, owned):
    near, owned = np.asarray(near), np.asarray(owned)
    if near.dtype != bool or owned.dtype != bool or near.ndim != 1 or near.shape != owned.shape:
        raise ValueError('Aligned boolean original-field selections required')
    removed = near & owned
    if not removed.any() or removed.all():
        raise ValueError('A nonempty strict source-and-proximity subset required')
    return removed


def request(config):
    photos = config.get('photographic_parts', {})
    sources = config.get('source_replacements', {})
    parts = config.get('replacement_parts', {})
    if (len(photos) != 1 or set(sources) != set(photos) or set(parts) != set(photos)
            or config.get('solid_parts') or config.get('source_only_owners')):
        raise ValueError('One explicit photographed background owner, named parts and matching source masks required')
    distance = copy.deepcopy(config)
    distance.pop('source_replacements')
    return distance


def prepare(session, source_manifest, geometry_job, config, output):
    from calibrated_scene_reconstruction.scene.background import field_replace, source_replace
    config, output, source_manifest = map(lambda p: Path(p).resolve(), (config, output, source_manifest))
    if (output.exists() or not output.is_relative_to(session.dir.resolve())
            or (session/'qa/independent_freeze.json').exists()):
        raise ValueError('New unfrozen session-local part proposal required')
    cfg = json.loads(config.read_text()); distance_config = request(cfg)
    output.mkdir(parents=True)
    path = output/'distance_request.json'; path.write_text(json.dumps(distance_config, indent=2)+'\n')
    # Existing audited implementation validates actual named meshes, their
    # photographic backing and current passed physical geometry.
    trial = output/'distance_proposal'
    proposed = field_replace.prepare(session, source_manifest, geometry_job, path, trial)
    cloud = source_manifest.parent/'context.ply'
    field = PlyData.read(str(cloud))['vertex'].data
    matrix = np.asarray(json.loads(source_manifest.read_text())['matrix'])
    points = (np.c_[field['x'], field['y'], field['z'], np.ones(len(field))] @ matrix.T)[:, :3]
    owned, source_inputs, report = source_replace.prepare(session, points, cfg['source_replacements'], cfg['photographic_parts'])
    near = np.load(trial/'removed.npy'); removed = intersection(near, owned)
    np.save(output/'removed.npy', removed)
    np.savez_compressed(output/'source_rows.npz', retained=np.flatnonzero(~removed), removed=np.flatnonzero(removed))
    PlyData([PlyElement.describe(field[~removed], 'vertex')], text=False).write(str(output/'context.ply'))
    with (output/'transcode.log').open('w') as log:
        subprocess.run([str(ROOT/'envs/splat/3dgrut/.venv/bin/python'), '-m', 'threedgrut.export.scripts.transcode',
                        str(output/'context.ply'), '-o', str(output/'context.usdz'), '-f', 'nurec'],
                       check=True, stdout=log, stderr=subprocess.STDOUT)
    snapshot = output/'solver.py'; snapshot.write_bytes(Path(__file__).read_bytes())
    inputs = {**proposed['input_sha256'], **source_inputs}
    for p in [config, trial/'manifest.json', output/'removed.npy', output/'source_rows.npz', snapshot]:
        inputs[str(p)] = digest(p)
    result = copy.deepcopy(proposed)
    original = json.loads(source_manifest.read_text())
    result.update(asset_sha256=digest(output/'context.usdz'), retained=int((~removed).sum()),
                  removed=original.get('removed', 0)+int(removed.sum()), input_sha256=inputs)
    result['native_surface_replacement'].update(
        removed_rows=int(removed.sum()), row_indices_relative_to=str(cloud),
        source_replacements=report, selection_operation='source_intersection_named_part_proximity',
        distance_proposal=str(trial/'manifest.json'), distance_rows=int(near.sum()), source_rows=int(owned.sum()),
        preserved_from_distance=int((near & ~owned).sum()), preserved_from_source=int((owned & ~near).sum()),
        scope=__doc__.strip())
    if any(digest(p) != sha for p, sha in inputs.items()):
        raise ValueError('Part replacement evidence changed')
    (output/'manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['session', 'source_manifest', 'geometry_job', 'config']:
        parser.add_argument(name)
    parser.add_argument('--output', required=True); a = parser.parse_args()
    result = prepare(Session(a.session), a.source_manifest, a.geometry_job, a.config, a.output)
    print('Unapproved source-and-proximity part replacement:', result['native_surface_replacement']['removed_rows'])


if __name__ == '__main__':
    main()
