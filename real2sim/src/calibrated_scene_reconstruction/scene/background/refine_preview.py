"""Preview source-fit background pose proposals without changing active models.

Writes a diagnostic USD and physical-probe geometry. This is never an accepted
native build; source/free-view and support checks precede canonical adoption.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from calibrated_scene_reconstruction.scene.background.blender_env import verify
from calibrated_scene_reconstruction.scene.background.environment_physics import current, extract
from calibrated_scene_reconstruction.scene.background.refine import rotation
from calibrated_scene_reconstruction.scene.review.ablation import copy_stage
from calibrated_scene_reconstruction.scene.review.environment import digest


def transform(points, before, after):
    points = np.asarray(points, float)
    scale = np.asarray(after['size'])/np.asarray(before['size'])
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Finite geometry and positive relative size required')
    old, new = rotation(np.radians(before.get('yaw_deg', 0))), rotation(np.radians(after.get('yaw_deg', 0)))
    return ((points-np.asarray(before['position']))@old*scale)@new.T+np.asarray(after['position'])


def prepare(session, proposal, output, entities=None):
    from pxr import Usd, UsdGeom, Vt
    from calibrated_scene_reconstruction.scene.review import frozen_scope
    from calibrated_scene_reconstruction.scene.review.frozen_scope_cache import same_geometry
    proposal, out = Path(proposal).resolve(), Path(output).resolve()
    if out.exists() or not out.is_relative_to(session.dir.resolve()) or (session/'qa/independent_freeze.json').exists():
        raise ValueError('New unfrozen session-local preview required')
    native = verify(session); audit, _ = current(session)
    data = json.loads(proposal.read_text())
    if any(digest(p) != sha for p, sha in data['input_sha256'].items()):
        raise ValueError('Source-fit proposal evidence changed')
    snapshot = json.loads((proposal.parent/'environment_input.json').read_text())
    originals = {e['id']: e for e in snapshot['parameters']['entities']}
    active = {e['id']: e for e in session.config('environment.json')['parameters']['entities']}
    rows = [r for r in data['results'] if entities is None or r['entity'] in entities]
    if not rows or any(r['entity'] not in active or originals[r['entity']] != active[r['entity']] for r in rows):
        raise ValueError('Preview only proposals from the current background entity definitions')
    if entities is not None and set(entities) != {r['entity'] for r in rows}:
        raise ValueError('Requested preview entity absent from proposal')
    out.mkdir(parents=True)
    stage = copy_stage(audit['usd'], out/'scene.usda', 'Unapproved source-fit background pose preview; frozen foreground unchanged')
    cache = UsdGeom.XformCache(); changes = {}
    for result in rows:
        owner = result['entity']; before = originals[owner]; count = 0
        fixed = result.get('fixed_parts', [])
        if (not isinstance(fixed,list)
                or any(not isinstance(p,str) or not p or '/' in p for p in fixed)
                or len(set(fixed)) != len(fixed)
                or (fixed and not str(result.get('articulation_review','')).strip())):
            raise ValueError('Fixed assembly parts require distinct names and an articulation review')
        fixed_meshes = {}; found = set()
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh) or not str(prim.GetPath()).startswith('/World/Environment/'):
                continue
            parent = prim; identity = None
            while parent and identity is None:
                identity = parent.GetAttribute('userProperties:r2s2r_entity').Get(); parent = parent.GetParent()
            if identity != owner:
                continue
            matched = set(fixed)&set(str(prim.GetPath()).split('/'))
            if matched:
                found.update(matched)
                fixed_meshes[str(prim.GetPath())] = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get()).copy()
                continue
            mesh = UsdGeom.Mesh(prim); matrix = np.asarray(cache.GetLocalToWorldTransform(prim)); inverse = np.linalg.inv(matrix)
            vertices = np.asarray(mesh.GetPointsAttr().Get(), float)
            world = (np.c_[vertices, np.ones(len(vertices))]@matrix)[:, :3]
            target = transform(world, before, result)
            local = (np.c_[target, np.ones(len(target))]@inverse)[:, :3]
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(local.astype('float32')))
            mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.array([local.min(0), local.max(0)], 'float32')))
            normals = mesh.GetNormalsAttr().Get()
            if normals is not None and len(normals):
                linear = (rotation(np.radians(before.get('yaw_deg', 0)))
                          @np.diag(np.asarray(result['size'])/np.asarray(before['size']))
                          @rotation(np.radians(result.get('yaw_deg', 0))).T)
                local_linear = matrix[:3, :3]@linear@inverse[:3, :3]
                normals = np.asarray(normals)@np.linalg.inv(local_linear).T
                normals /= np.maximum(np.linalg.norm(normals, axis=1)[:, None], 1e-12)
                mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals.astype('float32')))
            count += 1
        if not count:
            raise ValueError('Proposal owner has no actual background geometry')
        if found != set(fixed):
            raise ValueError('Fixed assembly part absent from actual exported geometry')
        if any(not np.array_equal(points, np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath(path)).GetPointsAttr().Get()))
               for path,points in fixed_meshes.items()):
            raise ValueError('Preview moved a fixed assembly part')
        changes[owner] = {'meshes': count, 'before': before, 'proposal': result,
                          'fixed_meshes_unchanged':list(fixed_meshes)}
    support = session.scene_graph['task']['support']; objects = session.scene_graph.get('dynamic_objects', {})
    original = Usd.Stage.Open(audit['usd'])
    if not same_geometry(frozen_scope.meshes(original, support, objects), frozen_scope.meshes(stage, support, objects)):
        raise ValueError('Preview changed frozen foreground triangles')
    stage.GetRootLayer().Save()
    parts, geometry = extract(stage, support, objects)
    np.savez_compressed(out/'geometry.npz', **geometry)
    job = {'geometry': str(out/'geometry.npz'), 'geometry_sha256': digest(out/'geometry.npz'),
           'parts': parts, 'contract': native['physics'], 'output': str(out/'physical_result.json'),
           'native_manifest_sha256': digest(session/'scene/environment/twin/current.json'),
           'external_entities': [support, *objects]}
    (out/'physical_job.json').write_text(json.dumps(job, indent=2)+'\n')
    inputs = {**data['input_sha256'], **{str(p): digest(p) for p in [proposal, Path(audit['usd']), Path(__file__).resolve()]}}
    record = {'diagnostic_only': True, 'approved': False, 'scene': str(out/'scene.usda'),
              'source_usd': audit['usd'], 'frozen_geometry_unchanged': True, 'changes': changes, 'input_sha256': inputs}
    (out/'preview.json').write_text(json.dumps(record, indent=2)+'\n')
    return record


def main():
    from calibrated_scene_reconstruction.scene.session import Session
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('session'); p.add_argument('proposal'); p.add_argument('--output', required=True)
    p.add_argument('--entities', nargs='+'); a = p.parse_args()
    result = prepare(Session(a.session), a.proposal, a.output, a.entities)
    print('Unapproved pose preview:', result['scene'])


if __name__ == '__main__':
    main()
