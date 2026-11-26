"""Compose explicit candidates for static native appearance review, never delivery.

Unlike a production export, this preserves failed source candidates to reveal
their appearance together. It records those failures, authors no object physics
or acceptance receipts, and writes only a new session QA directory.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

from calibrated_scene_reconstruction.assets.metric_mesh import asset
from calibrated_scene_reconstruction.assets.object_pose import rotation
from calibrated_scene_reconstruction.core.paths import resolve
from calibrated_scene_reconstruction.scene.background.blender_env import verify
from calibrated_scene_reconstruction.scene.capture.articulated_model import from_usd
from calibrated_scene_reconstruction.scene.capture.render_robot import capture_spec
from calibrated_scene_reconstruction.scene.review.environment import digest
from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.support.runtime_inputs import models
from calibrated_scene_reconstruction.robots.spec import load_spec
from calibrated_scene_reconstruction.sim.object_materials import add_surface
from calibrated_scene_reconstruction.sim.support_stage import add_support
from calibrated_scene_reconstruction.sim.usd import quat_from_matrix, set_pose


def run(session, request, output):
    request, output = Path(request).resolve(), Path(output).resolve()
    if output.exists() or not output.is_relative_to((session / 'qa').resolve()):
        raise ValueError('Candidate composition requires a new session QA directory')
    q = json.loads(request.read_text())
    from calibrated_scene_reconstruction.core.capabilities import require
    for key in q.keys() - {'object_rounds'}:
        require('candidate_fields', key)
    graph = session.scene_graph
    if set(q['object_rounds']) != set(graph['dynamic_objects']):
        raise ValueError('Explicit round selection must cover every scene object')
    environment = verify(session)
    inputs = {str(request): digest(request),
              str(session.config_dir / 'scene_graph.json'): digest(session.config_dir / 'scene_graph.json'),
              str(Path(__file__).resolve()): digest(__file__)}

    def bind(path):
        path = Path(path).resolve()
        inputs[str(path)] = digest(path)
        return path

    environment_path = bind(environment['asset'])
    output.mkdir(parents=True)
    stage = Usd.Stage.CreateNew(str(output / 'candidate.usdc'))
    world = UsdGeom.Xform.Define(stage, '/World')
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    env_prim = UsdGeom.Xform.Define(stage, '/World/Environment').GetPrim()
    env_prim.GetReferences().AddReference(str(environment_path))
    env_meta = dict(Usd.Stage.Open(str(environment_path)).GetRootLayer().customLayerData)
    support = graph['task']['support']
    support_model = models(session)[support]
    add_support(stage, '/World/Room/' + support, support_model, visual_only=True)
    for key in ('mesh', 'visual_glb'):
        bind(resolve(graph['furniture'][support]['fix'][key]))
    records = {}
    for name, number in q['object_rounds'].items():
        if type(number) is not int or number < 1:
            raise ValueError('Candidate rounds must be explicit positive integers')
        folder = session / f'objects/{name}/round_{number}'
        fit = json.loads(bind(folder / 'fit.json').read_text())
        mesh = bind(folder / 'mesh.npz')
        pose = fit['pose']
        spec = graph['dynamic_objects'][name].get('fix') or {}
        # A current appearance atlas belongs to one exact surface, not its name.
        look = spec.get('look') if spec.get('mesh') and resolve(spec['mesh']).resolve() == mesh else None
        kit = asset({'mesh': str(mesh), 'color': spec.get('color', [.75, .75, .75])})
        root = UsdGeom.Xform.Define(stage, '/World/Objects/' + name)
        set_pose(root.GetPrim(), [*pose['xy'], pose['support_z']], quat_from_matrix(rotation(pose)))
        face_uv = None
        if look:
            if look.get('source_mesh_sha256') != digest(mesh) or not look.get('preserves_surface'):
                raise ValueError('Candidate texture must preserve the exact selected surface')
            with np.load(bind(session / look['mesh'])) as data:
                face_uv = data['face_uv'].copy()
            bind(session / look['albedo'])
        offset = 0
        for part in kit['parts']:
            count = len(part['F'])
            surface = {**part, 'UV': face_uv[offset:offset+count] if face_uv is not None else part.get('UV')}
            add_surface(stage, str(root.GetPath()) + '/' + part['name'], surface,
                        look=look, look_base=session.dir)
            offset += count
        if face_uv is not None and offset != len(face_uv):
            raise ValueError('Candidate appearance face count differs from geometry')
        records[name] = {'round': number, 'mesh': str(mesh), 'pose': pose,
                         'source_metrics': fit.get('metrics'), 'source_pass': fit.get('pass', fit.get('ok')),
                         'appearance_atlas': bool(look), 'accepted': False}

    photo = graph['photo_robot']
    spec = capture_spec(session, load_spec(photo['spec']))
    robot_asset = bind(spec['usd'])
    bind(session / 'parsing/robot_visual_state.json')
    model = from_usd(robot_asset, points_per_link=8)
    fk = model.transforms(spec['home'])
    robot = UsdGeom.Xform.Define(stage, '/World/Robot').GetPrim()
    robot.GetReferences().AddReference(str(robot_asset))
    base = np.eye(4)
    base[:3, :3] = rotation({'yaw_deg': photo['base']['yaw_deg']})
    base[:3, 3] = [*photo['base']['xy'], spec['base']['height']]
    body_prims = {}
    for prim in Usd.PrimRange(robot):
        if prim.GetName() in fk:
            if prim.GetName() in body_prims:
                raise ValueError('Ambiguous robot CAD body names')
            body_prims[prim.GetName()] = prim
    if set(body_prims) != set(fk):
        raise ValueError('Candidate robot reference is missing FK bodies')
    cache = UsdGeom.XformCache()
    for name, transform in fk.items():
        prim = body_prims[name]
        parent = cache.GetLocalToWorldTransform(prim.GetParent())
        target = Gf.Matrix4d((base @ transform).T.tolist())
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTransformOp(opSuffix='candidate').Set(target * parent.GetInverse())
        cache.Clear()

    extra_records = {}
    from calibrated_scene_reconstruction.core.capabilities import entries, invoke
    for name in entries('candidate_composition'):
        extra_records.update(invoke('candidate_composition', name, q, stage, bind))
    # No dynamics in a static appearance review, even for referenced robot CAD.
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.CreateAttribute('physxArticulation:articulationEnabled', Sdf.ValueTypeNames.Bool).Set(False)
    # Track referenced USDs and texture assets, not just the top-level file.
    for layer in stage.GetUsedLayers():
        if not layer.anonymous and layer.realPath and Path(layer.realPath).resolve() != output / 'candidate.usdc':
            bind(layer.realPath)
    for prim in stage.Traverse():
        for attribute in prim.GetAttributes():
            value = attribute.Get() if attribute.GetTypeName() == Sdf.ValueTypeNames.Asset else None
            if value and value.resolvedPath and Path(value.resolvedPath).is_file():
                bind(value.resolvedPath)
    report = {'accepted': False, 'physics_validated': False, 'scope': __doc__.strip(),
              'objects': records, 'input_sha256': inputs,
              'environment_round': environment.get('round'), 'robot': photo, **extra_records}
    stage.GetRootLayer().customLayerData = {
        'r2s2r_candidate_only': True, 'r2s2r_candidate_physics_validated': False,
        'r2s2r_environment_render': env_meta['r2s2r_environment_render'],
        'r2s2r_environment': {'asset_sha256': digest(environment_path), 'variant': 'twin'}}
    stage.GetRootLayer().Save()
    if any(digest(path) != sha for path, sha in inputs.items()):
        raise ValueError('Composition input changed during assembly')
    report['output_sha256'] = digest(output / 'candidate.usdc')
    (output / 'review.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session')
    parser.add_argument('request')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(Session(args.session), args.request, args.output)
    print('Static candidate assembled; no source, physics or delivery approval.')


if __name__ == '__main__':
    main()
