"""Validate a photographic native backing before preserving it under GS.

A cutout can expose an unrelated dark inner wall. An explicitly captured skin
may instead stay opaque, retaining its complete original emission network. This
is source-lit appearance, not recovered reflectance or scene acceptance.
"""
import json
from pathlib import Path
import numpy as np
from calibrated_scene_reconstruction.scene.review.environment import digest, protected_state


def geometry_binding(prim, world_quad):
    """Check the actual exported pose and UVs against the baked source plane.

    A thin skin has front/rear faces and bevels. Allow at most 20 mm normal
    thickness and 3 mm tangent displacement; a shifted object or swapped UV
    orientation cannot reuse its old source projection.
    """
    from pxr import UsdGeom
    quad = np.asarray(world_quad, float)
    if (not prim or not prim.IsA(UsdGeom.Mesh) or quad.shape != (4, 3)
            or not np.isfinite(quad).all()):
        raise ValueError('Photographic backing needs its actual mesh and finite world quad')
    normal = np.cross(quad[1] - quad[0], quad[3] - quad[0])
    if np.linalg.norm(normal) < 1e-8:
        raise ValueError('Photographic world quad is degenerate')
    normal /= np.linalg.norm(normal)
    if abs((quad[2] - quad[0]) @ normal) > 1e-5:
        raise ValueError('Photographic world quad must be planar')
    mesh = UsdGeom.Mesh(prim)
    local = np.asarray(mesh.GetPointsAttr().Get(), float)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), int)
    uv = UsdGeom.PrimvarsAPI(prim).GetPrimvar('st')
    if not uv or not uv.HasValue() or not len(indices):
        raise ValueError('Photographic backing needs its actual UVs')
    coords = np.asarray(uv.ComputeFlattened(), float)
    if uv.GetInterpolation() in ('vertex', 'varying'):
        coords = coords[indices]
    elif uv.GetInterpolation() != 'faceVarying':
        raise ValueError('Photographic backing needs vertex or face-varying UVs')
    if coords.shape != (len(indices), 2) or not np.isfinite(coords).all():
        raise ValueError('Photographic backing has invalid UVs')
    transform = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
    world = (np.c_[local, np.ones(len(local))] @ transform)[:, :3][indices]
    u, v = coords.T
    expected = ((1-u)[:, None]*(1-v)[:, None]*quad[0]
                + u[:, None]*(1-v)[:, None]*quad[1]
                + u[:, None]*v[:, None]*quad[2]
                + (1-u)[:, None]*v[:, None]*quad[3])
    delta = world - expected
    thickness = delta @ normal
    tangent = delta - thickness[:, None]*normal
    if (np.max(abs(thickness)) > .020001
            or np.max(np.linalg.norm(tangent, axis=1)) > .003001):
        raise ValueError('Photographic projection no longer matches the actual mesh pose/UVs; rebake it')


def validate(session, proof_path, material, mesh=None):
    from pxr import UsdShade
    path=Path(proof_path).resolve();proof=json.loads(path.read_text());camera=session/'parsing/posed_frames/cameras.npz'
    source=Path(proof['source']).resolve();quad=np.asarray(proof.get('source_quad'),float)
    if (proof.get('version') not in (1,2) or proof.get('protected_task_state')!=protected_state(session)
            or proof.get('camera_sha256')!=digest(camera) or source!= (session/'splat/data/images'/source.name).resolve()
            or source.name not in set(map(str,np.load(camera)['names'])) or quad.shape!=(4,2) or not np.isfinite(quad).all()):
        raise ValueError('Captured backing needs current source/camera/task provenance')
    import cv2
    from PIL import Image
    width,height=Image.open(source).size
    if (not cv2.isContourConvex(quad.astype('float32')) or cv2.contourArea(quad.astype('float32'))<16
            or np.any(quad<0) or np.any(quad[:,0]>=width) or np.any(quad[:,1]>=height)):
        raise ValueError('Captured backing needs a bounded convex source quad')
    inputs={str(path):digest(path),str(Path(__file__).resolve()):digest(__file__),**proof['input_sha256']}
    for p,sha in [(source,proof['source_sha256']),(Path(proof['radiance']).resolve(),proof['radiance_sha256']),(camera,digest(camera))]:
        if inputs.get(str(p.resolve()))!=sha:raise ValueError('Captured backing files must be explicitly bound')
    for p,sha in inputs.items():
        if digest(p)!=sha:raise ValueError('Captured backing evidence changed')
    if proof['version'] == 2:
        geometry_binding(mesh, proof.get('world_quad'))
    surface=material.ComputeSurfaceSource()[0] if material else None
    emission=surface.GetInput('emissiveColor') if surface else None
    if not emission or not emission.HasConnectedSource():raise ValueError('Backing must retain its actual captured emission texture')
    connected=[emission.GetConnectedSource()[0]];seen=set();matched=False
    while connected:
        node=connected.pop();key=str(node.GetPath())
        if key in seen:continue
        seen.add(key);shader=UsdShade.Shader(node.GetPrim())
        if shader.GetIdAttr().Get()=='UsdUVTexture':
            value=shader.GetInput('file').Get();image=Path(value.resolvedPath or value.path).resolve()
            if image.is_file() and digest(image)==proof['radiance_sha256']:
                matched=True;inputs[str(image)]=digest(image)
        for plug in node.GetInputs():
            if plug.HasConnectedSource():connected.append(plug.GetConnectedSource()[0])
    if not matched:raise ValueError('Backing texture is not connected to the declared photographic radiance')
    return inputs


def prepare(session, surfaces, asset, frame, mesh_part, output):
    """Bind a freshly baked photographic plane to the actual native mesh.

    The bounded source quad describes a visible footprint, not full coverage.
    Unobserved texels keep the baker's declared fallback and coverage mask.
    """
    from PIL import Image
    from pxr import Usd, UsdShade, UsdGeom
    from calibrated_scene_reconstruction.scene.background.blender_env import verify
    out = Path(output).resolve(); surfaces = Path(surfaces).resolve()
    if (out.exists() or not out.is_relative_to(session.dir.resolve())
            or (session/'qa/independent_freeze.json').exists()):
        raise ValueError('New unfrozen session-local backing evidence required')
    native = verify(session)
    data = json.loads(surfaces.read_text())
    for p, sha in data['input_sha256'].items():
        if digest(p) != sha:
            raise ValueError('Photographic baking evidence changed')
    rows = [r for r in data['surfaces'] if r['id'] == asset and r.get('quad_world')]
    if len(rows) != 1 or asset not in native['captured_photos']:
        raise ValueError('One source-baked plane and actual captured radiance required')
    row = rows[0]; photo = native['captured_photos'][asset]
    if row['sha256'] != photo['source_sha256']:
        raise ValueError('Native photograph differs from the freshly baked plane')
    source = (session/'splat/data/images'/frame).resolve()
    camera = session/'parsing/posed_frames/cameras.npz'
    if Path(frame).name != frame or data['input_sha256'].get(str(source)) != digest(source):
        raise ValueError('The source frame must participate in the actual texture bake')
    cameras = np.load(camera); i = list(map(str, cameras['names'])).index(frame)
    quad = np.asarray(row['quad_world'], float)
    pc = (quad-cameras['centre'][i]) @ cameras['R_world_cam'][i]
    if np.any(pc[:, 2] <= .05):
        raise ValueError('Photographic plane is behind its source camera')
    pixels = pc @ cameras['K'][i].T; pixels = pixels[:, :2]/pixels[:, 2, None]
    width, height = Image.open(source).size
    lo = np.maximum(pixels.min(0), 0); hi = np.minimum(pixels.max(0), [width-1, height-1])
    if np.any(hi-lo < 4):
        raise ValueError('Photographic plane has no useful visible source footprint')
    stage = Usd.Stage.Open(native['asset'])
    meshes = [p for p in stage.Traverse() if mesh_part in str(p.GetPath()).split('/')
              and p.IsA(UsdGeom.Mesh)]
    if len(meshes) != 1:
        raise ValueError('One actual photographic mesh part required')
    coverage = Path(row['coverage']).resolve()
    inputs = dict(data['input_sha256'])
    paths = [surfaces, source, camera, coverage, Path(row['texture']),
             Path(photo['radiance']), Path(photo['response']), Path(__file__).resolve()]
    inputs.update({str(p.resolve()): digest(p) for p in paths})
    proof = {'version': 2, 'source': str(source), 'source_sha256': digest(source),
             'source_quad': [[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]],
             'world_quad': row['quad_world'], 'radiance': photo['radiance'],
             'radiance_sha256': photo['radiance_sha256'], 'camera_sha256': digest(camera),
             'protected_task_state': protected_state(session), 'input_sha256': inputs,
             'partial_capture': {'coverage': str(coverage), 'coverage_sha256': digest(coverage),
                                 'observed_fraction': row['observed_fraction'],
                                 'full_source_quad': pixels.tolist(), 'unseen_fill': row['unseen_fill']},
             'scope': prepare.__doc__.strip(), 'approved': False}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(proof, indent=2)+'\n')
    material = UsdShade.MaterialBindingAPI(meshes[0]).ComputeBoundMaterial()[0]
    try:
        validate(session, out, material, meshes[0])
    except BaseException:
        out.unlink()
        raise
    return proof


def main():
    import argparse
    from calibrated_scene_reconstruction.scene.session import Session
    parser = argparse.ArgumentParser(description=prepare.__doc__)
    for name in ['session', 'surfaces', 'asset', 'frame', 'mesh_part']:
        parser.add_argument(name)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    prepare(Session(args.session), args.surfaces, args.asset, args.frame, args.mesh_part, args.output)
    print('Unapproved photographic backing bound to actual mesh:', args.output)


if __name__ == '__main__':
    main()
