"""Compose native fluid dependencies and audit their static render surfaces.

Native surface generation must be checked in the runtime. Referencing only a
source World can discard its external PhysicsScene;
invisible particles then silently produce no visible fluid. Preserve the owner
explicitly and require native surfaces, without advancing the captured state.
"""
from pathlib import Path

import numpy as np
from pxr import Sdf, Usd, UsdGeom, UsdPhysics


def compose(stage, target, asset, *, source_root='/World', exclude=('Objects', 'Support')):
    asset = str(Path(asset).resolve())
    source = Usd.Stage.Open(asset)
    if (UsdGeom.GetStageMetersPerUnit(stage) != UsdGeom.GetStageMetersPerUnit(source)
            or UsdGeom.GetStageUpAxis(stage) != UsdGeom.GetStageUpAxis(source)):
        raise ValueError('Fluid references require matching stage units and up axis')
    origin = source.GetPrimAtPath(source_root)
    if not origin:
        raise ValueError('Fluid source root is missing')
    systems = [p for p in Usd.PrimRange(origin) if p.GetTypeName() == 'PhysxParticleSystem']
    if not systems:
        raise ValueError('Fluid candidate has no native particle systems')
    # Validate every owner before modifying the target.
    owners = {}
    for system in systems:
        paths = system.GetRelationship('simulationOwner').GetTargets()
        if (len(paths) != 1 or not source.GetPrimAtPath(paths[0])
                or not source.GetPrimAtPath(paths[0]).IsA(UsdPhysics.Scene)):
            raise ValueError('Fluid system needs one existing native PhysicsScene: '+str(system.GetPath()))
        owners[str(system.GetPath())] = paths[0]
    root = UsdGeom.Xform.Define(stage, target).GetPrim()
    root.GetReferences().SetReferences([Sdf.Reference(asset, source_root)])
    target, source_root = Sdf.Path(target), Sdf.Path(source_root)
    scenes = {}
    for system in systems:
        owner = owners[str(system.GetPath())]
        if owner.HasPrefix(source_root):
            mapped = owner.ReplacePrefix(source_root, target)
        else:
            if str(owner) not in scenes:
                mapped = target.AppendPath('PhysicsScenes/scene_'+str(len(scenes)))
                prim = stage.DefinePrim(mapped)
                prim.GetReferences().SetReferences([Sdf.Reference(asset, owner)])
                scenes[str(owner)] = mapped
            mapped = scenes[str(owner)]
        dest = stage.GetPrimAtPath(system.GetPath().ReplacePrefix(source_root, target))
        dest.GetRelationship('simulationOwner').SetTargets([mapped])
    for name in exclude:
        prim = stage.GetPrimAtPath(target.AppendChild(name))
        if prim:
            prim.SetActive(False)
    return {'systems': [str(p.GetPath().ReplacePrefix(source_root, target)) for p in systems],
            'source': asset, 'source_root': str(source_root),
            'external_scenes': {k: str(v) for k, v in scenes.items()},
            'accepted': False, 'physics_validated': False}


def snapshot(stage):
    """Capture all enabled fluid surfaces and their exact authored particles."""
    records = {}
    for prim in stage.Traverse():
        if prim.GetTypeName() != 'PhysxParticleSystem':
            continue
        enabled = prim.GetAttribute('physxParticleIsosurface:isosurfaceEnabled')
        if not enabled or enabled.Get() is not True:
            continue
        owners = prim.GetRelationship('simulationOwner').GetTargets()
        if (len(owners) != 1 or not stage.GetPrimAtPath(owners[0])
                or not stage.GetPrimAtPath(owners[0]).IsA(UsdPhysics.Scene)):
            raise ValueError('Unresolved fluid PhysicsScene: '+str(prim.GetPath()))
        particles = {}
        for points in stage.Traverse():
            if points.GetRelationship('physxParticle:particleSystem').GetTargets() != [prim.GetPath()]:
                continue
            name = 'positions' if points.IsA(UsdGeom.PointInstancer) else 'points'
            values = {}
            for key in (name, 'velocities', 'physxParticle:simulationPoints'):
                attr = points.GetAttribute(key)
                if attr and attr.HasAuthoredValue():
                    array = np.asarray(attr.Get(), float).copy()
                    if not np.isfinite(array).all():
                        raise ValueError('Nonfinite fluid state')
                    values[key] = array
            if name not in values or values[name].ndim != 2 or values[name].shape[1] != 3 or not len(values[name]):
                raise ValueError('Fluid has no finite particle positions')
            particles[str(points.GetPath())] = values
        if not particles:
            raise ValueError('Native surface has no particle set: '+str(prim.GetPath()))
        records[str(prim.GetPath())] = particles
    return records


def verify_surfaces(stage, before, *, output=None):
    """Fail a camera-only render if a fluid disappeared or its state moved."""
    after = snapshot(stage)
    if set(after) != set(before):
        raise ValueError('Fluid systems changed during static review')
    rows = {}
    for system, particles in before.items():
        if set(after[system]) != set(particles):
            raise ValueError('Fluid particle sets changed during static review')
        for path, attributes in particles.items():
            for name, values in attributes.items():
                if name not in after[system][path] or not np.array_equal(values, after[system][path][name]):
                    raise ValueError('Fluid state advanced during static review: '+path+'.'+name)
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(system+'/Isosurface'))
        if not mesh:
            raise ValueError('Native fluid isosurface missing: '+system)
        vertices = np.asarray(mesh.GetPointsAttr().Get(), float)
        faces = mesh.GetFaceVertexCountsAttr().Get()
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3
                or not np.isfinite(vertices).all() or not faces or mesh.ComputeVisibility() == 'invisible'):
            raise ValueError('Native fluid isosurface is empty, invalid or hidden: '+system)
        rows[system] = {'vertices': len(vertices), 'faces': len(faces),
                        'particles': sum(len(next(v for k, v in a.items() if k in ('points', 'positions'))) for a in particles.values()),
                        'particle_state_unchanged': True, 'native_surface_present': True}
        if output is not None:
            # Save the actual native render surface for diagnostics, rather
            # than reconstructing another mesh from particles after capture.
            from pathlib import Path
            folder = Path(output)
            folder.mkdir(parents=True, exist_ok=True)
            target = folder / f'surface_{len(rows)-1:02d}.npz'
            if target.exists():raise ValueError('Native surface snapshot already exists')
            transform = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(mesh.GetPrim()),float).T
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),np.int32)
            np.savez_compressed(target, points=vertices, face_counts=np.asarray(faces,np.int32),
                                face_indices=indices, world_transform=transform,
                                normals=np.asarray(mesh.GetNormalsAttr().Get() or [],float))
            import hashlib
            rows[system]['render_surface_snapshot'] = str(target.resolve())
            rows[system]['render_surface_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
    return rows


def verify_initialization(stage, before):
    """Bound an explicitly requested single-step render startup, not task physics.

The displacement limit is one quarter of the native particle spacing. Subsequent
render frames must use verify_surfaces against this new exact state, so this
allowance cannot conceal continuing drift or settling.
"""
    after = snapshot(stage)
    if set(before) != set(after):
        raise ValueError('Fluid systems changed during render initialization')
    result = {}
    for system, particles in before.items():
        if set(particles) != set(after[system]):
            raise ValueError('Fluid particle sets changed during render initialization')
        rest = stage.GetPrimAtPath(system).GetAttribute('fluidRestOffset').Get()
        if rest is None or not np.isfinite(rest) or rest <= 0:
            raise ValueError('Explicit native fluid resolution required for startup review')
        limit = .5 * rest
        rows = []
        for path, attributes in particles.items():
            key = 'points' if 'points' in attributes else 'positions'
            initial = attributes.get('physxParticle:simulationPoints', attributes[key])
            current = after[system][path].get('physxParticle:simulationPoints', after[system][path][key])
            if current.shape != initial.shape:
                raise ValueError('Fluid particle count changed during render initialization')
            distances = np.linalg.norm(current-initial, axis=1)
            if distances.max() > limit:
                raise ValueError('Fluid initialization moves beyond sub-resolution preview tolerance: '
                                 +system+f' max={distances.max():.9g}m limit={limit:.9g}m')
            rows.append({'path': path, 'particles': len(initial), 'max_displacement_m': float(distances.max()),
                         'p95_displacement_m': float(np.quantile(distances,.95)), 'limit_m': limit})
        result[system] = rows
    return result
