"""Read actual evaluated Blender geometry without modifying the editable source."""
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

if not __package__:  # Blender also executes this file directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from real2sim_harness.asset_metadata import asset_value

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from real2sim_harness.artifacts import RunStore, atomic_json  # noqa: E402
from real2sim_harness.blender_surface import is_asset_object  # noqa: E402
from real2sim_harness.measurement import inputs, load_metric_points, rigid_transform  # noqa: E402


def main():
    arguments = sys.argv[sys.argv.index('--') + 1:]
    store, output = RunStore(Path(arguments[0])), Path(arguments[1])
    spec = inputs(store)
    graph = bpy.context.evaluated_depsgraph_get()
    objects, all_points, vertices, triangles = [], [], [], []
    offset = 0
    for obj in bpy.context.scene.objects:
        if obj.type != 'MESH' or not is_asset_object(obj) or obj.hide_render:
            continue
        evaluated = obj.evaluated_get(graph)
        mesh = evaluated.to_mesh()
        mesh.calc_loop_triangles()
        local = np.asarray([vertex.co[:] for vertex in mesh.vertices], dtype=float).reshape(-1, 3)
        matrix = np.asarray(obj.matrix_world)
        world = local @ matrix[:3, :3].T + matrix[:3, 3]
        if len(world):
            objects.append({'name': obj.name, 'part': str(asset_value(obj, 'real2sim_measurement_part', obj.get('part', obj.get('semantic_part', '')))), 'points': world})
            all_points.append(world)
            vertices.extend(world.tolist())
            triangles.extend(tuple(offset + index for index in face.vertices) for face in mesh.loop_triangles)
            offset += len(world)
        evaluated.to_mesh_clear()
    if not all_points or not triangles:
        raise ValueError('No visible evaluated asset geometry to measure')
    points = np.concatenate(all_points)
    if not np.isfinite(points).all():
        raise ValueError('Nonfinite evaluated geometry')
    dimensions = []
    for constraint in spec.get('dimensions', []):
        selected = [row['points'] for row in objects if constraint['part'] is None
                    or constraint['part'] in {row['name'], row['part']}]
        if not selected:
            dimensions.append({**constraint, 'ok': False, 'error': 'requested part was not found'})
            continue
        extent = np.ptp(np.concatenate(selected), axis=0)
        measured = float(extent['XYZ'.index(constraint['axis'])])
        error = abs(measured - constraint['value_m'])
        dimensions.append({**constraint, 'actual_m': measured, 'absolute_error_m': error,
                           'suggested_uniform_scale': constraint['value_m'] / measured if measured > 0 else None,
                           'ok': error <= constraint['tolerance_m']})
    report = {'schema_version': 1, 'ok': all(row['ok'] for row in dimensions), 'dimensions': dimensions,
              'evaluated_vertices': len(points), 'evaluated_triangles': len(triangles),
              'bounds_m': {'min': points.min(axis=0).tolist(), 'max': points.max(axis=0).tolist()},
              'geometry_modified': False, 'scope': 'world-axis bounds and observed-point distances; no physical calibration'}
    if spec.get('surface'):
        surface = spec['surface']
        observed = load_metric_points(store.root / surface['path'])
        count = min(len(observed), surface['max_points'])
        selection = np.sort(np.random.default_rng(0).choice(len(observed), count, replace=False))
        observed = observed[selection]
        transform = rigid_transform(surface['measured_to_asset'])
        observed = observed @ transform[:3, :3].T + transform[:3, 3]
        tree = BVHTree.FromPolygons(vertices, triangles, all_triangles=True, epsilon=0.0)
        distances, closest = [], []
        for point in observed:
            position, _, _, distance = tree.find_nearest(Vector(point))
            if position is None or not np.isfinite(distance):
                raise ValueError('Invalid point-to-surface distance')
            distances.append(float(distance))
            closest.append(list(position))
        distances = np.asarray(distances)
        metrics = {'observed_points_total': int(len(load_metric_points(store.root / surface['path']))),
                   'evaluated_points': count, 'selection': 'deterministic uniform subset without replacement',
                   'mean_m': float(distances.mean()), 'rmse_m': float(np.sqrt(np.mean(distances**2))),
                   'p95_m': float(np.percentile(distances, 95)), 'max_m': float(distances.max()),
                   'within_tolerance_fraction': float(np.mean(distances <= surface['tolerance_m'])),
                   'tolerance_m': surface['tolerance_m'], 'registration': 'explicit rigid transform; no fitted scale or ICP',
                   'scope': 'observed points to exact evaluated triangles; no completeness or hidden-surface claim'}
        metrics['ok'] = metrics['p95_m'] <= surface['tolerance_m']
        report['surface'] = metrics
        report['ok'] = report['ok'] and metrics['ok']
        np.savez_compressed(output / 'surface_residuals.npz', observed_points_m=observed,
                            closest_points_m=np.asarray(closest), distance_m=distances, selected_indices=selection)
    atomic_json(output / 'measurement.json', report)


if __name__ == '__main__':
    main()
