"""Inspect actual replay geometry for transverse cloth/rigid surface crossings.

Runs in Blender on a source scene. This checks saved motion frames, not continuous
collision detection between them. Surface intersections are not penetration depths.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import bpy
from mathutils.bvhtree import BVHTree
from mathutils.geometry import intersect_ray_tri

if not __package__:  # Blender also executes this file directly.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from real2sim_harness.asset_metadata import asset_key

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'real2sim_harness'))
from blender_hybrid_harness import load_player  # noqa: E402


def snapshot(obj, depsgraph, *, raw=False):
    evaluated = obj if raw else obj.evaluated_get(depsgraph)
    mesh = obj.data if raw else evaluated.to_mesh()
    try:
        mesh.calc_loop_triangles()
        vertices = [evaluated.matrix_world @ vertex.co for vertex in mesh.vertices]
        triangles = [tuple(triangle.vertices) for triangle in mesh.loop_triangles]
        return vertices, triangles, BVHTree.FromPolygons(vertices, triangles, all_triangles=True, epsilon=0.0)
    finally:
        if not raw:
            evaluated.to_mesh_clear()


def crossing(first, second, tolerance=1e-7):
    """Find an edge crossing the other triangle plane, excluding mere touching."""
    for edges, face in ((first, second), (second, first)):
        normal = (face[1]-face[0]).cross(face[2]-face[0])
        if normal.length < 1e-14:
            continue
        normal.normalize()
        for index in range(3):
            start, end = edges[index], edges[(index+1) % 3]
            a, b = (start-face[0]).dot(normal), (end-face[0]).dot(normal)
            if not (min(a, b) < -tolerance and max(a, b) > tolerance):
                continue
            direction = end-start
            point = intersect_ray_tri(*face, direction, start, True)
            if point is not None and (point-start).dot(direction) <= direction.length_squared:
                return list(point)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--end-frame', type=int)
    parser.add_argument('--stride', type=int, default=1)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    configuration = json.loads((args.folder/'hybrid_configuration.json').read_text())
    collisions = json.loads((args.folder/'collision_meshes.json').read_text())
    apply, data, report, manifest = load_player(args.folder)
    control_name = configuration['control_object']
    surfaces = [('control', control_name)]
    if not bpy.data.objects[control_name].get(asset_key(bpy.data.objects[control_name], 'real2sim_sim_proxy')) and not bpy.data.objects[control_name].hide_render:
        surfaces.append(('visual', control_name))
    surfaces += [('visual', name) for name in manifest['deformable_visual_objects'] if name != control_name]
    rigid_names = sorted({part['object_name'] for part in collisions})
    end = len(data['times']) if args.end_frame is None else args.end_frame
    if args.stride < 1 or not 0 <= args.start_frame < end <= len(data['times']):
        raise ValueError('Invalid motion interval or stride')
    results = []
    pairs = Counter()
    for frame in range(args.start_frame, end, args.stride):
        apply(frame)
        depsgraph = bpy.context.evaluated_depsgraph_get()
        rigid = {name: snapshot(bpy.data.objects[name], depsgraph) for name in rigid_names}
        hits = []
        for layer, name in surfaces:
            vertices, triangles, tree = snapshot(bpy.data.objects[name], depsgraph, raw=layer == 'control')
            for other, (rigid_vertices, rigid_triangles, rigid_tree) in rigid.items():
                count, witnesses = 0, []
                for first, second in tree.overlap(rigid_tree):
                    witness = crossing([vertices[i] for i in triangles[first]],
                                       [rigid_vertices[i] for i in rigid_triangles[second]])
                    if witness is not None:
                        count += 1
                        if len(witnesses) < 3:
                            witnesses.append(witness)
                if count:
                    pairs[(layer, name, other)] += 1
                    hits.append({'surface_layer': layer, 'surface': name, 'rigid_object': other,
                                 'crossing_triangle_pairs': count, 'witness_points_m': witnesses})
        if hits:
            results.append({'cache_frame': frame, 'time_seconds': float(data['times'][frame]), 'crossings': hits})
        if frame % 12 == 0:
            print('INTERSECTION_FRAME', frame, 'pairs', len(hits), flush=True)
    checked = list(range(args.start_frame, end, args.stride))
    result = {'ok': not results, 'check': 'transverse_triangle_surface_crossings',
              'scope': 'cloth control and evaluated visual surfaces against authored rigid collider surfaces',
              'limitations': 'Saved-frame surface intersection check; not continuous collision detection, containment, self-contact or signed penetration depth. Contact-only touching is excluded.',
              'checked_cache_frames': checked, 'complete_cache_coverage': checked == list(range(len(data['times']))),
              'frames_with_crossings': len(results),
              'object_pair_frame_counts': [{'surface_layer': key[0], 'surface': key[1], 'rigid_object': key[2], 'frames': value}
                                            for key, value in pairs.most_common()], 'failures': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print('INTERSECTION_RESULT', result['ok'], len(results), 'frames', flush=True)


if __name__ == '__main__':
    main()
