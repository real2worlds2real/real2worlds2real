"""Geometric contact witnesses across one folded, continuous material sheet.

Different layers may touch opposing solids. Rays through the local stack check
each intervening gap instead of demanding that a single vertex touch both
solids. This measures proximity, not compressive force or frictional retention.
"""
import numpy as np

from calibrated_scene_reconstruction.assets.cloth_topology import measure as topology


from calibrated_scene_reconstruction.assets.ray_triangles import hits as _hits


def measure(points, faces, first_distance, second_distance, first_normals,
            second_triangles, *, tolerance):
    """Use actual native poses and outward normals, in one shared frame.

Each layer-to-layer and final layer-to-solid gap must be within the existing
contact tolerance. No new larger folded-stack gap tolerance is introduced.
Disconnected sheets cannot provide a single-sheet witness. Independent finite
thickness, intersection, passive retention and opening-release tests remain.
"""
    p = np.asarray(points, float)
    f = np.asarray(faces)
    a, b = np.asarray(first_distance, float), np.asarray(second_distance, float)
    normals = np.asarray(first_normals, float)
    solid = np.asarray(second_triangles, float)
    if (p.ndim != 2 or p.shape[1:] != (3,) or not len(p)
            or a.shape != (len(p),) or b.shape != a.shape or normals.shape != p.shape
            or solid.ndim != 3 or solid.shape[1:] != (3, 3) or not len(solid)
            or not all(np.isfinite(x).all() for x in (p, a, b, normals, solid))
            or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError('Finite native sheet/solid geometry and contact tolerance required')
    construction = topology(len(p), f)
    continuous = (construction['edge_connected_surfaces'] == 1
                  and construction['nonmanifold_edges'] == 0
                  and construction['duplicate_triangles'] == 0)
    first = np.flatnonzero(abs(a) <= tolerance)
    if len(first) and not np.allclose(np.linalg.norm(normals[first], axis=1), 1., atol=1e-5):
        raise ValueError('Unit outward normals required at first-side contacts')
    witnessed_first, witnessed_second, rays = set(), set(), []
    triangles = p[f]
    if continuous:
        for node in first:
            normal = normals[node]
            origin = p[node] - tolerance * normal
            stop, solid_ids = _hits(solid, origin, normal)
            if not len(stop) or stop[0] < tolerance - 1e-7:
                continue
            face = solid[solid_ids[0]]
            outward = np.cross(face[1] - face[0], face[2] - face[0])
            length = np.linalg.norm(outward)
            if length == 0 or outward @ normal / length > -.5:
                continue
            distances, ids = _hits(triangles, origin, normal)
            mask = (distances >= tolerance - 1e-7) & (distances <= stop[0] + 1e-7)
            distances, ids = distances[mask], ids[mask]
            if not len(ids) or abs(distances[0] - tolerance) > 1e-7:
                continue
            gaps = np.diff(np.r_[distances, stop[0]])
            if gaps.max() > tolerance:
                continue
            top = f[ids[-1]]
            top = top[abs(b[top]) <= tolerance]
            if not len(top):
                continue
            witnessed_first.add(int(node)); witnessed_second.update(map(int, top))
            rays.append({'first_node': int(node), 'layer_count': len(ids),
                         'maximum_gap_m': float(gaps.max()),
                         'stack_depth_m': float(distances[-1] - distances[0])})
    return {'pass': len(witnessed_first) >= 3 and len(witnessed_second) >= 3,
            'first_indices': sorted(witnessed_first), 'second_indices': sorted(witnessed_second),
            'same_node_both_sides': int(((abs(a) <= tolerance) & (abs(b) <= tolerance)).sum()),
            'rays': rays, 'topology': construction, 'contact_tolerance_m': float(tolerance),
            'scope': 'Opposed solid proximity through contiguous local sheet layers; '
                     'not a force measurement or retention/task approval'}
