"""Triangle self-intersection measurements on actual cloth states.

Shared-vertex faces are topological neighbours and excluded. This checks
crossing surfaces, not material thickness, near-contact clearance or CCD.
The Warp triangle predicate uses single precision; a clear result alone is
never sufficient to certify contact or a task.
"""
import numpy as np
import warp as wp


@wp.func
def _separates(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3, e: wp.vec3, f: wp.vec3,
               axis: wp.vec3):
    length = wp.length(axis)
    separated = False
    if length > 1.0e-12:
        lo1 = wp.min(wp.min(wp.dot(a, axis), wp.dot(b, axis)), wp.dot(c, axis))
        hi1 = wp.max(wp.max(wp.dot(a, axis), wp.dot(b, axis)), wp.dot(c, axis))
        lo2 = wp.min(wp.min(wp.dot(d, axis), wp.dot(e, axis)), wp.dot(f, axis))
        hi2 = wp.max(wp.max(wp.dot(d, axis), wp.dot(e, axis)), wp.dot(f, axis))
        tolerance = 1.0e-7*length
        separated = hi1 < lo2-tolerance or hi2 < lo1-tolerance
    return separated


@wp.func
def _edge(a: wp.vec3, b: wp.vec3, c: wp.vec3, i: int):
    result = b-a
    if i == 1:
        result = c-b
    elif i == 2:
        result = a-c
    return result


@wp.func
def _triangle_overlap(a: wp.vec3, b: wp.vec3, c: wp.vec3, d: wp.vec3, e: wp.vec3, f: wp.vec3):
    """Separating axes include in-plane edge normals for coplanar triangles."""
    n1, n2 = wp.cross(b-a, c-a), wp.cross(e-d, f-d)
    separated = _separates(a, b, c, d, e, f, n1) or _separates(a, b, c, d, e, f, n2)
    for i in range(3):
        edge1, edge2 = _edge(a, b, c, i), _edge(d, e, f, i)
        separated = separated or _separates(a, b, c, d, e, f, wp.cross(n1, edge1))
        separated = separated or _separates(a, b, c, d, e, f, wp.cross(n2, edge2))
        for j in range(3):
            separated = separated or _separates(a, b, c, d, e, f, wp.cross(edge1, _edge(d, e, f, j)))
    return not separated


@wp.kernel
def _crossings(mesh: wp.uint64, points: wp.array(dtype=wp.vec3),
               faces: wp.array(dtype=int), groups: wp.array(dtype=int), cross_only:int,
               counts: wp.array(dtype=int), partners: wp.array(dtype=int)):
    i = wp.tid()
    a, b, c = faces[3*i], faces[3*i+1], faces[3*i+2]
    p, q, r = points[a], points[b], points[c]
    lower = wp.min(wp.min(p, q), r)
    upper = wp.max(wp.max(p, q), r)
    query = wp.mesh_query_aabb(mesh, lower, upper)
    j = int(0)
    while wp.mesh_query_aabb_next(query, j):
        if j > i and (cross_only==0 or groups[i]!=groups[j]):
            x, y, z = faces[3*j], faces[3*j+1], faces[3*j+2]
            neighbour = (a == x or a == y or a == z or b == x or b == y or b == z
                         or c == x or c == y or c == z)
            if not neighbour:
                # The predicate has a fixed numerical epsilon. Millimetre
                # triangles must not all be treated as coplanar. Normalize each
                # pair after translation; the BVH still uses physical units.
                length = wp.max(wp.max(wp.length(q-p), wp.length(r-p)),
                                wp.max(wp.length(points[y]-points[x]), wp.length(points[z]-points[x])))
                scale = 1.0/wp.max(length, 1.0e-12)
                if (wp.intersect_tri_tri(wp.vec3(0.0), (q-p)*scale, (r-p)*scale,
                                        (points[x]-p)*scale, (points[y]-p)*scale, (points[z]-p)*scale) > 0
                        and _triangle_overlap(wp.vec3(0.0), (q-p)*scale, (r-p)*scale,
                                              (points[x]-p)*scale, (points[y]-p)*scale, (points[z]-p)*scale)):
                    counts[i] = counts[i]+1
                    if partners[i] < 0:
                        partners[i] = j


def self_intersections(points, faces, *, device='cuda:0', example_limit=32, face_groups=None):
    """Inspect every non-neighbour face pair using a mesh BVH, without writes."""
    points, faces = np.asarray(points, float), np.asarray(faces)
    if (points.ndim != 2 or points.shape[1:] != (3,) or not len(points) or not np.isfinite(points).all()
            or faces.ndim != 2 or faces.shape[1:] != (3,) or not len(faces)
            or faces.dtype.kind not in 'iu' or faces.min() < 0 or faces.max() >= len(points)
            or type(example_limit) is not int or not 0 <= example_limit <= 1000):
        raise ValueError('Finite cloth points, valid triangles and a bounded example count required')
    area = np.linalg.norm(np.cross(points[faces[:, 1]]-points[faces[:, 0]],
                                    points[faces[:, 2]]-points[faces[:, 0]]), axis=1)
    if (area <= 1e-14).any():
        raise ValueError('Degenerate cloth faces cannot certify self-contact geometry')
    group=np.zeros(len(faces),np.int32) if face_groups is None else np.asarray(face_groups)
    if group.shape!=(len(faces),) or group.dtype.kind not in 'iu':raise ValueError('One physical owner per triangle required')
    # Preserve small local edges even in a scene far from the origin.
    vertices = wp.array(np.asarray(points-points.mean(0), np.float32), dtype=wp.vec3, device=device)
    indices = wp.array(np.asarray(faces, np.int32).ravel(), dtype=int, device=device)
    mesh = wp.Mesh(points=vertices, indices=indices)
    counts = wp.zeros(len(faces), dtype=int, device=device)
    partners = wp.array(np.full(len(faces), -1, np.int32), dtype=int, device=device)
    groups=wp.array(group.astype(np.int32),dtype=int,device=device)
    wp.launch(_crossings, dim=len(faces), inputs=[mesh.id, vertices, indices,groups,int(face_groups is not None), counts, partners], device=device)
    values, examples = counts.numpy(), partners.numpy()
    selected = np.flatnonzero(values)[:example_limit]
    return {'pass': bool(values.sum() == 0), 'intersecting_pairs': int(values.sum()),
            'example_face_pairs': np.c_[selected, examples[selected]].tolist(),
            'triangles': len(faces), 'excluded': 'Pairs sharing any source vertex',
            'scope': 'Single-precision triangle intersection predicate; not a thickness, CCD or task certificate'}
