"""Blender BVH support and rigid-assembly audit. Never modifies the source scene."""
import json
from pathlib import Path
import sys

import numpy as np


def hull(points):
    points = sorted(set(map(tuple, np.asarray(points, float))))
    if len(points) <= 2:
        return np.asarray(points)
    def cross(a, b, c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    halves = []
    for order in (points, points[::-1]):
        half = []
        for p in order:
            while len(half) >= 2 and cross(half[-2], half[-1], p) <= 0:
                half.pop()
            half.append(p)
        halves.append(half[:-1])
    return np.asarray(halves[0]+halves[1])


def support_margin(point, polygon):
    if len(polygon) < 3:
        return None
    a = np.asarray(polygon, float)
    delta = np.roll(a, -1, axis=0)-a
    relative = np.asarray(point, float)-a
    lengths = np.linalg.norm(delta, axis=1)
    return float(np.min((delta[:, 0]*relative[:, 1]-delta[:, 1]*relative[:, 0])/lengths))


def polygon_clip(subject, boundary):
    """Intersect convex contact triangles, including edge/edge-only overlaps."""
    output=list(hull(subject));boundary=hull(boundary)
    if len(boundary)<3 or len(output)<3:return []
    def cross(a,b):return a[0]*b[1]-a[1]*b[0]
    for a,b in zip(boundary,np.roll(boundary,-1,axis=0)):
        previous=output;output=[]
        if not previous:break
        edge=b-a
        for p,q in zip(previous,previous[1:]+previous[:1]):
            dp,dq=cross(edge,p-a),cross(edge,q-a)
            if dp>=-1e-9:output.append(p)
            if (dp>=0)!=(dq>=0):output.append(p+(q-p)*dp/(dp-dq))
    return output


class Mesh:
    def __init__(self, vertices, faces, name):
        from mathutils.bvhtree import BVHTree
        self.v, self.f, self.name = np.asarray(vertices, float), np.asarray(faces, int), name
        self.lo, self.hi = self.v.min(0), self.v.max(0)
        self.tree = BVHTree.FromPolygons(self.v.tolist(), self.f.tolist(), all_triangles=True, epsilon=1e-7)
        edges=np.concatenate([self.f[:,[0,1]],self.f[:,[1,2]],self.f[:,[2,0]]])
        _,inverse,counts=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True,return_counts=True)
        directions=np.where(edges[:,0]<edges[:,1],1.,-1.)
        self.closed=bool(np.all(counts==2))
        self.consistent_winding=bool(np.all(np.bincount(inverse,weights=directions)==0))
        p = self.v-self.v.mean(0)
        tri = p[self.f]
        volumes = np.einsum('ij,ij->i', tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))/6
        total = volumes.sum()
        self.volume = abs(float(total))
        self.com = (np.einsum('i,ij->j', volumes, tri.sum(1)/4)/total+self.v.mean(0)) if abs(total) > 1e-10 else self.v.mean(0)
        world_tri=self.v[self.f]
        normals=np.cross(world_tri[:,1]-world_tri[:,0],world_tri[:,2]-world_tri[:,0])
        normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-12)
        self.bottom=world_tri[normals[:,2]<-.999]
        self.top=world_tri[normals[:,2]>.999]

    def inside(self, point):
        if self.volume < 1e-10 or np.any(point < self.lo-1e-7) or np.any(point > self.hi+1e-7):
            return False
        # Float32 BVH ray parity is unreliable for thin, rotated solids at
        # room-scale world offsets: repeated hits can flip outside to inside.
        # The signed solid angle of the actual triangles is translation-stable
        # in float64 and handles concave closed members without convex proxies.
        tri = self.v[self.f]-np.asarray(point, float)
        lengths = np.linalg.norm(tri, axis=2)
        numerator = np.einsum('ij,ij->i', tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))
        denominator = (np.prod(lengths, axis=1)
                       +np.einsum('ij,ij->i', tri[:, 0], tri[:, 1])*lengths[:, 2]
                       +np.einsum('ij,ij->i', tri[:, 1], tri[:, 2])*lengths[:, 0]
                       +np.einsum('ij,ij->i', tri[:, 2], tri[:, 0])*lengths[:, 1])
        winding = np.sum(2*np.arctan2(numerator, denominator))
        return bool(abs(winding) > 2*np.pi)


def islands(vertices, faces, name):
    """Treat disconnected islands inside a single USD mesh as separate members."""
    # USD may split points at normal/UV seams. Weld coincident positions only;
    # otherwise a closed cube exported with flat normals becomes six panels.
    _, ids, inverse = np.unique(np.round(vertices, 8), axis=0, return_index=True, return_inverse=True)
    vertices, faces = vertices[ids], inverse[faces]
    parents = np.arange(len(vertices))
    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return int(i)
    for face in faces:
        a = root(face[0])
        for i in face[1:]:
            parents[root(i)] = a
    groups = {}
    for face in faces:
        groups.setdefault(root(face[0]), []).append(face)
    for number, group in enumerate(groups.values()):
        group = np.asarray(group, int)
        ids, inverse = np.unique(group, return_inverse=True)
        yield Mesh(vertices[ids], inverse.reshape(-1, 3), name if len(groups) == 1 else f'{name}:island{number}')


def penetration(a, b, tolerance):
    """Conservative intersection diagnostic, including enclosed and crossing solids.

    Test vertices plus intersecting triangles' edge samples. A transverse surface
    crossing whose depth cannot be resolved is a failure, never an assumed touch.
    """
    if np.any(np.minimum(a.hi, b.hi)-np.maximum(a.lo, b.lo) <= tolerance):
        return 0., False
    overlaps = a.tree.overlap(b.tree)
    depth, unresolved = 0., False
    for mesh, other, column in ((a, b, 0), (b, a, 1)):
        ids = sorted({pair[column] for pair in overlaps})
        samples = [mesh.v]
        if ids:
            tri = mesh.v[mesh.f[ids]]
            samples.extend([tri.mean(1), (tri+np.roll(tri, 1, axis=1)).reshape(-1, 3)/2])
        for point in np.concatenate(samples):
            if other.inside(point):
                nearest = other.tree.find_nearest(tuple(point))
                if nearest[0] is not None:
                    depth = max(depth, float(nearest[3]))
    if depth <= tolerance:
        for i, j in overlaps:
            at, bt = a.v[a.f[i]], b.v[b.f[j]]
            an, bn = np.cross(at[1]-at[0], at[2]-at[0]), np.cross(bt[1]-bt[0], bt[2]-bt[0])
            if np.linalg.norm(an) < 1e-12 or np.linalg.norm(bn) < 1e-12:
                continue
            an /= np.linalg.norm(an); bn /= np.linalg.norm(bn)
            da, db = (at-bt[0])@bn, (bt-at[0])@an
            if da.min() < -tolerance and da.max() > tolerance and db.min() < -tolerance and db.max() > tolerance:
                unresolved = True
                break
    return depth, unresolved


def touching(a, b, tolerance):
    if np.linalg.norm(np.maximum(np.maximum(a.lo-b.hi, b.lo-a.hi), 0)) > tolerance:
        return False
    if a.tree.overlap(b.tree):
        return True
    for mesh, other in ((a, b), (b, a)):
        for point in mesh.v[::max(1, len(mesh.v)//128)]:
            found = other.tree.find_nearest(tuple(point))
            if found[0] is not None and found[3] <= tolerance or other.inside(point):
                return True
    return False


def components(parts, tolerance):
    groups = [{i} for i in range(len(parts))]
    for i, a in enumerate(parts):
        for j in range(i):
            if touching(a, parts[j], tolerance):
                first = next(g for g in groups if i in g)
                second = next(g for g in groups if j in g)
                if first is not second:
                    first.update(second); groups.remove(second)
    return [[parts[i].name for i in sorted(group)] for group in groups]


def contact(body, hosts, tolerance):
    from mathutils import Vector
    found, gaps, penetration = [], [], 0.
    body_lo = np.min([p.lo for p in body], axis=0)
    body_hi = np.max([p.hi for p in body], axis=0)
    for part in body:
        for point in part.v[::max(1, len(part.v)//2048)]:
            for host in hosts:
                if np.any(point[:2] < host.lo[:2]-tolerance) or np.any(point[:2] > host.hi[:2]+tolerance):
                    continue
                hit = host.tree.ray_cast(Vector(point)+Vector((0, 0, tolerance)), Vector((0, 0, -1)), 100.)
                if hit[0] is not None:
                    gap = float(point[2]-hit[0].z)
                    gaps.append(gap)
                    if abs(gap) <= tolerance:
                        found.append([point[0], point[1], float(hit[0].z)])
                if host.inside(point):
                    near = host.tree.find_nearest(tuple(point))
                    if near[0] is not None:
                        penetration = max(penetration, float(near[3]))
    # A small support can lie entirely inside a large bottom face: include the
    # support's upper vertices as well as the body's lower contact vertices.
    for host in hosts:
        for point in host.v[::max(1, len(host.v)//2048)]:
            if np.any(point[:2] < body_lo[:2]-tolerance) or np.any(point[:2] > body_hi[:2]+tolerance):
                continue
            for part in body:
                hit = part.tree.ray_cast(Vector(point)-Vector((0, 0, tolerance)), Vector((0, 0, 1)), 100.)
                if hit[0] is not None and abs(hit[0].z-point[2]) <= tolerance:
                    found.append([point[0], point[1], float(point[2])])
    for part in body:
        for host in hosts:
            if np.any(np.maximum(part.lo[:2],host.lo[:2])>np.minimum(part.hi[:2],host.hi[:2])):
                continue
            for bottom in part.bottom:
                for top in host.top:
                    if abs(bottom[:,2].mean()-top[:,2].mean())>tolerance:continue
                    for xy in polygon_clip(bottom[:,:2],top[:,:2]):found.append([*xy,float(top[:,2].mean())])
    return found, min((abs(g) for g in gaps), default=None), penetration


def audit(job):
    data = np.load(job['geometry'])
    bodies = {}
    for i, row in enumerate(job['parts']):
        bodies.setdefault(row['entity'], []).extend(islands(data[f'v{i}'], data[f'f{i}'], row['part']))
    cfg = job['contract']
    specs = cfg.get('entities', {})
    ground = cfg.get('ground')
    tolerance = float(cfg.get('contact_tolerance_m', .003))
    penetration_limit = float(cfg.get('penetration_tolerance_m', .002))
    stability = float(cfg.get('stability_margin_m', .002))
    if not 0 < tolerance <= .01 or not 0 < penetration_limit <= .005 or not 0 <= stability <= .02:
        raise ValueError('Physical tolerances must remain small metric reconstruction uncertainties')
    results = []
    external = set(job['external_entities'])
    grounded = set(external)
    if ground in bodies and 'ground_height' in cfg:
        z = float(cfg['ground_height'])
        lo, hi = np.min([p.lo for p in bodies[ground]], axis=0), np.max([p.hi for p in bodies[ground]], axis=0)
        if abs(hi[2]-z) <= tolerance and hi[2]-lo[2] < min(hi[:2]-lo[:2]):
            grounded.add(ground)
    for entity, body in bodies.items():
        if entity in external:
            continue
        spec = specs.get(entity, {})
        issues = []
        connected = components(body, tolerance)
        if len(connected) != 1:
            issues.append('Disconnected rigid assembly parts')
        mode, on = spec.get('mode'), spec.get('on', [])
        invalid=[p.name for p in body if not p.closed or not p.consistent_winding]
        if entity!=ground and invalid:
            issues.append('Nonclosed or inconsistently wound physical members: '+', '.join(invalid))
        if entity == ground:
            if entity not in grounded or mode != 'foundation':
                issues.append('Ground foundation does not match its observed height')
            results.append({'entity': entity, 'pass': not issues, 'issues': issues, 'components': connected, 'mode': mode})
            continue
        if mode not in ('resting', 'fixed') or not on or any(p not in bodies or p == entity for p in on):
            issues.append('Missing or invalid physical support declaration')
            results.append({'entity': entity, 'pass': False, 'issues': issues, 'components': connected, 'mode': mode})
            continue
        hosts = [part for support in on for part in bodies[support]]
        contacts, gap, depth = contact(body, hosts, tolerance)
        weights = np.asarray([max(p.volume, 1e-12) for p in body])
        com = np.average([p.com for p in body], weights=weights, axis=0)
        polygon = hull(np.asarray(contacts)[:, :2]) if contacts else np.empty((0, 2))
        margin = support_margin(com[:2], polygon)
        contact_z = float(np.median(np.asarray(contacts)[:, 2])) if contacts else None
        tip_angle = float(np.degrees(np.arctan2(max(0., margin or 0.), max(1e-9, com[2]-contact_z)))) if contacts else None
        if depth > penetration_limit:
            issues.append('Body penetrates its declared support')
        if mode == 'resting':
            if not contacts:
                issues.append('No actual support contact; body is floating')
            if margin is None or margin < stability:
                issues.append('Gravity centre is outside a stable support polygon')
            elif tip_angle < float(cfg.get('min_tip_angle_deg', 3.)):
                issues.append('Support is too marginal against small tipping perturbations')
        else:
            anchors = spec.get('anchors', [])
            if not anchors or any(not a.get('part') or not a.get('evidence') for a in anchors):
                issues.append('Fixed objects require explicit evidenced attachment members')
            for anchor in anchors:
                members = [p for p in body if p.name.split(':island')[0] == anchor.get('part')]
                if not members or not any(touching(a, b, tolerance) for a in members for b in hosts):
                    issues.append('Attachment member does not touch its declared support')
        results.append({'entity': entity, 'pass': not issues, 'issues': issues, 'mode': mode, 'on': on,
            'components': connected, 'centre_of_mass': com.tolist(), 'support_polygon': polygon.tolist(),
            'support_margin_m': margin, 'tip_angle_deg': tip_angle,
            'nearest_vertical_gap_m': gap, 'support_penetration_m': depth})
    collisions = []
    names = list(bodies)
    for i, entity in enumerate(names):
        for other in names[:i]:
            if entity in external and other in external:
                continue
            maximum, unresolved = 0., False
            for a in bodies[entity]:
                for b in bodies[other]:
                    depth, unknown = penetration(a, b, penetration_limit)
                    maximum, unresolved = max(maximum, depth), unresolved or unknown
            if maximum > penetration_limit or unresolved:
                collisions.append({'entities': [entity, other], 'penetration_m': maximum,
                                   'unresolved_surface_crossing': unresolved})
                for row in results:
                    if row['entity'] in (entity, other):
                        row['pass'] = False
                        row['issues'].append('Interpenetration with '+(other if row['entity'] == entity else entity))
    # A cycle of mutually supporting floating objects is not grounded support.
    for _ in bodies:
        previous = len(grounded)
        for row in results:
            if row['pass'] and any(s in grounded for s in row.get('on', [])):
                grounded.add(row['entity'])
        if len(grounded) == previous:
            break
    for row in results:
        if row['entity'] not in grounded:
            row['pass'] = False; row['issues'].append('Support chain does not reach an observed foundation')
    return {'pass': bool(results) and all(r['pass'] for r in results), 'entities': results, 'collisions': collisions,
            'method': 'World-space triangle BVHs, assembly connectivity, actual contact polygons and volume-weighted gravity centres. Fixed anchors need named contact geometry. This is static model plausibility; passive simulation is a separate check.',
            'dynamic_stability_verified': False}


def main():
    job = json.loads(Path(sys.argv[sys.argv.index('--')+1]).read_text())
    result = audit(job)
    Path(job['output']).write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('Physical support audit', result['pass'], len(result['entities']), flush=True)


if __name__ == '__main__':
    main()
