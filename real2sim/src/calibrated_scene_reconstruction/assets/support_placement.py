"""Place physical surfaces, rather than shell midplanes, on a support plane."""
import json

import numpy as np


def offset(mesh,thicknesses,*,normal=(0.,0.,1.),rotation=None):
    vertices=np.asarray(mesh['V'],float)
    if vertices.ndim!=2 or vertices.shape[1:]!=(3,) or not np.isfinite(vertices).all():
        raise ValueError('Finite actual candidate vertices required')
    n=np.asarray(normal,float);R=np.eye(3) if rotation is None else np.asarray(rotation,float)
    if (n.shape!=(3,) or R.shape!=(3,3) or not np.isfinite(np.r_[n,R.ravel()]).all()
            or not np.isclose(n@n,1.) or n[2]<=.2
            or not np.allclose(R.T@R,np.eye(3),atol=1e-8) or not np.isclose(np.linalg.det(R),1.)):
        raise ValueError('Upward unit support normal and proper object rotation required')
    distances=vertices@(R.T@n)
    if 'links_json' not in mesh:return -float(distances.min())/n[2]
    parts=json.loads(str(mesh['links_json']));lower=[]
    for part in parts:
        first,count=part['vertex_start'],part['vertex_count'];z=float(distances[first:first+count].min())
        h=thicknesses.get(part['name'],0.) if part.get('kind')=='cloth' else 0.
        if not np.isfinite(h) or h<0:raise ValueError('Finite nonnegative material thickness required')
        lower.append(z-h/2)
    if not lower:raise ValueError('Physical link inventory is empty')
    return -min(lower)/n[2]


class TriangleSurface:
    """Actual upward support triangles; holes and multiple height layers remain.

    This is an initialization query, never a contact/stability certificate.
    ``uppermost`` serves a table; ``interior_floor`` explicitly chooses the
    first upward surface from below, as the existing container seed does.
    """
    def __init__(self, vertices, faces):
        import trimesh
        v, f = np.asarray(vertices, float), np.asarray(faces)
        if (v.ndim != 2 or v.shape[1:] != (3,) or not len(v) or not np.isfinite(v).all()
                or f.ndim != 2 or f.shape[1:] != (3,) or not len(f)
                or f.dtype.kind not in 'iu' or f.min() < 0 or f.max() >= len(v)):
            raise ValueError('Finite actual support triangles required')
        tri = v[f]; normal = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
        keep = normal[:, 2] > 1e-10*np.linalg.norm(normal, axis=1)
        if not keep.any():
            raise ValueError('Actual support has no upward surface')
        self.origin = (v.min(0)+v.max(0))/2
        self.mesh = trimesh.Trimesh(v-self.origin, f[keep], process=False)
        self.low, self.high = self.mesh.vertices[:, 2].min(), self.mesh.vertices[:, 2].max()
        self.margin = max(float(np.ptp(v, axis=0).max()), 1.)

    def heights(self, xy, *, surface='uppermost'):
        xy = np.asarray(xy, float)
        if xy.ndim != 2 or xy.shape[1:] != (2,) or not np.isfinite(xy).all():
            raise ValueError('Finite support-query XY points required')
        if surface not in {'uppermost', 'interior_floor'}:
            raise ValueError('Choose the actual top or explicit interior floor')
        result = np.full(len(xy), np.nan)
        up = surface == 'interior_floor'
        for first in range(0, len(xy), 2048):
            section = xy[first:first+2048]-self.origin[:2]
            origins = np.c_[section, np.full(len(section), self.low-self.margin if up else self.high+self.margin)]
            directions = np.tile([0., 0., 1. if up else -1.], (len(section), 1))
            locations, rays, _ = self.mesh.ray.intersects_location(origins, directions, multiple_hits=False)
            if len(rays):
                result[first+rays] = locations[:, 2]+self.origin[2]
        return result


def on_triangles(vertices, faces, support, *, xy, rotation=None, surface='uppermost',
                 sample_spacing_m=.002, max_samples=200000):
    """Place an unchanged rigid geometry using its downward actual surfaces.

    A group's minimum vertex and a support height at its centre are insufficient
    on a slope. Query physical surface samples individually and choose one
    common vertical offset. Separate disconnected parts remain separate; this
    cannot establish that every part is supported or stable.
    """
    from .rigid_initial_state import surface_samples
    v, f = np.asarray(vertices, float), np.asarray(faces)
    R = np.eye(3) if rotation is None else np.asarray(rotation, float)
    xy = np.asarray(xy, float)
    if (v.ndim != 2 or v.shape[1:] != (3,) or not np.isfinite(v).all()
            or f.ndim != 2 or f.shape[1:] != (3,) or not len(f) or f.dtype.kind not in 'iu'
            or f.min() < 0 or f.max() >= len(v) or xy.shape != (2,) or not np.isfinite(xy).all()
            or R.shape != (3, 3) or not np.isfinite(R).all()
            or not np.allclose(R.T@R, np.eye(3), atol=1e-8) or not np.isclose(np.linalg.det(R), 1.)
            or not np.isfinite(sample_spacing_m) or not .0001 <= sample_spacing_m <= .01):
        raise ValueError('Finite rigid candidate, proper pose and bounded surface sampling required')
    world = v@R.T
    tri = world[f]; normal = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
    down = f[normal[:, 2] < -1e-10*np.linalg.norm(normal, axis=1)]
    if not len(down):
        raise ValueError('Rigid candidate has no downward material surface')
    used, inverse = np.unique(down, return_inverse=True)
    samples = surface_samples(world[used], inverse.reshape(-1, 3), sample_spacing_m, max_samples=max_samples)
    heights = support.heights(samples[:, :2]+xy, surface=surface)
    hit = np.isfinite(heights)
    if not hit.any():
        raise ValueError('Candidate samples miss the actual support; bounds cannot fill a hole')
    root_z = float(np.max(heights[hit]-samples[hit, 2]))
    return {'support_z': root_z, 'surface': surface, 'sample_spacing_m': float(sample_spacing_m),
            'samples': len(samples), 'supported_ray_samples': int(hit.sum()),
            'minimum_sampled_vertical_gap_m': float(np.min(samples[hit, 2]+root_z-heights[hit])),
            'proposal_only': True, 'contact_proven': False, 'native_validated': False,
            'scope': 'Actual upward support triangles and downward rigid-surface samples. '
                     'No hole filling, pins or runtime adjustment; full solid/native/source checks still required.'}
