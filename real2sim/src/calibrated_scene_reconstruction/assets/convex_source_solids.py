"""Exact convex cells from the authored solids, before same-body Boolean union.

Reject concavity rather than filling cavities. Large convex polytopes are cut
into smaller convex cells to respect the native vertex limit; internal shared
faces do not change the solid union. This still requires native fidelity checks.
"""
import json
import numpy as np
import trimesh
from scipy.spatial import ConvexHull


def _hull(points):
    """Keep Qhull's closed topology, including tiny source facets."""
    points=np.asarray(points,dtype=float)
    centre=points.mean(0);scale=np.ptp(points,axis=0).max()
    native=ConvexHull((points-centre)/scale)
    ids=np.unique(native.simplices);remap=np.full(len(points),-1,dtype=int)
    remap[ids]=np.arange(len(ids));faces=remap[native.simplices].copy()
    vertices=points[ids];tri=vertices[faces]
    backwards=(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])*native.equations[:,:3]).sum(1)<0
    faces[backwards]=faces[backwards,::-1]
    result=trimesh.Trimesh(vertices,faces,process=False)
    if not result.is_watertight or not result.is_winding_consistent:
        raise ValueError('Convex partition did not preserve closed topology')
    return result


def cells(vertices,faces,*,max_vertices=64,max_aspect_ratio=None,max_cells=256):
    if type(max_cells) is not int or not 1<=max_cells<=4096:
        raise ValueError('Explicit convex source-cell budget within1..4096 required')
    if max_aspect_ratio is not None and (not np.isfinite(max_aspect_ratio) or max_aspect_ratio<=1):
        raise ValueError('Finite convex cell aspect ratio greater than one required')
    source_dtype=np.asarray(vertices).dtype
    mesh=trimesh.Trimesh(vertices,faces,process=True)
    if not mesh.is_watertight or not mesh.is_winding_consistent or mesh.volume<=0:
        raise ValueError('Closed outward source solids required')
    hull=_hull(mesh.vertices)
    # Float32 transforms can make a nominal planar cap very slightly dented.
    # Relative volume exaggerates this for a broad thin tabletop. Bound every
    # source triangle against a hull support plane, in original metric units;
    # do not replace a resolved indentation or a cavity with a hull.
    roundoff=0.
    if (source_dtype==np.dtype('float32') or np.array_equal(
            np.asarray(vertices),np.asarray(vertices,dtype=np.float32).astype(source_dtype))):
        roundoff=min(1e-6,4*np.finfo(np.float32).eps*float(np.ptp(mesh.vertices,axis=0).max()))
    tri=np.asarray(mesh.triangles)
    depth=np.einsum('tfvi,fi->tfv',tri[:,None]-hull.triangles_center[None,:,None],-hull.face_normals)
    boundary_bound=float(depth.max(axis=2).min(axis=1).max())
    rounded=bool(roundoff and boundary_bound<=roundoff)
    if not np.isclose(hull.volume,mesh.volume,rtol=1e-6,atol=1e-13) and not rounded:
        # A star-shaped solid has an exact positive tetrahedral fan from a
        # point in its visibility kernel. Keep every original boundary face;
        # shared internal faces cancel. This preserves shallow source dents
        # (including float32 tilted caps) instead of filling them with a hull.
        centre=np.asarray(mesh.center_mass);tri=np.asarray(mesh.triangles)
        signed=np.einsum('ni,ni->n',tri[:,0]-centre,np.cross(tri[:,1]-centre,tri[:,2]-centre))/6
        scale=float(np.ptp(mesh.vertices,axis=0).max())
        if (signed<=100*np.finfo(float).eps*scale**3).any():
            raise ValueError('Concave source has no verified interior kernel; cannot replace it by a convex collider')
        faces0=np.array([[1,2,3],[0,2,1],[0,3,2],[0,1,3]])
        pending=[trimesh.Trimesh(np.r_[centre[None],face],faces0,process=False) for face in tri]
        if (len(pending)>256 or any(not tet.is_volume for tet in pending)
                or not np.isclose(sum(tet.volume for tet in pending),mesh.volume,rtol=1e-8,atol=1e-13)):
            raise ValueError('Exact source tetrahedra did not preserve the positive closed solid')
    else:pending=[hull]
    result=[]
    while pending:
        source=pending.pop();v=np.asarray(source.vertices)
        _,_,axes=np.linalg.svd(v-v.mean(0),full_matrices=False)
        extent=np.ptp(v@axes.T,axis=0)
        slender=max_aspect_ratio is not None and extent.max()/extent.min()>max_aspect_ratio
        if len(v)<=max_vertices and not slender:
            result.append({'V':v,'F':np.asarray(source.faces)});continue
        # Axial cuts shorten a rod but retain its circumferential vertices.
        # Once aspect is acceptable, choose the cut that actually reduces
        # vertex complexity. Float32 source facets need not be coplanar, so
        # deleting their cut vertices would silently change the source solid.
        candidates=[]
        for normal in axes[[int(extent.argmax())]] if slender else axes:
            projection=v@normal;plane=(projection.min()+projection.max())/2
            edges=v[source.edges_unique];a,b=edges[:,0],edges[:,1]
            crosses=(a@normal-plane)*(b@normal-plane)<0
            a,b=a[crosses],b[crosses];t=(plane-a@normal)/((b-a)@normal)
            cuts=a+t[:,None]*(b-a);pieces=[]
            for sign in [-1,1]:
                points=np.concatenate([v[sign*(projection-plane)>=0],cuts])
                pieces.append(_hull(points))
            candidates.append(pieces)
        pieces=min(candidates,key=lambda ps:(max(len(p.vertices) for p in ps),
                                             sum(len(p.vertices) for p in ps)))
        if not np.isclose(sum(p.volume for p in pieces),source.volume,rtol=1e-8,atol=1e-13):
            raise ValueError('Convex partition changed source volume')
        pending.extend(pieces)
        if len(pending)+len(result)>max_cells:raise ValueError('Convex source partition budget exceeded')
    if rounded:
        for part in result:
            part['source_roundoff_bound_m']=float(roundoff)
            part['source_boundary_error_bound_m']=boundary_bound
    return result


def load(path):
    with np.load(path) as data:
        rows=json.loads(str(data['parts_json']));result=[]
        for row in rows:
            a=row['vertex_start'];b=row['face_start']
            v=data['V'][a:a+row['vertex_count']];f=data['F'][b:b+row['triangle_count']]-a
            result.extend({**part,'source_name':row['name']} for part in cells(v,f))
    return result
