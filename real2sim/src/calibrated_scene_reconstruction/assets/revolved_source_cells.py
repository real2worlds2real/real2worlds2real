"""Convex cells of an existing closed mesh of revolution, preserving its cavity.

Infer the meridian cycle and angular rings from source topology; do not fit a
cylinder, invent a profile or accept arbitrary concave meshes. Source float32
rounding is explicitly bounded. Native cooking and functional tests are still
required after this geometric construction.
"""
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from .convex_source_solids import _hull


def _cross(a,b,c):
    u,v=b-a,c-b
    return float(u[0]*v[1]-u[1]*v[0])


def _area(points):
    return float(np.sum(points[:,0]*np.roll(points[:,1],-1)-points[:,1]*np.roll(points[:,0],-1))/2)


def _triangulate(points):
    ids=list(range(len(points)));triangles=[]
    if _area(points)<0:ids.reverse()
    tolerance=np.ptp(points,axis=0).max()**2*1e-12
    while len(ids)>3:
        for k,b in enumerate(ids):
            a,c=ids[k-1],ids[(k+1)%len(ids)]
            if _cross(points[a],points[b],points[c])<=tolerance:continue
            def inside(i):
                return min(_cross(points[a],points[b],points[i]),
                           _cross(points[b],points[c],points[i]),
                           _cross(points[c],points[a],points[i]))>=-tolerance
            if any(inside(i) for i in ids if i not in {a,b,c}):continue
            triangles.append([a,b,c]);ids.pop(k);break
        else:raise ValueError('Meridian is not a supported simple polygon')
    triangles.append(ids)
    return triangles


def _merge(points,polygons):
    polygons=[list(p) for p in polygons];tolerance=np.ptp(points,axis=0).max()**2*1e-12
    while True:
        changed=False
        for i in range(len(polygons)):
            for j in range(i):
                edges=[(a,b) for p in [polygons[i],polygons[j]] for a,b in zip(p,p[1:]+p[:1])]
                boundary=[e for e in edges if e[::-1] not in edges]
                if len(boundary)!=len(edges)-2:continue
                nxt=dict(boundary);cycle=[boundary[0][0]]
                while nxt[cycle[-1]]!=cycle[0]:
                    cycle.append(nxt[cycle[-1]])
                    if len(cycle)>len(boundary):raise ValueError('Invalid merged meridian')
                if len(cycle)!=len(boundary):continue
                if min(_cross(points[cycle[k-1]],points[a],points[cycle[(k+1)%len(cycle)]])
                       for k,a in enumerate(cycle)) < -tolerance:continue
                polygons[j]=cycle;polygons.pop(i);changed=True;break
            if changed:break
        if not changed:return polygons


def cells(vertices,faces,*,axis=None):
    vertices=np.asarray(vertices,float);faces=np.asarray(faces,int)
    if axis is not None:
        axis=np.asarray(axis,float)
        if axis.shape!=(3,) or not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-8:
            raise ValueError('Finite source revolution axis required')
        axis=axis/np.linalg.norm(axis)
        from scipy.spatial.transform import Rotation
        rotation=Rotation.align_vectors([[0,0,1]],[axis])[0].as_matrix()
        return [{**part,'V':part['V']@rotation} for part in cells(vertices@rotation.T,faces)]
    source=trimesh.Trimesh(vertices,faces,process=False)
    if not source.is_watertight or not source.is_winding_consistent or source.volume<=0:
        raise ValueError('Closed outward source mesh of revolution required')
    centre=(vertices[:,:2].min(0)+vertices[:,:2].max(0))/2
    xy=vertices[:,:2]-centre;radii=np.linalg.norm(xy,axis=1)
    meridian=np.c_[radii,vertices[:,2]]
    tolerance=float(np.ptp(vertices,axis=0).max()*1e-6)
    tree=cKDTree(meridian);group=np.full(len(vertices),-1,int);rings=[];profile=[]
    for index in range(len(vertices)):
        if group[index]>=0:continue
        ids=np.asarray(tree.query_ball_point(meridian[index],tolerance),int)
        if (group[ids]>=0).any():raise ValueError('Ambiguous revolution rings')
        group[ids]=len(rings);rings.append(ids);profile.append(meridian[ids].mean(0))
    profile=np.asarray(profile)
    if len(profile)<3:raise ValueError('Nondegenerate meridian required')
    edges=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    links={tuple(sorted((int(group[a]),int(group[b])))) for a,b in edges if group[a]!=group[b]}
    adjacent={i:[] for i in range(len(rings))}
    for a,b in links:adjacent[a].append(b);adjacent[b].append(a)
    # The meridian's closing edge on the axis sweeps zero area, so a welded
    # closed source mesh correctly has no surface triangles along that edge.
    ends=[i for i,neighbors in adjacent.items() if len(neighbors)==1]
    if len(ends)==2 and all(profile[i,0]<tolerance for i in ends):
        a,b=ends;adjacent[a].append(b);adjacent[b].append(a)
    if any(len(v)!=2 for v in adjacent.values()):raise ValueError('Source is not a single meridian cycle')
    order=[0];previous=-1
    while True:
        next_id=next(i for i in adjacent[order[-1]] if i!=previous)
        if next_id==0:break
        if next_id in order:raise ValueError('Disconnected meridian cycle')
        previous=order[-1];order.append(next_id)
    if len(order)!=len(rings):raise ValueError('Disconnected revolution surface')
    profile=profile[order];rings=[rings[i] for i in order]
    count=max(len(r) for r in rings)
    if count<8 or count>512:raise ValueError('Supported angular ring size is8..512')
    angular=None;phase=None;indexed=[]
    for row,ids in zip(profile,rings):
        if row[0]<tolerance:
            if len(ids)!=1:raise ValueError('Axis vertices must be welded')
            indexed.append(np.repeat(ids,count));continue
        if len(ids)!=count:raise ValueError('Angular rings have inconsistent vertex counts')
        angles=np.mod(np.arctan2(xy[ids,1],xy[ids,0]),2*np.pi)
        # Put the branch cut between vertices, avoiding rounding at angle0.
        if phase is None:phase=float(np.angle(np.mean(np.exp(1j*angles*count)))/count)
        bins=np.rint((angles-phase)*count/(2*np.pi)).astype(int)%count
        if len(set(bins.tolist()))!=count:raise ValueError('Source angular rings are not aligned uniform samples')
        indices=ids[np.argsort(bins)];observed=xy[indices]/radii[indices,None]
        expected=np.c_[np.cos(phase+2*np.pi*np.arange(count)/count),
                       np.sin(phase+2*np.pi*np.arange(count)/count)]
        if not np.allclose(observed,expected,atol=2e-6,rtol=0):
            raise ValueError('Source angular samples must be uniform')
        if angular is None:angular=observed
        if not np.allclose(observed,angular,atol=2e-6,rtol=0):raise ValueError('Twisted angular rings are not a revolution surface')
        indexed.append(indices)
    indexed=np.asarray(indexed);polygons=_merge(profile,_triangulate(profile));result=[]
    for sector in range(count):
        for polygon in polygons:
            ids=np.unique(indexed[polygon][:,[sector,(sector+1)%count]])
            piece=_hull(vertices[ids])
            result.append({'V':piece.vertices,'F':piece.faces})
    total=sum(trimesh.Trimesh(p['V'],p['F'],process=False).volume for p in result)
    if not np.isclose(total,source.volume,rtol=2e-6,atol=1e-13):
        # Rotated float32 rings need not remain exactly coplanar. Convex cell
        # triangulation may then differ by a few parts per million even though
        # every exported vertex is an original source vertex. Explain such a
        # discrepancy with a bounded reference, rather than increasing a global
        # relative-volume tolerance or changing the actual collision geometry.
        reference=np.empty_like(vertices)
        for row,ids in zip(profile,indexed):
            radius=0. if len(np.unique(ids))==1 else row[0]
            reference[ids]=np.c_[centre+radius*angular,np.full(count,row[1])]
        displacement=float(np.linalg.norm(reference-vertices,axis=1).max())
        budget=2*float(np.linalg.norm(np.spacing(vertices.astype(np.float32)).astype(float),axis=1).max())
        # Previously rounded geometry retains that error when rigidly moved.
        # Its current-coordinate ULP can shrink after centering, so it cannot
        # bound the inherited deviation by itself. The existing source-ring
        # recognition tolerance supplies a second, geometric bound. Actual
        # cells still contain only source vertices; the ideal reference is
        # used solely to bound and explain the volume discrepancy.
        partition_budget=max(budget,tolerance)
        original_tree=cKDTree(vertices);reference_volumes=[];rounding_change=0.
        for piece in result:
            distance,indices=original_tree.query(piece['V'])
            if distance.max()>1e-12:raise ValueError('Revolution cell invented a source vertex')
            actual=trimesh.Trimesh(piece['V'],piece['F'],process=False).volume
            ideal=trimesh.Trimesh(reference[indices],piece['F'],process=False).volume
            reference_volumes.append(ideal);rounding_change+=abs(actual-ideal)
        ideal_source=trimesh.Trimesh(reference,faces,process=False).volume
        if (displacement>partition_budget or ideal_source<=0
                or not np.isclose(sum(reference_volumes),ideal_source,rtol=1e-9,atol=1e-15)
                or abs(total-source.volume)>rounding_change+abs(ideal_source-source.volume)+1e-15):
            raise ValueError('Revolution partition changed source volume beyond bounded source-ring deviation')
        for piece in result:
            piece.update(source_roundoff_bound_m=displacement,source_float32_roundoff_budget_m=budget,
                         source_revolution_recognition_bound_m=tolerance,
                         source_partition_deviation_budget_m=partition_budget,
                         source_partition_scope='Exact source vertices; volume discrepancy bounded by the existing source-ring recognition tolerance or current float32 encoding. No reference geometry is exported.',
                         source_volume_relative_error=float(total/source.volume-1),
                         reference_partition_relative_volume_error=float(sum(reference_volumes)/ideal_source-1))
    return result
