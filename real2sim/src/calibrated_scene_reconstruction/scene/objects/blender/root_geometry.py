"""Blender-side exact modeled receptacle containment for real physical bonds.

Concave closest-face normals are not an inside/outside classifier. These
checks inspect actual closed triangles; no target silhouette or task pass is
used to create a connection through empty space.
"""
import numpy as np


class TriangleSolid:
    def __init__(self,vertices,faces):
        from mathutils.bvhtree import BVHTree
        self.vertices=np.asarray(vertices,float);self.faces=np.asarray(faces,int)
        self.bvh=BVHTree.FromPolygons(self.vertices.tolist(),self.faces.tolist(),all_triangles=True)
    def find_nearest(self,*args):return self.bvh.find_nearest(*args)
    def ray_cast(self,*args):return self.bvh.ray_cast(*args)
    def inside_winding(self,point):
        tri=self.vertices[self.faces]-np.asarray(point,float)
        a,b,c=tri[:,0],tri[:,1],tri[:,2]
        la,lb,lc=np.linalg.norm(a,axis=1),np.linalg.norm(b,axis=1),np.linalg.norm(c,axis=1)
        numerator=np.einsum('ij,ij->i',a,np.cross(b,c))
        denominator=la*lb*lc+np.einsum('ij,ij->i',a,b)*lc+np.einsum('ij,ij->i',b,c)*la+np.einsum('ij,ij->i',c,a)*lb
        winding=np.arctan2(numerator,denominator).sum()/(2*np.pi)
        return abs(winding)>.5


def tree(vertices, faces):
    return TriangleSolid(vertices,faces)


def signed_distance(surface, point):
    from mathutils import Vector
    point=Vector(point);nearest,_,_,distance=surface.find_nearest(point)
    if nearest is None:raise ValueError('Physical root receptacle has no closest triangle')
    if distance<=1e-8:return 0.
    votes=[]
    for values in ((1.,.371,.128),(.219,1.,.613),(.527,.183,1.)):
        direction=Vector(values).normalized();origin=point.copy();crossings=0;seen=set()
        step=max(1e-7,8*float(np.spacing(np.float32(max(abs(v) for v in point)))))
        for _ in range(128):
            hit,_,face,_=surface.ray_cast(origin,direction)
            if hit is None:
                votes.append(bool(crossings%2));break
            if face in seen:step*=4
            else:crossings+=1;seen.add(face)
            origin=hit+direction*step
    # Even parity can also mean inside two overlapping, consistently oriented
    # closed members. A composite rigid body is not the XOR of its solids.
    # Signed winding retains hollow-shell cancellation and handles such unions.
    if len(votes)!=3 or len(set(votes))!=1 or not votes[0]:
        inside=surface.inside_winding(point)
    else:inside=votes[0]
    return -distance if inside else distance


def embedded_roots(vertices, faces, rigid_vertices, rigid_faces, seeds, *, tolerance_m=.00005,surface=None):
    surface=tree(rigid_vertices,rigid_faces) if surface is None else surface
    signed=np.array([signed_distance(surface,p) for p in vertices])
    seeds=np.asarray(seeds,int)
    if not len(seeds) or signed[seeds].max()>tolerance_m:
        raise ValueError(f'Physical root outside actual receptacle: maximum gap {signed[seeds].max():.9g}m')
    neighbors=[set() for _ in vertices]
    for face in faces:
        for a,b in zip(face,np.roll(face,1)):neighbors[a].add(b);neighbors[b].add(a)
    attached=set();pending=seeds.tolist()
    while pending:
        i=pending.pop()
        if i in attached or signed[i]>tolerance_m:continue
        attached.add(i);pending.extend(neighbors[i]-attached)
    return [int(i) for i in sorted(attached)]


def resolve_root_boundary(vertices,faces,rigid_vertices,rigid_faces,seeds,*,clearance_m,
                          maximum_levels=3,maximum_vertices=4096,surface=None):
    """Resolve real root/free transitions without moving the source surface.

    A free face can approach a curved backing between otherwise clear nodes.
    Conforming subdivision exposes such locations to the existing real-root
    classifier. No vertex is projected, and a root must remain connected to
    the original evidenced patch within the declared physical clearance.
    """
    from calibrated_scene_reconstruction.assets.surface_refinement import uniform
    if not np.isfinite(clearance_m) or not 0<clearance_m<=.001:
        raise ValueError('Explicit positive physical sheet half-thickness required')
    solid=tree(rigid_vertices,rigid_faces) if surface is None else surface
    v=np.asarray(vertices,float);f=np.asarray(faces,int);roots=list(map(int,seeds));history=[]
    for level in range(maximum_levels+1):
        roots=embedded_roots(v,f,rigid_vertices,rigid_faces,roots,tolerance_m=clearance_m,surface=solid)
        free=f[~np.isin(f,roots).any(1)]
        if not len(free):raise ValueError('Root refinement must retain an exposed free sheet')
        tri=v[free];samples=np.r_[v[np.unique(free)],tri.mean(1),
            (tri[:,0]+tri[:,1])/2,(tri[:,1]+tri[:,2])/2,(tri[:,2]+tri[:,0])/2]
        distances=np.array([signed_distance(solid,p) for p in samples])
        minimum=float(distances.min());history.append({'level':level,'vertices':len(v),
            'free_faces':len(free),'minimum_sampled_free_clearance_m':minimum})
        if minimum>=clearance_m-1e-6:
            return v,f,roots,{'source_surface_changed':False,'root_boundary_history':history,
                'clearance_m':clearance_m,'physical_or_source_approval':False}
        if level<maximum_levels:v,f,roots,_=uniform(v,f,roots,1,maximum_vertices=maximum_vertices)
    raise ValueError('Source-preserving root subdivision cannot resolve physical clearance; revise the construction')


def verify(vertices, faces, parts, bindings, *, tolerance_m=.00005):
    by_name={p['name']:p for p in parts};trees={};result={}
    for binding in bindings:
        rigid=by_name[binding['rigid']];surface=by_name[binding['surface']]
        if binding['rigid'] not in trees:
            a,n,b,count=(rigid[k] for k in ('vertex_start','vertex_count','face_start','triangle_count'))
            trees[binding['rigid']]=tree(vertices[a:a+n],faces[b:b+count]-a)
        start=surface['vertex_start']
        maximum=max(signed_distance(trees[binding['rigid']],vertices[start+i]) for i in binding['vertices'])
        if maximum>tolerance_m:raise ValueError(f"{binding['surface']}: modeled bond crosses empty space ({maximum:.9g}m)")
        result[binding['surface']]={'maximum_root_gap_m':float(max(0,maximum)),'tolerance_m':tolerance_m}
    return result


def exposed_patch(vertices, faces, rigid_vertices, rigid_faces, *, attributes=None):
    """Model only the flexible material exposed outside a rigid receptacle.

    Embedded portions belong to the composite rigid root. Preserve free rims;
    intersections become real surface attachment boundaries, not air pins.
    """
    vertices=np.asarray(vertices,float);faces=np.asarray(faces,int);solid=tree(rigid_vertices,rigid_faces)
    attributes=None if attributes is None else np.asarray(attributes,float)
    if attributes is not None and (len(attributes)!=len(vertices) or not np.isfinite(attributes).all()):
        raise ValueError('Clipped vertex attributes must match finite source vertices')
    fields=None if attributes is None else attributes.tolist()
    distance=[signed_distance(solid,p) for p in vertices];points=vertices.tolist();roots=set();crossings={}
    def intersection(a,b):
        key=tuple(sorted((a,b)))
        if key in crossings:return crossings[key]
        if abs(distance[a])<=1e-6:roots.add(a);crossings[key]=a;return a
        if abs(distance[b])<=1e-6:roots.add(b);crossings[key]=b;return b
        low,high=0.,1.;left=distance[a]
        for _ in range(24):
            middle=(low+high)/2;value=signed_distance(solid,vertices[a]*(1-middle)+vertices[b]*middle)
            if (value>=0)==(left>=0):low=middle
            else:high=middle
        t=(low+high)/2;point=vertices[a]*(1-t)+vertices[b]*t
        if np.linalg.norm(point-vertices[a])<=1e-6:index=a
        elif np.linalg.norm(point-vertices[b])<=1e-6:index=b
        else:
            index=len(points);points.append(point.tolist())
            if fields is not None:fields.append((attributes[a]*(1-t)+attributes[b]*t).tolist())
        roots.add(index);crossings[key]=index;return index
    result=[]
    for face in faces:
        polygon=[]
        for a,b in zip(face,np.roll(face,-1)):
            if distance[a]>=0:polygon.append(int(a))
            if (distance[a]>=0)!=(distance[b]>=0):polygon.append(intersection(int(a),int(b)))
        polygon=list(dict.fromkeys(polygon))
        for i in range(1,len(polygon)-1):
            triangle=[polygon[0],polygon[i],polygon[i+1]];p=np.asarray([points[j] for j in triangle])
            area=np.linalg.norm(np.cross(p[1]-p[0],p[2]-p[0]))/2
            if area<=1e-14:
                if not roots.intersection(triangle):raise ValueError('Degenerate free petal surface')
                continue  # Only an unresolved cut sliver belongs to the rigid root.
            result.append(triangle)
    if len(result)<2:raise ValueError('No resolved exposed flexible patch')
    used=sorted(set(np.asarray(result).ravel()));mapping={old:new for new,old in enumerate(used)}
    boundary=sorted(mapping[i] for i in roots if i in mapping)
    if not boundary:raise ValueError('Exposed appendage has no real receptacle boundary')
    output=(np.asarray(points)[used],np.asarray([[mapping[j] for j in f] for f in result]),boundary)
    return output if fields is None else (*output,np.asarray(fields)[used])
def root_strip_faces(faces,columns,root_rows):
    """Attach root rows without changing any existing triangle's orientation."""
    f=np.asarray(faces)
    if (f.ndim!=2 or f.shape[1:]!=(3,) or not len(f) or f.dtype.kind not in 'iu'
            or type(columns) is not int or columns<2 or type(root_rows) is not int or root_rows<2):
        raise ValueError('Triangular strip and resolved added root rows required')
    directions=[]
    edges=np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]])
    for a in range(columns-1):
        selected=edges[((edges[:,0]==a)&(edges[:,1]==a+1))
                       |((edges[:,1]==a)&(edges[:,0]==a+1))]
        if len(selected)!=1:
            raise ValueError('Declared first row must be an actual consistently oriented strip boundary')
        directions.append(1 if selected[0,0]==a else -1)
    if len(set(directions))!=1:
        raise ValueError('Root boundary triangle winding is inconsistent')
    new=[]
    for i in range(root_rows-1):
        for j in range(columns-1):
            a=i*columns+j;b=a+columns;new.extend([(a,b,b+1),(a,b+1,a+1)])
    new=np.asarray(new,int)
    # Fixed historical ordering has a positive last-row boundary; the old
    # source edge must run the other way across the same physical seam.
    if directions[0]>0:new=new[:,::-1]
    return np.concatenate([new,f+(root_rows-1)*columns])


def clamped_strip(points,faces,solid_vertices,solid_faces,*,columns,root_rows=2,footprint_depth_m,
                  embed_depth_m,free_clearance_m,max_displacement_m,minimum_span_m):
    """Add a real clamped base using one rigid root transform, before rest charts.

    Independent closest-point projections can collapse neighboring material
    nodes. Preserve the root cross-section and every original free vertex;
    translate the inferred base into the actual solid as one physical patch.
    """
    import numpy as np
    from mathutils import Vector
    original=np.asarray(points,float)
    if (columns<2 or root_rows<2 or len(original)%columns or len(original)//columns<3
            or min(footprint_depth_m,embed_depth_m,free_clearance_m,max_displacement_m,minimum_span_m)<=0):
        raise ValueError('Resolved structured strip and positive physical bond footprint required')
    solid=tree(solid_vertices,solid_faces);front=original[:columns]
    near,normal,_,_=solid.find_nearest(Vector(front.mean(0)));normal=np.asarray(normal)
    tangent=original[columns:2*columns].mean(0)-front.mean(0);tangent-=normal*(tangent@normal)
    length=np.linalg.norm(tangent)
    if length<minimum_span_m:raise ValueError('Unresolved material root direction')
    tangent/=length
    base=np.concatenate([front-tangent*footprint_depth_m*(1-i/(root_rows-1)) for i in range(root_rows)])
    solution=None
    for sign in (-1.,1.):
        depth=0.
        for _ in range(80):
            target=base+sign*normal*depth
            distance=max(signed_distance(solid,q) for q in target)
            if distance<=-embed_depth_m:
                solution=sign*normal*depth;break
            depth+=max(.000025,1.05*(distance+embed_depth_m))
            if depth>max_displacement_m:break
        if solution is not None:break
    if solution is None:raise ValueError('A finite bonded cross-section cannot fit within the hidden construction prior')
    base+=solution;v=np.concatenate([base,original[columns:]])
    f=root_strip_faces(faces,columns,root_rows);roots=set(range(root_rows*columns))
    for j in range(root_rows*columns,len(v)):
        if signed_distance(solid,v[j])<=0:roots.add(j)
    roots=np.array(sorted(roots));span=np.linalg.svd(v[roots]-v[roots].mean(0),compute_uv=False)
    if span[1]<minimum_span_m:raise ValueError('Root patch has insufficient physical width')
    return v,f,roots.tolist(),{'root_rows':root_rows,'root_nodes':len(roots),
        'root_footprint_singular_values_m':span.tolist(),'inferred_root_translation_m':solution.tolist(),
        'maximum_construction_displacement_m':float(np.linalg.norm(solution)),
        'original_free_vertices_changed':False,'minimum_embed_depth_m':embed_depth_m,
        'source_and_native_review_required':True}
