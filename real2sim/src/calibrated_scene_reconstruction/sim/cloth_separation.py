"""Finite-thickness source-geometry preparation, never a native state writer."""
import numpy as np
import warp as wp


def separate_triangles(points,faces,gap,*,device='cuda:0',iterations=4):
    """Correct intersecting triangle proposals; never modify a native state.

    Point separation can miss an edge passing through a triangle. Move the
    actual disjoint faces along a separating normal, then require independent
    material, solid-contact and native checks on the resulting proposal.
    """
    from calibrated_scene_reconstruction.assets.cloth_contact import self_intersections
    points=np.asarray(points,float).copy();faces=np.asarray(faces)
    if (not np.isfinite(points).all() or points.ndim!=2 or points.shape[1:]!=(3,)
            or faces.ndim!=2 or faces.shape[1:]!=(3,) or faces.dtype.kind not in 'iu'
            or not len(faces) or faces.min()<0 or faces.max()>=len(points)
            or not np.isfinite(gap) or gap<=0 or type(iterations) is not int or not 1<=iterations<=32):
        raise ValueError('Finite triangle proposal, physical separation and bounded correction budget required')
    for _ in range(iterations):
        crossing=self_intersections(points,faces,device=device,example_limit=256)
        if crossing['pass']:break
        move=np.zeros_like(points);count=np.zeros(len(points))
        for first,second in crossing['example_face_pairs']:
            aa,bb=faces[first],faces[second];a,b=points[aa],points[bb]
            normals=np.array([np.cross(q[1]-q[0],q[2]-q[0]) for q in [a,b]])
            lengths=np.linalg.norm(normals,axis=1)
            if (lengths<1e-14).any():raise ValueError('Degenerate proposal faces require a new source geometry')
            normals/=lengths[:,None]
            if np.dot(*normals)<0:normals[1]*=-1
            axis=normals.sum(0)
            if np.linalg.norm(axis)<1e-12:axis=normals[0]
            axis/=np.linalg.norm(axis);u,v=a@axis,b@axis
            before=u.max()-v.min()+gap;after=v.max()-u.min()+gap
            delta=(-.5*before if before<=after else .5*after)*axis
            move[aa]+=delta;move[bb]-=delta;count[aa]+=1;count[bb]+=1
        points+=move/np.maximum(count,1)[:,None]
    return points


@wp.kernel
def _separate(grid:wp.uint64,points:wp.array(dtype=wp.vec3),rest:wp.array(dtype=wp.vec3),
              diameter:float,output:wp.array(dtype=wp.vec3),ambiguous:wp.array(dtype=int)):
    i=wp.tid()
    query=wp.hash_grid_query(grid,points[i],diameter)
    move=wp.vec3(0.0);count=float(0.0)
    for j in query:
        if i!=j and wp.length(rest[i]-rest[j])>1.05*diameter:
            delta=points[i]-points[j];distance=wp.length(delta)
            if distance<diameter:
                if distance<1.e-10:
                    wp.atomic_add(ambiguous,0,1)
                else:
                    move+=.5*(diameter-distance)*delta/distance
                    count+=1.0
    move=move/wp.max(count,1.0)
    length=wp.length(move)
    if length>.05*diameter:move*=.05*diameter/length
    output[i]=points[i]+move


class Separation:
    """Separate distant material samples while retaining local fold neighbours.

    The caller must still check actual triangles for self intersections,
    intrinsic strain, rigid contacts and particle coverage, then run fresh
    native dynamics. Point separation alone is not a cloth certificate.
    """
    def __init__(self,rest,radius,*,device='cuda:0',iterations=8):
        rest=np.asarray(rest,float)
        if (rest.ndim!=2 or rest.shape[1:]!=(3,) or len(rest)<3 or not np.isfinite(rest).all()
                or not np.isfinite(radius) or radius<=0 or type(iterations) is not int or not 1<=iterations<=100):
            raise ValueError('Finite material coordinates, positive physical radius and bounded iterations required')
        wp.init();self.device=device;self.radius=float(radius);self.iterations=iterations
        self.rest=wp.array(rest.astype('float32'),dtype=wp.vec3,device=device)
        self.points=wp.empty(len(rest),dtype=wp.vec3,device=device)
        self.output=wp.empty_like(self.points);self.ambiguous=wp.zeros(1,dtype=int,device=device)
        self.grid=wp.HashGrid(128,128,128,device=device)

    def __call__(self,points):
        points=np.asarray(points,float)
        if points.shape!=(len(self.rest),3) or not np.isfinite(points).all():raise ValueError('Matching finite source geometry required')
        self.points.assign(points.astype('float32'));self.ambiguous.zero_()
        for _ in range(self.iterations):
            self.grid.build(self.points,2*self.radius)
            wp.launch(_separate,dim=len(points),inputs=[self.grid.id,self.points,self.rest,2*self.radius,
                       self.output,self.ambiguous],device=self.device)
            self.points,self.output=self.output,self.points
        if self.ambiguous.numpy()[0]:raise ValueError('Coincident distant material points need a new source geometry proposal')
        return self.points.numpy().astype(float)
