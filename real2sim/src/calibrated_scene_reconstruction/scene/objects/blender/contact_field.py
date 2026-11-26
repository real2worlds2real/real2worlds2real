"""Differentiable distances for source geometry proposals, never native pins."""
import numpy as np
import torch
import warp as wp

from calibrated_scene_reconstruction.assets.surface_distance import SignedSurface


@wp.kernel
def _query(mesh:wp.uint64,points:wp.array(dtype=wp.vec3),distances:wp.array(dtype=float),
           normals:wp.array(dtype=wp.vec3),hits:wp.array(dtype=int)):
    i=wp.tid();p=points[i]
    query=wp.mesh_query_point_sign_winding_number(mesh,p,10000.)
    hits[i]=int(query.result)
    if query.result:
        closest=wp.mesh_eval_position(mesh,query.face,query.u,query.v)
        delta=p-closest;length=wp.length(delta)
        distances[i]=query.sign*length
        normal=wp.mesh_eval_face_normal(mesh,query.face)
        if length>1.e-10:normal=query.sign*delta/length
        normals[i]=normal


class _Distance(torch.autograd.Function):
    @staticmethod
    def forward(ctx,points,surface):
        distances,normals=surface.query(points)
        ctx.save_for_backward(normals)
        return distances

    @staticmethod
    def backward(ctx,gradient):
        return gradient[:,None]*ctx.saved_tensors[0],None


class ContactField:
    """Exact authored-solid queries with piecewise signed-distance gradients.

    Closest-triangle distance is nondifferentiable on medial axes and edges.
    Projection uses a detached normal, a first-order tangent update. Full
    triangle/material/self-contact acceptance still runs on actual proposals.
    """
    def __init__(self,vertices,faces,*,device='cuda:0'):
        self.device=torch.device(device)
        self.surface=SignedSurface(vertices,faces,device=str(self.device))

    def query(self,points):
        if (points.ndim!=2 or points.shape[1]!=3 or points.dtype!=torch.float32
                or points.device!=self.device or not torch.isfinite(points).all()):
            raise ValueError('Finite float32 source points on the field device required')
        p=points.contiguous();d=torch.empty(len(p),device=p.device);n=torch.empty_like(p)
        hits=torch.empty(len(p),dtype=torch.int32,device=p.device)
        kwargs=dict(inputs=[self.surface.mesh.id,wp.from_torch(p,dtype=wp.vec3),wp.from_torch(d),
                            wp.from_torch(n,dtype=wp.vec3),wp.from_torch(hits)],device=str(self.device))
        if p.is_cuda:
            with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(p.device))):
                wp.launch(_query,dim=len(p),**kwargs)
        else:wp.launch(_query,dim=len(p),**kwargs)
        if not bool(hits.all()) or not bool(torch.isfinite(d).all()):
            raise ValueError('Actual solid query failed; cannot assume free space')
        return d,n

    def __call__(self,points):return _Distance.apply(points,self)

    def project(self,points,clearance):
        if not np.isfinite(clearance) or clearance<=0:raise ValueError('Positive geometry clearance required')
        distance,normals=self.query(points)
        # Keep distance differentiation while treating the local normal as a
        # geometric tangent approximation, not an applied material force.
        distance=_Distance.apply(points,self)
        return points+torch.relu(clearance-distance)[:,None]*normals.detach()
