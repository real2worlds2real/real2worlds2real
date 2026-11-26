"""Optional GPU proposal rasterizer; acceptance always uses the CPU evidence path.

The same actual triangles and every cached camera are used. This accelerates
parameter search, without replacing exact integer-pixel-centre CPU acceptance.
"""
import numpy as np


class ProposalRaster:
    def __init__(self, *, legacy_antialias=False):
        import torch
        import nvdiffrast.torch as dr
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        self.torch, self.dr = torch, dr
        self.context = dr.RasterizeCudaContext()
        self.last_mesh = None
        self.cameras = {}
        self.legacy_antialias = bool(legacy_antialias)

    def __call__(self, mesh, K, R, c, window, shape):
        torch = self.torch
        if self.last_mesh is not mesh:
            self.vertices = torch.as_tensor(np.asarray(mesh.vertices), dtype=torch.float32, device='cuda')
            self.faces = torch.as_tensor(np.asarray(mesh.faces).astype(np.int32), device='cuda')
            self.last_mesh = mesh
            if self.legacy_antialias:
                self.topology = self.dr.antialias_construct_topology_hash(self.faces)
        key = (np.asarray(K).tobytes(), np.asarray(R).tobytes(), np.asarray(c).tobytes(), tuple(window), tuple(shape))
        if key not in self.cameras:
            x0, y0, _, _, scale = window
            camera_K = np.array([[scale, 0., -x0*scale], [0., scale, -y0*scale], [0., 0., 1.]]) @ K
            self.cameras[key] = [torch.as_tensor(v, dtype=torch.float32, device='cuda') for v in (camera_K, R, c)]
        K, R, c = self.cameras[key]
        camera = (self.vertices-c) @ R
        h, w = shape
        z = camera[:, 2]
        # OpenCV pixel centres are integer coordinates. GPU sample centres are
        # at half pixels, hence the +.5 shift of the projected principal point.
        x = 2*K[0, 0]*camera[:, 0]/w + (2*(K[0, 2]+.5)/w-1)*z
        y = 2*K[1, 1]*camera[:, 1]/h + (2*(K[1, 2]+.5)/h-1)*z
        near, far = .05, 100.
        depth = (far+near)/(far-near)*z-2*far*near/(far-near)
        clip = torch.stack([x, y, depth, z], dim=-1)[None].contiguous()
        raster, _ = self.dr.rasterize(self.context, clip, self.faces, shape, grad_db=False)
        self.raster=raster
        if self.legacy_antialias:
            # Historical diagnostics only: positive antialias coverage expands
            # the silhouette. Current proposals sample the hard face IDs.
            coverage = (raster[..., 3:4]>0).float()
            coverage = self.dr.antialias(coverage, raster, clip, self.faces, topology_hash=self.topology)
            return (coverage[0, ..., 0]>0).cpu().numpy()
        return (raster[0, ..., 3]>0).cpu().numpy()

    def visible(self,mesh,K,R,c,window,shape,*,target_faces):
        self(mesh,K,R,c,window,shape)
        ids=self.raster[0,...,3].to(self.torch.int64).cpu().numpy()-1
        visible=ids>=0
        visible[visible]&=np.asarray(target_faces,bool)[ids[visible]]
        return visible


def optional(backend='auto'):
    if backend not in {'auto','cpu','cuda'}:
        raise ValueError('Proposal backend must be auto, cpu or cuda')
    if backend=='cpu':
        return None
    try:
        return ProposalRaster()
    except (ImportError, RuntimeError) as exc:
        if backend=='cuda':
            raise
        print(f'[objects.fit] GPU proposals unavailable: {exc}; using CPU', flush=True)
        return None
