"""Torch forward kinematics matching the metric CAD model; no measured state input."""
import numpy as np


class TorchModel:
    def __init__(self, model, names, *, device='cpu', dtype=None):
        import torch
        self.torch = torch
        self.dtype = dtype or torch.float32
        self.device = device
        self.root = model.root
        self.names = list(names)
        parents = {joint.child: joint for joint in model.joints}
        required = {model.root}
        for name in names:
            while name != model.root:
                required.add(name)
                name = parents[name].parent
        self.joints = [joint for joint in model.joints if joint.child in required]
        self.constants = {joint.name: (self.tensor(joint.frame0), self.tensor(joint.frame1_inverse),
                                       self.tensor(joint.axis)) for joint in self.joints}

    def tensor(self, value):
        return self.torch.as_tensor(np.asarray(value), dtype=self.dtype, device=self.device)

    def transforms(self, values):
        torch = self.torch
        transforms = {self.root: torch.eye(4, dtype=self.dtype, device=self.device)}
        zero = self.tensor(0.)
        for joint in self.joints:
            frame0, frame1, axis = self.constants[joint.name]
            q = values.get(joint.name, zero)
            motion = torch.eye(4, dtype=self.dtype, device=self.device)
            if joint.kind == 'revolute':
                x, y, z = axis.unbind()
                K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero]).reshape(3, 3)
                R = torch.eye(3, dtype=self.dtype, device=self.device)+torch.sin(q)*K+(1-torch.cos(q))*(K@K)
                motion[:3, :3] = R
            elif joint.kind == 'prismatic':
                motion[:3, 3] = axis*q
            elif joint.kind != 'fixed':
                raise ValueError('Unsupported differentiable CAD joint kind')
            transforms[joint.child] = transforms[joint.parent]@frame0@motion@frame1
        return transforms

    def points(self, values, surfaces):
        transforms = self.transforms(values)
        return self.torch.cat([surfaces[name]@transforms[name][:3, :3].T+transforms[name][:3, 3]
                               for name in self.names])
