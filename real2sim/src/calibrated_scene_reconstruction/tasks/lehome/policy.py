"""An official Diffusion Policy checkpoint (configs/dp_official/lehome_garment_*.yaml, served by
tools/dp_official_server.py) as a LeHome Challenge policy: `--policy_type dp_official` of the challenge's scripts.eval
(registered by apps/tasks/lehome_run.py), the server's port in $SCENE_RECONSTRUCTION_DP_PORT. The control is
calibrated_scene_reconstruction.policy.dp_official_client's: observed and planned at the export's 10 Hz, executed every env step.

    PYTHONPATH=src envs/dp_official/bin/python tools/dp_official_server.py <run>/checkpoints/latest.ckpt --port 5591 &
    cd envs/lehome/lehome-challenge
    SCENE_RECONSTRUCTION_DP_PORT=5591 CUDA_VISIBLE_DEVICES=<g> ../../../tools/run_isaac.sh ../../../apps/tasks/lehome_run.py -m scripts.eval \
        --policy_type dp_official --garment_type top_long --num_episodes 2 --device cpu --enable_cameras --headless
"""
from __future__ import annotations

import os

import numpy as np
from scripts.eval_policy.base_policy import BasePolicy
from scripts.eval_policy.registry import PolicyRegistry



from calibrated_scene_reconstruction.core.environment import setting

class DPOfficialPolicy(BasePolicy):
    client_factory = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.client_factory is None:
            raise RuntimeError("Register the policy with an injected client factory before constructing it")
        self.client = self.client_factory(int(setting("DP_PORT", "5581")))

    def reset(self):
        self.client.reset()

    def select_action(self, observation):
        images = {c: observation[f"observation.images.{c}"] for c in self.client.cams}
        return np.asarray(self.client.act(images, observation["observation.state"]), np.float32)


def register_policy(client_factory):
    """Register the challenge adapter with an application-owned client implementation."""
    DPOfficialPolicy.client_factory = staticmethod(client_factory)
    PolicyRegistry.register("dp_official")(DPOfficialPolicy)
    return DPOfficialPolicy
