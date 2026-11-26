"""The official Diffusion Policy (third_party/diffusion_policy, envs/dp_official) on our datasets: the dataset class
its training workspace instantiates from configs/dp_official/*.yaml, and a DiT variant of its hybrid image policy.

    PYTHONPATH=src envs/dp_official/bin/python third_party/diffusion_policy/train.py --config-dir configs/dp_official \
        --config-name place_hybrid training.device=cuda:0

ReplayImageDataset is the official real-robot image dataset (diffusion_policy.dataset.real_pusht_image_dataset
.RealPushTImageDataset) reading a replay buffer tools/export_dp_official.py wrote (a .zarr.zip, copied into memory
with its images still JPEG-compressed) in place of converting recorded videos: the same sampler, masks, normalizer
(limits to [-1, 1] for the action and the low-dim observations, images to [0, 1]) and samples; low_dim_dropout
adds the LeRobot baseline's state dropout (configs/dp_official/*_sd05.yaml).

DiffusionDiTHybridImagePolicy is the official hybrid image policy (robomimic's encoder) with a diffusion transformer
(DiT, adaLN-Zero: Peebles & Xie 2023) as the noise model in place of its 1D conditional UNet, called the same way (the
noisy chunk, the diffusion step, the observations as the global condition) and everything else unchanged
(configs/dp_official/*_dit.yaml): the chunk's steps are the tokens, the step's and the observations' embeddings
modulate every block.
"""
import copy
from typing import Dict

import numpy as np
import torch
import zarr
from torch import nn
from threadpoolctl import threadpool_limits

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy

register_codecs()


class ReplayImageDataset(BaseImageDataset):
    def __init__(self, shape_meta: dict, dataset_path: str, horizon=1, pad_before=0, pad_after=0, n_obs_steps=None,
                 n_latency_steps=0, seed=42, val_ratio=0.0, max_train_episodes=None, low_dim_dropout=0.0):
        with zarr.ZipStore(dataset_path, mode="r") as store:
            replay_buffer = ReplayBuffer.copy_from_store(src_store=store, store=zarr.MemoryStore())
        rgb_keys = [k for k, v in shape_meta["obs"].items() if v.get("type", "low_dim") == "rgb"]
        lowdim_keys = [k for k, v in shape_meta["obs"].items() if v.get("type", "low_dim") == "low_dim"]
        key_first_k = {k: n_obs_steps for k in rgb_keys + lowdim_keys} if n_obs_steps is not None else {}
        val_mask = get_val_mask(n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = downsample_mask(mask=~val_mask, max_n=max_train_episodes, seed=seed)
        self.sampler = SequenceSampler(replay_buffer=replay_buffer, sequence_length=horizon + n_latency_steps,
                                       pad_before=pad_before, pad_after=pad_after, episode_mask=train_mask,
                                       key_first_k=key_first_k)
        self.replay_buffer, self.shape_meta = replay_buffer, shape_meta
        self.rgb_keys, self.lowdim_keys = rgb_keys, lowdim_keys
        self.n_obs_steps, self.val_mask, self.horizon = n_obs_steps, val_mask, horizon
        self.n_latency_steps, self.pad_before, self.pad_after = n_latency_steps, pad_before, pad_after
        # low_dim_dropout: the share of training samples whose low-dim observations (all their steps) are the
        # dataset's mean (as the LeRobot baseline's state_dropout): a policy that can follow a trajectory from its
        # state alone has to find the cup in the image
        self.low_dim_dropout = float(low_dim_dropout)
        self.low_dim_mean = {k: np.asarray(replay_buffer[k][:]).mean(0).astype(np.float32) for k in lowdim_keys}

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(replay_buffer=self.replay_buffer,
                                          sequence_length=self.horizon + self.n_latency_steps,
                                          pad_before=self.pad_before, pad_after=self.pad_after,
                                          episode_mask=self.val_mask)
        val_set.val_mask = ~self.val_mask
        val_set.low_dim_dropout = 0.0
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(self.replay_buffer["action"])
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(self.replay_buffer[key])
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)
        T_slice = slice(self.n_obs_steps)
        obs_dict = {}
        for key in self.rgb_keys:  # T, H, W, C uint8 -> T, C, H, W float in [0, 1]
            obs_dict[key] = np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.0
            del data[key]
        drop = self.low_dim_dropout > 0 and np.random.random() < self.low_dim_dropout
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            if drop:
                obs_dict[key][:] = self.low_dim_mean[key]
            del data[key]
        action = data["action"].astype(np.float32)
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]
        return {"obs": dict_apply(obs_dict, torch.from_numpy), "action": torch.from_numpy(action)}


def _modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class _DiTBlock(nn.Module):
    def __init__(self, d, heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        h = int(d * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(d, h), nn.GELU(approximate="tanh"), nn.Linear(h, d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[-1].weight)  # adaLN-Zero: each block starts as the identity
        nn.init.zeros_(self.ada[-1].bias)

    def forward(self, x, c):
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(c).chunk(6, -1)
        y = _modulate(self.norm1(x), sh1, sc1)
        x = x + g1.unsqueeze(1) * self.attn(y, y, y, need_weights=False)[0]
        return x + g2.unsqueeze(1) * self.mlp(_modulate(self.norm2(x), sh2, sc2))


class DiTDenoiser(nn.Module):
    """ConditionalUnet1D's call (sample (B, T, A), timestep, global_cond (B, C)) -> (B, T, A), as a DiT."""

    def __init__(self, input_dim, horizon, global_cond_dim, d_model=768, depth=12, heads=12, mlp_ratio=4.0):
        super().__init__()
        d = d_model
        self.x_in = nn.Linear(input_dim, d)
        self.pos = nn.Parameter(torch.randn(1, horizon, d) * 0.02)
        self.t_emb = nn.Sequential(SinusoidalPosEmb(d), nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.c_emb = nn.Sequential(nn.Linear(global_cond_dim, d), nn.SiLU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList(_DiTBlock(d, heads, mlp_ratio) for _ in range(depth))
        self.norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.out = nn.Linear(d, input_dim)
        for lin in (self.ada[-1], self.out):
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, sample, timestep, local_cond=None, global_cond=None, **kwargs):
        t = timestep if torch.is_tensor(timestep) else torch.tensor([timestep], device=sample.device)
        t = (t[None] if t.ndim == 0 else t).to(sample.device).expand(sample.shape[0])  # as ConditionalUnet1D
        c = self.t_emb(t) + self.c_emb(global_cond)
        h = self.x_in(sample) + self.pos[:, :sample.shape[1]]
        for blk in self.blocks:
            h = blk(h, c)
        shift, scale = self.ada(c).chunk(2, -1)
        return self.out(_modulate(self.norm(h), shift, scale))


class DiffusionDiTHybridImagePolicy(DiffusionUnetHybridImagePolicy):
    """The official DiffusionUnetHybridImagePolicy with DiTDenoiser (`dit`: its d_model, depth, heads, mlp_ratio) as
    the noise model; the observations must be the global condition (obs_as_global_cond)."""

    def __init__(self, *args, dit=None, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.obs_as_global_cond:
            raise ValueError("DiffusionDiTHybridImagePolicy: obs_as_global_cond only")
        self.model = DiTDenoiser(self.action_dim, self.horizon, self.obs_feature_dim * self.n_obs_steps,
                                 **dict(dit or {}))
        print(f"DiT noise model: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M parameters", flush=True)
