"""Diffusion Policy (LeRobot) on our exported datasets: features, splits and the training loop.

    python -m calibrated_scene_reconstruction.policy.diffusion configs/policy/diffusion_pick_place.json [--dataset DIR] [--out DIR]

The dataset is one calibrated_scene_reconstruction.datagen.export.lerobot output (one robot, LeRobot v2.1). Inputs: the robot cameras (`cameras`,
default every camera but the third-person one, resized to `image_size`), observation.state (joint positions) and, for
pick-and-place with `use_goal` (default true), observation.goal as LeRobot's observation.environment_state; without
it the policy finds the goal in the images (the goal marker of a batch collection). Output: action (joint targets).
Episodes of the scenes in `held_out_tags` (cousin_05, and its randomized configurations cousin_05__d<k>; kept for
the evaluation) are not trained on.
Augmentation (`augment`, training only, the same draw for every frame of a sample's observation window): colour
jitter (brightness, contrast, saturation, hue), a small random affine (shift, rotation, scale) and, in the policy,
DP's random crop (`crop_shape`); the renders already vary in light, camera mount and scene (calibrated_scene_reconstruction.dr). The joint
state gets Gaussian noise (`state_noise`, rad) and, with probability `state_dropout` per sample, is replaced by the
dataset's mean state: a policy that can replay a trajectory from its joints alone ignores the images (it fitted the
demonstrations to 0.008 rad and failed every closed-loop rollout).
Optimisation: AdamW, linear warmup (`lr_warmup` steps) then cosine decay (`lr_schedule: "cosine"`), and an EMA of the
weights (`ema`, DP's schedule: decay min(ema_max, 1 - (1 + step) ** -0.75)); checkpoints step_N hold the EMA
weights (what calibrated_scene_reconstruction.policy.runners and apps/episode.py --policy load), step_N_raw the trained ones.

Variants (every option defaults to the earlier behaviour):
* `state_mode`: "joints" (the joint state of every observation step), "current" (the current step's only: the earlier
  steps repeat it, so the state holds no velocity to extrapolate) or "none" (the dataset's mean state, a constant:
  vision only); `state_joints` a regex of the joints the state keeps (default every joint).
* `action_space`: "joints" (the commanded joint targets) or "ee": the hand's pose in the robot's base frame
  (observation.tcp_base, xyz + 6D rotation: calibrated_scene_reconstruction.datagen.export.lerobot --base-frame) and the hand's joints
  (`hand_joints`); the evaluation solves the IK.
* `pretrained_backbone_weights`: an ImageNet ResNet (BatchNorm kept).
* `frame_cache`: the cameras decoded once at `image_size` into a uint8 array beside the dataset (<dataset>/cache/),
  read by a light dataset instead of decoding videos per sample (the loader, not the GPU, bounded the step);
  `frame_cache_quality`: the frames as JPEG of that quality (frame_cache_jpeg; 90: ~1.7 / 255 mean error), small
  enough to stay in memory, in place of the raw cache read from disk.
* `init_from`: a checkpoint (step_N) whose weights start the run (fine-tuning, a fresh schedule).
* `channels_last`, `compile` (torch.compile of the loss, the normalisation left eager; the first compiled loss is
  checked against the eager one), `amp: "bf16"` (the forward pass under bfloat16 autocast; weights, optimiser and
  EMA in float32): throughput.
* `balance_goal_dir` (a power p): training frames drawn with weight 1 / n ** p, n the frames of their episode's
  goal sector (the mat's direction from the cup in the base frame, 8 sectors): 1 draws every sector alike, 0.5 halfway.
* `ee_label: "command"` (end-effector actions): the hand pose each frame's joint command puts the hand at
  (apps/command_tcp.py) as the action label, in place of the pose it reached (observation.tcp_base): the reached
  pose trails the command and, while the fingers close, moves with the contact; followed through IK it pushes the
  cup away.
* `obs_stride` (frames): the n_obs_steps observations that far apart (15 at 30 Hz: 0.5 s), a history the policy can
  tell motion and waiting by; `visual_history_grad: false`: the backbone's gradient from the current frame only, the
  earlier frames encoded without (less memory for a longer visual history).
* `action_history` (end-effector actions, frame cache): the actions executed before, `action_history_steps` (default
  n_obs_steps) of them `action_history_stride` frames apart (default obs_stride), ending at the current frame;
  `history_inject: "conv"` (the only one left): normalised as the action is, encoded by a temporal convolution into
  a 128-d embedding added to the network's conditioning; `action_history_dropout` (default state_dropout): the share
  of samples shown the mean action instead. calibrated_scene_reconstruction.policy.runners.PolicyRunner keeps the observations and its
  returned actions to match.
The resolved layout (state / arm / hand indices, the mean state) goes into train_config.json; `Predictor` rebuilds
a checkpoint's inputs and outputs from it (calibrated_scene_reconstruction.policy.runners).

Needs `lerobot` with the v2.1 dataset format (the 0.3 releases: lerobot.datasets / lerobot.policies) and torch; the
feature / split helpers here are plain Python.
"""
import argparse
import copy
import json
import math
import re
import time
from pathlib import Path

GOAL_KEY, ENV_KEY = "observation.goal", "observation.environment_state"


def load_meta(root):
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    eps = [json.loads(line) for line in (root / "meta" / "episodes.jsonl").read_text().splitlines() if line]
    return info, eps


def policy_cameras(info, cameras=None):
    vids = sorted(k[len("observation.images."):] for k, f in info["features"].items() if f["dtype"] == "video")
    cams = cameras or [c for c in vids if c != "third"]
    missing = set(cams) - set(vids)
    if missing:
        raise ValueError(f"cameras {sorted(missing)} not in the dataset ({vids})")
    return cams


def dp_features(info, cameras=None, image_size=None):
    """(input_features, output_features) as {key: {"type", "shape"}} for DiffusionConfig: images (C, H, W), state,
    the goal as environment state when the dataset has one, the action."""
    f = info["features"]
    inp = {}
    for c in policy_cameras(info, cameras):
        h, w, ch = f[f"observation.images.{c}"]["shape"]
        h, w = image_size or (h, w)
        inp[f"observation.images.{c}"] = {"type": "VISUAL", "shape": [ch, h, w]}
    inp["observation.state"] = {"type": "STATE", "shape": list(f["observation.state"]["shape"])}
    if GOAL_KEY in f:
        inp[ENV_KEY] = {"type": "ENV", "shape": list(f[GOAL_KEY]["shape"])}
    return inp, {"action": {"type": "ACTION", "shape": list(f["action"]["shape"])}}


def held_out(tag, held_out_tags, session=None):
    """A scene tag (twin, cousin_03, cousin_03__d02: a randomized configuration of cousin_03) of a held-out scene;
    `<session>:<tag>` holds out that session's scene only (a dataset over several sessions, whose
    scene tags repeat)."""
    def hit(h):
        if ":" in h:
            s, h = h.split(":", 1)
            if session is None or Path(str(session)).name != s:
                return False
        return tag == h or str(tag).startswith(f"{h}__")
    return any(hit(h) for h in held_out_tags)


def split(eps, held_out_tags=(), only_success=True):
    """(train, held-out) episode indices: held-out by the episode's source scene (held_out)."""
    train, held = [], []
    for e in eps:
        src = e.get("source") or {}
        if only_success and src.get("success") is False:
            continue
        (held if held_out(src.get("tag"), held_out_tags, src.get("session")) else train).append(e["episode_index"])
    return train, held


AUGMENT = {"brightness": 0.3, "contrast": 0.3, "saturation": 0.3, "hue": 0.04, "translate": 0.04, "degrees": 3.0,
           "scale": [0.95, 1.05], "p": 0.8}


def augment_options(augment):
    """`augment` (true or a dict over AUGMENT's keys) with AUGMENT's defaults."""
    a = augment if isinstance(augment, dict) else {}
    if set(a) - set(AUGMENT):
        raise ValueError(f"augment keys {sorted(set(a) - set(AUGMENT))}: not among {sorted(AUGMENT)}")
    return {**AUGMENT, **a}


def image_transforms(image_size=None, augment=None):
    """Resize to the policy's input, then (training) colour jitter and a small affine with probability p; v2
    transforms draw once per call, and LeRobot calls them on a camera's (n_obs_steps, C, H, W) stack."""
    from torchvision.transforms import v2

    ops = [v2.Resize(tuple(image_size), antialias=True)] if image_size else []
    if augment:
        a = augment_options(augment)
        ops.append(v2.RandomApply([v2.ColorJitter(a["brightness"], a["contrast"], a["saturation"], a["hue"]),
                                   v2.RandomAffine(a["degrees"], (a["translate"], a["translate"]), tuple(a["scale"]))],
                                  p=a["p"]))
    return v2.Compose(ops) if ops else None


def gpu_augment(x, augment):
    """image_transforms' augmentation on the GPU, batched: x (B, n_obs, C, H, W) in [0, 1], one draw per sample for
    all its frames, applied with probability p: brightness, contrast, saturation, hue (a rotation of the chroma in
    YIQ), then an affine (rotation, shift, scale; outside filled with 0) by grid sampling."""
    import torch

    a = augment_options(augment)
    B, n, C, H, W = x.shape
    dev = x.device
    u = lambda lo, hi: torch.empty(B, device=dev).uniform_(lo, hi)  # noqa: E731
    on = (torch.rand(B, device=dev) < a["p"]).float()
    y = x.reshape(B, n * C, H, W)
    sel = lambda v, one: (on * v + (1 - on) * one).view(B, 1, 1, 1, 1)  # noqa: E731
    z = y.view(B, n, C, H, W)
    z = z * sel(u(1 - a["brightness"], 1 + a["brightness"]), 1.0)
    gw = torch.tensor([0.299, 0.587, 0.114], device=dev).view(1, 1, 3, 1, 1)
    grey = (z * gw).sum(2, keepdim=True)
    m = grey.mean((1, 2, 3, 4), keepdim=True)
    z = ((z - m) * sel(u(1 - a["contrast"], 1 + a["contrast"]), 1.0) + m).clamp(0, 1)
    grey = (z * gw).sum(2, keepdim=True)
    z = ((z - grey) * sel(u(1 - a["saturation"], 1 + a["saturation"]), 1.0) + grey).clamp(0, 1)
    th = 2 * math.pi * u(-a["hue"], a["hue"]) * on
    to_yiq = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]], device=dev)
    c, s = torch.cos(th), torch.sin(th)
    rot = torch.zeros(B, 3, 3, device=dev)
    rot[:, 0, 0], rot[:, 1, 1], rot[:, 1, 2], rot[:, 2, 1], rot[:, 2, 2] = 1, c, -s, s, c
    M = torch.linalg.inv(to_yiq) @ rot @ to_yiq  # (B, 3, 3)
    z = torch.einsum("bij,bnjhw->bnihw", M, z).clamp(0, 1)
    ang = torch.deg2rad(u(-a["degrees"], a["degrees"])) * on
    sc = 1 + (u(*a["scale"]) - 1) * on
    tx, ty = (2 * a["translate"] * (torch.rand(B, device=dev) * 2 - 1) * on for _ in range(2))
    ca, sa = torch.cos(ang) / sc, torch.sin(ang) / sc  # output -> input sampling: the inverse transform
    theta = torch.stack([torch.stack([ca, sa * H / W, -tx], 1), torch.stack([-sa * W / H, ca, -ty], 1)], 1)
    grid = torch.nn.functional.affine_grid(theta, (B, n * C, H, W), align_corners=False)
    return torch.nn.functional.grid_sample(z.reshape(B, n * C, H, W), grid, mode="bilinear", padding_mode="zeros",
                                           align_corners=False).view(B, n, C, H, W)


def frame_offsets(cfg):
    """(observation, action) frame offsets: observations at -(n_obs_steps - 1) .. 0 times `obs_stride` frames (a
    history reaching back (n_obs_steps - 1) x obs_stride frames), actions at -(n_obs_steps - 1) .. horizon -
    n_obs_steps, moved by `action_shift` frames: a recorded frame holds the state after its own command
    (apps/episode.py commands, steps the physics, then records), so the command issued from that state is the next
    frame's (action_shift 1)."""
    n_obs, hz, sh = cfg["n_obs_steps"], cfg["horizon"], int(cfg.get("action_shift", 0))
    st = int(cfg.get("obs_stride", 1))
    return [(i - n_obs + 1) * st for i in range(n_obs)], [i + sh for i in range(1 - n_obs, hz - n_obs + 1)]


def history_offsets(cfg):
    """The action history's frame offsets: action_history_steps actions action_history_stride frames apart, ending
    at the current frame (by default the observation steps')."""
    k = int(cfg.get("action_history_steps") or cfg["n_obs_steps"])
    st = int(cfg.get("action_history_stride") or cfg.get("obs_stride", 1))
    return [(i - k + 1) * st for i in range(k)]


def delta_timestamps(cfg, fps, keys, action_keys=("action",)):
    """LeRobot's delta_timestamps of frame_offsets: `keys` at the observation steps, `action_keys` at the actions'."""
    obs, act = frame_offsets(cfg)
    return {**{k: [i / fps for i in obs] for k in keys}, **{k: [i / fps for i in act] for k in action_keys}}


def widen_stats(stats, keys=("observation.state", "action"), min_range=0.2):
    """MIN_MAX normalisation of a joint that barely moves (head, a tucked arm: 0 - 0.002 rad over a collection) maps its
    simulator jitter onto all of [-1, 1] and any offset far outside (0.05 rad of state noise became a std of 50 -
    500000): such a dimension's min / max are widened to its centre +- min_range / 2. Returns (stats, moving mask per
    key: the dimensions whose own range was at least a quarter of min_range)."""
    import numpy as np

    moving = {}
    for k in keys:
        if k not in stats:
            continue
        st = dict(stats[k])
        lo, hi = (np.asarray(st[n], np.float32) for n in ("min", "max"))
        rng = hi - lo
        moving[k] = rng >= 0.25 * min_range
        c = (lo + hi) / 2
        small = rng < min_range
        lo, hi = np.where(small, c - min_range / 2, lo), np.where(small, c + min_range / 2, hi)
        st["min"], st["max"] = lo.astype(np.float32), hi.astype(np.float32)
        stats[k] = st
    return stats, moving


# --- variants: what the policy sees and outputs ----------------------------------------------------------------------
STATE_MODES = ("joints", "current", "none")
TCP_KEY = "observation.tcp_base"
AHIST_KEY = "observation.action_history"  # (B, steps, action dim): the actions executed at the history's steps
HISTORY_INJECT = ("conv",)
# options of earlier variants, since removed: a config (or a checkpoint's train_config.json) that sets one is refused
REMOVED = ("privileged_relative", "privileged_noise", "estimated_layout", "aux_xy", "aux_phase", "image_mask",
           "backbone_lr_scale", "backbone_dilate", "generator", "flow_target", "flow_t", "denoiser", "dit",
           "action_relative", "episode_share", "camera_dropout", "crop_random", "imagenet_norm")


def check_options(cfg):
    """A config (or a checkpoint's) with a removed option or augment key refused, rather than trained or loaded
    without it."""
    gone = [k for k in REMOVED if cfg.get(k) is not None]
    if gone:
        raise ValueError(f"options {gone}: removed (an earlier variant of the policy)")
    if str(cfg.get("vision_backbone", "resnet18")).startswith("dinov2"):
        raise ValueError("vision_backbone dinov2: removed (a ResNet only)")
    augment_options(cfg.get("augment"))


def layout(info, cfg):
    """The variant's index layout over the dataset's joints: state_index (the joints the state keeps), arm_index /
    hand_index (the moving arm's and hand's joints), state_mode, action_space, use_goal."""
    names = info["features"]["observation.state"]["names"]
    pick = lambda rx: [i for i, n in enumerate(names) if re.search(rx, n)]  # noqa: E731
    lay = {"state_mode": cfg.get("state_mode", "joints"), "action_space": cfg.get("action_space", "joints"),
           "state_index": pick(cfg["state_joints"]) if cfg.get("state_joints") else list(range(len(names))),
           "arm_index": pick(cfg.get("arm_joints", "^R_arm_")), "hand_index": pick(cfg.get("hand_joints", "^right_")),
           "use_goal": bool(cfg.get("use_goal", True)) and GOAL_KEY in info["features"],
           "obs_stride": int(cfg.get("obs_stride", 1)), "action_history": bool(cfg.get("action_history"))}
    lay["n_state"] = len(lay["state_index"])
    if lay["action_history"]:
        off = history_offsets(cfg)
        lay.update(action_history_steps=len(off), action_history_stride=int(off[1] - off[0]) if len(off) > 1 else 1,
                   history_inject=cfg.get("history_inject", "state"))
        if lay["history_inject"] not in HISTORY_INJECT:
            raise ValueError(f"history_inject {lay['history_inject']!r}: one of {HISTORY_INJECT}")
    if lay["state_mode"] not in STATE_MODES:
        raise ValueError(f"state_mode {lay['state_mode']!r}: one of {STATE_MODES}")
    if lay["action_space"] == "ee" and TCP_KEY not in info["features"]:
        raise ValueError("action_space ee needs observation.tcp_base (calibrated_scene_reconstruction.datagen.export.lerobot --add-base-frame)")
    if lay["action_history"] and (lay["action_space"] != "ee" or not cfg.get("frame_cache")):
        raise ValueError("action_history: end-effector actions read through the frame cache")
    return lay


def action_dims(lay):
    """(arm, hand) dimensions of the policy's action: the arm joints or the hand pose (xyz + 6D), the hand joints."""
    if lay["action_space"] == "ee":
        return list(range(9)), list(range(9, 9 + len(lay["hand_index"])))
    return list(lay["arm_index"]), list(lay["hand_index"])


def policy_stats(stats, lay):
    """The dataset's stats cut to the policy's state and action (ee: tcp_base then the hand's joints)."""
    import numpy as np

    cut = lambda d, idx: {n: np.asarray(d[n], np.float32)[..., idx] for n in ("min", "max", "mean", "std")}  # noqa
    st = dict(stats)
    st["observation.state"] = cut(st["observation.state"], lay["state_index"])
    if lay["action_space"] == "ee":
        a, tcp = cut(st["action"], lay["hand_index"]), cut(st[TCP_KEY], list(range(9)))
        st["action"] = {n: np.concatenate([tcp[n], a[n]]) for n in a}
    return st


def to_policy(batch, lay, s_mean, noise=0.0, drop=0.0, mask=None, a_drop=0.0):
    """A raw batch (the dataset's full joint state, action, tcp_base at the action steps) as the policy's: the state
    cut to state_index, with Gaussian `noise` (on `mask`) and, per sample with probability `drop`, the mean state,
    then state_mode applied; the ee action built; the goal as the environment state (use_goal). With the layout's
    action_history, the actions executed at the history's steps as `AHIST_KEY` (as given, else built from the action
    columns the dataset gives at those steps, `<key>.hist`), replaced per sample with probability `a_drop` by the
    mean action."""
    import torch

    out = dict(batch)
    n_s = len(lay["state_index"])
    x = batch["observation.state"][..., lay["state_index"]]
    sm = s_mean[:n_s]
    if noise:
        x = x + noise * torch.randn_like(x) * (1.0 if mask is None or mask.numel() == 1 else mask[..., :n_s])
    if drop:
        d = torch.rand(x.shape[0], 1, 1, device=x.device) < drop
        x = torch.where(d, sm.view(1, 1, -1).expand_as(x), x)
    if lay["state_mode"] == "current":
        x = x[:, -1:].expand_as(x).contiguous()
    elif lay["state_mode"] == "none":
        x = sm.view(1, 1, -1).expand_as(x).contiguous()
    if lay.get("action_history"):  # into the conditioning by the policy's history encoder (history_modules)
        ah = out.pop(AHIST_KEY, None)
        if ah is None:
            ah = torch.cat([out.pop(TCP_KEY + ".hist"), out.pop("action.hist")[..., lay["hand_index"]]], -1)
        if a_drop:
            am = torch.as_tensor(lay["action_mean"], dtype=ah.dtype, device=ah.device).view(1, 1, -1)
            d = torch.rand(ah.shape[0], 1, 1, device=ah.device) < a_drop
            ah = torch.where(d, am.expand_as(ah), ah)
        out[AHIST_KEY] = ah.to(x.dtype)
    out["observation.state"] = x
    if "action" in batch and lay["action_space"] == "ee":
        out["action"] = torch.cat([batch[TCP_KEY], batch["action"][..., lay["hand_index"]]], -1)
    g = out.pop(GOAL_KEY, None)
    if g is not None and lay["use_goal"]:
        out[ENV_KEY] = g
    return out


# --- frame cache --------------------------------------------------------------------------------------------------
def _decoded(video, n, size):
    """The episode's frames, resized to `size` (H, W) as image_transforms does: (3, H, W) uint8 tensors."""
    import av
    import torch

    torch.set_num_threads(1)
    k = 0
    with av.open(str(video)) as c:
        for f in c.decode(video=0):
            im = torch.from_numpy(f.to_ndarray(format="rgb24")).permute(2, 0, 1)[None].float()
            if size:
                im = torch.nn.functional.interpolate(im, size=tuple(size), mode="bilinear", antialias=True,
                                                     align_corners=False)
            yield im[0].round().clamp(0, 255).byte()
            k += 1
    if k != n:
        raise RuntimeError(f"{video}: {k} frames decoded, the episode has {n}")


def _decode_into(video, path, start, n, size):
    import numpy as np

    mm = np.load(path, mmap_mode="r+")
    for k, im in enumerate(_decoded(video, n, size)):
        mm[start + k] = im.permute(1, 2, 0).numpy()
    mm.flush()


def _encode_episode(video, n, size, quality):
    """JPEG bytes of an episode's frames, decoded and resized as _decode_into (one worker's share)."""
    from torchvision.io import encode_jpeg

    return [encode_jpeg(im.contiguous(), quality=quality).numpy().tobytes() for im in _decoded(video, n, size)]


def _encode_episode_args(a):
    return _encode_episode(*a)


def _episode_videos(root, info, cam):
    """[(video, length)] of the camera, in the dataset's frame order."""
    eps = [json.loads(line) for line in (root / "meta" / "episodes.jsonl").read_text().splitlines() if line]
    return [(root / info["video_path"].format(episode_chunk=e["episode_index"] // info["chunks_size"],
                                              episode_index=e["episode_index"],
                                              video_key=f"observation.images.{cam}"), e["length"]) for e in eps]


def frame_cache(root, info, cam, size, workers=16):
    """<root>/cache/<camera>_<H>x<W>.npy: every frame of the camera, resized to `size` (H, W) as image_transforms
    does, uint8 (N, H, W, 3) in the dataset's frame order; built once (in parallel over episodes)."""
    import multiprocessing as mp

    import numpy as np

    root = Path(root)
    h, w = size or info["features"][f"observation.images.{cam}"]["shape"][:2]
    path = root / "cache" / f"{cam}_{h}x{w}.npy"
    if path.exists():
        return path
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    np.lib.format.open_memmap(tmp, "w+", np.uint8, (info["total_frames"], h, w, 3)).flush()
    jobs, start = [], 0
    for v, n in _episode_videos(root, info, cam):
        jobs.append((v, tmp, start, n, (h, w) if size else None))
        start += n
    with mp.get_context("spawn").Pool(workers) as pool:
        pool.starmap(_decode_into, jobs)
    tmp.rename(path)
    print(f"frame cache {path}: {start} frames", flush=True)
    return path


def _encode_range(src, lo, hi, quality):
    """JPEG bytes of frames lo..hi of a frame cache (one worker's share)."""
    import numpy as np
    import torch
    from torchvision.io import encode_jpeg

    torch.set_num_threads(1)
    mm = np.load(src, mmap_mode="r")
    return [encode_jpeg(torch.from_numpy(np.array(mm[i])).permute(2, 0, 1), quality=quality)
            .numpy().tobytes() for i in range(lo, hi)]


def _encode_args(a):
    return _encode_range(*a)


def frame_cache_jpeg(root, info, cam, size, quality, workers=12):
    """<root>/cache/<camera>_<H>x<W>_q<quality>.jpg: the frame cache's frames as JPEG (~20 KB a frame at 90 against
    276 KB raw: the whole cache fits the page cache, where the raw one is read from disk at ~1 GB/s for as long as
    a run trains), concatenated, and .idx.npy their byte offsets (N + 1); encoded from the raw cache (frame_cache)
    when there is one, else straight from the videos, the same frames without the raw cache on disk."""
    import multiprocessing as mp

    import numpy as np

    root = Path(root)
    h, w = size or info["features"][f"observation.images.{cam}"]["shape"][:2]
    path = root / "cache" / f"{cam}_{h}x{w}_q{quality}.jpg"
    idx = path.with_suffix(".idx.npy")
    if path.exists() and idx.exists():  # without the raw cache too (it may have been removed once this one was built)
        return path
    path.parent.mkdir(exist_ok=True)
    src = path.parent / f"{cam}_{h}x{w}.npy"
    if src.exists():
        n = int(np.load(src, mmap_mode="r").shape[0])
        chunk = 4096
        fn, jobs = _encode_args, [(src, lo, min(lo + chunk, n), quality) for lo in range(0, n, chunk)]
    else:
        vids = _episode_videos(root, info, cam)
        n = sum(k for _, k in vids)
        fn, jobs = _encode_episode_args, [(v, k, (h, w) if size else None, quality) for v, k in vids]
        workers = max(workers, 16)  # decoding as frame_cache does
    tmp = path.with_suffix(".tmp")
    off = [0]
    with open(tmp, "wb") as f, mp.get_context("spawn").Pool(workers) as pool:
        for part in pool.imap(fn, jobs):
            for b in part:
                f.write(b)
                off.append(off[-1] + len(b))
    np.save(idx, np.asarray(off, np.int64))
    tmp.rename(path)
    print(f"frame cache {path}: {n} frames, {off[-1] / 1e9:.1f} GB", flush=True)
    return path


class _JpegFrames:
    """frame_cache_jpeg's frames, read (pread) and decoded: (n, H, W, 3) uint8 as _FrameFile.read."""

    def __init__(self, path):
        import os

        import numpy as np

        self.off = np.load(Path(path).with_suffix(".idx.npy"))
        self.fd = os.open(path, os.O_RDONLY)

    def read(self, idx):
        import os

        import torch
        from torchvision.io import decode_jpeg

        out = []
        for i in idx:
            a, b = int(self.off[i]), int(self.off[i + 1])
            buf = os.pread(self.fd, b - a, a)
            out.append(decode_jpeg(torch.frombuffer(bytearray(buf), dtype=torch.uint8)).permute(1, 2, 0))
        return torch.stack(out).numpy()


def _columns(root, info, keys):
    """{key: (N, ...) array} of the dataset's parquet columns in frame order."""
    import numpy as np
    import pyarrow.parquet as pq

    files = sorted((Path(root) / "data").glob("chunk-*/episode_*.parquet"))
    tabs = [pq.read_table(f, columns=list(keys)) for f in files]
    out = {}
    for k in keys:
        cols = [tb[k].combine_chunks() for tb in tabs]
        if hasattr(cols[0], "flatten") and str(cols[0].type).startswith("fixed_size_list"):
            n = cols[0].type.list_size
            out[k] = np.concatenate([c.flatten().to_numpy().reshape(-1, n) for c in cols]).astype(np.float32)
        else:
            out[k] = np.concatenate([c.to_numpy() for c in cols])
    return out


def command_tcp(eps):
    """The commanded hand pose of every frame, in frame order, in observation.tcp_base's layout: apps/command_tcp.py's
    forward kinematics of each frame's joint command, beside each source episode (sessions resolve against the
    repository)."""
    import numpy as np

    repo, out = Path(__file__).resolve().parents[3], []
    for e in sorted(eps, key=lambda e: e["episode_index"]):
        src = e["source"]
        f = Path(src["session"]) if Path(src["session"]).is_absolute() else repo / src["session"]
        f = f / "episodes" / src["episode"] / "raw" / "command_tcp.npz"
        if not f.exists():
            raise FileNotFoundError(f"{f}: run apps/command_tcp.py on the session")
        a = np.load(f)["tcp_base"]
        if len(a) != e["length"]:
            raise ValueError(f"{f}: {len(a)} frames, the dataset's episode {e['episode_index']} has {e['length']}")
        out.append(a)
    return np.concatenate(out).astype(np.float32)


def goal_sectors(eps):
    """{episode_index: the goal mat's direction from the cup in the robot's base frame, one of 8 sectors (0 front,
    2 left, 4 back: toward the robot, 6 right)} from each source episode's result.json (None without a goal / base)."""
    import json as json_

    import numpy as np

    repo, out = Path(__file__).resolve().parents[3], {}
    for e in eps:
        src = e["source"]
        f = Path(src["session"]) if Path(src["session"]).is_absolute() else repo / src["session"]
        r = json_.loads((f / "episodes" / src["episode"] / "result.json").read_text())
        if not r.get("goal") or not r.get("base"):
            out[e["episode_index"]] = None
            continue
        yaw = np.radians(float(r["base"][2]))
        d = np.asarray(r["goal"]["position"][:2], float) - np.asarray(r["object"][:2], float)
        a = np.degrees(np.arctan2(-np.sin(yaw) * d[0] + np.cos(yaw) * d[1], np.cos(yaw) * d[0] + np.sin(yaw) * d[1]))
        out[e["episode_index"]] = int(((a + 22.5) % 360) // 45)
    return out


class _FrameFile:
    """A frame cache's frames read one pread each, the kernel's read-ahead off: a memory-mapped cache far larger than
    the memory a run may use read at random faults in read-ahead pages that are evicted before they are used, and
    the workers stall on the disk."""

    def __init__(self, path):
        import os

        import numpy as np

        mm = np.load(path, mmap_mode="r")  # the header: shape, dtype, where the data starts
        assert mm.flags.c_contiguous
        self.shape, self.dtype, self.offset = mm.shape, mm.dtype, mm.offset
        del mm
        self.frame = int(np.prod(self.shape[1:])) * self.dtype.itemsize
        self.fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_RANDOM)

    def read(self, idx):
        import os

        import numpy as np

        out = np.empty((len(idx), *self.shape[1:]), self.dtype)
        buf = out.reshape(len(idx), -1).view(np.uint8)
        for j, i in enumerate(idx):
            n = os.preadv(self.fd, [buf[j]], self.offset + int(i) * self.frame)
            assert n == self.frame, (n, self.frame)
        return out


class FrameDataset:
    """Training samples as LeRobotDataset would give them (observation windows, action chunks, `<key>_is_pad`, edges
    clamped to the episode) from the parquet columns and the frame cache: no video decoding per sample."""

    def __init__(self, root, info, frames, obs_keys, act_keys, cfg, cams, transform=None, uint8=False, columns=None,
                 hist=False):
        import numpy as np

        self.root, self.frames, self.transform, self.uint8 = Path(root), list(frames), transform, uint8
        self.hist = hist  # the action keys at the observation steps too (<key>.hist): the action history
        self.obs_keys, self.act_keys, self.cams = list(obs_keys), list(act_keys), list(cams)
        given = dict(columns or {})  # {key: (N, ...)}: a column in place of the dataset's, or one it does not have
        self.cols = _columns(root, info, [k for k in [*self.obs_keys, *self.act_keys] if k not in given]
                             + ["episode_index"])
        self.cols.update(given)
        ep = self.cols.pop("episode_index").astype(np.int64)
        lo = np.r_[0, np.flatnonzero(np.diff(ep)) + 1]
        hi = np.r_[lo[1:], len(ep)]
        self.lo, self.hi = np.repeat(lo, hi - lo), np.repeat(hi, hi - lo)
        self.obs_off, self.act_off = (np.asarray(o) for o in frame_offsets(cfg))
        self.hist_off = np.asarray(history_offsets(cfg))
        q = cfg.get("frame_cache_quality")  # JPEG frames (frame_cache_jpeg) in place of the raw cache
        self.paths = {c: frame_cache_jpeg(root, info, c, cfg.get("image_size"), int(q)) if q else
                      frame_cache(root, info, c, cfg.get("image_size")) for c in self.cams}
        self.mm = None

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, k):
        import numpy as np
        import torch

        if self.mm is None:  # per worker
            self.mm = {c: _JpegFrames(p) if p.suffix == ".jpg" else _FrameFile(p) for c, p in self.paths.items()}
        i = self.frames[k]
        lo, hi = self.lo[i], self.hi[i]
        oi, ai = i + self.obs_off, i + self.act_off
        out = {}
        for key in self.obs_keys:
            out[key] = torch.from_numpy(np.asarray(self.cols[key][np.clip(oi, lo, hi - 1)]))
        apad = (ai < lo) | (ai >= hi)
        for key in self.act_keys:
            out[key] = torch.from_numpy(np.asarray(self.cols[key][np.clip(ai, lo, hi - 1)]))
            out[f"{key}_is_pad"] = torch.from_numpy(apad)
            if self.hist:  # a frame's action is the command that brought it about: known when it is observed
                hi_ = np.clip(i + self.hist_off, lo, hi - 1)
                out[f"{key}.hist"] = torch.from_numpy(np.asarray(self.cols[key][hi_]))
        for c in self.cams:
            im = torch.from_numpy(self.mm[c].read(np.clip(oi, lo, hi - 1))).permute(0, 3, 1, 2)
            if self.uint8:  # (n, C, H, W) uint8: to float (and augmented) on the GPU
                out[f"observation.images.{c}"] = im
                continue
            im = im.float() / 255.0
            out[f"observation.images.{c}"] = self.transform(im) if self.transform else im
        return out


# --- model -------------------------------------------------------------------------------------------------------
def history_modules(policy, cfg, stats=None):
    """The history options on a built DiffusionPolicy (applied again on load): visual_history_grad false (the
    backbone's gradient from the current frame only), action_history (encoded into the conditioning by a temporal
    convolution, history_inject conv: an encoder `diffusion.hist_encoder`, the UNet rebuilt for the wider
    conditioning)."""
    import types

    import torch
    from torch import nn

    m = policy.diffusion
    n_obs = policy.config.n_obs_steps
    if cfg.get("visual_history_grad", True) is False and n_obs > 1:
        enc, n_cam = m.rgb_encoder, len(policy.config.image_features)
        base = enc.forward

        def forward(self, x):  # x ((b s n), C, H, W): the last n rows of each sample are the current frame
            if not (torch.is_grad_enabled() and self.training):
                return base(x)
            y = x.unflatten(0, (-1, n_obs * n_cam))
            cur = base(y[:, -n_cam:].flatten(0, 1)).unflatten(0, (-1, n_cam))
            with torch.no_grad():
                past = base(y[:, :-n_cam].flatten(0, 1)).unflatten(0, (-1, (n_obs - 1) * n_cam))
            return torch.cat([past, cur], 1).flatten(0, 1)

        enc.forward = types.MethodType(forward, enc)
    if not cfg.get("action_history"):
        return policy
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionConditionalUnet1d

    k, a = len(history_offsets(cfg)), policy.config.action_feature.shape[0]

    class HistEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            lo, hi = torch.zeros(a), torch.zeros(a)
            if stats is not None:
                lo = torch.as_tensor(stats["action"]["min"], dtype=torch.float32).flatten()
                hi = torch.as_tensor(stats["action"]["max"], dtype=torch.float32).flatten()
            self.register_buffer("lo", lo)
            self.register_buffer("hi", hi)
            # over time, the action's dimensions as channels
            self.conv = nn.Sequential(nn.Conv1d(a, 64, 3, padding=1), nn.Mish(), nn.Conv1d(64, 64, 3, padding=1),
                                      nn.Mish())
            self.net = nn.Sequential(nn.Flatten(), nn.Linear(64 * k, 128))
            self.out_dim = 128

        def forward(self, h):  # (B, k, a) raw actions -> (B, out_dim)
            x = 2 * (h - self.lo) / (self.hi - self.lo).clamp_min(1e-6) - 1
            return self.net(self.conv(x.transpose(1, 2)))

    m.hist_encoder = HistEncoder()
    pc = policy.config  # the conditioning as DiffusionModel sizes it, then the history's embedding
    cond = pc.robot_state_feature.shape[0] + m.rgb_encoder.feature_dim * len(pc.image_features) \
        + (pc.env_state_feature.shape[0] if pc.env_state_feature else 0)
    m.unet = DiffusionConditionalUnet1d(pc, global_cond_dim=cond * n_obs + m.hist_encoder.out_dim)
    base_cond = m._prepare_global_conditioning

    def prepare(self, batch):
        return torch.cat([base_cond(batch), self.hist_encoder(batch[AHIST_KEY])], -1)

    m._prepare_global_conditioning = types.MethodType(prepare, m)
    return policy


def build_policy(cfg, inp, outp, stats=None, weights=True):
    """The DiffusionPolicy of a config: features `inp` / `outp` ({key: {type, shape}}), `stats` its normalisation;
    `weights` False: the architecture only (no pretrained backbone; the checkpoint's weights follow)."""
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

    def feat(d):
        return {k: PolicyFeature(type=FeatureType[v["type"]], shape=tuple(v["shape"])) for k, v in d.items()}

    check_options(cfg)
    pc = DiffusionConfig(
        input_features=feat(inp), output_features=feat(outp), n_obs_steps=cfg["n_obs_steps"], horizon=cfg["horizon"],
        n_action_steps=cfg["n_action_steps"], crop_shape=tuple(cfg["crop_shape"]) if cfg.get("crop_shape") else None,
        vision_backbone=cfg.get("vision_backbone", "resnet18"),
        # an ImageNet-pretrained backbone ("ResNet18_Weights.IMAGENET1K_V1") keeps its BatchNorm: GroupNorm in its place
        # would void the weights
        pretrained_backbone_weights=cfg.get("pretrained_backbone_weights") if weights else None,
        use_group_norm=not cfg.get("pretrained_backbone_weights"),
        noise_scheduler_type=cfg.get("noise_scheduler", "DDIM"),
        num_inference_steps=cfg.get("num_inference_steps", 10),
        normalization_mapping={"VISUAL": NormalizationMode.MEAN_STD, "STATE": NormalizationMode.MIN_MAX,
                               "ENV": NormalizationMode.MIN_MAX, "ACTION": NormalizationMode.MIN_MAX})
    return history_modules(DiffusionPolicy(pc, dataset_stats=stats), cfg, stats)


def checkpoint_config(ckpt):
    """The train_config.json of a checkpoint (beside it, or in it)."""
    ck = Path(ckpt)
    for f in (ck / "train_config.json", ck.parent / "train_config.json"):
        if f.exists():
            return json.loads(f.read_text())
    return {}


def load_policy(ckpt, cfg=None, device="cuda"):
    """A checkpoint's DiffusionPolicy (eval mode)."""
    import torch
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from safetensors.torch import load_file

    ck = Path(ckpt)
    cfg = cfg if cfg is not None else checkpoint_config(ck)
    check_options(cfg)
    if cfg.get("action_history"):  # the history encoder: built as trained, then the weights
        pc = json.loads((ck / "config.json").read_text())
        inp = {k: {"type": v["type"], "shape": v["shape"]} for k, v in pc["input_features"].items()}
        outp = {k: {"type": v["type"], "shape": v["shape"]} for k, v in pc["output_features"].items()}
        policy = build_policy(cfg, inp, outp, None, weights=False)
        policy.load_state_dict(load_file(ck / "model.safetensors"))
    else:
        policy = DiffusionPolicy.from_pretrained(ck)
    return policy.to(torch.device(device)).eval()


class Predictor:
    """A checkpoint as raw batch -> action chunk: the dataset's keys in (full joint state; images resized to
    image_size, (B, n_obs, C, H, W) in [0, 1]), the variant's state transform (layout from train_config.json), DP's
    normalisation, sampling, unnormalisation. `decode` turns one action into joint targets or a hand pose."""

    def __init__(self, ckpt, device="cuda"):
        import torch

        self.torch, self.dev = torch, torch.device(device)
        self.cfg = checkpoint_config(ckpt)
        self.policy = load_policy(ckpt, self.cfg, device)
        lay = self.cfg.get("layout")
        if lay is None:  # a checkpoint trained before layouts: every joint, joint actions
            info, _ = load_meta(self.cfg["dataset"])
            lay = layout(info, {**self.cfg, "state_mode": "joints", "action_space": "joints"})
        self.lay = lay
        self.cams = [k[len("observation.images."):] for k in self.policy.config.image_features]
        sm = lay.get("state_mean")
        self.s_mean = torch.as_tensor(sm, dtype=torch.float32, device=self.dev) if sm is not None else None

    def policy_batch(self, batch):
        return to_policy({k: v.to(self.dev) if hasattr(v, "to") else v for k, v in batch.items()}, self.lay, self.s_mean)

    def predict(self, batch, seed=None):
        """(normalised, unnormalised) actions (B, n_action_steps, A) from the current observation on; `seed` fixes
        the sampling noise (torch.manual_seed before sampling)."""
        t = self.torch
        with t.inference_mode():
            b = self.policy.normalize_inputs(self.policy_batch(batch))
            b["observation.images"] = t.stack([b[k] for k in self.policy.config.image_features], dim=-4)
            if seed is not None:
                t.manual_seed(seed)
            a = self.policy.diffusion.generate_actions(b)
            return a, self.policy.unnormalize_outputs({"action": a})["action"]

    def targets(self, batch):
        """The batch's normalised policy actions (B, horizon, A)."""
        with self.torch.inference_mode():
            return self.policy.normalize_targets(self.policy_batch(batch))["action"]

    def decode(self, a):
        """One unnormalised action (A,): joint targets (D,) or {"p", "R": the hand pose in the base frame, "hand":
        {joint index: target}}."""
        import numpy as np

        a = np.asarray(a, float)
        if self.lay["action_space"] != "ee":
            return a
        from calibrated_scene_reconstruction.core.geometry.rotation6d import rot_from6d

        return {"p": a[:3], "R": rot_from6d(a[3:9]), "hand": dict(zip(self.lay["hand_index"], a[9:].tolist()))}


def train(cfg, dataset, out, resume=None):
    import numpy as np
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    check_options(cfg)
    info, eps = load_meta(dataset)
    inp, outp = dp_features(info, cfg.get("cameras"), cfg.get("image_size"))
    if not cfg.get("use_goal", True):  # vision only: the goal is the marker in the images
        inp.pop(ENV_KEY, None)
    lay = layout(info, cfg)
    inp["observation.state"]["shape"] = [len(lay["state_index"])]
    if lay["action_space"] == "ee":
        outp["action"]["shape"] = [9 + len(lay["hand_index"])]
    tr, held = split(eps, cfg.get("held_out_tags", ()), cfg.get("only_success", True))
    print(f"{dataset}: {len(tr)} train episodes, {len(held)} held out, inputs {list(inp)}, state "
          f"{len(lay['state_index'])} ({lay['state_mode']}), action {outp['action']['shape'][0]} ({lay['action_space']})",
          flush=True)
    meta = LeRobotDatasetMetadata(Path(dataset).name, root=dataset)
    raw_stats = dict(meta.stats)
    cmd = None
    if cfg.get("ee_label", "reached") == "command":  # the hand pose commanded, not the one reached (apps/command_tcp.py)
        if lay["action_space"] != "ee" or not cfg.get("frame_cache"):
            raise ValueError("ee_label command: end-effector actions read through the frame cache")
        cmd = command_tcp(eps)
        raw_stats[TCP_KEY] = {"min": cmd.min(0), "max": cmd.max(0), "mean": cmd.mean(0), "std": cmd.std(0)}
    lay["ee_label"] = cfg.get("ee_label", "reached")
    stats = policy_stats(dict(raw_stats), lay)
    if GOAL_KEY in stats:
        stats[ENV_KEY] = stats.pop(GOAL_KEY)
    stats, moving = widen_stats(stats, min_range=float(cfg.get("min_norm_range", 0.0))) \
        if cfg.get("min_norm_range") else (stats, {})
    lay["state_mean"] = np.asarray(stats["observation.state"]["mean"], np.float32).flatten().tolist()
    lay["action_mean"] = np.asarray(stats["action"]["mean"], np.float32).flatten().tolist()
    dev = torch.device(cfg.get("device", "cuda"))
    policy = build_policy(cfg, inp, outp, stats).to(dev).train()
    cams = [k[len("observation.images."):] for k in inp if k.startswith("observation.images.")]
    obs_keys = ["observation.state"] + ([GOAL_KEY] if ENV_KEY in inp else [])
    act_keys = ["action"] + ([TCP_KEY] if lay["action_space"] == "ee" else [])
    # every episode, then the training frames as a Subset: lerobot 0.3's episode_data_index is positional, so an
    # `episodes=` subset other than 0..k-1 looks up the wrong episode bounds
    lengths = {e["episode_index"]: e["length"] for e in eps}
    starts = np.cumsum([0] + [lengths[i] for i in sorted(lengths)])
    frames = [i for e in tr for i in range(int(starts[e]), int(starts[e] + lengths[e]))]
    if cfg.get("frame_cache"):  # uint8 frames to the GPU, augmented there (gpu_augment)
        data = FrameDataset(dataset, info, frames, obs_keys, act_keys, cfg, cams, uint8=True,
                            columns={TCP_KEY: cmd} if cmd is not None else None, hist=lay["action_history"])
    else:
        tfm = image_transforms(cfg.get("image_size"), cfg.get("augment"))  # every frame, before the policy's crop
        ds = LeRobotDataset(Path(dataset).name, root=dataset, image_transforms=tfm,
                            delta_timestamps=delta_timestamps(cfg, info["fps"],
                                                              [f"observation.images.{c}" for c in cams] + obs_keys,
                                                              act_keys),
                            video_backend=cfg.get("video_backend"))  # None: torchcodec (needs FFmpeg's shared libs)
        data = torch.utils.data.Subset(ds, frames)
    bb = [p_ for n, p_ in policy.named_parameters() if n.startswith("diffusion.rgb_encoder.backbone.")]
    rest = [p_ for n, p_ in policy.named_parameters() if not n.startswith("diffusion.rgb_encoder.backbone.")]
    # two parameter groups, the backbone's second: the optimiser state (train_state.pt) of runs that scaled its rate
    opt = torch.optim.AdamW([{"params": rest}, {"params": bb}], lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-6))
    warm, total = int(cfg.get("lr_warmup", 0)), int(cfg["steps"])

    def lr_factor(k):
        if k < warm:
            return (k + 1) / warm
        if cfg.get("lr_schedule") == "cosine":
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, (k - warm) / max(1, total - warm))))
        return 1.0

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    ema = copy.deepcopy(policy).eval() if cfg.get("ema") else None
    if ema is not None:
        for p_ in ema.parameters():
            p_.requires_grad_(False)
    step = 0
    if cfg.get("init_from") and not resume:  # fine-tuning: an earlier checkpoint's trained and EMA weights, step 0
        from safetensors.torch import load_file

        rd = Path(cfg["init_from"])
        raw = rd.with_name(rd.name + "_raw") if rd.with_name(rd.name + "_raw").exists() else rd

        def init(m, f):  # what another image size changes (the spatial softmax's coordinate grid) and the
            # normalisation (this run's dataset stats and labels) stay as built
            own = m.state_dict()
            sd = {k: v for k, v in load_file(f).items() if k in own and own[k].shape == v.shape
                  and not k.startswith(("normalize_", "unnormalize_"))
                  and not k.endswith(("hist_encoder.lo", "hist_encoder.hi"))}
            m.load_state_dict(sd, strict=False)
            return sorted(set(own) - set(sd))

        kept = init(policy, raw / "model.safetensors")
        if ema is not None:
            init(ema, rd / "model.safetensors")
        print(f"initialised from {rd} (as built: {kept})", flush=True)
    if resume:  # a checkpoint step_N: its trained (step_N_raw) and EMA (step_N) weights, then the optimiser and LR
        # schedule from train_state.pt -- or, for a checkpoint saved before that file existed, the schedule moved
        # to N and the optimiser's moments rebuilt from zero
        from safetensors.torch import load_file

        rd = Path(resume)
        raw = rd.with_name(rd.name + "_raw") if (rd.with_name(rd.name + "_raw")).exists() else rd
        policy.load_state_dict(load_file(raw / "model.safetensors"))
        if ema is not None:
            ema.load_state_dict(load_file(rd / "model.safetensors"))
        step = int(rd.name.split("_")[1])
        st = rd / "train_state.pt"
        if st.exists():
            state = torch.load(st, map_location=dev)
            opt.load_state_dict(state["optimizer"])
            sched.load_state_dict(state["scheduler"])
        else:
            for _ in range(step):
                sched.step()
        print(f"resumed at step {step} from {rd}" + ("" if st.exists() else " (optimizer moments reset)"), flush=True)
    s_noise, s_drop = float(cfg.get("state_noise", 0.0)), float(cfg.get("state_dropout", 0.0))
    a_drop = float(cfg.get("action_history_dropout", s_drop))
    if lay["state_mode"] == "none":
        s_noise = s_drop = 0.0
    # the noise on the joints that move only (a fixed one's jitter is not a state to be robust to)
    s_mask = torch.as_tensor(moving["observation.state"], dtype=torch.float32, device=dev) \
        if "observation.state" in moving else torch.ones(1, device=dev)
    s_mean = torch.as_tensor(lay["state_mean"], dtype=torch.float32, device=dev)
    sampler, fw = None, None  # fw: the training frames' draw weights
    if cfg.get("balance_goal_dir"):  # frames drawn with weight 1 / (frames of their episode's goal sector) ** power
        sec = goal_sectors([e for e in eps if e["episode_index"] in set(tr)])
        per = {}
        for e in tr:
            if sec[e] is not None:
                per[sec[e]] = per.get(sec[e], 0) + lengths[e]
        pw = float(cfg["balance_goal_dir"])
        big = max(per.values())  # an episode without a goal direction: the commonest sector's weight
        fw = np.concatenate([np.full(lengths[e], per.get(sec[e], big) ** -pw) for e in tr])
        share = {k: round(float(per[k] ** (1 - pw)) / sum(v ** (1 - pw) for v in per.values()), 3) for k in sorted(per)}
        print(f"balance_goal_dir {pw}: sector share of the draws {share}", flush=True)
        lay["balance_goal_dir"] = {"power": pw, "share": share}
    if fw is not None:
        sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(fw, dtype=torch.double), len(fw),
                                                         replacement=True)
    dl = torch.utils.data.DataLoader(data, batch_size=cfg["batch_size"], shuffle=sampler is None, sampler=sampler,
                                     drop_last=True,
                                     num_workers=cfg.get("num_workers", 4), pin_memory=dev.type == "cuda",
                                     persistent_workers=cfg.get("num_workers", 4) > 0)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_config.json").write_text(json.dumps({**cfg, "dataset": str(dataset), "train_episodes": tr,
                                                        "held_out_episodes": held, "layout": lay}, indent=1))
    # `amp` "bf16": the forward pass under bfloat16 autocast (the weights, the optimiser and the EMA stay float32)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    amp = cfg.get("amp") == "bf16"
    if cfg.get("channels_last"):
        policy.to(memory_format=torch.channels_last)
    fwd = policy.forward
    if cfg.get("compile"):
        # the loss compiled, LeRobot's normalisation eager: torch.compile of the whole forward skips Normalize (its
        # @torch.no_grad forward), training on raw images and actions; the first batch's compiled loss (mean of 4
        # noise draws) is checked against eager's (the graph orders its draws otherwise: no seed makes them equal)
        loss_c = torch.compile(policy.diffusion.compute_loss)

        def fwd(b):  # (loss, None) as policy.forward
            b = dict(policy.normalize_inputs(b))
            b["observation.images"] = torch.stack([b[k] for k in policy.config.image_features], dim=-4)
            return loss_c(policy.normalize_targets(b)), None
    t_log = time.time()
    e_par, p_par = (list(ema.parameters()), list(policy.parameters())) if ema is not None else ([], [])
    while step < cfg["steps"]:
        for batch in dl:
            batch = {k: v.to(dev, non_blocking=True) if hasattr(v, "to") else v for k, v in batch.items()}
            if cfg.get("frame_cache"):
                for c in cams:
                    im = batch[f"observation.images.{c}"].float() / 255.0
                    im = gpu_augment(im, cfg["augment"]) if cfg.get("augment") else im
                    batch[f"observation.images.{c}"] = im
            batch = to_policy(batch, lay, s_mean, s_noise, s_drop, s_mask, a_drop)
            if cfg.get("compile") and fwd is not policy.forward and not getattr(fwd, "checked", False):
                with torch.no_grad(), torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                    l_c = sum(float(fwd(batch)[0]) for _ in range(4)) / 4
                    # the eager pass on a quarter of the batch: beside the compiled one's memory a whole batch
                    # ran out of GPU memory at sizes the training itself fits
                    part = {k_: v_[:max(1, len(v_) // 4)] if torch.is_tensor(v_) else v_ for k_, v_ in batch.items()}
                    torch.cuda.empty_cache()
                    l_e = sum(float(policy.forward(part)[0]) for _ in range(8)) / 8
                    del part
                    torch.cuda.empty_cache()
                if not 0.5 < l_c / max(l_e, 1e-12) < 2.0:
                    raise RuntimeError(f"the compiled loss {l_c:.6f} is not the eager one {l_e:.6f}")
                print(f"compiled loss checked: {l_c:.6f} (eager {l_e:.6f})", flush=True)
                fwd.checked = True
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp):
                loss = fwd(batch)[0]
            loss = loss.float()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.get("grad_clip", 10.0))
            opt.step()
            sched.step()
            step += 1
            if ema is not None:
                decay = min(float(cfg.get("ema_max", 0.9999)), 1.0 - (1.0 + step) ** -0.75)
                with torch.no_grad():
                    torch._foreach_mul_(e_par, decay)
                    torch._foreach_add_(e_par, p_par, alpha=1.0 - decay)
                    for be, bp in zip(ema.buffers(), policy.buffers()):
                        be.copy_(bp)
            if step % cfg.get("log_every", 100) == 0:
                dt = (time.time() - t_log) / cfg.get("log_every", 100)
                t_log = time.time()
                print(f"step {step}: loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} {dt:.3f} s/step",
                      flush=True)
            if step % cfg.get("save_every", 10000) == 0 or step >= cfg["steps"]:
                if ema is not None:
                    ema.save_pretrained(out / f"step_{step:06d}")
                    policy.save_pretrained(out / f"step_{step:06d}_raw")
                else:
                    policy.save_pretrained(out / f"step_{step:06d}")
                torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(), "step": step},
                           out / f"step_{step:06d}" / "train_state.pt")  # what --resume continues from
            if step >= cfg["steps"]:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--dataset", help="overrides the config's dataset")
    ap.add_argument("--out", help="overrides the config's out")
    ap.add_argument("--resume", help="a checkpoint directory (<out>/step_N) to continue from at step N")
    ap.add_argument("--batch-size", type=int, help="overrides the config's batch_size (the learning rate scaled by the "
                                                   "square root of the ratio, the steps and the checkpoint interval by its "
                                                   "inverse: the same samples seen): a smaller batch where the larger does "
                                                   "not fit")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    if a.batch_size and a.batch_size != cfg["batch_size"]:
        r = a.batch_size / cfg["batch_size"]
        cfg["lr"] = cfg["lr"] * math.sqrt(r)
        for k in ("steps", "save_every", "lr_warmup"):
            if cfg.get(k):
                cfg[k] = round(cfg[k] / r)
        cfg["batch_size"] = a.batch_size
    train(cfg, a.dataset or cfg["dataset"], a.out or cfg["out"], a.resume)


if __name__ == "__main__":
    main()
