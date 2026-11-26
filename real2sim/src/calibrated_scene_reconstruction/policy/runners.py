"""Policies as apps/episode.py --policy drives them: obs -> command, one call per simulation frame.

* PolicyRunner: a LeRobot Diffusion Policy checkpoint (calibrated_scene_reconstruction.policy.diffusion, envs/policy) loaded in the Isaac
  python itself;
* ServedDPRunner: an official Diffusion Policy checkpoint (configs/dp_official, envs/dp_official) served by
  tools/dp_official_server.py (apps/episode.py --dp-server);
* ServedOpenpiRunner: a fine-tuned pi0.5 checkpoint (calibrated_scene_reconstruction.policy.openpi, envs/openpi) served by
  tools/openpi_serve.py (apps/episode.py --openpi-server).

Both return joint targets (D,) or, for end-effector actions, {"p", "R": the hand pose in the robot's base frame,
"hand": {joint index: target}} for the caller's IK; reset() per rollout.

lerobot is not installed in the Isaac python: `lerobot_path()` appends the site-packages of envs/policy
(setup/setup_policy_env.sh, the same Python minor version) to the END of sys.path, so torch / torchvision / numpy stay
Isaac's (torch 2.7.0+cu128 in both) and only what Isaac lacks (lerobot, diffusers, draccus, huggingface_hub,
safetensors, ...) comes from envs/policy. `openpi_client_path()` does the same for openpi's client (its source in
third_party/openpi, its websockets / msgpack from envs/openpi).
"""
import collections
import os
import sys
from pathlib import Path

import numpy as np

GOAL_KEY = "observation.goal"
AHIST_KEY = "observation.action_history"  # calibrated_scene_reconstruction.policy.diffusion.AHIST_KEY
ROOT = Path(__file__).resolve().parents[3]


def lerobot_path(env=None):
    """Make `lerobot` importable: as is when installed, else from the policy env's site-packages (appended last;
    its .pth files are not processed). `env`: default $POLICY_ENV or envs/policy."""
    try:
        import lerobot  # noqa: F401
        return
    except ImportError:
        pass
    env = Path(env or os.environ.get("POLICY_ENV") or ROOT / "envs" / "policy")
    site = env / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    if not (site / "lerobot").is_dir():
        raise ImportError(f"lerobot not found in {site} (setup/setup_policy_env.sh; the env's Python must be "
                          f"{sys.version_info.major}.{sys.version_info.minor}, as this interpreter)")
    sys.path.append(str(site))


def openpi_client_path(env=None):
    """Make `openpi_client` importable: as is when installed, else from third_party/openpi's source and the openpi
    env's site-packages (appended last, for what this interpreter lacks). `env`: default $OPENPI_ENV or envs/openpi."""
    try:
        import openpi_client  # noqa: F401
        return
    except ImportError:
        pass
    env = Path(env or os.environ.get("OPENPI_ENV") or ROOT / "envs" / "openpi")
    sys.path.append(str(ROOT / "third_party" / "openpi" / "packages" / "openpi-client" / "src"))
    sys.path.append(str(env / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"))


class PolicyRunner:
    """A LeRobot Diffusion Policy checkpoint: images (H, W, 3) uint8 per camera, the full joint state (D,), goal (7,)
    or None in. The variant (calibrated_scene_reconstruction.policy.diffusion.Predictor, from the checkpoint's train_config.json) decides what
    the policy sees (the state cut, its current step only, or none) and what comes out. Keeps the observations of the
    last (n_obs_steps - 1) x obs_stride + 1 calls (the policy sees every obs_stride-th), the actions it executed over
    the action history's span (its action_history_steps, every action_history_stride-th, when the variant has one)
    and the queued actions between calls."""

    def __init__(self, checkpoint, device="cuda", image_size=None):
        import torch

        lerobot_path()
        from calibrated_scene_reconstruction.policy.diffusion import Predictor

        self.torch = torch
        self.pred = Predictor(checkpoint, device)
        self.policy = self.pred.policy  # n_action_steps / num_inference_steps overrides act on it
        self.cams = self.pred.cams
        self.image_size = image_size or self.pred.cfg.get("image_size")
        self.reset()

    def reset(self):
        lay = self.pred.lay
        self.stride = int(lay.get("obs_stride", 1))
        self.a_stride = int(lay.get("action_history_stride", self.stride))
        k = int(lay.get("action_history_steps", self.policy.config.n_obs_steps))
        self.obs = collections.deque(maxlen=(self.policy.config.n_obs_steps - 1) * self.stride + 1)
        self.executed = collections.deque(maxlen=(k - 1) * self.a_stride + 1)  # the actions returned, oldest first
        self.actions = collections.deque()

    def __call__(self, images, state, goal=None):
        t = self.torch
        o = {"observation.state": t.as_tensor(np.asarray(state), dtype=t.float32)}
        for c in self.cams:
            im = t.as_tensor(images[c]).permute(2, 0, 1).float()[None] / 255.0
            if self.image_size:
                im = t.nn.functional.interpolate(im, size=tuple(self.image_size), mode="bilinear", antialias=True)
            o[f"observation.images.{c}"] = im[0]
        if goal is not None:
            o[GOAL_KEY] = t.as_tensor(goal, dtype=t.float32)
        while len(self.obs) < self.obs.maxlen:  # the first observation fills the window (as LeRobot's queues)
            self.obs.append(o)
        self.obs.append(o)
        if not self.actions:
            seen = list(self.obs)[::self.stride]
            batch = {k: t.stack([x[k] for x in seen])[None] for k in o}
            if self.pred.lay.get("action_history"):
                # the action that brought each observed state about: the one returned the call before; before the
                # first, the mean action (what the training's dropout of the history shows)
                lay = self.pred.lay
                mean = np.asarray(lay["action_mean"] if "action_mean" in lay else lay["state_mean"][lay["n_state"]:],
                                  np.float32)
                n, ex = self.executed.maxlen, list(self.executed)
                ex = [ex[0]] * (n - len(ex)) + ex if ex else [mean] * n
                batch[AHIST_KEY] = t.as_tensor(np.stack(ex[::self.a_stride]), dtype=t.float32)[None]
            _, a = self.pred.predict(batch)
            self.actions.extend(a[0, :self.policy.config.n_action_steps].cpu().numpy())
        a = self.actions.popleft()
        self.executed.append(np.asarray(a, np.float32))
        return self.pred.decode(a)


class ServedDPRunner:
    """An official Diffusion Policy checkpoint served by tools/dp_official_server.py, at the export's rate
    (tools/export_dp_official.py: every `every`-th frame, 10 Hz): an observation (the camera resized to the export's
    frames, the hand's pose in the robot's base frame -- `tcp`: (p, R), as observation.tcp_base: xyz + the first two
    columns -- and the hand's joints) kept every `every` calls, the last n_obs_steps of them sent when the queue is
    empty (the server passes the policy the keys it reads); the first `n_action_steps` predicted actions (the command
    1 frame after each of those observations, as the labels) executed over `every` frames each, interpolated toward
    the next."""

    wants_tcp = True  # apps/episode.py passes the hand's pose

    def __init__(self, server="localhost:5581", n_action_steps=None):
        from multiprocessing.connection import Client

        import torch

        host, port = server.rsplit(":", 1)
        self.conn = Client((host, int(port)), authkey=b"calibrated_scene_reconstruction-dp")
        self.lay = self.ask({"layout": True})["layout"]
        self.torch = torch
        self.cams = [self.lay["camera"]]
        self.every, self.n_obs = int(self.lay["every"]), int(self.lay["n_obs_steps"])
        self.n_action_steps = int(n_action_steps or self.lay["n_action_steps"])
        self.hand_idx = list(self.lay["hand_index"])
        self.reset()

    def ask(self, req):
        self.conn.send(req)
        out = self.conn.recv()
        if "error" in out:
            raise RuntimeError(f"Diffusion Policy server: {out['error']}")
        return out

    def reset(self):
        self.obs = collections.deque(maxlen=(self.n_obs - 1) * self.every + 1)  # every call's; every `every`-th seen
        self.actions = collections.deque()

    def __call__(self, images, state, goal=None, tcp=None):
        from calibrated_scene_reconstruction.core.geometry.rotation6d import rot_from6d

        t = self.torch
        im = t.as_tensor(images[self.cams[0]]).permute(2, 0, 1).float()[None]
        h, w = self.lay["image_hw"]
        if tuple(im.shape[-2:]) != (h, w):  # as the frame cache (calibrated_scene_reconstruction.policy.diffusion.frame_cache)
            im = t.nn.functional.interpolate(im, size=(h, w), mode="bilinear", antialias=True)
        im = im[0].round().clamp(0, 255).to(t.uint8).permute(1, 2, 0).numpy()
        p, R = (np.asarray(v, np.float32) for v in tcp)
        o = {self.cams[0]: im, "robot_eef_pose": np.r_[p, R[:, 0], R[:, 1]].astype(np.float32),
             "hand_qpos": np.asarray(state, np.float32)[self.hand_idx]}
        while len(self.obs) < self.obs.maxlen:  # the first observation fills the window (as the padding in training)
            self.obs.append(o)
        self.obs.append(o)
        if not self.actions:
            seen = list(self.obs)[::self.every]
            A = np.asarray(self.ask({"obs": {k: np.stack([x[k] for x in seen]) for k in o}})["action"], float)
            for j in range(self.n_action_steps):
                b = A[min(j + 1, len(A) - 1)]
                for m in range(self.every):  # toward the next action, as the frames between two samples moved
                    x = A[j] + (b - A[j]) * (m / self.every)
                    self.actions.append({"p": x[:3], "R": rot_from6d(x[3:9]),
                                         "hand": {i: float(v) for i, v in zip(self.hand_idx, x[9:])}})
        return self.actions.popleft()


class ServedOpenpiRunner:
    """A fine-tuned pi0.5 checkpoint served by tools/openpi_serve.py (openpi's websocket server and client), at the
    dataset's rate (tools/export_openpi.py: every frame): the camera image as rendered (the model resizes it with
    padding, as in training), the state's joints (the server's `state_index` into the full state) and the dataset's
    task text sent when the queue is empty; the first `n_action_steps` of the returned chunk (absolute joint targets)
    executed one per frame, as openpi's examples execute an open-loop part of each chunk (DEFAULT_ACTION_STEPS: half a
    second at 30 Hz, as its real ALOHA's 25 of 50 at 50 Hz). The joints the policy does not drive stay where the
    rollout started."""

    DEFAULT_ACTION_STEPS = 15

    def __init__(self, server="localhost:8000", n_action_steps=None):
        openpi_client_path()
        from openpi_client import websocket_client_policy

        host, port = server.rsplit(":", 1)
        self.client = websocket_client_policy.WebsocketClientPolicy(host, int(port))
        self.lay = self.client.get_server_metadata()
        self.cams = [self.lay["camera"]]
        self.fps = int(self.lay["fps"])
        self.idx = np.asarray(self.lay["state_index"], int)
        self.n_action_steps = int(n_action_steps or self.DEFAULT_ACTION_STEPS)
        self.reset()

    def reset(self):
        self.actions = collections.deque()
        self.hold = None  # the full joint state at the rollout's first call

    def __call__(self, images, state, goal=None):
        state = np.asarray(state, np.float32)
        if self.hold is None:
            self.hold = state.copy()
        if not self.actions:
            out = self.client.infer({"observation/image": np.asarray(images[self.cams[0]], np.uint8),
                                     "observation/state": state[self.idx], "prompt": self.lay["prompt"]})
            self.actions.extend(np.asarray(out["actions"], np.float32)[:self.n_action_steps])
        q = self.hold.copy()
        q[self.idx] = self.actions.popleft()
        return q
