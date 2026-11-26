"""A joint-space official Diffusion Policy (configs/dp_official/lehome_garment_*.yaml, trained on
tools/export_dp_official_deformable.py's buffers) served by tools/dp_official_server.py, as a controller called at the
recording's frame rate: `act(images, state)` -> the joint targets for this frame. Stdlib and numpy only (it runs in
the Isaac image's python; OpenCV, when there, resizes frames that are not the export's size).

At the export's rate (every `every`-th frame: 10 Hz of 30) it keeps an observation (the cameras, the joints as
robot_qpos; the server drops what the policy does not read); when its plan is empty it sends the last n_obs_steps of
them (the first one repeated to fill the window, as the training's padding) and plans the first n_action_steps
predicted targets, each reached over `every` frames, moving linearly from the previous target (the joints at the
start).
"""
from __future__ import annotations

import collections
from multiprocessing.connection import Client

import numpy as np

AUTH = b"calibrated_scene_reconstruction-dp"


class DPOfficialClient:
    def __init__(self, port, host="127.0.0.1"):
        self.conn = Client((host, int(port)), authkey=AUTH)
        lay = self.ask({"layout": True})["layout"]
        self.layout = lay
        self.every, self.n_obs, self.n_act = int(lay["every"]), int(lay["n_obs_steps"]), int(lay["n_action_steps"])
        self.cams, self.hw = list(lay["cameras"]), tuple(lay["image_hw"])
        self.reset()

    def ask(self, req):
        self.conn.send(req)
        out = self.conn.recv()
        if "error" in out:
            raise RuntimeError(f"Diffusion Policy server: {out['error']}")
        return out

    def reset(self):
        self.hist = collections.deque(maxlen=self.n_obs)
        self.plan = collections.deque()
        self.prev, self.frame = None, 0

    def image(self, im):
        im = np.ascontiguousarray(np.asarray(im)[..., :3], dtype=np.uint8)
        if im.shape[:2] != self.hw:
            import cv2

            im = cv2.resize(im, (self.hw[1], self.hw[0]), interpolation=cv2.INTER_AREA)
        return im

    def act(self, images, state):
        state = np.asarray(state, np.float32)
        if self.frame % self.every == 0:  # an observation at the policy's rate (a plan ends on one)
            x = {c: self.image(images[c]) for c in self.cams}
            x["robot_qpos"] = state
            while len(self.hist) < self.n_obs - 1:
                self.hist.append(x)
            self.hist.append(x)
        self.frame += 1
        if not self.plan:
            obs = {k: np.stack([h[k] for h in self.hist]) for k in self.hist[0]}
            targets = np.asarray(self.ask({"obs": obs})["action"], np.float32)[: self.n_act]
            start = self.prev if self.prev is not None else state
            for t in targets:
                for k in range(1, self.every + 1):
                    self.plan.append(start + (t - start) * k / self.every)
                start = t
        self.prev = self.plan.popleft()
        return self.prev
