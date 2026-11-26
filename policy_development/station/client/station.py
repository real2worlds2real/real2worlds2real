"""Client of the task station (numpy only).  Connects to the socket the station tool exposes in this workspace.

    from station import Station
    st = Station()                      # $STATION_SOCKET or ./.station.sock
    info = st.info()                    # task text, num_envs, action_dim, limits, label names, partition sizes, your usage and caps
    obs = st.reset_train()              # every env starts a fresh random training episode (counts against your training budget)
    obs, ended, all_done = st.step(actions)   # actions: (num_envs, info["action_dim"]); cloud stations: 29 absolute joint targets (rad)
    obs = st.reset_partition("dev", [51, 52, ...])   # sealed starts; one episode per listed id, envs beyond the list stay idle

obs is a dict of arrays with a leading env axis.  Cloud stations (interface cloud_q29):
    cloud (N, P, 3) float16 metres, labels (N, P) uint8, q29 (N, 29), palm_pos (N, 3), palm_quat (N, 4) wxyz,
    tactile_n (N, k) pad force magnitudes, step (N,) steps since the episode start, active (N,) bool
State-vector stations (e.g. nut): obs (N, D) plus q29, palm_pos, palm_quat, step, active; info["observation_layout"] names D.
ended is a list of {"env", "id", "success", "term", "steps"} for episodes that finished on this step.  In training mode an env
that finished starts its next episode immediately (its observation in this reply already belongs to the new episode).
"""
import json
import os
import socket
import struct
import threading

import numpy as np


class StationError(RuntimeError):
    pass


class Station:
    def __init__(self, path=None, timeout=600.0):
        self.path = path or os.environ.get("STATION_SOCKET", ".station.sock")
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.sock.connect(self.path)
        self._lock = threading.Lock()     # 11-11: program threads call fk/ik concurrently; one request/reply at a time on the socket

    # ------------------------------------------------------------ wire
    def _send(self, header, arrays=None):
        arrays = arrays or {}
        meta, blobs = [], []
        for k, v in arrays.items():
            v = np.ascontiguousarray(v)
            meta.append({"name": k, "dtype": str(v.dtype), "shape": list(v.shape)})
            blobs.append(v.tobytes())
        hb = json.dumps({**header, "arrays": meta}).encode()
        self.sock.sendall(struct.pack("<II", len(hb), sum(len(b) for b in blobs)) + hb + b"".join(blobs))

    def _recvn(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.sock.recv(min(1 << 20, n - len(buf)))
            if not chunk:
                raise StationError("station closed the connection")
            buf += chunk
        return bytes(buf)

    def _call(self, header, arrays=None):
        with self._lock:
            self._send(header, arrays)
            hl, bl = struct.unpack("<II", self._recvn(8))
            h = json.loads(self._recvn(hl)); body = self._recvn(bl)
        out = {}; off = 0
        for m in h.pop("arrays", []):
            dt = np.dtype(m["dtype"]); n = int(np.prod(m["shape"])) * dt.itemsize
            out[m["name"]] = np.frombuffer(body[off:off + n], dtype=dt).reshape(m["shape"]).copy(); off += n
        if "error" in h:
            raise StationError(h["error"])
        return h, out

    @staticmethod
    def _obs(arrays):
        return {k[4:]: v for k, v in arrays.items() if k.startswith("obs.")}

    # ------------------------------------------------------------ commands
    def info(self):
        return self._call({"cmd": "info"})[0]

    def reset_train(self):
        return self._obs(self._call({"cmd": "reset_train"})[1])

    def reset_partition(self, partition, ids):
        return self._obs(self._call({"cmd": "reset_partition", "partition": partition, "ids": [int(i) for i in ids]})[1])

    def step(self, actions):
        h, a = self._call({"cmd": "step"}, {"actions": np.asarray(actions, np.float32)})
        return self._obs(a), h["ended"], h["all_done"]

    def fk(self, q7):
        """palm pose of arm joint vectors (M, 7) -> pos (M, 3), quat (M, 4) wxyz, in the task frame of the observations."""
        a = self._call({"cmd": "fk"}, {"q7": np.asarray(q7, np.float32).reshape(-1, 7)})[1]
        return a["pos"], a["quat"]

    def ik(self, q7_seed, pos, quat, iters=10):
        """arm joints (M, 7) reaching palm poses pos (M, 3) / quat (M, 4) wxyz from seeds (M, 7); also returns the position error (m)."""
        a = self._call({"cmd": "ik", "iters": int(iters)}, {"q7_seed": np.asarray(q7_seed, np.float32).reshape(-1, 7),
                                                            "pos": np.asarray(pos, np.float32).reshape(-1, 3),
                                                            "quat": np.asarray(quat, np.float32).reshape(-1, 4)})[1]
        return a["q7"], a["pos_error_m"]

    def close(self):
        try:
            self._call({"cmd": "close"})
        finally:
            self.sock.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
