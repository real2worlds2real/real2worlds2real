"""Generic station server: one task backend behind the station protocol (numpy only; no simulator import here).

Interface given to a client (identical for every method, and identical to what the real robot provides): the backend's
observation and action contract, the station's safety limits, the outcome of an episode only at its end (success,
termination reason, steps; the success rule is not exposed).  Commands (length-prefixed JSON header + raw arrays):
info, reset_train, reset_partition, step, fk, ik, close.

Starts: train = fresh random starts of the task's distribution; partitions dev / unseen / heldout = sealed starts from the
station root's partition file (heldout only with the coordinator's token).  Every finished episode is appended to
<run>/ledger.jsonl; <run>/policy.json (written by the coordinator's tool) caps the client's usage.
"""
from __future__ import annotations

import faulthandler
import json
import os
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np

from server.backend import Refusal


# ---------------------------------------------------------------- wire format
def send(conn, header, arrays=None):
    arrays = arrays or {}
    meta, blobs = [], []
    for k, v in arrays.items():
        v = np.ascontiguousarray(v)
        meta.append({"name": k, "dtype": str(v.dtype), "shape": list(v.shape)})
        blobs.append(v.tobytes())
    hb = json.dumps({**header, "arrays": meta}).encode()
    conn.sendall(struct.pack("<II", len(hb), sum(len(b) for b in blobs)) + hb + b"".join(blobs))


def recvn(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(min(1 << 20, n - len(buf)))
        if not chunk:
            raise ConnectionError("client closed")
        buf += chunk
    return bytes(buf)


def recv(conn):
    hl, bl = struct.unpack("<II", recvn(conn, 8))
    h = json.loads(recvn(conn, hl))
    body = recvn(conn, bl)
    arrays, off = {}, 0
    for m in h.pop("arrays", []):
        dt = np.dtype(m["dtype"])
        n = int(np.prod(m["shape"])) * dt.itemsize
        arrays[m["name"]] = np.frombuffer(body[off:off + n], dtype=dt).reshape(m["shape"])
        off += n
    return h, arrays


def split(reply):
    arrays = {}
    if "obs" in reply:
        arrays.update({f"obs.{k}": v for k, v in reply.pop("obs").items()})
    for k in ("pos", "quat", "q7", "pos_error_m"):
        if k in reply:
            arrays[k] = reply.pop(k)
    return reply, arrays


class StationServer:
    def __init__(self, backend, run_dir, sealed_path, token_path):
        self.b = backend
        self.b.read_sealed = self.sealed
        self.N = backend.num_envs
        self.run = Path(run_dir)
        self.run.mkdir(parents=True, exist_ok=True)
        self.sealed_path = Path(sealed_path)
        self.token = Path(token_path).read_text().strip()
        self.active = np.zeros(self.N, dtype=bool)          # rows whose current episode counts (partition mode)
        self.mode = {"kind": None, "ids": [None] * self.N, "part": None}
        self.commands = {"info": self.cmd_info, "reset_train": self.cmd_reset_train, "reset_partition": self.cmd_reset_partition,
                         "step": self.cmd_step, "fk": self.cmd_fk, "ik": self.cmd_ik}

    # ------------------------------------------------------------------ sealed partitions, usage policy, ledger
    def sealed(self):
        return json.loads(self.sealed_path.read_text())

    def usage(self):
        f = self.run / "ledger.jsonl"
        rows = [json.loads(line) for line in f.read_text().splitlines()] if f.exists() else []
        return {"train_episodes": sum(r["mode"] == "train" for r in rows if "mode" in r),
                "partition_runs": [r for r in rows if r.get("kind") == "partition_run"]}

    def policy(self):
        f = self.run / "policy.json"
        return json.loads(f.read_text()) if f.exists() else {}

    def log(self, row):
        with open(self.run / "ledger.jsonl", "a") as fh:
            fh.write(json.dumps({"time": time.time(), **row}) + "\n")

    def _set_active(self):
        self.b.set_active(self.active.copy())

    # ------------------------------------------------------------------ commands
    def cmd_info(self, _h, _a):
        s = self.sealed()
        return {**self.b.info(live=self.mode["kind"] is not None), "num_envs": self.N,
                "partitions": {k: len(v) for k, v in s["ids"].items() if k != "heldout"},
                "usage": {"train_episodes": self.usage()["train_episodes"]},
                "policy": {k: v for k, v in self.policy().items() if k != "heldout_token"}}

    def cmd_reset_train(self, _h, _a):
        pol, used = self.policy(), self.usage()["train_episodes"]
        if pol.get("train_episode_budget") is not None and used >= pol["train_episode_budget"]:
            return {"error": f"training-episode budget exhausted ({used} of {pol['train_episode_budget']})"}
        self.active[:] = True
        self._set_active()
        self.mode.update(kind="train", ids=[None] * self.N, part=None)
        self.b.reset(list(range(self.N)), [None] * self.N)
        return {"obs": self.b.observe()}

    def cmd_reset_partition(self, h, _a):
        part, ids = h["partition"], [int(i) for i in h["ids"]]
        s = self.sealed()
        if part not in s["ids"] or not set(ids) <= set(s["ids"][part]) or len(ids) > self.N or len(set(ids)) != len(ids):
            return {"error": f"ids must be distinct members of partition {part}, at most {self.N} per batch"}
        if part == "heldout" and h.get("token") != self.token:
            return {"error": "the held-out partition is coordinator-only"}
        allowed = self.policy().get("partitions", ["dev"])
        if part not in allowed and part != "heldout":
            return {"error": f"partition {part} is not open to this client (open: {allowed})"}
        self.active[:] = False
        self.active[:len(ids)] = True
        self._set_active()
        self.mode.update(kind="partition", ids=ids + [None] * (self.N - len(ids)), part=part)
        self.b.reset(list(range(len(ids))), [s["starts"][part][str(i)] for i in ids])
        self.log({"kind": "partition_run", "mode": part, "ids": ids})
        return {"obs": self.b.observe()}

    def cmd_step(self, _h, a):
        acts = np.asarray(a["actions"], np.float32)
        err = self.b.validate_actions(acts)
        if err:
            return {"error": err}
        if self.mode["kind"] is None:
            return {"error": "reset first"}
        was = self.active.copy()
        done = np.asarray(self.b.step(acts), bool)
        oc = self.b.outcomes([int(r) for r in np.nonzero(done)[0]])
        ended = []
        for r, rec in oc.items():
            if not was[r]:
                continue
            rid = self.mode["ids"][r]
            ended.append({"env": r, "id": rid, **rec})
            self.log({"mode": self.mode["kind"] if self.mode["kind"] == "train" else self.mode["part"], "id": rid, **rec})
            if self.mode["kind"] == "partition":
                self.active[r] = False
        if self.mode["kind"] == "train":
            rows = [e["env"] for e in ended]
            used, budget = self.usage()["train_episodes"], self.policy().get("train_episode_budget")
            fresh = [r for r in rows if budget is None or used < budget]
            for r in rows:
                if r not in fresh:
                    self.active[r] = False
            if fresh:
                self.b.after_train_episodes(fresh)
        self._set_active()
        return {"obs": self.b.observe(), "ended": ended, "all_done": bool(not self.active.any())}

    def cmd_fk(self, _h, a):
        try:
            pos, quat = self.b.fk(np.asarray(a["q7"], np.float32))
        except Refusal as e:
            return {"error": str(e)}
        return {"pos": np.asarray(pos), "quat": np.asarray(quat)}

    def cmd_ik(self, h, a):
        try:
            q7, err = self.b.ik(np.asarray(a["q7_seed"], np.float32), np.asarray(a["pos"], np.float32),
                                np.asarray(a["quat"], np.float32), int(h.get("iters", 10)))
        except Refusal as e:
            return {"error": str(e)}
        return {"q7": np.asarray(q7), "pos_error_m": np.asarray(err)}

    # ------------------------------------------------------------------ validation path and serving
    def check(self, partition: str, ids: list) -> dict:
        """The backend's reference controller through the external control path (coordinator validation of a station)."""
        res = []
        self.b.first_reset()
        for b0 in range(0, len(ids), self.N):
            rep = self.cmd_reset_partition({"partition": partition, "ids": ids[b0:b0 + self.N], "token": self.token}, {})
            assert "error" not in rep, rep
            for _ in range(self.b.max_steps + 5):
                tg = self.b.validation_targets()
                if tg is None:
                    raise SystemExit("this backend has no reference controller")
                rep = self.cmd_step({}, {"actions": np.asarray(tg, np.float32)})
                res += rep["ended"]
                if rep["all_done"]:
                    break
        terms = {}
        for e in res:
            terms[e["term"]] = terms.get(e["term"], 0) + 1
        (self.run / "check_act.json").write_text(json.dumps(res, indent=1))
        return {"partition": partition, "n": len(res), "success": sum(e["success"] for e in res), "terms": terms}

    def handle(self, h, a):
        fn = self.commands.get(h.get("cmd"))
        # Dump thread stacks when a command stalls.
        faulthandler.dump_traceback_later(120, repeat=True, file=sys.stderr)
        t0 = time.time()
        try:
            reply = fn(h, a) if fn else {"error": f"unknown command {h.get('cmd')}"}
        except Exception as e:  # noqa: BLE001  (a bad request must not kill the station)
            reply = {"error": f"{type(e).__name__}: {e}"[:400]}
        finally:
            faulthandler.cancel_dump_traceback_later()
        if time.time() - t0 > 20:
            print(f"[STATION] slow {h.get('cmd')}: {time.time() - t0:.1f} s", flush=True)
        return reply

    def serve(self, sock_path, idle_exit: float, exit_fn=os._exit):
        self.b.first_reset()
        sock_path = Path(sock_path)
        if sock_path.exists():
            sock_path.unlink()
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(str(sock_path))
        srv.listen(1)
        srv.settimeout(idle_exit)
        (self.run / "READY").write_text(str(os.getpid()))
        print(f"[STATION] {getattr(self.b, 'task', '?')} ready on {sock_path} with {self.N} envs", flush=True)
        while True:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                print("[STATION] idle, exiting", flush=True)
                (self.run / "READY").unlink(missing_ok=True)
                sock_path.unlink(missing_ok=True)
                return exit_fn(0)
            conn.settimeout(None)
            try:
                while True:
                    h, a = recv(conn)
                    if h.get("cmd") == "close":
                        send(conn, {"ok": True})
                        break
                    hdr, arr = split(self.handle(h, a))
                    send(conn, hdr, arr)
            except ConnectionError:
                pass
            finally:
                conn.close()
                self.mode.update(kind=None)
                self.active[:] = False
                self._set_active()
