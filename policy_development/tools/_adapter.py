"""Shared adapter plumbing: argv contract (--binding --request --result), receipts, the station world and the sandbox.

Adapters run on the controller host.  Agent-authored or learned policy code never runs here: it runs in the station
root's bubblewrap sandbox (workspace writable, no network) through the same runner the baselines use.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

POLICY_DEV = Path(__file__).resolve().parents[1]
REPO = POLICY_DEV.parent
RUNTIME = POLICY_DEV / "runtime"                     # modules that run inside the sandbox or inside candidates
STATION_CLIENT = POLICY_DEV / "station" / "client"   # the station's client side (program contract, evaluation runner)
sys.path[:0] = [str(POLICY_DEV / "src"), str(RUNTIME), str(STATION_CLIENT)]

from astra_station import binding as binding_mod  # noqa: E402
from astra_station.catalog import request_sha  # noqa: E402
from astra_station.common import read  # noqa: E402
from astra_station.site import Site  # noqa: E402

HERE = Path(__file__).resolve().parent
LINK_LOST = "station closed the connection"          # station.py StationError when the relay or the network drops


class Job:
    def __init__(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("--binding", required=True); ap.add_argument("--request", required=True); ap.add_argument("--result", required=True)
        a = ap.parse_args()
        self.binding = binding_mod.verify(a.binding)
        self.request = read(a.request)
        self.result_path = Path(a.result)
        self.args = self.request["arguments"]
        self.ws = Path(self.request["workspace"])
        self.jd = Path(self.request["job_dir"])
        self.meta = self.request["station_meta"]
        self.site = Site(self.request["site"])
        self.task_spec = self.site.task_spec(self.binding["task_id"])
        self.usage = {}
        self.link_retries = 0
        self.t0 = time.time()

    @property
    def tt(self):
        return self.site.tt

    def charge(self, key, amount):
        self.usage[key] = self.usage.get(key, 0) + amount

    def finish(self, facts=None, created=None, status="completed", error=None):
        rec = {"schema": "astra_station_receipt_v1", "status": status, "tool": self.request["tool"],
               "request_sha256": request_sha(self.request), "facts": dict(facts or {}), "created": created or {}, "usage": self.usage}
        if self.link_retries:
            rec["facts"]["station_link_retries"] = self.link_retries
        if error:
            rec["error"] = error
        self.result_path.write_text(json.dumps(rec, indent=1, allow_nan=False))

    def fail(self, error):
        self.finish(status="failed", error=error)

    # ------------------------------------------------------------------ station + sandbox
    def ensure_station(self, ws=None, meta=None):
        """The campaign's station (or a sub-session's: ws, meta) up and linked.  The site may pin the station hosts; a
        sub-session's own task_tool calls then find this station alive and reuse it."""
        try:
            hosts = self.site.cfg.get("station_hosts")
            self.tt.ensure_station(ws or self.ws, meta or self.meta, wait_s=3 * 3600, hosts=tuple(hosts) if hosts else None)
        except SystemExit as e:                       # task_tool reports PENDING / unreachable hosts by exiting
            raise RuntimeError(f"station unavailable: {e}")

    def sandbox(self, argv, log_path, timeout, gpu=None) -> int:
        """gpu=None: the site's sandbox_gpu (false while this machine's GPU serves the real robot)."""
        use = self.site.cfg.get("sandbox_gpu", True) if gpu is None else gpu
        return self.tt.sandbox(self.ws, argv, gpu=use, timeout=timeout, log_path=log_path)

    def link_lost(self, log_path) -> bool:
        """The run died because the station link dropped (relay / network), not because of the candidate."""
        return LINK_LOST in self.tt.tail(log_path)

    def eval_partition(self, entry, part, ids, out_root, is_program, tag, art):
        """The baselines' evaluation runner on a sealed partition (task_tool.run_eval, with the site's GPU rule).  A run cut
        by a dropped station link is repeated once after the link is re-established (an evaluation has no side effects)."""
        out_root.mkdir(parents=True, exist_ok=True)
        res, log = out_root / f"{tag}_summary.json", out_root / f"{tag}.log"
        argv = ["/station/eval_runner.py", str(entry), part, ",".join(map(str, ids)), str(res), "--artifacts", str(art)]
        if is_program:
            argv.append("--program")
        for attempt in range(2):
            self.ensure_station()
            rc = self.sandbox(argv, log, timeout=self.tt.RUN_TIMEOUT_S)
            if rc == 0 and res.exists():
                return json.loads(res.read_text())
            if attempt == 0 and self.link_lost(log):
                self.link_retries += 1
                shutil.rmtree(art, ignore_errors=True)
                continue
            break
        return {"error": f"evaluation failed (exit {rc}); see {log}", "log_tail": self.tt.tail(log)}

    @property
    def out(self) -> Path:
        """Job output directory INSIDE the workspace: sandboxed runners can only write there (the job dir is outside)."""
        d = self.ws / "runs" / self.request["job_id"]
        d.mkdir(parents=True, exist_ok=True)
        return d

    def runtime(self) -> Path:
        """A fresh copy of the in-sandbox runners inside the workspace (the sandbox sees only the workspace and /station)."""
        d = self.ws / ".runtime" / self.request["job_id"]
        d.mkdir(parents=True, exist_ok=True)
        for f in ("rollout_runner.py", "diagnostics.py"):
            shutil.copy2(RUNTIME / f, d / f)
        shutil.copy2(STATION_CLIENT / "program_runtime.py", d / "program_runtime.py")
        return d

    # ------------------------------------------------------------------ task facts the tools need
    def kinematics_dir(self) -> Path | None:
        """The arm model (kin.py + kin.npz) identified from this task's station fk, if the task has one."""
        k = (self.task_spec.get("station") or {}).get("kinematics_model")
        return self.site.task_path(self.task_spec, k) if k else None

    def real_cost(self) -> dict | None:
        """The real-robot worker's per-step cost model (ik_ms, fk_ms, step_budget_ms) from the task spec."""
        return (self.task_spec.get("real") or {}).get("worker_cost")


def main(fn):
    job = None
    try:
        job = Job()
        fn(job)
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-2500:]}"
        if job is not None:
            job.fail(msg)
        print(msg, file=sys.stderr)
        sys.exit(1)
