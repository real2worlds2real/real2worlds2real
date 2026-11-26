"""Process-group and container lifecycle for orchestration jobs."""
import os
import subprocess
import time
from calibrated_scene_reconstruction.core.paths import ROOT

LOG_MAX = 64 * 2**20
LOG_KEEP = (8 * 2**20, 2**20)


def run(cmd, log, prefix=(), mem="32G", timeout=None, stall=None, who="collect", max_log=LOG_MAX):
    """Run a job to its log; its whole process group is killed (rc -9) past `timeout` s, when the log has not grown
    for `stall` s -- an Isaac process that ran out of GPU memory hangs in the renderer for good, silently, and Kit
    sometimes hangs in its start -- or when the log passes `max_log` bytes (Kit stuck repeating an error, which fills
    the disk; the log is then cut to its first and last LOG_KEEP bytes). The job's Isaac container
    (tools/run_isaac.sh, ISAAC_CONTAINER) is killed with it: killing docker run leaves the container running, holding
    its GPU memory."""
    import signal
    import uuid

    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    box = f"calibrated_scene_reconstruction-{who}-{uuid.uuid4().hex[:12]}"
    env = {**os.environ, "ISAAC_CONTAINER": box, **({"MEM": mem} if mem else {})}
    with open(log, "w") as f:  # MEM: tools/run_capped.sh's cap of the job (run_isaac.sh's default is 24G)
        p = subprocess.Popen([*prefix, *cmd], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env,
                             start_new_session=True)
        size, seen, why = -1, time.time(), None
        while p.poll() is None:
            time.sleep(10)
            now, sz = time.time(), log.stat().st_size
            if sz != size:
                size, seen = sz, now
            if timeout and now - t0 > timeout:
                why = f"past {timeout:.0f} s"
            elif stall and now - seen > stall:
                why = f"no output for {stall:.0f} s"
            elif max_log and sz > max_log:
                why = f"log past {max_log / 2**20:.0f} MB"
            if why:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
                try:  # no such container: a host run, or one that never started
                    subprocess.run(["docker", "kill", box], capture_output=True, timeout=60)
                except (OSError, subprocess.TimeoutExpired):
                    pass
                if max_log and log.stat().st_size > max_log:
                    with open(log, "rb") as r:
                        head = r.read(LOG_KEEP[0])
                        r.seek(-LOG_KEEP[1], os.SEEK_END)
                        tail = r.read()
                    f.seek(0)
                    f.truncate()
                    f.write(head.decode(errors="replace") + f"\n[{who}] ... cut ...\n" + tail.decode(errors="replace"))
                f.write(f"\n[{who}] killed: {why}\n")
                print(f"[{who}] {log.stem} killed: {why}", flush=True)
                return -9, time.time() - t0
        rc = p.returncode
    return rc, time.time() - t0
