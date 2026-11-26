#!/usr/bin/env python3
"""Station lifecycle, sandbox execution and scoped policy-development tools."""
from __future__ import annotations

import argparse, hashlib, json, os, random, re, shlex, shutil, signal, socket, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONF = json.loads((HERE / "station_root.json").read_text()) if (HERE / "station_root.json").exists() else {}
RUNS = Path(CONF.get("workspaces_root", HERE / "workspaces")).expanduser()
VENV = Path(CONF.get("venv", HERE / ".venv")).expanduser()
UVPY = Path(CONF.get("uv_python", Path.home() / ".local/share/uv/python")).expanduser()
STATION_LIB = HERE / "station"
LEROBOT_SRC = Path(CONF["lerobot_src"]).expanduser() if CONF.get("lerobot_src") else None
TASKS = Path(CONF.get("tasks_dir", Path.home() / "astra-tasks/tasks")).expanduser()   # the task registry (task directory)
DEV, UNSEEN = list(CONF.get("dev", range(51, 66))), list(CONF.get("unseen", range(66, 81)))
HELDOUT = list(CONF.get("heldout", range(1, 51)))
ASPIRE_MAX_NUMBERED, ASPIRE_MAX_PROBES = 9, 30
ENPIRE_TRAIN_EPISODES, ENPIRE_MAX_EVALS, RUN_TIMEOUT_S, WALL_BUDGET_H = 2000, 20, 3 * 3600, 10
ROBOT_PROCS = CONF.get("robot_procs", r"bringup_|inference/run.py|replay\.py|deploy_cup_act|sharpa_runner|collect_cup|collect_nut|teleop_record|run_desk_demo")


def task_conf(task: str) -> dict:
    """The task's spec in the registry (station block: num_envs, need_mb, hosts, partition_kind, backend)."""
    f = TASKS / task / "task.json"
    if not f.exists():
        sys.exit(f"unknown task {task!r}: no {f}")
    return json.loads(f.read_text())


def task_names() -> list:
    return sorted(p.parent.name for p in TASKS.glob("*/task.json")) if TASKS.is_dir() else []


def task_station(task: str) -> dict:
    return task_conf(task).get("station") or {}


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def meta(ws: Path) -> dict:
    return json.loads((ws / ".task.json").read_text())


def cdir(m: dict) -> Path:
    d = HERE / "campaigns" / m["campaign"] / m["method"] / f"{m['task']}_{m['run_id']}"; d.mkdir(parents=True, exist_ok=True); return d


def ledger(m: dict) -> list:
    f = cdir(m) / "ledger.jsonl"
    return [json.loads(l) for l in f.read_text().splitlines()] if f.exists() else []


def log(m: dict, row: dict) -> None:
    with open(cdir(m) / "ledger.jsonl", "a") as fh:
        fh.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **row}) + "\n")


def rname(m: dict) -> str:
    return f"{m['campaign']}_{m['method']}_{m['task']}_{m['run_id']}"


# ---------------------------------------------------------------- station lifecycle
def robot_busy() -> bool:
    p = subprocess.run(["pgrep", "-af", ROBOT_PROCS], capture_output=True, text=True)
    return any(l and "pgrep" not in l for l in p.stdout.splitlines())


def station_policy(m: dict) -> dict:
    if m["method"] == "astra":                    # 11-12: our harness (station path); its budget comes from the campaign
        return {"partitions": ["dev", "unseen"], "train_episode_budget": int(m.get("train_episode_budget", ENPIRE_TRAIN_EPISODES))}
    if m["method"] == "aspire":
        return {"partitions": ["dev", "unseen"], "train_episode_budget": 0}
    return {"partitions": ["dev"], "train_episode_budget": ENPIRE_TRAIN_EPISODES}


def local_ok(ws: Path) -> bool:
    s = ws / ".station.sock"
    if not s.exists():
        return False
    try:
        c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); c.settimeout(20); c.connect(str(s))
        hb = json.dumps({"cmd": "info", "arrays": []}).encode(); c.sendall(len(hb).to_bytes(4, "little") + (0).to_bytes(4, "little") + hb)
        ok = len(c.recv(8)) == 8; c.close(); return ok
    except OSError:
        return False


# Station hosts: station_root.json "hosts" (written by install.py from the site file's "hosts"). Without one, this
# machine alone: the station starts from this root's sim/server/station_up.sh and its socket is relayed with socat.
# Each entry: name, ssh (None = this machine), root (where `up` runs), stations (run directories), slots, gpu, up.
HOSTS = CONF.get("hosts") or [
    {"name": "local", "ssh": None, "root": str(HERE / "sim/server"), "stations": str(HERE / "stations"), "slots": 1, "gpu": 0,
     "up": "./station_up.sh {task} {rd} {rs} {nenv}"},
]


def hssh(h: dict, cmd: str, timeout=180) -> str:
    if h["ssh"] is None:
        return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout).stdout
    for _ in range(5):
        p = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", h["ssh"], cmd], capture_output=True, text=True, timeout=timeout)
        if p.returncode != 255:
            return p.stdout
        time.sleep(15)
    return ""


def host_free(h: dict, task: str) -> bool:
    """A slot on this host: fewer live stations of ours on it than its slot count, and (shared GPUs) enough free memory."""
    tag = h["name"]
    live = hssh(h, f"n=0; for d in {h['stations']}/*/; do [ \"$(cat $d/HOST 2>/dev/null)\" = {tag} ] && [ -f $d/READY ] && kill -0 $(cat $d/READY) 2>/dev/null && n=$((n+1)); done; echo $n").strip()
    if not live or int(live) >= h["slots"]:
        return False
    if h["gpu"] is not None:
        out = hssh(h, f"nvidia-smi -i {h['gpu']} --query-gpu=memory.total,memory.used --format=csv,noheader,nounits").strip().split(",")
        if len(out) != 2 or int(out[0]) - int(out[1]) < int(task_station(task).get("need_mb", 12000)):
            return False
    return True


def release_stations(m: dict) -> None:
    """Stop the stations of this run (its workspace's and its jobs'): a frozen deliverable needs no more simulator time
    from them, and left alone each holds its host's slot until the server's idle exit (half an hour), which the sealed
    evaluation on a one-slot site would spend waiting."""
    base = rname(m)
    for h in HOSTS:
        hssh(h, f"for d in {h['stations']}/{base} {h['stations']}/{base}j*; do [ -f $d/READY ] || continue; p=$(cat $d/READY); "
                f"kill $p 2>/dev/null; for i in $(seq 20); do kill -0 $p 2>/dev/null || break; sleep 0.5; done; kill -9 $p 2>/dev/null; "
                f"rm -f $d/READY; done; true")


def ensure_station(ws: Path, m: dict, wait_s=540, hosts=None) -> None:
    if local_ok(ws):
        return
    pf = ws / ".tunnel.pid"
    if pf.exists():
        try:
            os.kill(int(pf.read_text()), signal.SIGTERM)
        except (OSError, ValueError):
            pass
        pf.unlink()
    hf = cdir(m) / "station_host.json"
    cur = json.loads(hf.read_text()) if hf.exists() else None
    h = next((x for x in HOSTS if cur and x["name"] == cur["host"]), None)
    rd = rs = None
    if h is not None:
        rd = f"{h['stations']}/{rname(m)}"; rs = f"{rd}/station.sock"
        alive = any(hssh(h, f"test -f {rd}/READY && kill -0 $(cat {rd}/READY) 2>/dev/null && echo yes").strip() == "yes" for _ in range(3))
        if not alive:
            h = None
    if h is None:
        t0 = time.time()
        pool = [x for x in HOSTS if not hosts or x["name"] in hosts]
        only = task_station(m["task"]).get("hosts")
        if only:
            pool = [x for x in pool if x["name"] in only]
        while True:
            h = next((x for x in pool if host_free(x, m["task"])), None)
            if h is not None:
                break
            if time.time() - t0 > wait_s:
                sys.exit("PENDING: every simulator slot is busy (other campaigns are running). Nothing failed; run the same command again in a few minutes.")
            time.sleep(30)
        rd = f"{h['stations']}/{rname(m)}"; rs = f"{rd}/station.sock"
        pol = json.dumps(station_policy(m))
        hssh(h, f"for p in $(pgrep -f -- '--socket {rs}'); do [ $p != $$ ] && kill -9 $p; done; true")   # never two servers for one workspace
        up = h["up"].format(task=m["task"], rd=rd, rs=rs, nenv=int(task_station(m["task"]).get("num_envs", 16)))
        hssh(h, f"mkdir -p {rd} && echo {shlex.quote(pol)} > {rd}/policy.json && echo {h['name']} > {rd}/HOST && rm -f {rd}/READY && cd {h['root']} && "
                f"(nohup {up} > {rd}/server.log 2>&1 < /dev/null &)")
        hf.write_text(json.dumps({"host": h["name"], "since": time.strftime("%Y-%m-%dT%H:%M:%S%z")}))
        t0 = time.time()
        while hssh(h, f"test -f {rd}/READY && echo yes").strip() != "yes":
            if time.time() - t0 > 900:
                sys.exit(f"the station did not come up within 15 min (host {h['name']}); retry the command")
            time.sleep(10)
    (ws / ".station.sock").unlink(missing_ok=True)
    relay = (["socat", f"UNIX-LISTEN:{ws}/.station.sock,fork", f"UNIX-CONNECT:{rs}"] if h["ssh"] is None else
             ["ssh", "-N", "-o", "BatchMode=yes", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30", "-L", f"{ws}/.station.sock:{rs}", h["ssh"]])
    p = subprocess.Popen(relay,
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=open(cdir(m) / "tunnel.log", "a"), start_new_session=True)
    pf.write_text(str(p.pid))
    for _ in range(90):
        if local_ok(ws):
            return
        time.sleep(1)
    tail = (cdir(m) / "tunnel.log").read_text()[-300:].strip().splitlines()
    sys.exit("the station tunnel did not open" + (f" ({tail[-1]})" if tail else "") + "; retry the command")


def sandbox(ws: Path, argv: list, gpu: bool, timeout: int, log_path: Path) -> int:
    # Bind GPU devices only for a GPU-enabled sandbox.
    cmd = ["bwrap", "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64", "--ro-bind", "/bin", "/bin",
           "--ro-bind", "/sbin", "/sbin", "--ro-bind", "/etc", "/etc", "--ro-bind", "/sys", "/sys", "--proc", "/proc", "--dev", "/dev",
           "--ro-bind-try", str(UVPY), str(UVPY), "--ro-bind", str(VENV), str(VENV),
           "--ro-bind", str(STATION_LIB), "/station",
           "--bind", str(ws), str(ws), "--tmpfs", "/tmp", "--unshare-all", "--die-with-parent", "--new-session",
           "--setenv", "HOME", str(ws), "--setenv", "PYTHONPATH", "/station", "--setenv", "STATION_SOCKET", f"{ws}/.station.sock",
           "--setenv", "OMP_NUM_THREADS", "8", "--setenv", "MPLCONFIGDIR", "/tmp", "--setenv", "STATION_GPU_FRACTION", "0.45", "--chdir", str(ws)]
    if LEROBOT_SRC is not None:
        cmd.extend(["--ro-bind", str(LEROBOT_SRC), str(LEROBOT_SRC)])
    if gpu:
        for d in sorted(Path("/dev").glob("nvidia*")):
            cmd += ["--dev-bind", str(d), str(d)]
    else:
        cmd += ["--setenv", "CUDA_VISIBLE_DEVICES", ""]
    cmd = ["nice", "-n", "19", *cmd, str(VENV / "bin/python"), "-u", *argv]
    with open(log_path, "w") as fh:
        try:
            p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            fh.write(f"\n[station] killed after the {timeout} s limit\n"); rc = 124
    return rc


def tail(p: Path, n=40) -> str:
    try:
        return "\n".join(p.read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def run_eval(ws: Path, m: dict, prog: Path, part: str, ids: list, out_root: Path, is_program: bool, tag: str, art: Path | None = None) -> dict:
    ensure_station(ws, m)
    out_root.mkdir(parents=True, exist_ok=True)
    res = out_root / f"{tag}_summary.json"
    argv = ["/station/eval_runner.py", str(prog), part, ",".join(map(str, ids)), str(res), "--artifacts", str(art or out_root / f".{tag}_runs")]
    if is_program:
        argv.append("--program")
    rc = sandbox(ws, argv, gpu=True, timeout=RUN_TIMEOUT_S, log_path=out_root / f"{tag}.log")
    if rc != 0 or not res.exists():
        return {"error": f"evaluation failed (exit {rc}); see {out_root / (tag + '.log')}", "log_tail": tail(out_root / f"{tag}.log")}
    return json.loads(res.read_text())


# ---------------------------------------------------------------- ASPIRE: cap-harness on the task station
def cap(ws: Path, rest: list):
    m = meta(ws); task = ws / m["task_dir"]
    if not rest:
        sys.exit("usage: cap-harness run|run-batch|status|score|select|py|skill-sha|validate-skill-report ...")
    verb, rest = rest[0], rest[1:]
    L = [r for r in ledger(m) if r.get("kind") in ("numbered", "probe", "unseen")]
    if verb == "status":
        print(json.dumps({"numbered_versions_run": sorted({r["program"] for r in L if r["kind"] == "numbered"}), "numbered_budget": ASPIRE_MAX_NUMBERED,
                          "probe_runs": sum(r["kind"] == "probe" for r in L), "probe_budget": ASPIRE_MAX_PROBES,
                          "unseen_check_done": any(r["kind"] == "unseen" for r in L)}, indent=1)); return
    if verb == "score":
        sweep = Path(rest[0]).resolve(); rows = [json.loads(l) for l in open(sweep / "results.jsonl")]
        for r in sorted(rows, key=lambda r: r["seed"]):
            if not r["run_dir"]:
                print(r["seed"], "NO ARTIFACT", r["error"]); continue
            o = json.load(open(r["run_dir"] + "/outcome.json")); print(r["seed"], o["program_ok"], o["task_success"], o["termination_reason"])
        print("passed", sum(1 for r in rows if r["run_dir"] and json.load(open(r["run_dir"] + "/outcome.json"))["task_success"]), "of", len(rows)); return
    if verb == "py":
        scr = Path(rest[0]).resolve() if rest else None
        if scr is None or scr.parent != task / "code" or not scr.name.startswith("analysis_") or not scr.is_file():
            sys.exit("REFUSED: analysis scripts live in $TASK_DIR/code/analysis_<slug>.py")
        lp = task / "debug" / "logs" / f"{scr.stem}.log"; lp.parent.mkdir(parents=True, exist_ok=True)
        log(m, {"kind": "analysis", "script": scr.name, "sha256": sha(scr)})
        rc = sandbox(ws, [str(scr), *rest[1:]], gpu=False, timeout=1800, log_path=lp)
        print(f"exit {rc}; log: {lp.relative_to(ws)}\n--- last lines ---\n{tail(lp, 80)}"); return
    if verb in ("skill-sha", "validate-skill-report"):
        script = ws / ".claude/skills/iterative-debugging/scripts/campaign.py"
        a = ["--skill-library-sha"] if verb == "skill-sha" else ["--validate-skill-report", str(Path(rest[0]).resolve())]
        sys.exit(subprocess.run(["/usr/bin/python3", "-I", str(script), *a], cwd=ws).returncode)
    if verb == "select":
        src = Path(rest[0]).resolve()
        if src.parent != task / "code" or not any(r["program"] == src.name and r["program_sha256"] == sha(src) for r in L if r["kind"] == "numbered"):
            sys.exit("REFUSED: select a numbered version in code/ exactly as it was run on the development seeds")
        shutil.copy2(src, task / "fix_code.py"); log(m, {"kind": "select", "selected": src.name, "program_sha256": sha(src)}); print(f"fix_code.py := {src.name}"); return
    if verb not in ("run", "run-batch"):
        sys.exit(f"unknown command {verb}")
    p = argparse.ArgumentParser(prog=f"cap-harness {verb}")
    p.add_argument("--program", required=True); p.add_argument("--output-root", required=True)
    p.add_argument("--seed", type=int); p.add_argument("--seeds", type=int, nargs="+"); p.add_argument("--results-jsonl")
    for flag in ("--benchmark", "--suite", "--task-id", "--init-mode", "--workers", "--log-dir", "--max-steps"):
        p.add_argument(flag)
    p.add_argument("--flat-run-dir", action="store_true"); p.add_argument("--no-videos", action="store_true")
    b = p.parse_args(rest)
    seeds = sorted(b.seeds if b.seeds else [b.seed] if b.seed is not None else [])
    if not seeds:
        sys.exit("give --seed or --seeds")
    prog = Path(b.program).resolve(); out = Path(b.output_root).resolve()
    if not prog.is_file() or task not in prog.parents:
        sys.exit(f"REFUSED: the program must be a file inside {task}")
    if task / "debug" not in [out, *out.parents]:
        sys.exit(f"REFUSED: --output-root must be inside {task / 'debug'}")
    if set(seeds) & set(HELDOUT) or not set(seeds) <= set(DEV) | set(UNSEEN):
        sys.exit("REFUSED: held-out seeds are coordinator-owned; a worker may run only its development seeds and, once, the unseen-check seeds")
    name = prog.name
    if set(seeds) & set(UNSEEN):
        if seeds != UNSEEN or prog != task / "fix_code.py" or out != task / "debug" / "holdout":
            sys.exit("REFUSED: the unseen-check is one run of fix_code.py on all unseen-check seeds into debug/holdout")
        if any(r["kind"] == "unseen" for r in L):
            sys.exit("REFUSED: the unseen-check has already been run; it is a single measurement")
        kind, part = "unseen", "unseen"
    elif re.fullmatch(r"v\d\d_\w+\.py", name):
        if seeds != DEV:
            sys.exit("REFUSED: every numbered version runs on all assigned development seeds")
        done = {r["program"] for r in L if r["kind"] == "numbered"}
        if name not in done and len(done) >= ASPIRE_MAX_NUMBERED:
            sys.exit(f"REFUSED: budget used ({ASPIRE_MAX_NUMBERED} numbered versions = v00 + 8); select among the versions you have")
        kind, part = "numbered", "dev"
    elif name == "fix_code.py":
        sys.exit("REFUSED: fix_code.py is run only for the unseen-check")
    else:
        if sum(r["kind"] == "probe" for r in L) >= ASPIRE_MAX_PROBES:
            sys.exit(f"REFUSED: probe budget used ({ASPIRE_MAX_PROBES} probe runs)")
        kind, part = "probe", "dev"
    job = f"{m['run_id']}_{len(L) + 1:03d}"
    log(m, {"kind": kind, "job": job, "program": name, "program_sha256": sha(prog), "seeds": seeds})
    print(f"running job {job} ({kind}, {len(seeds)} seed(s)) on the station ...", flush=True)
    res = run_eval(ws, m, prog, part, seeds, out, True, job)
    if "error" in res:
        log(m, {"kind": "result", "job": job, "error": res["error"]}); print(res["error"]); print(res.get("log_tail", "")); sys.exit(1)
    rows = []
    for s in seeds:
        src = out / f".{job}_runs" / str(s); dst = out / str(s) / job
        if src.exists():
            dst.parent.mkdir(parents=True, exist_ok=True); shutil.move(str(src), str(dst)); rows.append({"seed": s, "run_dir": str(dst), "error": None})
        else:
            rows.append({"seed": s, "run_dir": None, "error": "no artifact (station-side failure, not the program)"})
    shutil.rmtree(out / f".{job}_runs", ignore_errors=True)
    rj = Path(b.results_jsonl).resolve() if b.results_jsonl else out / "results.jsonl"; rj.parent.mkdir(parents=True, exist_ok=True)
    with open(rj, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    log(m, {"kind": "result", "job": job, "passed": res["success"], "of": res["n"], "terms": res["terms"]})
    for r in rows:
        print(f"run: {r['run_dir']}" if r["run_dir"] else f"seed {r['seed']}: {r['error']}")
    print(f"job {job}: {res['success']}/{res['n']} task_success   results: {rj}")


# ---------------------------------------------------------------- ENPIRE: policy researcher on the task station
def station_cmd(ws: Path, rest: list):
    m = meta(ws)
    if not rest:
        sys.exit("usage: station status|run|eval|score|freeze ...")
    verb, rest = rest[0], rest[1:]
    L = ledger(m)
    started = next((r["time"] for r in L if r.get("kind") == "session_start"), None)
    if verb == "status":
        info = {}
        if local_ok(ws):
            info = json.loads(subprocess.run([str(VENV / "bin/python"), "-c", "import sys,json;sys.path.insert(0,'" + str(STATION_LIB) + "');from station import Station;"
                                              f"s=Station('{ws}/.station.sock');print(json.dumps(s.info()));s.close()"], capture_output=True, text=True).stdout or "{}")
        evals = [r for r in L if r.get("kind") == "eval_result"]
        print(json.dumps({"training_episodes_used": (info.get("usage") or {}).get("train_episodes"), "training_episode_budget": ENPIRE_TRAIN_EPISODES,
                          "evaluations": [{k: r[k] for k in ("name", "policy", "success", "trials")} for r in evals], "evaluation_budget": ENPIRE_MAX_EVALS,
                          "station_up": bool(info)}, indent=1)); return
    if verb == "run":
        p = argparse.ArgumentParser(prog="station run"); p.add_argument("script"); p.add_argument("--gpu", action="store_true")
        p.add_argument("--timeout", type=int, default=RUN_TIMEOUT_S); p.add_argument("args", nargs=argparse.REMAINDER)
        b = p.parse_args(rest); scr = (ws / b.script).resolve() if not Path(b.script).is_absolute() else Path(b.script).resolve()
        if ws not in scr.parents or not scr.is_file():
            sys.exit("REFUSED: the script must be a file inside this workspace")
        if b.timeout > RUN_TIMEOUT_S:
            sys.exit(f"REFUSED: at most {RUN_TIMEOUT_S} s per run")
        ensure_station(ws, m)
        n = sum(1 for r in L if r.get("kind") == "run") + 1; lp = ws / "runs" / f"run_{n:03d}_{scr.stem}.log"; lp.parent.mkdir(exist_ok=True)
        log(m, {"kind": "run", "n": n, "script": str(scr.relative_to(ws)), "sha256": sha(scr), "gpu": b.gpu})
        t0 = time.time(); rc = sandbox(ws, [str(scr), *b.args], b.gpu, b.timeout, lp)
        log(m, {"kind": "run_end", "n": n, "rc": rc, "wall_s": round(time.time() - t0, 1)})
        print(f"exit {rc} after {time.time() - t0:.0f} s; full log: {lp.relative_to(ws)}\n--- last lines ---\n{tail(lp)}"); return
    if verb == "eval":
        p = argparse.ArgumentParser(prog="station eval"); p.add_argument("--name", required=True); p.add_argument("--policy", required=True)
        p.add_argument("--program", action="store_true"); b = p.parse_args(rest)
        if not re.fullmatch(r"[a-z][a-z0-9_]{0,15}", b.name) or any(r.get("kind") == "eval" and r["name"] == b.name for r in L):
            sys.exit("REFUSED: --name must be new, lower-case")
        if sum(r.get("kind") == "eval" for r in L) >= ENPIRE_MAX_EVALS:
            sys.exit(f"REFUSED: evaluation budget used ({ENPIRE_MAX_EVALS} evaluations)")
        pol = (ws / b.policy).resolve()
        if ws not in pol.parents or not pol.is_file():
            sys.exit("REFUSED: the policy file must be inside this workspace")
        snap = ws / "runs" / b.name / "deliverable"; shutil.copytree(pol.parent, snap, ignore=shutil.ignore_patterns("__pycache__", "*.log"))
        log(m, {"kind": "eval", "name": b.name, "policy": str(pol.relative_to(ws)), "program": b.program, "snapshot_sha256": tree_sha(snap)})
        res = run_eval(ws, m, snap / pol.name, "dev", DEV, ws / "runs" / b.name, b.program, b.name, art=ws / "runs" / b.name / "trials")
        if "error" in res:
            log(m, {"kind": "eval_result", "name": b.name, "policy": str(pol.relative_to(ws)), "success": None, "trials": 0, "error": res["error"]})
            print(res["error"]); print(res.get("log_tail", "")); sys.exit(1)
        log(m, {"kind": "eval_result", "name": b.name, "policy": str(pol.relative_to(ws)), "success": res["success"], "trials": res["n"], "terms": res["terms"]})
        print(json.dumps({k: res[k] for k in ("partition", "n", "success", "terms")}, indent=1)); print(f"per-trial artifacts: runs/{b.name}/"); return
    if verb == "score":
        evals = [r for r in L if r.get("kind") == "eval_result" and r["trials"]]
        flags = [True] * 0
        print(json.dumps({"evaluations": [{"name": r["name"], "success": r["success"], "trials": r["trials"], "rate": round(r["success"] / r["trials"], 3)} for r in evals]}, indent=1)); return
    if verb == "freeze":
        p = argparse.ArgumentParser(prog="station freeze"); p.add_argument("--eval", required=True); b = p.parse_args(rest)
        ev = next((r for r in L if r.get("kind") == "eval_result" and r["name"] == b.eval and r["trials"]), None)
        if ev is None:
            sys.exit("REFUSED: freeze names a completed evaluation of this campaign (its deliverable snapshot is what gets frozen)")
        e0 = next(r for r in L if r.get("kind") == "eval" and r["name"] == b.eval)
        d = cdir(m) / "frozen"; shutil.rmtree(d, ignore_errors=True); shutil.copytree(ws / "runs" / b.eval / "deliverable", d)
        rec = {"evaluation": b.eval, "policy_file": Path(e0["policy"]).name, "program": e0["program"], "dev_success": ev["success"], "dev_trials": ev["trials"],
               "snapshot_sha256": tree_sha(d), "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        (cdir(m) / "frozen.json").write_text(json.dumps(rec, indent=1)); log(m, {"kind": "frozen", **rec}); print(json.dumps(rec, indent=1)); return
    sys.exit(f"unknown command {verb}")


def tree_sha(d: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(p for p in Path(d).rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        h.update(str(f.relative_to(d)).encode()); h.update(f.read_bytes())
    return h.hexdigest()


# ---------------------------------------------------------------- coordinator
def cmd_heldout(a):
    ws = RUNS / a.campaign / f"{a.method}_{a.task}_{a.run_id}"; m = meta(ws); d = cdir(m)
    if (d / "heldout.json").exists():
        sys.exit("the held-out partition has already been run for this deliverable")
    fr = json.loads((d / "frozen.json").read_text())
    target = d / "frozen" / fr["policy_file"]
    is_prog = fr["program"]
    token = (HERE / "sim/sealed/heldout_token").read_text().strip()
    # the coordinator's own evaluation: the runner is given the token through an argument file outside every workspace
    hw = HERE / "campaigns" / m["campaign"] / "_heldout_ws" / f"{a.method}_{a.task}_{a.run_id}"; hw.mkdir(parents=True, exist_ok=True)
    shutil.copytree(target.parent, hw / "deliverable", dirs_exist_ok=True)
    (hw / ".task.json").write_text(json.dumps({**m, "run_id": f"{a.run_id}_heldout"}))
    held = ",".join(x["name"] for x in HOSTS)
    release_stations(m)
    hm = meta(hw); ensure_station(hw, hm, wait_s=6 * 3600, hosts=tuple(os.environ.get("TT_HELDOUT_HOSTS", held).split(",")))
    out = hw / "result"; out.mkdir(exist_ok=True)
    runner = hw / "heldout_runner.py"
    runner.write_text((STATION_LIB / "eval_runner.py").read_text().replace('st.reset_partition(part, batch)', f'st._obs(st._call({{"cmd": "reset_partition", "partition": part, "ids": batch, "token": "{token}"}})[1])'))
    argv = [str(runner), str(hw / "deliverable" / target.name), "heldout", ",".join(map(str, HELDOUT)), str(out / "summary.json")] + (["--program"] if is_prog else [])
    # the caller's site may keep this machine's GPU for the real robot (site sandbox_gpu=false: learned policies run on CPU in
    # the sandbox, as in that site's own evaluations); the baselines' command line has no such field and keeps the GPU
    rc = sandbox(hw, argv, gpu=getattr(a, "sandbox_gpu", True), timeout=RUN_TIMEOUT_S, log_path=out / "heldout.log"); runner.unlink()
    if rc != 0:
        sys.exit(f"held-out evaluation failed: {tail(out / 'heldout.log')}")
    res = json.loads((out / "summary.json").read_text())
    rec = {"heldout_n": res["n"], "success": res["success"], "terms": res["terms"], "deliverable_sha256": tree_sha(hw / "deliverable")}
    for key, levels in (task_station(a.task).get("heldout_progress") or {}).items():
        # a judge's graded progress per episode beside the binary success (evaluation-side only; nut: tightening turns)
        tu = [float(e[key]) for e in res.get("episodes", []) if key in e]
        if tu:
            import statistics as S
            rec[key] = {"per_episode": {str(e["id"]): e[key] for e in res["episodes"] if key in e},
                        "mean": round(S.mean(tu), 3), "median": round(S.median(tu), 3), "max": round(max(tu), 3),
                        "at_least": {str(k): sum(x >= k for x in tu) for k in levels}}
    (d / "heldout.json").write_text(json.dumps(rec, indent=1)); print(json.dumps(rec))


# ---------------------------------------------------------------- long commands run as detached jobs
# 11-11: a 15-start evaluation can take longer than the agent's 10-minute shell limit. A long command therefore runs as a
# detached job of this workspace; the call follows its output for at most FOLLOW_S and then returns, and `wait <id>` keeps
# following it. One job at a time per workspace (the station serves one connection).
LONG_VERBS = {"cap": ("run", "run-batch", "py"), "station": ("run", "eval")}
FOLLOW_S = 540


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0); return True
    except OSError:
        return False


def follow_job(jd: Path, jid: str, tool: str, show_all=True) -> None:
    lp, rcp = jd / f"{jid}.log", jd / f"{jid}.rc"
    meta_j = json.loads((jd / f"{jid}.json").read_text())
    off = 0
    if not show_all and lp.exists():
        lines = lp.read_text(errors="replace").splitlines()
        if len(lines) > 60:
            print(f"[job {jid}: last 60 of {len(lines)} lines so far]")
            off = len("\n".join(lines[:-60])) + 1
    t0 = time.time()
    while True:
        if lp.exists():
            with open(lp, errors="replace") as fh:
                fh.seek(off); chunk = fh.read(); off = fh.tell()
            if chunk:
                print(chunk, end="", flush=True)
        if rcp.exists():
            rc = int(rcp.read_text().strip() or 1); print(f"\n[job {jid} finished, exit {rc}]"); sys.exit(rc)
        if not _alive(meta_j["pid"]):
            time.sleep(2)
            if not rcp.exists():
                print(f"\n[job {jid} ended without an exit status]"); sys.exit(1)
            continue
        if time.time() - t0 > FOLLOW_S:
            print(f"\n[job {jid} ({meta_j['cmd']}) is still running after {FOLLOW_S // 60} min. It continues in the background and keeps "
                  f"its result. Run `./tools/{tool} wait {jid}` to keep following it (each call follows up to {FOLLOW_S // 60} min); "
                  f"another long command is refused until it ends.]", flush=True)
            sys.exit(0)
        time.sleep(2)


def start_job(ws: Path, verb: str, rest: list) -> None:
    tool = "cap-harness" if verb == "cap" else "station"
    jd = ws / ".jobs"; jd.mkdir(exist_ok=True)
    act = jd / "active.json"
    if act.exists():
        a = json.loads(act.read_text())
        if _alive(a["pid"]) and not (jd / f"{a['id']}.rc").exists():
            sys.exit(f"REFUSED: job {a['id']} ({a['cmd']}) is still running on your station. Run `./tools/{tool} wait {a['id']}` "
                     f"to follow it until it ends; one long command at a time.")
    jid = f"j{len(list(jd.glob('j*.json'))) + 1:03d}"
    env = {**os.environ, "TT_JOB": jid, "TT_JOBDIR": str(jd), "PYTHONUNBUFFERED": "1"}
    with open(jd / f"{jid}.log", "w") as fh:
        p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]], env=env, stdout=fh, stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL, start_new_session=True)
    rec = {"id": jid, "pid": p.pid, "cmd": " ".join(rest)[:200], "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (jd / f"{jid}.json").write_text(json.dumps(rec)); act.write_text(json.dumps(rec))
    print(f"[job {jid} started: {rec['cmd']}]", flush=True)
    follow_job(jd, jid, tool)


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("cap", "station"):
        verb = sys.argv[1]; argv = sys.argv[2:]
        if len(argv) < 2 or argv[0] != "--workspace":
            sys.exit("bad tool invocation")
        rest = argv[2:]; rest = rest[1:] if rest[:1] == ["--"] else rest
        ws = Path(argv[1]).resolve()
        m = meta(ws)
        if not any(r.get("kind") == "session_start" for r in ledger(m)):
            log(m, {"kind": "session_start"})
        tool = "cap-harness" if verb == "cap" else "station"
        if rest[:1] == ["wait"]:
            jd = ws / ".jobs"
            if len(rest) < 2 or not (jd / f"{rest[1]}.json").exists():
                sys.exit(f"usage: {tool} wait <job id>  (job ids are printed when a long command starts)")
            follow_job(jd, rest[1], tool, show_all=False); return
        if rest[:1] and rest[0] in LONG_VERBS[verb] and not os.environ.get("TT_JOB"):
            start_job(ws, verb, rest); return
        if os.environ.get("TT_JOB"):           # the detached job itself: record its exit status for the followers
            jd, jid = Path(os.environ["TT_JOBDIR"]), os.environ["TT_JOB"]; rc = 1
            try:
                (cap if verb == "cap" else station_cmd)(ws, rest); rc = 0
            except SystemExit as e:
                rc = e.code if isinstance(e.code, int) else (0 if e.code is None else (print(e.code, flush=True) or 1))
            except BaseException:  # noqa: BLE001
                import traceback; traceback.print_exc(); rc = 1
            finally:
                sys.stdout.flush(); (jd / f"{jid}.rc").write_text(str(rc))
            return
        (cap if verb == "cap" else station_cmd)(ws, rest); return
    sys.exit("Usage: task_tool.py {cap,station} --workspace PATH -- COMMAND")


if __name__ == "__main__":
    main()
