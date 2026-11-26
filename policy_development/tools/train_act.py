"""Tool `train_act`: the lab's point-cloud ACT (CVAE, PointNeXt encoder, chunk 32) on any mix of datasets.

Datasets (real, recorded from candidates, or both, each with a repeat weight) -> act_prep.py on the training host ->
the lab trainer -> stamped checkpoints -> two self-contained candidates `<name>_best` (lowest loss on the held-out
episodes) and `<name>_last` (final epoch).  Lab rule of thumb: for sim-only data the held-out loss is a poor selector."""
import json
import shlex
import shutil
import subprocess
import time
from pathlib import Path

from _adapter import RUNTIME, main
from astra_station.catalog import request_sha


def sh(host, cmd, timeout=600, check=True):
    p = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, cmd], capture_output=True, text=True, timeout=timeout)
    if check and p.returncode != 0:
        raise RuntimeError(f"ssh {host} failed ({p.returncode}): {p.stderr[-800:]} {p.stdout[-400:]}")
    return p.stdout


def rsync(src, dst, timeout=7200):
    p = subprocess.run(["rsync", "-a", "--protect-args", "-e", "ssh -o BatchMode=yes", "--", src, dst], capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"rsync {src} -> {dst} failed: {p.stderr[-800:]}")


def package(job, name, stamped: Path, rt: dict, note: str) -> Path:
    d = job.ws / "candidates" / name
    (d / "act").mkdir(parents=True)
    shutil.copy2(stamped / "policy_stamped.pt", d / "act" / "policy_stamped.pt")
    shutil.copy2(stamped / "SHA256", d / "act" / "SHA256")
    shutil.copy2(RUNTIME / "act_policy_station.py", d / "policy.py")
    src = Path(rt["pkg"])
    (d / "pkg" / "il" / "pointnet2_batch").mkdir(parents=True)
    shutil.copy2(src / "point_act_cvae.py", d / "pkg")
    for f in ("act_policy.py", "point_features.py", "pointnext_encoder.py"):
        shutil.copy2(src / "il" / f, d / "pkg" / "il" / f)
    for f in (src / "il" / "pointnet2_batch").glob("*"):
        if f.is_file():
            shutil.copy2(f, d / "pkg" / "il" / "pointnet2_batch" / f.name)
    if rt.get("libstdcxx"):
        shutil.copy2(rt["libstdcxx"], d / "pkg" / "libstdc++.so.6")
    (d / "CANDIDATE.md").write_text(note + "\n")
    return d


def reserve_remote_dir(host, cfg, request):
    """Only one invocation may write outputs for this exact campaign request."""
    reserve = """\
import sys
from pathlib import Path

root = Path(sys.argv[1]).expanduser().resolve()
root.mkdir(parents=True, exist_ok=True)
job = root / sys.argv[2]
try:
    job.mkdir()
except FileExistsError:
    sys.exit(f"ACT job directory already reserved: {job}; submit a new job to retry")
(job / "data").mkdir()
print(job)
"""
    key = "act-" + request_sha(request)
    return sh(host, shlex.join([cfg["python"], "-c", reserve, cfg["jobs_root"], key])).strip()


def run(job):
    a = job.args
    cfg, rt = job.site.compute("act_train"), job.site.compute("act_runtime")
    host = cfg["ssh"]
    rd = reserve_remote_dir(host, cfg, job.request)
    cd = f"cd {shlex.quote(rd)} && "
    specs = []
    for index, item in enumerate(a["datasets"]):
        ds = job.request["resolved"]["datasets"][item["name"]]
        dataset_dir = f"data/{index:02d}"
        rsync(ds["path"].rstrip("/") + "/", f"{host}:{rd}/{dataset_dir}/")
        specs.extend(["--dataset", f"{dataset_dir}:{int(item.get('repeat', 1))}"])
    rsync(str(RUNTIME / "act_prep.py"), f"{host}:{rd}/act_prep.py")
    prep_args = [cfg["python"], "-u", "act_prep.py", "prep", *specs, "--dev-fraction", str(a.get("dev_fraction", 0.1)),
                 "--pad-hold", str(int(a.get("pad_hold", 0))), "--seed", str(int(a.get("seed", 2026)))]
    if a.get("mask_hand", True):
        prep_args.append("--mask-hand")
    t0 = time.time()
    sh(host, cd + shlex.join(prep_args) + " > prep.log 2>&1", timeout=3 * 3600)
    prep = json.loads(sh(host, cd + "cat prep/prep_summary.json"))
    for dataset, item in zip(prep["datasets"], a["datasets"], strict=True):
        dataset["name"] = item["name"]
    # one simulator per campaign: free the campaign's station GPU memory while training (it restarts on the next station tool)
    import subsession
    subsession.stop_station(job, job.meta)
    # wait for GPU memory, then train in the background on the host and follow it
    need = int(cfg.get("need_free_mb", 9000))
    while True:
        free = sh(host, cfg["gpu_free_cmd"], check=False).strip().splitlines()
        if free and int(free[0].strip() or 0) >= need:
            break
        if time.time() - t0 > 4 * 3600:
            raise RuntimeError(f"no GPU memory ({need} MB) on the training host for 4 h")
        time.sleep(60)
    train = [x.format(prep=f"{rd}/prep", out=f"{rd}/act", seed=int(a.get("seed", 2026)), epochs=int(a.get("epochs", 400)))
             for x in cfg["train"]]
    env = [f"{k}={v}" for k, v in (cfg.get("env") or {}).items()]
    command = shlex.join(["env", *env, *train]) + " > train.log 2>&1; echo $? > train.rc"
    sh(host, cd + f"(nohup bash -c {shlex.quote(command)} > /dev/null 2>&1 < /dev/null &)")
    tg = time.time()
    while not sh(host, cd + "cat train.rc 2>/dev/null", check=False).strip():
        time.sleep(60)
    gpu_h = (time.time() - tg) / 3600.0
    job.charge("gpu_hours", gpu_h)
    rc = int(sh(host, cd + "cat train.rc").strip() or 1)
    if rc != 0 or sh(host, cd + "test -f act/policy_best.pt && echo yes", check=False).strip() != "yes":
        raise RuntimeError(f"training failed (rc {rc}): {sh(host, cd + 'tail -20 train.log', check=False)}")
    mh = bool(a.get("mask_hand", True))
    stamp = (f"import hashlib,torch;from pathlib import Path\n"
             f"for k in ('best','last'):\n"
             f" c=torch.load('act/policy_'+k+'.pt',map_location='cpu',weights_only=False)\n"
             f" c['metadata']['action_target']='delta_to_measured_state'\n"
             f" c['metadata']['gear_input']={{'arm':'measured','hand':'{'zero' if mh else 'measured'}'}}\n"
             f" c['metadata']['astra_job']={job.request['job_id']!r}\n"
             f" c['metadata']['astra_request_sha256']={request_sha(job.request)!r}\n"
             f" d=Path('stamped_'+k);d.mkdir();torch.save(c,d/'policy_stamped.pt')\n"
             f" (d/'SHA256').write_text(hashlib.sha256((d/'policy_stamped.pt').read_bytes()).hexdigest())\n")
    sh(host, cd + shlex.join([cfg["python"], "-c", stamp]), timeout=1800)
    summary = ("import json;d=json.load(open('act/train_summary.json'));h=d.get('history') or [];"
               "print(json.dumps({'best_loss':d.get('best_loss'),'epochs':len(h),'last':h[-1] if h else None,"
               "'num_trajs':d.get('num_trajs'),'point_ops_backend':d.get('point_ops_backend')}))")
    summ = sh(host, cd + shlex.join([cfg["python"], "-c", summary]), check=False)
    created = []
    for k in ("best", "last"):
        loc = job.jd / f"stamped_{k}"
        loc.mkdir()
        rsync(f"{host}:{rd}/stamped_{k}/", str(loc) + "/")
        nm = f"{a['name']}_{k}"
        note = (f"point-cloud ACT ({k} checkpoint) on {[(i['name'], i.get('repeat', 1)) for i in a['datasets']]}, seed {a.get('seed', 2026)}, "
                f"{a.get('epochs', 400)} epochs, hand inputs {'masked' if mh else 'measured'}, pad_hold {a.get('pad_hold', 0)}")
        d = package(job, nm, loc, rt, note)
        created.append({"name": nm, "kind": "policy", "path": str(d), "entry": "policy.py", "note": note, "method": "act",
                        "params": a})
    try:
        ts = json.loads(summ.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        ts = {"raw": summ[:600]}
    job.finish(facts={"prep": prep, "train": ts, "gpu_hours": round(gpu_h, 2), "host": host, "remote_job_dir": rd},
               created={"candidates": created})


if __name__ == "__main__":
    main(run)
