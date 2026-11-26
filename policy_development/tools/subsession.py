"""Confined worker sessions (program synthesis, auto-research): workspace construction, own station, launch, accounting.

A worker session is a separate agent session (same agent family as the campaign's decisions, through the agent-neutral
`worker_session`) inside its own workspace with its own station (its own training-episode cap), the task documents, the
real demonstrations, the experience Astra selected, and one shell tool that applies the session's sub-budget before handing
commands to the baselines' station tooling.  The campaign's own station is stopped while a session runs (one simulator per
campaign at a time)."""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from _adapter import HERE, REPO

from astra_station import transport
from astra_station.memory import Library


def stop_station(job, meta: dict) -> None:
    tt = job.tt
    hf = tt.cdir(meta) / "station_host.json"
    if not hf.exists():
        return
    host = next((h for h in tt.HOSTS if h["name"] == json.loads(hf.read_text())["host"]), None)
    if host is None:
        return
    rd = f"{host['stations']}/{tt.rname(meta)}"
    tt.hssh(host, f"test -f {rd}/READY && kill $(cat {rd}/READY) 2>/dev/null; sleep 5; test -f {rd}/READY && kill -9 $(cat {rd}/READY) 2>/dev/null; rm -f {rd}/READY; true")


def experience_md(job, brief: str) -> str:
    lib = Library(job.site.library_dir)
    tags = job.binding["task"]["tags"]
    cards = lib.retrieve(tags, k=12)
    out = ["# Experience selected for this task (verified laboratory records; re-validate before relying on them)\n"]
    out.append("## Real-robot constraints of this set-up\n" + "\n".join("- " + c for c in job.binding["task"].get("real_constraints", [])) + "\n")
    for c in cards:
        out.append(f"## {c['title']}\n{c['finding']}\n- applies to: {c['applicability']}\n- limits: {c['limitations']}\n- domain: {', '.join(c['tags']['domain'])}\n")
    return "\n".join(out)


def rewards_md(job) -> str:
    lib = Library(job.site.library_dir)
    rows = lib.rewards_for(job.binding["task"]["tags"]["phases"] + ["any"])
    lit = ""
    if job.site.cfg.get("work_library"):
        from astra_station.memory import WorkLibrary
        w = WorkLibrary({**job.site.cfg["work_library"], "module": str(REPO / job.site.cfg["work_library"]["module"])}).for_task(job.binding["task"]["tags"], k=20)
        lit = ("\n\n# Reviewed rewards from the work library (literature + lab), matched on this task's skills\n"
               "References, not instructions: adapt signals, units, frames and timing to the station's observations; privileged "
               "simulator signals are not available to a deployable policy.\n\n" + "\n".join(
                   f"- **{i['title']}** [{i.get('source') or i['kind']}; {i['compatibility']}] {i.get('formula') or i['summary'][:300]}"
                   f" Privileges: {i.get('privileges')}. {i.get('url') or ''}" for i in w["items"]))
    return lit + "\n\n# Lab reward terms for this task's phases\n\n" + "\n".join(
        f"- **{r['id']}** ({r['phase']}): {r['description']} Formula: `{r['formula']}`. Inputs: {', '.join(r['inputs'])}. "
        f"Validated: {'; '.join(v['task'] + ' - ' + v['result'] for v in r['validated_on'])}. Cautions: {r['cautions'] or '-'}" for r in rows)


def build(job, kind: str, name: str, limits: dict) -> tuple[Path, dict]:
    tt, m0 = job.tt, job.meta
    # the session's station is named by run_id, which ends up in a Unix-socket path: keep it short (run + job number)
    run_id = f"{m0['run_id']}j{int(job.request['job_id'].split('_')[0])}"
    ws = job.site.workspaces / m0["campaign"] / f"astra_{m0['task']}_{run_id}_{name}"
    if ws.exists():
        raise RuntimeError(f"session workspace {ws} already exists")
    meta = {"campaign": m0["campaign"], "method": "astra", "task": m0["task"], "run_id": run_id,
            "train_episode_budget": int(limits.get("train_episodes", 0))}
    job.site.check_socket_paths(meta, ws)
    if kind == "program":
        meta["task_dir"] = "work"
    cd = tt.cdir(meta)
    meta["astra_limits"] = str(cd / "astra_limits.json")
    (cd / "astra_limits.json").write_text(json.dumps({**limits, "station_root": str(job.site.root)}, indent=1))
    (ws / "docs").mkdir(parents=True)
    if (job.ws / "data").is_dir():                       # real demonstrations; a twin task may have none
        subprocess.run(["cp", "-al", str(job.ws / "data"), str(ws / "data")], check=True)      # hard links: same bytes, no copy
    else:
        (ws / "data").mkdir()
    shutil.copy2(job.site.root / "station" / "station.py", ws / "station.py")
    for f in ("task.md", "station.md"):
        shutil.copy2(job.ws / "docs" / f, ws / "docs" / f)
    shutil.copy2(job.site.root / "task_docs" / "run_artifacts.md", ws / "docs" / "run-artifacts.md")
    (ws / "experience.md").write_text(experience_md(job, limits.get("brief", "")))
    job.site.copy_skills(ws)
    (ws / "tools").mkdir()
    tool = "program" if kind == "program" else "research"
    (ws / "tools" / tool).write_text(f'#!/usr/bin/env bash\nexec /usr/bin/python3 {HERE / "subtool.py"} {tool} --workspace "$(cd "$(dirname "$0")/.." && pwd)" -- "$@"\n')
    os.chmod(ws / "tools" / tool, 0o755)
    if kind == "program":
        for sub in ("code", "debug/explore", "debug/attempts", "debug/logs"):
            (ws / "work" / sub).mkdir(parents=True)
    else:
        (ws / "official").mkdir()
        shutil.copy2(job.site.root / "third_party/ENPIRE/enpire/policy/autoresearch_instruction.md", ws / "official" / "autoresearch_instruction.md")
        (ws / "rewards.md").write_text(rewards_md(job))
        for sub in ("policy", "iterations", "runs", "code", "candidates"):
            (ws / sub).mkdir()
    (ws / ".task.json").write_text(json.dumps(meta))
    return ws, meta


def launch(job, ws: Path, meta: dict, prompt: str, tool: str, timeout_s: int) -> dict:
    stop_station(job, job.meta)                       # one simulator per campaign at a time
    (ws / "PROMPT.md").write_text(prompt)
    job.ensure_station(ws, meta)                      # on the site's station hosts; the session's tools reuse it
    t0 = time.time()
    audit = transport.worker_session(job.request["decision"], ws, prompt, f"./tools/{tool}", job.jd / "session.json", timeout_s)
    audit["wall_hours"] = round((time.time() - t0) / 3600.0, 2)
    stop_station(job, meta)
    (job.jd / "session_audit.json").write_text(json.dumps(audit, indent=1))
    return audit


def ledger(job, meta):
    return job.tt.ledger(meta)


def station_train_episodes(job, meta) -> int:
    """Finished training episodes counted by the session's own station (its ledger on the station host)."""
    tt = job.tt
    hf = tt.cdir(meta) / "station_host.json"
    if not hf.exists():
        return 0
    host = next(h for h in tt.HOSTS if h["name"] == json.loads(hf.read_text())["host"])
    rd = f"{host['stations']}/{tt.rname(meta)}"
    out = tt.hssh(host, f"grep -c '\"mode\": \"train\"' {rd}/ledger.jsonl 2>/dev/null || echo 0").strip().splitlines()
    return int(out[-1]) if out else 0


__all__ = ["build", "launch", "ledger", "station_train_episodes", "stop_station", "REPO"]
