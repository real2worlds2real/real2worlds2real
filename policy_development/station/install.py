"""Assemble a station root from this repository and a task directory, or compare one with them.

The station root (e.g. ~/astra_real_arena) is an installed instance: generic code from this repository (station client,
server core and backends contract, coordinator, generic real worker), task code from the operator's task directory (station
backends, real chains, task docs and data pointers), plus run state that never enters a repository (sealed partitions,
task data, campaigns, deliverables).  This script lays the code out, writes station_root.json for the coordinator, and in
--check mode only reports how an existing root differs (nothing is written).

    python install.py --root ~/astra_real_arena --tasks <task directory> [--site policy_development/configs/station/site.example.json] --check
    python install.py --root <new or idle root> --tasks <task directory> [--site ...] --apply

--apply refuses to overwrite files a running or unfinished campaign pinned (the site's pinned_files) unless --force-pinned:
a binding pins those files by hash, so replacing them ends every campaign bound to the old ones.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent              # policy_development/station
REPO = HERE.parents[1]


def layout(tasks: Path) -> dict:
    """station-root path -> source file."""
    out = {"task_tool.py": HERE / "coordinator/task_tool.py"}
    for f in ("station.py", "eval_runner.py", "program_runtime.py", "sitecustomize.py"):
        out[f"station/{f}"] = HERE / "client" / f
    for f in ("__init__.py", "core.py", "backend.py", "serve.py", "twin.py", "partitions.py", "station_up.sh"):
        out[f"sim/server/{f}"] = HERE / "server" / f
    out["sim/server/delivery_files.py"] = HERE.parent / "src/astra_station/delivery_files.py"
    for f in sorted((tasks / "station_backends").glob("*.py")):
        out[f"sim/backends/{f.name}"] = f
    out["real_cup/station_worker.py"] = REPO / "policy_development/real_robot/station_worker.py"
    out["real_cup/program_runtime.py"] = HERE / "client/program_runtime.py"
    out["real_cup/cup_observer.py"] = tasks / "real/cup/cup_observer.py"
    for f in sorted((tasks / "real/cup").glob("station_info_*.json")):
        out[f"real_cup/{f.name}"] = f
    for f in sorted((tasks / "real/cup/tools").glob("*.py")):
        out[f"real_cup/{f.name}"] = f
    out["task_docs/run_artifacts.md"] = HERE / "docs/run_artifacts.md"     # every sub-session workspace gets it (tools/subsession.py)
    return out


def sha(p: Path) -> str | None:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True); ap.add_argument("--tasks", required=True)
    ap.add_argument("--site", default=str(REPO / "policy_development/configs/station/site.example.json"))
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true"); mode.add_argument("--apply", action="store_true")
    ap.add_argument("--force-pinned", action="store_true")
    a = ap.parse_args(argv)
    root, tasks = Path(a.root).expanduser(), Path(a.tasks).expanduser()
    site = json.loads(Path(a.site).read_text())
    pinned = set(site.get("pinned_files", []))
    files = layout(tasks)
    # the coordinator reads the same task registry as the site (site tasks_dir, relative to this repository's root);
    # a site without one: the task directory's own registry, <tasks>/tasks
    registry = Path(site["tasks_dir"]).expanduser() if site.get("tasks_dir") else tasks / "tasks"
    registry = registry if registry.is_absolute() else REPO / registry
    conf = {"tasks_dir": str(registry), "workspaces_root": site.get("workspaces_root"),
            "dev": site["splits"]["development"], "unseen": site["splits"]["unseen_check"]}
    report = {"same": [], "differs": [], "missing": [], "pinned_differs": []}
    files = {rel: src for rel, src in files.items() if Path(src).is_file()}      # a task directory without a real chain has none of its files
    for rel, src in files.items():
        cur = sha(root / rel)
        key = "same" if cur == sha(src) else "missing" if cur is None else "differs"
        report[key].append(rel)
        if key == "differs" and rel in pinned:
            report["pinned_differs"].append(rel)
    legacy = {"sim/station_server.py": "replaced by sim/server/serve.py + sim/backends/*",
              "real_cup/cup_policy_worker.py": "replaced by real_cup/station_worker.py + cup_observer.py"}
    report["superseded_in_root"] = {k: v for k, v in legacy.items() if (root / k).exists()}
    if a.check:
        print(json.dumps({k: (v if isinstance(v, dict) else sorted(v)) for k, v in report.items()}, indent=1))
        return 0
    if report["pinned_differs"] and not a.force_pinned:
        sys.exit(f"refused: pinned files would change {report['pinned_differs']}; campaigns bound to them would stop "
                 "(finish them, then --force-pinned)")
    for rel, src in files.items():
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    # Preserve operator settings from earlier installations. Explicit site settings win.
    old = json.loads((root / "station_root.json").read_text()) if (root / "station_root.json").is_file() else {}
    conf = {**{k: v for k, v in old.items() if k not in conf}, **conf}
    for key in ("hosts", "venv", "lerobot_src", "uv_python"):
        if site.get(key):
            conf[key] = site[key]
    (root / "station_root.json").write_text(json.dumps(conf, indent=1) + "\n")
    print(json.dumps({"installed": len(files), "root": str(root)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
