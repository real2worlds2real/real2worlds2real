"""Site configuration and the world provider (the station infrastructure that the baselines also use).

The site file names where the station code, task documents, real demonstrations and sealed partitions live, which files
pin the station's identity, and which compute hosts the method adapters may use.  The world provider wraps that
infrastructure (`task_tool.py` of the station root: station hosts, socket relay, bubblewrap sandbox, sealed held-out) so
that our campaigns get exactly the same station, judge, safety limits and evaluation runner as every baseline.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

from .common import digest, read, validate

SITE_SCHEMA = {
    "type": "object",
    "required": ["schema", "name", "station_root", "workspaces_root", "pinned_files", "sealed_files", "splits",
                 "worker_python", "tasks_dir", "tools_dir", "library_dir"],
    "properties": {
        "schema": {"const": "astra_station_site_v1"},
        "name": {"type": "string", "minLength": 1},
        "station_root": {"type": "string"}, "workspaces_root": {"type": "string"},
        "pinned_files": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "sealed_files": {"type": "array", "items": {"type": "string"}},
        "splits": {"type": "object", "required": ["development", "unseen_check", "sealed_count"],
                   "properties": {"development": {"type": "array", "items": {"type": "integer"}, "minItems": 1},
                                  "unseen_check": {"type": "array", "items": {"type": "integer"}},
                                  "sealed_count": {"type": "integer", "minimum": 1}}},
        "worker_python": {"type": "string"},
        "venv": {"type": "string"}, "lerobot_src": {"type": "string"}, "uv_python": {"type": "string"},
        "tasks_dir": {"type": "string"}, "tools_dir": {"type": "string"}, "library_dir": {"type": "string"},
        "tool_code": {"type": "array", "items": {"type": "string"}},
        "compute": {"type": "object"},
        "decision": {"type": "object"},
        "skills_dir": {"type": "string"},
        "prompts": {"type": "object"},
        "campaign_note": {"type": "string"},
    },
}

REPO = Path(__file__).resolve().parents[3]            # repository root
SRC = REPO / "policy_development" / "src"               # this package's source root (PYTHONPATH of tool adapters)


def _abs(p: str, base: Path) -> Path:
    q = Path(p).expanduser()
    return q if q.is_absolute() else (base / q)


class Site:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.cfg = read(self.path)
        validate(self.cfg, SITE_SCHEMA)
        self.root = _abs(self.cfg["station_root"], REPO).resolve()
        self.workspaces = _abs(self.cfg["workspaces_root"], REPO)
        self.tasks_dir = _abs(self.cfg["tasks_dir"], REPO)
        self.tools_dir = _abs(self.cfg["tools_dir"], REPO)
        self.library_dir = _abs(self.cfg["library_dir"], REPO)
        self.worker_python = self.cfg["worker_python"]
        self.sha256 = digest(self.cfg)
        self._tt = None

    def task_spec_path(self, task_id: str) -> Path:
        """Task specs live in the task directory: <tasks_dir>/<task>/task.json (a flat <tasks_dir>/<task>.json is
        still read, for sites that predate the split)."""
        for p in (self.tasks_dir / task_id / "task.json", self.tasks_dir / f"{task_id}.json"):
            if p.is_file():
                return p
        raise ValueError(f"no task spec for {task_id!r} under {self.tasks_dir}; add <task>/task.json (schema astra_station_task_v1)")

    def task_spec(self, task_id: str) -> dict:
        p = self.task_spec_path(task_id)
        return {**read(p), "_dir": str(p.parent)}

    def task_path(self, task: dict, rel: str) -> Path:
        """A path named by a task spec: `station:<path>` = in the station root (demonstrations, large or sealed data),
        an absolute or ~ path as is, anything else = relative to the task spec's own directory (documents, models)."""
        if rel.startswith("station:"):
            return self.root / rel[len("station:"):]
        q = Path(rel).expanduser()
        return q if q.is_absolute() else (Path(task["_dir"]) / q).resolve()

    def sealed_files(self, task: dict) -> list:
        """The sealed files a task's binding pins: its own start-partition file and the coordinator token
        (task spec station.partition_kind); the site's list when the spec predates that field."""
        kind = (task.get("station") or {}).get("partition_kind")
        if not kind:
            return list(self.cfg["sealed_files"])
        return [f"sim/sealed/{kind}_partitions.json", "sim/sealed/heldout_token"]

    def compute(self, name: str) -> dict:
        c = (self.cfg.get("compute") or {}).get(name)
        if c is None:
            raise ValueError(f"site has no compute entry {name!r}")
        return c

    # ------------------------------------------------------------------ world provider
    @property
    def tt(self):
        """The station root's task_tool module (station hosts, relay, sandbox, sealed held-out)."""
        if self._tt is None:
            spec = importlib.util.spec_from_file_location("astra_station_task_tool", self.root / "task_tool.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            self._tt = mod
        return self._tt

    def copy_skills(self, ws: Path) -> list:
        """The site's skill library (site `skills_dir`: <name>/skill.py + SKILL.md) as it is now, copied into a workspace
        (`skills/`): programs reach it through `load_skill`, sandboxed runs can read nothing outside the workspace."""
        src = self.cfg.get("skills_dir")
        dst = ws / "skills"
        dst.mkdir(exist_ok=True)
        names = []
        if src:
            for d in sorted(Path(src).expanduser().glob("*/skill.py")):
                shutil.copytree(d.parent, dst / d.parent.name, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
                names.append(d.parent.name)
        return names

    SOCK_MAX = 107                                    # Linux sun_path limit (108 bytes incl. the terminating NUL)

    def check_socket_paths(self, meta: dict, ws: Path, heldout: bool = False) -> None:
        """A station listens on <host stations>/<campaign>_<method>_<task>_<run>/station.sock and the relay on
        <workspace>/.station.sock; a path over the Unix-socket limit makes the station server die at bind while the
        client waits for it.  Refuse such names up front.  `heldout` adds the paths of the sealed evaluation that
        follows a campaign (its own station and the coordinator's workspace under the station root), so a campaign
        cannot finish and then be impossible to evaluate."""
        tt = self.tt
        if not hasattr(tt, "rname"):
            return
        allowed = self.cfg.get("station_hosts") or None
        hosts = [h for h in getattr(tt, "HOSTS", []) if not allowed or h["name"] in allowed]
        paths = [f"{h['stations']}/{tt.rname(meta)}/station.sock" for h in hosts] + [str(Path(ws) / ".station.sock")]
        if heldout:
            held = {**meta, "run_id": f"{meta['run_id']}_heldout"}
            paths += [f"{h['stations']}/{tt.rname(held)}/station.sock" for h in getattr(tt, "HOSTS", [])]
            paths.append(str(Path(self.root).resolve() / "campaigns" / meta["campaign"] / "_heldout_ws"
                             / f"{meta['method']}_{meta['task']}_{meta['run_id']}" / ".station.sock"))
        bad = [q for q in paths if len(q.encode()) > self.SOCK_MAX]
        if bad:
            raise ValueError(f"station socket path over {self.SOCK_MAX} bytes ({len(bad[0].encode())}): {bad[0]}; "
                             "use a shorter campaign or run name")

    def workspace(self, campaign: str, task: dict, run_id: str, train_episode_budget: int) -> tuple[Path, dict]:
        """The campaign's station workspace (created once): real demonstrations, station client, task documents.
        Candidates and datasets live inside it so that the sandboxed evaluation runner can read them."""
        # same naming as the baselines' workspaces (<method>_<station task>_<run>), so the sealed held-out tool finds it
        ws = self.workspaces / campaign / f"astra_{task['station_task']}_{run_id}"
        meta = {"campaign": campaign, "method": "astra", "task": task["station_task"], "run_id": run_id,
                "train_episode_budget": int(train_episode_budget)}
        self.check_socket_paths(meta, ws, heldout=True)
        if not ws.exists():
            (ws / "docs").mkdir(parents=True)
            data = self.task_path(task, task["data"])
            if data.is_dir():
                shutil.copytree(data, ws / "data", symlinks=False)
            shutil.copy2(self.root / "station" / "station.py", ws / "station.py")
            (ws / "docs" / "task.md").write_text(self.task_path(task, task["doc"]).read_text())
            (ws / "docs" / "station.md").write_text(self.task_path(task, task["station_doc"]).read_text())
            for sub in ("candidates", "datasets", "runs"):
                (ws / sub).mkdir()
            self.copy_skills(ws)
            (ws / ".task.json").write_text(json.dumps(meta))
        elif json.loads((ws / ".task.json").read_text()) != meta:
            raise ValueError(f"workspace {ws} belongs to a different campaign configuration")
        return ws, meta
