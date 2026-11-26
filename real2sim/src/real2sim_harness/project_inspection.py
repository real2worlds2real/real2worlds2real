"""Portable, task-scoped read access for native agent investigation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from .artifacts import atomic_json
from .codex_transport import toml_literal

SOURCE_DIRECTORIES = (
    "src",
    "scripts",
    "skills",
    "docs",
    "tests",
    "examples",
    "requirements",
    "benchmarks",
)
SOURCE_FILES = ("README.md", "THIRD_PARTY.md", "pyproject.toml")


def configure(argv, config, *, provider, project, run, workspace, auth_dir=None):
    project, run, workspace = (Path(p).resolve() for p in (project, run, workspace))
    if workspace == run or not workspace.is_relative_to(run):
        raise ValueError("Use a separate writable workspace inside this task's records")
    key = f"permissions.{provider}.filesystem"
    filesystem = dict(config.get(key, {}))
    # A readonly base avoids implicit host /tmp writes and the redundant denied
    # /tmp ancestor that can prevent the CLI from mounting nested task grants.
    filesystem.pop(":tmpdir", None)
    filesystem.pop(":slash_tmp", None)
    roots = dict(filesystem.get(":workspace_roots", {}))
    roots.update(
        {".": "write", ".git": "read", ".agents": "read", ".codex": "read", ".aws": "read"}
    )
    filesystem[":workspace_roots"] = roots
    readable = [
        project / name for name in SOURCE_DIRECTORIES + SOURCE_FILES if (project / name).exists()
    ]
    if any(not path.resolve().is_relative_to(project) for path in readable):
        raise ValueError("Project source entries must stay inside the selected checkout")
    for path in readable:
        filesystem[str(path)] = "read"
    filesystem[str(run)] = "read"
    filesystem[str(workspace)] = "write"
    filesystem[str(run / "codex/home")] = "deny"
    if auth_dir:
        auth = Path(auth_dir).resolve()
        if auth.is_relative_to(run) or run.is_relative_to(auth):
            raise ValueError("Keep authorization storage separate from task evidence and workspace")
        if any(
            path.resolve() == auth
            or path.resolve().is_relative_to(auth)
            or auth.is_relative_to(path.resolve())
            for path in readable
        ):
            raise ValueError("Keep authorization storage outside readable project source")
        filesystem[str(auth)] = "deny"
    environment = dict(config.get("shell_environment_policy.set", {}))
    analysis = workspace / ".analysis"
    for name in ("home", "cache", "tmp"):
        (analysis / name).mkdir(parents=True, exist_ok=True)
    environment.update(
        HOME=str(analysis / "home"),
        XDG_CACHE_HOME=str(analysis / "cache"),
        TMPDIR=str(analysis / "tmp"),
        PYTHONPYCACHEPREFIX=str(analysis / "cache/python"),
    )
    imports = [str(project / "src")] if (project / "src").is_dir() else []
    imports += environment.get("PYTHONPATH", "").split(os.pathsep)
    environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(path for path in imports if path))
    overrides = {
        key: filesystem,
        f"permissions.{provider}.extends": ":read-only",
        "shell_environment_policy.set": environment,
    }
    updated = {**config, **overrides}
    command = list(argv)
    for option, value in overrides.items():
        argument = option + "=" + toml_literal(value)
        for index in range(1, len(command)):
            if command[index - 1] == "-c" and command[index].startswith(option + "="):
                command[index] = argument
                break
        else:
            command.extend(["-c", argument])
    atomic_json(
        workspace / "project_access.json",
        {
            "schema_version": 1,
            "project_source_read_paths": [str(p) for p in readable],
            "task_records_read_root": str(run),
            "workspace_write_root": str(workspace),
            "analysis_environment": {
                "python_executable": sys.executable,
                "project_import_path": str(project / "src"),
                "temporary_root": str(analysis / "tmp"),
                "cache_root": str(analysis / "cache"),
            },
            "purpose": "Optional on-demand source/log/parameter/evidence investigation; choose what to inspect.",
            "excluded": "Authorization stores and agent transport home are not task evidence. Other user files are not granted by this interface.",
        },
    )
    return command, updated
