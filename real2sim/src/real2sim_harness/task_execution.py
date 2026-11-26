"""Linux process boundary for generated code, shared by both model transports.

This module executes no model calls. System/runtime files and the current task
are visible; personal directories and transport credentials are not mounted.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import sys
import tempfile
from pathlib import Path

from .artifacts import atomic_json
from .project_inspection import SOURCE_DIRECTORIES, SOURCE_FILES

OUTPUT_DIRECTORIES = (
    "attempts",
    "environments",
    "scenes",
    "scene-assets",
    "behavior_tools",
    "reference_tools",
    "reference_geometry",
    "camera_solutions",
    "measurements",
    "cousins",
    "scene-cousins",
    "gaussian_tools",
    "codex/workspace",
    "workspace",
    ".tool-runtime",
)
SYSTEM_PATHS = (
    "/usr",
    "/bin",
    "/sbin",
    "/lib",
    "/lib64",
    "/sys",
    "/etc/ld.so.cache",
    "/etc/ld.so.conf",
    "/etc/ld.so.conf.d",
    "/etc/fonts",
    "/etc/localtime",
    "/etc/vulkan",
    "/etc/OpenCL",
    "/proc/driver/nvidia",
)
ENVIRONMENT_KEYS = (
    "LANG",
    "LC_ALL",
    "LD_LIBRARY_PATH",
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
)


def _runtime_paths(command=None):
    paths = {Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()}
    if command:
        executable = Path(command).resolve()
        paths.add(executable)
        # Blender's bundled Python, shared libraries and resources are runtime
        # dependencies, rather than a grant to its installation's entire parent.
        for sibling in executable.parent.iterdir():
            if sibling.is_file() and ".so" in sibling.name:
                paths.add(sibling)
            if sibling.is_dir() and (
                sibling.name in {"lib", "datafiles", "textures", "usd"}
                or sibling.name.replace(".", "").isdigit()
            ):
                paths.add(sibling.resolve())
    for name in ("PYTHONPATH", "LD_LIBRARY_PATH"):
        paths.update(Path(p).resolve() for p in os.environ.get(name, "").split(os.pathsep) if p)
    extra = json.loads(os.environ.get("REAL2SIM_TOOL_READ_PATHS", "[]"))
    if not isinstance(extra, list) or not all(isinstance(p, str) for p in extra):
        raise ValueError("REAL2SIM_TOOL_READ_PATHS must be a JSON list of installed runtime paths")
    paths.update(Path(p).resolve() for p in extra)
    home = Path.home().resolve()
    if any(p == Path("/") or p == home or home.is_relative_to(p) for p in paths):
        raise ValueError("Use specific installed runtime paths, not a home or filesystem root")
    return sorted(str(p) for p in paths if p.exists())


def policy_for(run, *, executable=None, writable=None, auth_dir=None):
    project = Path(__file__).resolve().parents[2]
    run = Path(run).resolve()
    runtime = run / ".tool-runtime"
    for directory in (runtime / "home", runtime / "cache", runtime / "tmp"):
        if not directory.resolve().is_relative_to(run):
            raise ValueError("Task temporary/cache paths must not follow links outside this task")
        directory.mkdir(parents=True, exist_ok=True)
    writes = (
        [run / name for name in OUTPUT_DIRECTORIES]
        if writable is None
        else [Path(p).resolve() for p in writable]
    )
    writes.append(runtime)
    if any(not p.is_relative_to(run) or p == run for p in writes):
        raise ValueError("Tool outputs must stay in declared directories inside this task")
    readable = [
        project / name
        for name in SOURCE_DIRECTORIES + SOURCE_FILES + ("vendor",)
        if (project / name).exists()
    ]
    readable += [Path(p) for p in _runtime_paths(executable)]
    environment = {key: os.environ[key] for key in ENVIRONMENT_KEYS if key in os.environ}
    environment.update(
        PATH=os.pathsep.join([str(Path(sys.executable).parent), "/usr/bin", "/bin"]),
        HOME=str(runtime / "home"),
        XDG_CACHE_HOME=str(runtime / "cache"),
        TMPDIR=str(runtime / "tmp"),
        PYTHONPYCACHEPREFIX=str(runtime / "cache/python"),
        PYTHONPATH=os.pathsep.join(
            [
                str(project / "src"),
                *filter(None, os.environ.get("PYTHONPATH", "").split(os.pathsep)),
            ]
        ),
    )
    return {
        "schema_version": 1,
        "run": str(run),
        "readable": [str(p) for p in readable],
        "writable": [str(p) for p in dict.fromkeys(writes)],
        "denied": [str(run / "codex/home"), *([str(Path(auth_dir).resolve())] if auth_dir else [])],
        "environment": environment,
    }


def _interpreter_links():
    """Symbolic links crossed on the way to this interpreter, as {link path: its target}.

    A managed interpreter is commonly reached through aliases: a virtual environment's ``python`` points into a
    ``cpython-3.12-...`` directory that is itself a link to ``cpython-3.12.x-...``. The policy binds the resolved
    installation, so inside the boundary those alias names must exist as well or the interpreter cannot start.
    Only the link names are reproduced; they expose no file beyond what the policy already grants.
    """
    links, pending = {}, [Path(os.path.abspath(sys.executable))]
    while pending:
        path = pending.pop()
        current = Path(path.anchor)
        for index, part in enumerate(path.parts[1:], start=2):
            current = current / part
            if str(current) in links or not current.is_symlink():
                continue
            target = os.readlink(current)
            links[str(current)] = target
            pending.append(Path(os.path.abspath(current.parent / target)).joinpath(*path.parts[index:]))
            break
    return links


def sandbox_command(command, policy, *, cwd=None):
    if sys.platform != "linux" or not (bwrap := shutil.which("bwrap")):
        raise RuntimeError(
            "Generated tool code requires Linux bubblewrap; install it before running this tool"
        )
    run = Path(policy["run"]).resolve()
    work = Path(cwd or run).resolve()
    if not work.is_relative_to(run):
        raise ValueError("Generated tool commands use a working directory inside this task")
    argv = [
        bwrap,
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--cap-drop",
        "ALL",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--tmpfs",
        "/dev/shm",
    ]
    paths = [Path(p) for p in SYSTEM_PATHS if Path(p).exists()]
    paths += [Path(p).resolve() for p in policy["readable"] if Path(p).exists()]
    paths.append(run)
    for path in sorted(set(paths), key=lambda p: (len(p.parts), str(p))):
        argv += ["--ro-bind", str(path), str(path)]
    bound = set(paths)
    for link, target in sorted(_interpreter_links().items()):
        if not any(Path(link) == p or Path(link).is_relative_to(p) for p in bound):
            argv += ["--symlink", target, link]
    for name in ("/dev/dri", *map(str, Path("/dev").glob("nvidia*"))):
        if Path(name).exists():
            argv += ["--dev-bind", name, name]
    for value in policy["writable"]:
        path = Path(value).resolve()
        if not path.is_relative_to(run) or path == run:
            raise ValueError("A tool policy cannot grant writes outside its declared task outputs")
        path.mkdir(parents=True, exist_ok=True)
        argv += ["--bind", str(path), str(path)]
    # Mask private transport state even though the rest of this task is readable.
    for value in policy["denied"]:
        path = Path(value)
        if path.is_dir():
            argv += ["--tmpfs", str(path)]
        elif path.is_file():
            argv += ["--ro-bind", "/dev/null", str(path)]
    for key, value in policy["environment"].items():
        argv += ["--setenv", key, value]
    # Clean the environment before bubblewrap itself starts, including its PID
    # namespace supervisor. This works on releases without --clearenv.
    env_program = shutil.which("env", path=os.defpath)
    if not env_program:
        raise RuntimeError("The tool boundary requires the standard env utility")
    return [
        env_program,
        "-i",
        *[key + "=" + value for key, value in policy["environment"].items()],
        *argv,
        "--chdir",
        str(work),
        "--",
        *map(str, command),
    ]


def blender_proxy(blender, run, *, auth_dir=None):
    """Return a task-bound executable used by all downstream Blender subprocesses."""
    if sys.platform != "linux" or not shutil.which("bwrap"):
        raise RuntimeError("Generated tool code requires Linux bubblewrap; install it before starting generation")
    run, blender = Path(run).resolve(), Path(blender).resolve()
    folder = run / "execution"
    folder.mkdir(parents=True, exist_ok=True)
    policy = policy_for(run, executable=blender, auth_dir=auth_dir)
    policy["blender"] = str(blender)
    atomic_json(folder / "blender-policy.json", policy)
    runner = Path(__file__).resolve().parents[2] / "scripts/run_task_blender.py"
    wrapper = folder / "blender"
    content = (
        "#!/bin/sh\nexec "
        + " ".join(
            shlex.quote(str(p)) for p in (sys.executable, runner, folder / "blender-policy.json")
        )
        + ' -- "$@"\n'
    )
    # A retained wrapper is intentionally read-only to generated code. Replace
    # it atomically on resume instead of opening that read-only file for writing.
    descriptor, temporary = tempfile.mkstemp(prefix=".blender-", dir=folder)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(content)
        temporary.chmod(0o555)
        temporary.replace(wrapper)
    finally:
        temporary.unlink(missing_ok=True)
    return wrapper


def run_blender_proxy(policy_path, arguments):
    policy = json.loads(Path(policy_path).read_text())
    if policy.get("schema_version") != 1:
        raise ValueError("Unsupported tool execution policy")
    command = sandbox_command([policy["blender"], "--disable-autoexec", *arguments], policy)
    # No inherited model credentials, personal home, host IPC or network.
    os.execve(command[0], command, policy["environment"])
