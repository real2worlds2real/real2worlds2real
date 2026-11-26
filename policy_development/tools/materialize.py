"""Materialise any candidate as a station policy-module directory (programs are wrapped by program_policy.py with the
task's identified arm model), so candidates of every kind can be stages, residual bases or policy-worker deliverables."""
import shutil
from pathlib import Path

from _adapter import RUNTIME, STATION_CLIENT


def kin_dir(job, dest: Path) -> Path:
    src = job.kinematics_dir()
    if src is None:
        raise RuntimeError(f"task {job.binding['task_id']} has no arm model (task spec station.kinematics_model); "
                           "programs cannot run behind the policy interface on it")
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "kin.npz", dest / "kin.npz")                     # the task's identified parameters
    shutil.copy2(RUNTIME / "poe_kinematics.py", dest / "kin.py")         # the generic PoE code that evaluates them
    shutil.copy2(RUNTIME / "kinematics.py", dest / "kinematics.py")
    return dest


def as_policy(job, cand: dict, dest: Path) -> Path:
    src = Path(cand["path"])
    if cand["kind"] == "policy":
        shutil.copytree(src, dest, ignore=shutil.ignore_patterns("__pycache__"))
        if cand["entry"] != "policy.py":
            shutil.copy2(src / cand["entry"], dest / "policy.py")
        return dest
    dest.mkdir(parents=True)
    shutil.copy2(src / cand["entry"], dest / "program.py")
    shutil.copy2(RUNTIME / "program_policy.py", dest / "policy.py")
    shutil.copy2(STATION_CLIENT / "program_runtime.py", dest / "program_runtime.py")
    kin_dir(job, dest)
    return dest
