"""Stage an unfrozen, isolated derivative of an existing scene without approving it.

The session basename is retained in a separate code checkout so that recorded
``sessions/<name>`` references retain their meaning. This copies assets, never
hardlinks them. An inherited freeze is archived as provenance, never installed
as the derivative's active freeze. Missing inherited files remain explicit.

Run with --max-bytes to inspect a copy plan; add --execute to stage the copy.
Native semantic review, physical checks, exports and a new freeze are separate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat


FREEZE = "qa/independent_freeze.json"
PROVENANCE = "provenance/derivative_restore"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _inventory(root: Path, repository: Path) -> list[dict]:
    """Resolve internal links, rejecting external links, cycles and special files."""
    rows = []

    def visit(path: Path, relative: str, ancestors: frozenset[Path]) -> None:
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(repository):
            raise ValueError(f"Asset escapes source repository: {path}")
        mode = resolved.stat().st_mode
        if stat.S_ISDIR(mode):
            if resolved in ancestors:
                raise ValueError(f"Cyclic directory link: {path}")
            for child in sorted(path.iterdir()):
                if child.name == "__pycache__" or child.suffix == ".pyc":
                    continue
                visit(child, f"{relative}/{child.name}".lstrip("/"), ancestors | {resolved})
        elif stat.S_ISREG(mode):
            info = resolved.stat()
            rows.append({"path": relative, "resolved_source": str(resolved),
                         "bytes": info.st_size, "mtime_ns": info.st_mtime_ns})
        else:
            raise ValueError(f"Unsupported non-regular asset: {path}")

    visit(root, "", frozenset())
    return rows


def _roots(source_repo: Path, destination_repo: Path, session: str) -> tuple[Path, Path]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", session) or session in {".", ".."}:
        raise ValueError("A single session basename is required")
    source_repo, destination_repo = source_repo.resolve(), destination_repo.resolve()
    if (source_repo.is_relative_to(destination_repo)
            or destination_repo.is_relative_to(source_repo)):
        raise ValueError("Use separate, non-overlapping source and destination checkouts")
    for root in (source_repo, destination_repo):
        if not (root / "src/calibrated_scene_reconstruction/scene/session.py").is_file():
            raise ValueError(f"Expected an existing scene repository checkout: {root}")
    return source_repo, destination_repo


def _destination_check(plan: dict) -> None:
    destination_repo = Path(plan["destination_repo"])
    session = plan["session"]
    destination = destination_repo / "sessions" / session
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"Derivative session must not already exist: {destination}")
    for name in ("sessions", "configs"):
        parent = destination_repo / name
        if parent.exists() and (parent.is_symlink() or parent.resolve() != parent):
            raise ValueError(f"Destination asset parent must be a real directory: {parent}")
    config = destination_repo / "configs" / session
    if config.is_symlink():
        raise ValueError("Destination configuration must not be a symlink")
    if config.exists():
        actual = _inventory(config, destination_repo)
        expected = {r["path"]: r["sha256"] for r in plan["files"] if r["base"] == "config"}
        if {r["path"]: digest(Path(r["resolved_source"])) for r in actual} != expected:
            raise ValueError("Existing destination configuration differs; preserve it and use another checkout")


def _freeze_audit(source_repo: Path, session: str, rows: list[dict]) -> dict | None:
    source = source_repo / "sessions" / session / FREEZE
    if not source.is_file():
        return None
    saved = json.loads(source.read_text(encoding="utf-8"))
    inventory = {(r["base"], r["path"]): r for r in rows}
    audit = []
    for row in saved["files"]:
        base, raw = row["base"], row["path"]
        relative = PurePosixPath(raw.replace("\\", "/"))
        if base not in {"session", "config"} or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid inherited freeze path")
        current = inventory.get((base, relative.as_posix()))
        verdict = "missing" if current is None else "match" if current["sha256"] == row["sha256"] else "changed"
        audit.append({"base": base, "path": raw, "status": verdict,
                      "expected_sha256": row["sha256"],
                      "actual_sha256": current["sha256"] if current else None})
    return {"sha256": digest(source), "frozen_at": saved.get("frozen_at"),
            "code": saved.get("code"), "files": audit,
            "counts": {status: sum(r["status"] == status for r in audit)
                       for status in ("match", "missing", "changed")}}


def plan_restore(source_repo: Path, destination_repo: Path, session: str, *, max_bytes: int) -> dict:
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes <= 0:
        raise ValueError("An explicit positive byte budget is required")
    source_repo, destination_repo = _roots(Path(source_repo), Path(destination_repo), session)
    rows = []
    for base, directory in (("session", "sessions"), ("config", "configs")):
        root = source_repo / directory / session
        if not root.is_dir():
            raise ValueError(f"Source {base} directory is missing: {root}")
        for row in _inventory(root, source_repo):
            row["base"] = base
            rows.append(row)
    if any(r["base"] == "session" and r["path"].startswith(PROVENANCE + "/") for r in rows):
        raise ValueError("Source is already a staged derivative; preserve its lineage with a new explicit plan")
    total = sum(row["bytes"] for row in rows)
    if total > max_bytes:
        largest = sorted(rows, key=lambda r: r["bytes"], reverse=True)[:8]
        raise ValueError(f"Copy requires {total} bytes, exceeds {max_bytes}; largest assets: "
                         + json.dumps([{k: r[k] for k in ("base", "path", "bytes")} for r in largest]))
    for row in rows:
        row["sha256"] = digest(Path(row["resolved_source"]))
    result = {"version": 1, "status": "planned_requires_native_review", "session": session,
              "source_repo": str(source_repo), "destination_repo": str(destination_repo),
              "max_bytes": max_bytes, "bytes": total, "files": rows,
              "inherited_freeze": _freeze_audit(source_repo, session, rows),
              "scope": "Asset restoration only; inherited reviews are historical evidence, not a new approval. "
                       "No geometry, policy, success criteria, or semantic/physical verdicts are modified."}
    _destination_check(result)
    return result


def execute_restore(plan: dict) -> dict:
    """Copy a current plan with preserved mtimes; never rewrite the source."""
    source_repo, destination_repo = _roots(Path(plan["source_repo"]), Path(plan["destination_repo"]), plan["session"])
    # Replanning catches new files, changed links, files, or configuration before any writes.
    current = plan_restore(source_repo, destination_repo, plan["session"], max_bytes=plan["max_bytes"])
    if current != plan:
        raise ValueError("Source or restore plan changed; inspect a new plan before execution")
    session_dir = destination_repo / "sessions" / plan["session"]
    config_dir = destination_repo / "configs" / plan["session"]
    copy_config = not config_dir.exists()
    session_dir.mkdir(parents=True, exist_ok=False)
    manifest = session_dir / PROVENANCE / "manifest.json"
    manifest.parent.mkdir(parents=True)
    record = dict(plan, status="copy_incomplete_requires_native_review",
                  staged_at=datetime.now(timezone.utc).isoformat())
    manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    for row in plan["files"]:
        if row["base"] == "config" and not copy_config:
            continue
        root = session_dir if row["base"] == "session" else config_dir
        target = root / row["path"]
        if row["base"] == "session" and row["path"] == FREEZE:
            target = session_dir / PROVENANCE / "inherited_independent_freeze.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise ValueError(f"Refusing to overwrite an asset during restoration: {target}")
        source = Path(row["resolved_source"])
        # copy2 preserves review chronology without linking later edits to the source.
        shutil.copy2(source, target)
        if digest(target) != row["sha256"] or digest(source) != row["sha256"]:
            raise ValueError(f"Asset changed during copy; partial derivative retained for diagnosis: {source}")
    record["status"] = "staged_requires_native_review"
    manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return {"status": record["status"], "manifest": str(manifest),
            "files": len(plan["files"]), "bytes": plan["bytes"],
            "active_freeze_installed": False, "qualified": False,
            "inherited_freeze_counts": (plan["inherited_freeze"] or {}).get("counts")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_repo", type=Path)
    parser.add_argument("destination_repo", type=Path)
    parser.add_argument("session")
    parser.add_argument("--max-bytes", type=int, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--report", type=Path, help="Optional new plan/report JSON; never overwrite")
    args = parser.parse_args()
    if args.report and args.report.resolve().is_relative_to(args.source_repo.resolve()):
        parser.error("Report must not modify the source checkout")
    plan = plan_restore(args.source_repo, args.destination_repo, args.session, max_bytes=args.max_bytes)
    result = execute_restore(plan) if args.execute else plan
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
    if not args.execute:
        result = {key: value for key, value in plan.items() if key != "files"}
        result["files"] = len(plan["files"])
        if result["inherited_freeze"]:
            result["inherited_freeze"] = {key: value for key, value in result["inherited_freeze"].items() if key != "files"}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
