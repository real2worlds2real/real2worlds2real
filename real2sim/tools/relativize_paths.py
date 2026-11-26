"""Rewrite stored absolute paths so sessions, outputs and assets can move between checkouts.

    python tools/relativize_paths.py [roots ...] [--source-root /old/checkout/real2sim] [--dry-run]

The scan defaults to sessions, outputs and assets. Pass the absolute POSIX path of the previous Real2Sim
directory as --source-root when moving from another checkout. The previous directory may be on another machine.
The tool recognises the current directory layout, historical checkout paths and the historical YCB asset layout.
JSON paths become relative to the Real2Sim directory. URDF mesh filenames become relative to their URDF file.
"""
import argparse
import os
import posixpath
import re
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
from calibrated_scene_reconstruction.core.paths import rel  # noqa: E402

QUOTED = re.compile(r'"((?:bodex:)?)(/[^"\n]*)"')
URDF_MESH = re.compile(r'(filename=")(/[^"]*)(")')
REPO_TAIL = re.compile(r"/r2s2r_datagen/(?:\.claude/worktrees/[^/]+/)?(.*)$")
YCB_TAIL = re.compile(r"/ocir-grasp-synthesis/data/raw_data/ycb/models/(.*)$")


def repo_relative(p, *, source_root=None):
    """The repository-relative form of an absolute path, or None when it is not a file of this repository."""
    absolute = Path(os.path.abspath(p))
    if absolute.is_relative_to(ROOT):
        return rel(absolute)
    if source_root is not None:
        recorded = PurePosixPath(posixpath.normpath(p))
        if recorded != source_root and recorded.is_relative_to(source_root):
            return recorded.relative_to(source_root).as_posix()
    m = REPO_TAIL.search(p)
    if m and m.group(1):
        return m.group(1)
    m = YCB_TAIL.search(p)
    if m:
        return "assets/ycb/models/" + m.group(1)
    return None


def fix_json(text, *, source_root=None):
    n = 0

    def sub(m):
        nonlocal n
        r = repo_relative(m.group(2), source_root=source_root)
        if r is None:
            return m.group(0)
        n += 1
        return f'"{m.group(1)}{r}"'

    return QUOTED.sub(sub, text), n


def fix_urdf(text, folder, *, source_root=None):
    n = 0

    def sub(m):
        nonlocal n
        r = repo_relative(m.group(2), source_root=source_root)
        if r is None:
            return m.group(0)
        n += 1
        return m.group(1) + Path(os.path.relpath(ROOT / r, folder)).as_posix() + m.group(3)

    return URDF_MESH.sub(sub, text), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="*", default=["sessions", "outputs", "assets"])
    ap.add_argument("--source-root", help="absolute POSIX path of the old Real2Sim directory in recorded paths")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    source_root = None
    if a.source_root is not None:
        source_root = PurePosixPath(posixpath.normpath(a.source_root))
        if not source_root.is_absolute():
            ap.error("--source-root must be an absolute POSIX path")
    files = paths = 0
    for r in a.roots:
        for f in sorted((ROOT / r).rglob("*")):
            if f.suffix not in (".json", ".urdf") or not f.is_file() or "/logs/" in f.as_posix():
                continue
            text = f.read_text(errors="surrogateescape")
            if "/" not in text:
                continue
            if f.suffix == ".json":
                new, n = fix_json(text, source_root=source_root)
            else:
                new, n = fix_urdf(text, f.resolve().parent, source_root=source_root)
            if n:
                files += 1
                paths += n
                if not a.dry_run:
                    f.write_text(new, errors="surrogateescape")
    print(f"{'would rewrite' if a.dry_run else 'rewrote'} {paths} paths in {files} files")


if __name__ == "__main__":
    main()
