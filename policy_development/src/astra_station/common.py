"""Small shared helpers: JSON files, hashes, a hash-linked event ledger and a JSON-schema subset validator (stdlib only)."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path


def digest(value):
    """sha256 of the canonical JSON of a value (the same definition the end-to-end oversight records use)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value) -> None:
    """Atomic replace (state files)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_new(path, value) -> None:
    """Immutable record: refuses to overwrite."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as fh:
        fh.write(json.dumps(value, indent=1, ensure_ascii=False, allow_nan=False) + "\n")


def tree_hash(root) -> str:
    """sha256 over relative paths and bytes of every file below root (caches excluded)."""
    root = Path(root)
    h = hashlib.sha256()
    for f in sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        h.update(str(f.relative_to(root)).encode())
        h.update(file_hash(f).encode())
    return h.hexdigest()


class Ledger:
    """Append-only JSONL; every row carries the hash of the previous row, so edits and deletions are detectable."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def rows(self) -> list:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def append(self, event: str, **details) -> dict:
        rows = self.rows()
        prev = rows[-1]["sha256"] if rows else None
        row = {"n": len(rows) + 1, "time": now(), "event": event, "prev": prev, "details": details}
        row["sha256"] = digest(row)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        return row

    def verify(self) -> None:
        prev = None
        for row in self.rows():
            body = {k: v for k, v in row.items() if k != "sha256"}
            if row.get("prev") != prev or digest(body) != row.get("sha256"):
                raise ValueError(f"ledger {self.path} broken at row {row.get('n')}")
            prev = row["sha256"]


_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}


def validate(value, schema, path="") -> None:
    """The JSON-schema subset the tool manifests use: type, properties, required, additionalProperties, enum, const,
    minimum/maximum, minLength/maxLength, pattern, items, minItems/maxItems, anyOf.  Raises ValueError."""
    import re
    if "anyOf" in schema:
        errs = []
        for sub in schema["anyOf"]:
            try:
                validate(value, sub, path)
                break
            except ValueError as e:
                errs.append(str(e))
        else:
            raise ValueError(f"{path or '/'}: matches none of anyOf ({'; '.join(errs)})")
    t = schema.get("type")
    if t is not None:
        if t == "integer":
            ok = isinstance(value, int) and not isinstance(value, bool)
        elif t == "number":
            ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        else:
            ok = isinstance(value, _TYPES[t])
        if not ok:
            raise ValueError(f"{path or '/'}: expected {t}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path or '/'}: must be one of {schema['enum']}")
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path or '/'}: must be {schema['const']!r}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ValueError(f"{path or '/'}: below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValueError(f"{path or '/'}: above maximum {schema['maximum']}")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", 10 ** 9):
            raise ValueError(f"{path or '/'}: length out of range")
        if "pattern" in schema and not re.fullmatch(schema["pattern"], value):
            raise ValueError(f"{path or '/'}: does not match {schema['pattern']}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", 10 ** 9):
            raise ValueError(f"{path or '/'}: item count out of range")
        if "items" in schema:
            for i, item in enumerate(value):
                validate(item, schema["items"], f"{path}/{i}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                raise ValueError(f"{path or '/'}: missing {key}")
        extra = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in props:
                validate(item, props[key], f"{path}/{key}")
            elif extra is False:
                raise ValueError(f"{path or '/'}: unexpected property {key}")
            elif isinstance(extra, dict):
                validate(item, extra, f"{path}/{key}")
