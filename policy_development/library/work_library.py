"""Persistent task experience for agent retrieval, not automatic execution rights.

Model-authored proposals remain drafts. Only trusted local worker integration
calls ``record_verified`` with a bounded, non-holdout evidence receipt. Receipts
preserve source identity; this Python API is a trust boundary, not cryptographic
proof that an arbitrary caller is a worker. Do not expose it as an agent tool.

Record contents and evidence receipts are immutable. Mutable quarantine/stale
state is append-only audited and propagates to descendants. Search is read-only,
deterministic and distinguishes exact contracts from ideas requiring adaptation.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time


BINDING_KEYS = ("task_id", "stage_id", "backend_id", "observation_contract",
                "action_contract", "success_protocol")
KINDS = ("reward", "tool", "recipe", "task_experience", "failure", "lesson")
LEVELS = ("source_verified", "offline_validation", "execution_failure", "development_evaluation")
HASH = re.compile(r"[0-9a-f]{64}\Z")
TAG = re.compile(r"[a-z0-9][a-z0-9_.:-]{0,79}\Z")
FORBIDDEN = re.compile(r"(?:^|_)(?:final|holdout|heldout|held_out|test_set|test_success)(?:_|$)", re.I)
SECRET = re.compile(r"^(?:api_key|apikey|authorization|password|cookie|secret|credential|access_token|refresh_token|private_key)$", re.I)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def library_root(root=None):
    return Path(root if root is not None else os.environ.get(
        "ASTRA_WORK_LIBRARY_ROOT", str(Path.home()/"astra_harness_v1"/"work_library"))).resolve()


def profile_tags(task_id, stage_id=None):
    """Retrieval hints only: shared family tags never certify contract parity."""
    base = str(task_id).split("@", 1)[0]
    tags = {
        "peg_insert": ["family:insertion", "contact:rigid", "skill:alignment", "skill:insertion"],
        "gear_insert": ["family:insertion", "contact:rigid", "skill:alignment", "skill:insertion", "geometry:gear"],
        "cup_nesting": ["family:stacking", "contact:rigid", "skill:grasp", "skill:alignment", "skill:release"],
        "tissue_open_extract_wipe": ["family:container_manipulation", "contact:deformable", "skill:bimanual", "skill:wiping"],
        "cap_tighten": ["family:threaded_assembly", "contact:rigid", "skill:rotation", "skill:bimanual"],
    }.get(base, [])
    tags = list(tags)
    if stage_id is not None and isinstance(stage_id, str) and TAG.fullmatch("stage:"+stage_id):
        tags.append("stage:"+stage_id)
    return sorted(set(tags))


def _keys(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(label + " fields do not match")


def _text(value, label, maximum=2000):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(label + " must be a bounded nonempty string")
    return value


def _hash(value, label="library identity"):
    if not isinstance(value, str) or not HASH.fullmatch(value):
        raise ValueError(label + " must be a SHA-256 identity")
    return value


def _bounded(value, depth=0):
    if depth > 12:
        raise ValueError("library JSON exceeds nesting limit")
    if value is None or type(value) is bool:
        return
    if type(value) in (int, float):
        if abs(value) > 1e100 or not math.isfinite(value):
            raise ValueError("library numbers must be finite and bounded")
        return
    if isinstance(value, str):
        if len(value) > 16000:
            raise ValueError("library strings must be bounded")
        return
    if isinstance(value, list):
        if len(value) > 256:
            raise ValueError("library arrays must be bounded")
        for child in value:
            _bounded(child, depth+1)
        return
    if isinstance(value, dict):
        if len(value) > 128:
            raise ValueError("library objects must be bounded")
        for key, child in value.items():
            if not isinstance(key, str) or not key or len(key) > 160:
                raise ValueError("invalid library field name")
            if SECRET.fullmatch(key):
                raise ValueError("credential fields cannot enter the work library")
            if FORBIDDEN.search(key):
                raise ValueError("final/holdout information cannot enter the reusable library")
            if key in ("split", "evaluation_split") and child not in ("none", "train", "development", "dev", "offline"):
                raise ValueError("final/holdout splits cannot enter the reusable library")
            _bounded(child, depth+1)
        return
    raise ValueError("library records must contain bounded JSON values")


def validate_record(record):
    _keys(record, ("kind", "title", "binding", "tags", "summary", "payload", "parents"), "library record")
    if record["kind"] not in KINDS:
        raise ValueError("unregistered library record kind")
    _text(record["title"], "record title", 240)
    _text(record["summary"], "record summary", 6000)
    _keys(record["binding"], BINDING_KEYS, "record binding")
    for key, value in record["binding"].items():
        _text(value, key)
    tags = record["tags"]
    if not isinstance(tags, list) or len(tags) > 32 or any(not isinstance(t, str) or not TAG.fullmatch(t) for t in tags):
        raise ValueError("record tags must be up to 32 normalized identifiers")
    parents = record["parents"]
    if not isinstance(parents, list) or len(parents) > 32 or any(not isinstance(p, str) for p in parents) or len(set(parents)) != len(parents):
        raise ValueError("record lineage needs unique parent identities")
    for parent in parents:
        _hash(parent, "parent identity")
    if not isinstance(record["payload"], dict):
        raise ValueError("record payload must be an object")
    _bounded(record)
    normalized = deepcopy(record)
    normalized["tags"] = sorted(set(tags))
    normalized["parents"] = sorted(parents)
    if len(canonical(normalized).encode()) > 98304:
        raise ValueError("library record exceeds 96 KiB")
    return normalized


def validate_evidence(evidence):
    _keys(evidence, ("level", "source_ref", "source_sha256", "split", "metrics"), "evidence receipt")
    if evidence["level"] not in LEVELS:
        raise ValueError("unknown evidence level")
    _text(evidence["source_ref"], "evidence source reference", 2000)
    _hash(evidence["source_sha256"], "evidence source hash")
    if evidence["split"] not in ("none", "development"):
        raise ValueError("only non-holdout evidence can enter the work library")
    if not isinstance(evidence["metrics"], dict):
        raise ValueError("evidence metrics must be an object")
    if evidence["level"] == "development_evaluation":
        if evidence["split"] != "development":
            raise ValueError("development evaluation requires the development split")
        metrics = evidence["metrics"]
        _keys(metrics, ("episodes", "successes", "failure_counts"), "development metrics")
        episodes, successes = metrics["episodes"], metrics["successes"]
        if type(episodes) is not int or type(successes) is not int or not 1 <= episodes <= 10**7 or not 0 <= successes <= episodes:
            raise ValueError("invalid development evaluation denominator")
        failures = metrics["failure_counts"]
        if not isinstance(failures, dict) or any(not isinstance(k, str) or not k or type(v) is not int or v < 0 for k, v in failures.items()) or sum(failures.values()) != episodes-successes:
            raise ValueError("development evidence must retain every failed episode")
    elif evidence["split"] == "development" and evidence["level"] not in ("execution_failure",):
        raise ValueError("source/offline checks cannot claim a development evaluation")
    _bounded(evidence)
    if len(canonical(evidence).encode()) > 32768:
        raise ValueError("evidence receipt exceeds 32 KiB")
    return deepcopy(evidence)


@contextmanager
def _database(root=None, *, write=False):
    directory = library_root(root)
    path = directory/"library.sqlite3"
    if write:
        directory.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(path, timeout=30)
    elif path.is_file():
        db = sqlite3.connect(path.as_uri()+"?mode=ro", uri=True, timeout=30)
    else:
        yield None
        return
    db.row_factory = sqlite3.Row
    try:
        if write:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY,content TEXT NOT NULL,created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS parents(child TEXT NOT NULL,parent TEXT NOT NULL,PRIMARY KEY(child,parent));
                CREATE TABLE IF NOT EXISTS evidence(id TEXT PRIMARY KEY,record_id TEXT NOT NULL,receipt TEXT NOT NULL,
                    source_ref TEXT NOT NULL,source_sha TEXT NOT NULL,created REAL NOT NULL,
                    UNIQUE(record_id,source_ref,source_sha));
                CREATE TABLE IF NOT EXISTS invalidations(id INTEGER PRIMARY KEY,record_id TEXT NOT NULL,
                    status TEXT NOT NULL,reason TEXT NOT NULL,cause TEXT NOT NULL,created REAL NOT NULL);
            """)
            db.execute("BEGIN IMMEDIATE")
        yield db
        if write:
            db.commit()
    except BaseException:
        if write:
            db.rollback()
        raise
    finally:
        db.close()


def _state(db, identifier):
    events = db.execute("SELECT status,reason,cause FROM invalidations WHERE record_id=? ORDER BY id", (identifier,)).fetchall()
    if events:
        level = "quarantined" if any(r["status"] == "quarantined" for r in events) else "stale"
        return level, [dict(row) for row in events]
    known = db.execute("SELECT id FROM evidence WHERE record_id=? LIMIT 1", (identifier,)).fetchone()
    return ("verified" if known else "draft"), []


def _entry(db, row):
    record = json.loads(row["content"])
    if digest(record) != row["id"]:
        raise ValueError("work library record integrity check failed")
    receipts = []
    for ev in db.execute("SELECT id,receipt FROM evidence WHERE record_id=? ORDER BY created,id", (row["id"],)):
        receipt = json.loads(ev["receipt"])
        if digest({"record_id": row["id"], "receipt": receipt}) != ev["id"]:
            raise ValueError("work library evidence integrity check failed")
        receipts.append({"id": ev["id"], **receipt})
    state, reasons = _state(db, row["id"])
    evidence_level = max((x["level"] for x in receipts), key=LEVELS.index, default="agent_hypothesis")
    return {"id": row["id"], "status": state, "record": record, "evidence": receipts,
            "evidence_level": evidence_level, "invalidations": reasons, "execution_authorized": False}


def _insert(db, record):
    identity = digest(record)
    for parent in record["parents"]:
        if not db.execute("SELECT id FROM records WHERE id=?", (parent,)).fetchone():
            raise ValueError("library parent is not registered")
    db.execute("INSERT OR IGNORE INTO records VALUES(?,?,?)", (identity, canonical(record), time.time()))
    for parent in record["parents"]:
        db.execute("INSERT OR IGNORE INTO parents VALUES(?,?)", (identity, parent))
        state, reasons = _state(db, parent)
        if state in ("quarantined", "stale"):
            for item in reasons:
                if not db.execute("SELECT id FROM invalidations WHERE record_id=? AND status=? AND cause=?",
                                  (identity, item["status"], item["cause"])).fetchone():
                    db.execute("INSERT INTO invalidations(record_id,status,reason,cause,created) VALUES(?,?,?,?,?)",
                               (identity, item["status"], "Inherited dependency invalidation: "+item["reason"], item["cause"], time.time()))
    return identity


def propose(record, *, root=None):
    """Store model-authored knowledge without granting it worker evidence."""
    record = validate_record(record)
    with _database(root, write=True) as db:
        identity = _insert(db, record)
        result = _entry(db, db.execute("SELECT * FROM records WHERE id=?", (identity,)).fetchone())
    return result


def record_verified(record, evidence, *, root=None):
    """Trusted worker integration only; public agent tools must call propose."""
    record, evidence = validate_record(record), validate_evidence(evidence)
    with _database(root, write=True) as db:
        identity = _insert(db, record)
        eid = digest({"record_id": identity, "receipt": evidence})
        previous = db.execute("SELECT id FROM evidence WHERE record_id=? AND source_ref=? AND source_sha=?",
                              (identity, evidence["source_ref"], evidence["source_sha256"])).fetchone()
        if previous and previous["id"] != eid:
            raise ValueError("same evidence source cannot be rewritten with different metrics")
        db.execute("INSERT OR IGNORE INTO evidence VALUES(?,?,?,?,?,?)",
                   (eid, identity, canonical(evidence), evidence["source_ref"], evidence["source_sha256"], time.time()))
        result = _entry(db, db.execute("SELECT * FROM records WHERE id=?", (identity,)).fetchone())
    return result


def get(identifier, *, root=None):
    _hash(identifier)
    with _database(root) as db:
        row = db.execute("SELECT * FROM records WHERE id=?", (identifier,)).fetchone() if db else None
        if row is None:
            raise ValueError("unknown work library record")
        return _entry(db, row)


def validate_query(query):
    if not isinstance(query, dict) or set(query)-{"binding", "tags", "kinds", "text", "include_drafts"}:
        raise ValueError("unknown library query field")
    result = {"binding": {}, "tags": [], "kinds": [], "text": "", "include_drafts": False, **deepcopy(query)}
    binding = result["binding"]
    if not isinstance(binding, dict) or set(binding)-set(BINDING_KEYS):
        raise ValueError("unknown query binding field")
    for key, value in binding.items():
        _text(value, key)
    if not isinstance(result["tags"], list) or len(result["tags"]) > 32 or any(not isinstance(t, str) or not TAG.fullmatch(t) for t in result["tags"]):
        raise ValueError("invalid query tags")
    if not isinstance(result["kinds"], list) or len(result["kinds"]) > len(KINDS) or any(k not in KINDS for k in result["kinds"]):
        raise ValueError("invalid query kinds")
    if not isinstance(result["text"], str) or len(result["text"]) > 1000 or type(result["include_drafts"]) is not bool:
        raise ValueError("invalid query text or draft flag")
    result["tags"] = sorted(set(result["tags"]))
    result["kinds"] = sorted(set(result["kinds"]))
    return result


def _match(entry, query):
    record, binding = entry["record"], query["binding"]
    if query["kinds"] and record["kind"] not in query["kinds"]:
        return None
    matches = [key for key, value in binding.items() if record["binding"][key] == value]
    differences = [key for key, value in binding.items() if record["binding"][key] != value]
    overlap = sorted(set(record["tags"]) & set(query["tags"]))
    text = " ".join([record["title"], record["summary"], *record["tags"]]).casefold()
    tokens = [x for x in re.split(r"\s+", query["text"].casefold()) if x]
    token_matches = sum(token in text for token in tokens)
    same_task = "task_id" in matches
    relevance = same_task or overlap or token_matches
    if (binding or query["tags"] or tokens) and not relevance:
        return None
    complete = set(binding) == set(BINDING_KEYS)
    if complete and not differences:
        compatibility = "exact_contract"
    elif same_task or overlap:
        compatibility = "adaptation_required"
    else:
        compatibility = "inspiration_only"
    # Shared schemas alone are not semantic task similarity. Exact applicability
    # also never overrides a runtime adapter's current qualification checks.
    score = (1000 if compatibility == "exact_contract" else 0) + (200 if same_task else 0)
    score += 10*len(overlap)+3*len(matches)+token_matches
    score += (LEVELS.index(entry["evidence_level"])+1 if entry["evidence_level"] in LEVELS else 0)
    return {**entry, "compatibility": compatibility, "differences": differences,
            "matched_tags": overlap, "score": score,
            "reuse_rule": "Agent must choose and the current task runtime must validate; memory does not authorize execution."}


def search(query, limit=8, *, root=None):
    query = validate_query(query)
    if type(limit) is not int or not 1 <= limit <= 25:
        raise ValueError("library search limit must be 1..25")
    found = []
    with _database(root) as db:
        if db:
            for row in db.execute("SELECT * FROM records ORDER BY id"):
                entry = _entry(db, row)
                if entry["status"] in ("quarantined", "stale") or (entry["status"] == "draft" and not query["include_drafts"]):
                    continue
                match = _match(entry, query)
                if match is not None:
                    found.append(match)
    found.sort(key=lambda x: (-x["score"], x["id"]))
    return {"items": found[:limit], "query": query, "limit": limit, "total_matches": len(found)}


def _invalidate(identifier, reason, state, root):
    _hash(identifier)
    _text(reason, "invalidation reason", 2000)
    with _database(root, write=True) as db:
        if not db.execute("SELECT id FROM records WHERE id=?", (identifier,)).fetchone():
            raise ValueError("unknown work library record")
        descendants = {identifier}
        frontier = [identifier]
        while frontier:
            parent = frontier.pop()
            for row in db.execute("SELECT child FROM parents WHERE parent=?", (parent,)):
                if row[0] not in descendants:
                    descendants.add(row[0])
                    frontier.append(row[0])
        for child in sorted(descendants):
            if not db.execute("SELECT id FROM invalidations WHERE record_id=? AND status=? AND cause=? AND reason=?",
                              (child, state, identifier, reason)).fetchone():
                db.execute("INSERT INTO invalidations(record_id,status,reason,cause,created) VALUES(?,?,?,?,?)",
                           (child, state, reason, identifier, time.time()))
    return {"id": identifier, "status": state, "affected_ids": sorted(descendants), "reason": reason}


def quarantine(identifier, reason, *, root=None):
    return _invalidate(identifier, reason, "quarantined", root)


def mark_stale(identifier, reason, *, root=None):
    return _invalidate(identifier, reason, "stale", root)


def status(*, root=None):
    counts = {key: 0 for key in ("draft", "verified", "quarantined", "stale")}
    kinds = {key: 0 for key in KINDS}
    evidence_count = 0
    with _database(root) as db:
        if db:
            for row in db.execute("SELECT * FROM records"):
                entry = _entry(db, row)
                counts[entry["status"]] += 1
                kinds[entry["record"]["kind"]] += 1
            evidence_count = db.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
    return {"records": sum(counts.values()), "states": counts, "kinds": kinds,
            "evidence_receipts": evidence_count, "scope": "retrieval_memory_not_execution_authority"}
