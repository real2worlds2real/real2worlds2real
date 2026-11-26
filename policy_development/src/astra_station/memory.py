"""Experience and reward libraries with tag-based retrieval.

Experience cards (`astra_experience_card_v1`) are verified laboratory records: what was tried on which scene / objects /
phases, what happened in simulation and on the real robot, and when it applies.  Reward terms (`astra_reward_term_v1`) are
reusable shaping components, each tied to the phase it rewards and the tasks it was validated on, so a new task can borrow
the terms of tasks that share its phases.  Retrieval ranks by overlap with the task's tags; similar experience is a
hypothesis to re-validate in the current campaign, never a result.
"""
from __future__ import annotations

import re
from pathlib import Path

from .common import read, validate, write_new

CARD_SCHEMA = {
    "type": "object",
    "required": ["schema", "id", "title", "tags", "finding", "evidence", "applicability", "limitations", "source", "status"],
    "properties": {
        "schema": {"const": "astra_experience_card_v1"},
        "id": {"type": "string", "pattern": "[a-z0-9_\\-]+"},
        "title": {"type": "string", "minLength": 5},
        "tags": {"type": "object", "required": ["scene", "objects", "phases", "methods", "domain"],
                 "properties": {"domain": {"type": "array", "items": {"enum": ["sim", "real"]}}}},
        "finding": {"type": "string", "minLength": 20},
        "evidence": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "applicability": {"type": "string"}, "limitations": {"type": "string"},
        "source": {"type": "string"}, "status": {"enum": ["verified", "candidate"]},
        "date": {"type": "string"},
    },
}

REWARD_SCHEMA = {
    "type": "object",
    "required": ["schema", "id", "phase", "description", "inputs", "formula", "validated_on", "cautions"],
    "properties": {
        "schema": {"const": "astra_reward_term_v1"},
        "id": {"type": "string", "pattern": "[a-z0-9_]+"},
        "phase": {"type": "string"}, "description": {"type": "string"},
        "inputs": {"type": "array", "items": {"type": "string"}},
        "formula": {"type": "string"},
        "validated_on": {"type": "array", "items": {"type": "object", "required": ["task", "result"]}},
        "cautions": {"type": "string"},
        "implementation": {"type": "string"},
    },
}


def _load(d: Path, schema: dict) -> list:
    out = []
    for p in sorted(d.glob("*.json")) if d.is_dir() else []:
        row = read(p)
        validate(row, schema)
        out.append({**row, "_path": str(p)})
    return out


class Library:
    def __init__(self, root):
        self.root = Path(root)
        self.cards = _load(self.root / "cards", CARD_SCHEMA)
        self.rewards = _load(self.root / "rewards", REWARD_SCHEMA)

    def retrieve(self, task_tags: dict, filters: dict | None = None, k: int = 12) -> list:
        """Rank cards by tag overlap with the task.  filters: exact-match constraints on card tags (lists: any overlap)
        and/or 'text' (case-insensitive substring of title / finding)."""
        filters = filters or {}
        scored = []
        for c in self.cards:
            if not _passes(c, filters):
                continue
            t = c["tags"]
            score = 3.0 * (t.get("scene") == task_tags.get("scene"))
            score += 2.0 * len(set(t.get("phases", [])) & set(task_tags.get("phases", [])))
            score += 1.0 * len(set(t.get("objects", [])) & set(task_tags.get("objects", [])))
            if score == 0 and not filters:          # no scene, phase or object in common: another task's experience
                continue
            score += 1.0 * (t.get("robot") == task_tags.get("robot"))
            score += 0.5 * ("real" in t.get("domain", []))
            score += 0.5 * (c["status"] == "verified")
            scored.append((score, c))
        scored.sort(key=lambda x: (-x[0], x[1]["id"]))
        return [{"score": round(s, 1), **{k2: v for k2, v in c.items() if k2 != "_path"}} for s, c in scored[:k]]

    def rewards_for(self, phases: list) -> list:
        """Reward terms whose phase is one of the task's phases (borrowable shaping)."""
        return [{k: v for k, v in r.items() if k != "_path"} for r in self.rewards if r["phase"] in set(phases)]

    def record(self, card: dict) -> str:
        validate(card, CARD_SCHEMA)
        p = self.root / "cards" / f"{card['id']}.json"
        write_new(p, card)
        self.cards.append({**card, "_path": str(p)})
        return str(p)


def _passes(card: dict, filters: dict) -> bool:
    for key, want in filters.items():
        if key == "text":
            if not re.search(re.escape(str(want)), card["title"] + " " + card["finding"], re.I):
                return False
            continue
        have = card["tags"].get(key)
        if isinstance(have, list):
            wants = want if isinstance(want, list) else [want]
            if not set(have) & set(wants):
                return False
        elif have != want:
            return False
    return True


# ---------------------------------------------------------------------- the harness work library (literature + lab rewards)
PHASE_SKILLS = {     # station task phases -> skill tags used by the work library's reviewed records
    "reach": ["skill:reach", "skill:reaching"], "grasp": ["skill:grasp", "skill:grasping", "skill:gripper_close"],
    "lift": ["skill:lift", "skill:lift_bucket"], "hold": ["skill:settle"], "carry": ["skill:object_transport", "skill:carry"],
    "place": ["skill:place", "skill:pick_and_place", "skill:stacking"], "release": ["skill:release", "skill:place"],
    "insert": ["skill:insertion", "skill:alignment"], "align": ["skill:alignment"], "in_hand_rotate": ["skill:rotation"],
}


class WorkLibrary:
    """Read-only bridge to the policy-development work library. Quarantined and stale records are excluded by the
    library itself.  Records are references: retrieval never authorises execution; a reward is executed only after it is
    transcribed into station-observable terms (rewards.py `expression`)."""

    def __init__(self, cfg: dict):
        import importlib.util
        import os
        os.environ["ASTRA_WORK_LIBRARY_ROOT"] = str(Path(cfg["root"]).expanduser())
        spec = importlib.util.spec_from_file_location("astra_work_library", Path(cfg["module"]).expanduser())
        self.wl = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.wl)

    def status(self) -> dict:
        return self.wl.status()

    def search(self, tags=None, text="", kinds=("reward",), k=10) -> dict:
        res = self.wl.search({"kinds": list(kinds), "tags": list(tags or [])[:32], "text": text[:1000]}, limit=max(1, min(25, k)))
        return {"total_matches": res["total_matches"], "items": [compact(i) for i in res["items"]]}

    def for_task(self, task_tags: dict, k=12) -> dict:
        skills = sorted({s for p in task_tags.get("phases", []) for s in PHASE_SKILLS.get(p, [])})
        return {"query_tags": skills, **self.search(tags=skills, k=k)}


def compact(item: dict) -> dict:
    """Model-facing summary of one work-library record (references, not instructions)."""
    rec = item["record"]
    p = rec.get("payload") or {}
    lit = p.get("literature_reward") or {}
    out = {"id": item["id"][:16], "title": rec["title"][:120], "kind": rec["kind"], "status": item.get("status"),
           "compatibility": item.get("compatibility"), "matched_tags": item.get("matched_tags"),
           "summary": rec["summary"][:500]}
    if lit:
        out.update(formula=str(lit.get("formula", ""))[:600], skills=lit.get("skills"), privileges=lit.get("privileges"),
                   required_signals=[s["name"] if isinstance(s, dict) else s for s in lit.get("required_signals", [])][:10],
                   risks=[str(r)[:160] for r in lit.get("risks", [])][:4], source=lit.get("source_id"),
                   url=next((loc.get("url") for loc in lit.get("source_locators", []) if loc.get("url")), None))
    elif p.get("candidate"):
        out.update(native_parameters=p["candidate"].get("parameters"), binding=rec.get("binding"))
    return out
