"""Execute one fixed candidate across agent-selected fit/validation experiments."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from .artifacts import atomic_json
from .physics_inference import ParameterSpace, _digest, run_trial


def evaluate_candidate(cases, parameters, candidate, output):
    """Execute existing native runners; no parameter proposals or model calls.

    Each case has id, role ('fit'/'validation'), experiment and runner. Validation
    additionally needs independence_evidence. This records an author's rationale,
    not proof of independent data. Exact observation reuse across roles is rejected.
    Failures retain suite progress and abort; incomplete comparisons cannot produce
    a complete role score. Scores remain separate for fit and validation.
    """
    candidate = ParameterSpace(parameters).validate(candidate)
    if not cases:
        raise ValueError("Select at least one experiment")
    specs, runners, ids, hashes = [], [], set(), {}
    for case in cases:
        name, role = case["id"], case["role"]
        if not isinstance(name, str) or not name.strip() or name in ids:
            raise ValueError("Case IDs must be nonempty and unique")
        if role not in {"fit", "validation"} or not callable(case["runner"]):
            raise ValueError("Each case needs a fit/validation role and native runner")
        if role == "validation" and (not isinstance(case.get("independence_evidence"), str)
                                     or not case["independence_evidence"].strip()):
            raise ValueError("State why validation observations are independent of fitting")
        spec = copy.deepcopy({k: v for k, v in case.items() if k != "runner"})
        fingerprint = _digest(spec["experiment"]["observations"])
        if fingerprint in hashes and hashes[fingerprint] != role:
            raise ValueError("Identical observations cannot be both fit and held-out validation")
        hashes[fingerprint] = role
        ids.add(name)
        specs.append(spec)
        runners.append(case["runner"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "request.json", {"cases": specs, "parameters": parameters,
                                         "candidate": candidate, "candidate_sha256": _digest(candidate)})
    results = []

    def persist():
        roles = {}
        for role in ("fit", "validation"):
            expected = sum(s["role"] == role for s in specs)
            rows = [r for r in results if r["role"] == role]
            complete = expected > 0 and len(rows) == expected and all(r.get("complete", False) for r in rows)
            roles[role] = {"expected": expected, "finished": len(rows), "complete": complete,
                           "mean_score": float(np.mean([r["score"] for r in rows])) if complete else None}
        report = {"schema_version": 1, "candidate": candidate, "cases": results,
                  "roles": roles, "acceptance": False,
                  "complete": len(results) == len(specs) and all(r.get("complete", False) for r in results),
                  "score_definition": "Separate per-role means of per-experiment normalized track Huber; inspect each case and event errors",
                  "independence": "Caller-declared evidence; exact cross-role observation reuse checked, not general overlap or leakage"}
        atomic_json(output / "report.json", report)
        return report

    persist()
    for i, (spec, runner) in enumerate(zip(specs, runners, strict=True)):
        trial_dir = f"case-{i:03d}"
        row = {"id": spec["id"], "role": spec["role"], "trial": trial_dir}
        try:
            report = run_trial(spec["experiment"], parameters, candidate, runner, output / trial_dir)
            row.update(status="completed", complete=report["complete"], score=report["score"],
                       events=report["events"], observation_sha256=report["observation_sha256"],
                       experiment_sha256=report["experiment_sha256"])
        except Exception as exc:
            row.update(status="failed", complete=False, score=None, error=f"{type(exc).__name__}: {exc}")
            results.append(row)
            persist()
            raise
        results.append(row)
        persist()
    return persist()
