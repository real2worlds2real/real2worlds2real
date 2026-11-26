"""Declared per-participant support-response requirements and report audit."""
from __future__ import annotations

import math

METHOD = "observed_free_drop_and_settle_on_declared_horizontal_support"


def support_requirements(rows, object_ids, support_ids):
    if not isinstance(rows, list) or len(rows) > 1024:
        raise ValueError("support_requirements must be a bounded list")
    result, seen, releases = [], set(), {}
    for row in rows:
        keys = {"asset_id", "support_id", "release_height_m", "minimum_drop_m",
                "separation_tolerance_m", "max_late_speed_m_s"}
        if not isinstance(row, dict) or set(row) != keys:
            raise ValueError("Each support requirement needs IDs, release height and explicit drop/separation/speed bounds")
        asset, support = row["asset_id"], row["support_id"]
        if not isinstance(asset, str) or not isinstance(support, str) or asset not in object_ids or support not in support_ids:
            raise ValueError("Support requirements must name an existing asset and static support")
        if (asset, support) in seen:
            raise ValueError("Duplicate required support relation")
        item = dict(row)
        for key in keys - {"asset_id", "support_id"}:
            value = item[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Support bounds and release height must be finite nonnegative numbers")
            item[key] = float(value)
        if item["minimum_drop_m"] <= 0 or item["max_late_speed_m_s"] <= 0:
            raise ValueError("Minimum drop and maximum late speed must be positive")
        if asset in releases and releases[asset] != item["release_height_m"]:
            raise ValueError("One asset must have one declared release height per scene trial")
        releases[asset] = item["release_height_m"]
        seen.add((asset, support))
        result.append(item)
    return result


def audit_support_requirements(report, requirements):
    by_pair = {}
    for check in report.get("checks", []):
        for row in check.get("support_response_tests", []):
            if isinstance(row, dict):
                by_pair.setdefault((row.get("id"), row.get("support")), []).append(row)
    results = []
    for requirement in requirements:
        asset, support = requirement["asset_id"], requirement["support_id"]
        eligible = []
        reasons = []
        for row in by_pair.get((asset, support), []):
            try:
                keys = ("initial_bottom_m", "final_bottom_m", "support_top_m", "late_speed_m_s")
                if any(isinstance(row[k], bool) or not isinstance(row[k], (int, float)) for k in keys):
                    raise ValueError("Malformed support numbers")
                initial, final, top, speed = [float(row[k]) for k in keys]
                valid = (all(math.isfinite(v) for v in (initial, final, top, speed))
                         and initial - final >= requirement["minimum_drop_m"]
                         and abs(final - top) <= requirement["separation_tolerance_m"]
                         and 0 <= speed <= requirement["max_late_speed_m_s"]
                         and row.get("footprint_inside") is True
                         and row.get("method") == METHOD
                         and row.get("ok") is True and not row.get("failures"))
            except (KeyError, TypeError, ValueError):
                valid = False
            if valid:
                eligible.append(row)
            else:
                reasons.append("response_missing_or_outside_declared_bounds")
        results.append({"asset_id": asset, "support_id": support, "ok": bool(eligible),
                        "eligible_rows": len(eligible),
                        "failures": [] if eligible else reasons or ["required_support_response_missing"]})
    return {"ok": all(row["ok"] for row in results), "requirements": results,
            "scope": "Each declared static-support relation requires an actual sampled drop/settle response. "
                     "The footprint flag is the adapter's reference-point test, not full contact-patch geometry. "
                     "No force calibration or continuous-time penetration certification."}
