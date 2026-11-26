"""The shell tool of a confined worker session (`./tools/program` or `./tools/research` in its workspace).

It applies the session's sub-budget (set by the harness in the session's state directory, outside the workspace) and then
hands the command to the station root's baseline tooling unchanged (cap-harness verbs for program sessions, station verbs
for research sessions), so workers run on the same station, sandbox and evaluation runner as every baseline.

    subtool.py program|research --workspace WS -- <verb> [args]
"""
import json
import os
import re
import sys
import time
from pathlib import Path

ALLOWED = {"program": {"run", "run-batch", "py", "score", "status", "wait"},
           "research": {"run", "eval", "score", "status", "wait", "freeze"}}


def seed_args(rest: list) -> list:
    """The seeds of a run / run-batch command line: the values right after --seed / --seeds, up to the next option.
    (11-13: a regex over everything after --seeds also read the digits of `--output-root work/debug/attempts/v01` as
    seed 1 and refused every numbered evaluation of a session.)"""
    out = []
    for flag in ("--seeds", "--seed"):
        if flag in rest:
            for tok in rest[rest.index(flag) + 1:]:
                if tok.startswith("--"):
                    break
                out += [int(x) for x in re.findall(r"\d+", tok)]
    return out


def main():
    kind, ws = sys.argv[1], Path(sys.argv[3]).resolve()
    rest = sys.argv[5:] if sys.argv[4:5] == ["--"] else sys.argv[4:]
    meta = json.loads((ws / ".task.json").read_text())
    limits = json.loads(Path(meta["astra_limits"]).read_text())
    root = Path(limits["station_root"])
    sys.path.insert(0, str(root))
    import task_tool as tt  # noqa: E402
    if not rest or rest[0] not in ALLOWED[kind]:
        sys.exit(f"usage: {kind} " + "|".join(sorted(ALLOWED[kind])) + " ...")
    verb = rest[0]
    L = tt.ledger(tt.meta(ws))
    if kind == "program":
        numbered = sorted({r["program"] for r in L if r.get("kind") == "numbered"})
        probes = sum(r.get("kind") == "probe" for r in L)
        if verb == "status":
            print(json.dumps({"numbered_versions_run": numbered, "numbered_budget": limits["versions"], "probe_runs": probes,
                              "probe_budget": limits["probes"], "deadline_in_min": round((limits["deadline"] - time.time()) / 60)}, indent=1))
            return
        if verb in ("run", "run-batch"):
            if time.time() > limits["deadline"]:
                sys.exit("REFUSED: the session's time is up; write work/RESULT.md naming your best version")
            prog = rest[rest.index("--program") + 1] if "--program" in rest else ""
            name = Path(prog).name
            seeds = seed_args(rest)
            if set(seeds) - set(tt.DEV):
                sys.exit(f"REFUSED: only the development seeds {tt.DEV[0]}-{tt.DEV[-1]} are available to this session")
            if re.fullmatch(r"v\d\d_\w+\.py", name):
                if name not in numbered and len(numbered) >= limits["versions"]:
                    sys.exit(f"REFUSED: this session's budget of {limits['versions']} numbered versions is used; finish with work/RESULT.md")
            elif probes >= limits["probes"]:
                sys.exit(f"REFUSED: probe budget used ({limits['probes']} runs)")
        os.execv(sys.executable, [sys.executable, str(root / "task_tool.py"), "cap", "--workspace", str(ws), "--", *rest])
    else:
        evals = sum(r.get("kind") == "eval" for r in L)
        if verb == "status":
            print(json.dumps({"evaluations_used": evals, "evaluation_budget": limits["evals"],
                              "training_episode_budget_of_session": limits["train_episodes"],
                              "training_episodes_used_by_campaign_station": "see the station's info()['usage'] from your scripts",
                              "deadline_in_min": round((limits["deadline"] - time.time()) / 60)}, indent=1))
            return
        if verb in ("run", "eval") and time.time() > limits["deadline"]:
            sys.exit("REFUSED: the session's time is up; freeze your best evaluation")
        if verb == "eval" and evals >= limits["evals"]:
            sys.exit(f"REFUSED: evaluation budget used ({limits['evals']}); freeze one of your evaluations")
        os.execv(sys.executable, [sys.executable, str(root / "task_tool.py"), "station", "--workspace", str(ws), "--", *rest])


if __name__ == "__main__":
    main()
