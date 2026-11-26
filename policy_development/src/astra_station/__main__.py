"""Command line for the station path of the harness.

    python -m astra_station bind    --site S --task T --out binding.json
    python -m astra_station init    --binding B --campaign C --run R [--budget JSON] [--caps JSON] [--decision JSON]
    python -m astra_station tools   --state DIR
    python -m astra_station exec    --state DIR --tool T --args JSON      (operator: run one tool by hand, e.g. a smoke test)
    python -m astra_station run     --state DIR [--steps N]               (the agent's decision loop)
    python -m astra_station prompt  --state DIR                           (print the next decision prompt; no model call)
    python -m astra_station status  --state DIR
    python -m astra_station heldout --state DIR                           (coordinator: sealed evaluation, once)
    python -m astra_station handoff --state DIR                           (operator: hand the frozen deliverable to the task's real chain)
    python -m astra_station new-twin-task --session S --config C --task-id T --factory F --policy-io JSON --out DIR
    python -m astra_station new-twin-task --delivery RECORD     --task-id T --factory F --policy-io JSON --out DIR [--setup FILE]
                                                                                (a task spec for a frozen real2sim twin; see twin_task.py)
"""
from __future__ import annotations

import argparse
import json
import sys

from . import binding as binding_mod
from . import catalog
from .campaign import Campaign
from .runner import Runner


def main(argv=None):
    ap = argparse.ArgumentParser(prog="astra-station")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("bind"); p.add_argument("--site", required=True); p.add_argument("--task", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("init"); p.add_argument("--binding", required=True); p.add_argument("--campaign", required=True); p.add_argument("--run", required=True)
    p.add_argument("--budget", default="{}"); p.add_argument("--caps", default="{}"); p.add_argument("--decision", default="")
    for name in ("tools", "status", "heldout", "prompt", "handoff"):
        p = sub.add_parser(name); p.add_argument("--state", required=True)
    p = sub.add_parser("exec"); p.add_argument("--state", required=True); p.add_argument("--tool", required=True); p.add_argument("--args", required=True)
    p = sub.add_parser("run"); p.add_argument("--state", required=True); p.add_argument("--steps", type=int, default=0)
    p = sub.add_parser("new-twin-task")
    for k in ("--task-id", "--factory", "--policy-io", "--out"):
        p.add_argument(k, required=True)
    p.add_argument("--session"); p.add_argument("--config")
    p.add_argument("--delivery", help="real2sim delivery record (real2sim-harness deliver); replaces --session/--config")
    p.add_argument("--setup", help="with --delivery: a JSON file with the station author's interaction setup for the "
                   "factory (robot, base, target, kind, thresholds); kept in the spec as twin.setup")
    p.add_argument("--robot", default="vega_right_sharpa"); p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=600)
    a = ap.parse_args(argv)
    if a.cmd == "bind":
        b = binding_mod.create(a.site, a.task, a.out)
        print(json.dumps(binding_mod.public(b), indent=1, ensure_ascii=False))
    elif a.cmd == "init":
        c = Campaign.create(a.binding, a.campaign, a.run, json.loads(a.budget), json.loads(a.caps), json.loads(a.decision) if a.decision else None)
        print(c.dir)
    elif a.cmd == "tools":
        c = Campaign(a.state)
        print(json.dumps(catalog.menu(c.ready_tools()), indent=1, ensure_ascii=False))
    elif a.cmd == "status":
        c = Campaign(a.state)
        c.ledger.verify()
        print(json.dumps({"used": c.s["used"], "budget": c.cfg["budget"], "caps_used": c.s["cap_used"], "datasets": list(c.s["datasets"]),
                          "frozen": (c.s["frozen"] or {}).get("candidate"), "unseen": c.s["unseen"], "ended": c.s["ended"]}, indent=1))
        for r in c.leaderboard():
            print(" ", r["candidate"], r["kind"], r["method"], json.dumps({k: f"{v['success']}/{v['n']}" for k, v in r["evaluations"].items()}))
    elif a.cmd == "exec":
        c = Campaign(a.state)
        print(json.dumps(c.execute(a.tool, json.loads(a.args)), indent=1, ensure_ascii=False))
    elif a.cmd == "prompt":
        print(Runner(Campaign(a.state), agent=object()).prompt())
    elif a.cmd == "run":
        c = Campaign(a.state)
        r = Runner(c)
        if a.steps:
            for _ in range(a.steps):
                if c.s["frozen"] or c.s["ended"]:
                    break
                out = r.step()
                print(json.dumps(out, ensure_ascii=False)[:1500], flush=True)
        else:
            print(json.dumps(r.run(), ensure_ascii=False, default=str)[:3000])
    elif a.cmd == "heldout":
        print(json.dumps(Campaign(a.state).heldout(), indent=1))
    elif a.cmd == "new-twin-task":
        from pathlib import Path
        from . import twin_task
        from .site import REPO
        if bool(a.delivery) == bool(a.session and a.config):
            ap.error("new-twin-task takes either --delivery or both --session and --config")
        if a.delivery:
            spec = twin_task.spec_from_delivery(Path(a.delivery), a.task_id, a.factory, json.loads(a.policy_io), a.robot,
                                                a.num_envs, a.max_steps,
                                                json.loads(Path(a.setup).read_text()) if a.setup else None)
        else:
            spec = twin_task.spec_from_twin(Path(a.session), Path(a.config), a.task_id, a.factory, json.loads(a.policy_io),
                                            a.robot, a.num_envs, a.max_steps)
        print(twin_task.write_task(spec, Path(a.out), REPO / "policy_development/station/server/twin.py"))
    elif a.cmd == "handoff":
        print(json.dumps(Campaign(a.state).handoff(), indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
