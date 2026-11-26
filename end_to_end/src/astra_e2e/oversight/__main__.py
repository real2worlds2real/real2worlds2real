"""Agent-operated real2sim2real loop: intake, plan, execute, review, and refine."""
import argparse
import json

from .contracts import PACKET_SCHEMA, REPORT_SCHEMA, SPEC_SCHEMA, intake
from .controller import Controller, read
from .knowledge import lookup, record
from .planning import build_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("schemas")
    for name in ("intake", "plan"):
        p = sub.add_parser(name)
        p.add_argument("spec")
    p = sub.add_parser("init")
    p.add_argument("spec")
    p.add_argument("run")
    for name in ("status", "next"):
        p = sub.add_parser(name)
        p.add_argument("run")
    for name in ("run", "review", "feedback"):
        p = sub.add_parser(name)
        p.add_argument("run")
        p.add_argument("request")
    for name in ("learn", "recall"):
        p = sub.add_parser(name)
        p.add_argument("knowledge")
        p.add_argument("request")
    p = sub.add_parser("recover")
    p.add_argument("run")
    p.add_argument("module")
    p.add_argument("--reason", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "schemas":
            result = {"intake": SPEC_SCHEMA, "packet": PACKET_SCHEMA, "report": REPORT_SCHEMA}
        elif args.command == "intake":
            result = intake(read(args.spec))
        elif args.command == "plan":
            result = build_plan(read(args.spec))
        elif args.command == "init":
            result = Controller.create(args.run, read(args.spec)).status()
        elif args.command in ("status", "next"):
            ctrl = Controller(args.run)
            result = ctrl.status()
            if args.command == "next":
                plan = read(ctrl.root / "plan.json")
                result["work"] = {m: plan["modules"][m] for m in result["ready"]}
        elif args.command in ("run", "review", "feedback"):
            ctrl = Controller(args.run)
            operation = {"run": ctrl.run, "review": ctrl.accept, "feedback": ctrl.feedback}[args.command]
            result = operation(read(args.request))
        elif args.command == "learn":
            result = record(args.knowledge, read(args.request))
        elif args.command == "recover":
            result = Controller(args.run).recover(args.module, args.reason)
        else:
            result = lookup(args.knowledge, read(args.request))
        print(json.dumps(result, indent=2, allow_nan=False))
        if args.command == "intake" and not result["ready"]:
            return 2
        if args.command == "review" and not result["accepted"]:
            return 2
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
