"""Expose a unified task-independent harness to Astra or another external agent."""
import argparse
import json
from pathlib import Path
import sys

from .service import Harness, serve, tool_schemas


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("schemas", "serve", "call"))
    parser.add_argument("--state", type=Path)
    parser.add_argument("--execute", action="store_true", help="Permit contracted simulation commands; no hardware tools exist.")
    parser.add_argument("--learning-root", type=Path)
    parser.add_argument("--request", type=Path)
    args = parser.parse_args(argv)
    if args.command == "schemas":
        print(json.dumps({"tools": tool_schemas()}, indent=2))
        return 0
    if args.state is None:
        parser.error("--state is required for durable request IDs and knowledge stores")
    harness = Harness(args.state, execute=args.execute, learning_root=args.learning_root)
    if args.command == "serve":
        serve(sys.stdin, sys.stdout, harness)
        return 0
    if args.request is None:
        parser.error("call requires --request containing a tool request")
    import io
    output = io.StringIO()
    serve(io.StringIO(args.request.read_text(encoding="utf-8") + "\n"), output, harness)
    print(output.getvalue(), end="")
    return 2 if "error" in json.loads(output.getvalue()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
