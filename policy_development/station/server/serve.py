"""Start one station server: a task backend behind the generic station protocol.

    python serve.py --backend <file.py>:<Class> --task <task> --num_envs 16 --socket <path.sock> --run <dir>
                    [--sealed <partitions.json>] [--token <file>] [--runtime isaac|python] [--check_act <partition> --check_ids ...]
                    [backend options] [Isaac app options]

The coordinator's launcher (station_up.sh) fills these from the task spec's `station` block.  With --runtime isaac the
Isaac Sim app is launched before the backend module is imported (Isaac requires that order).
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))                  # the `server` package next to this file (repository or station root)


def load_class(spec: str):
    path, _, name = spec.partition(":")
    p = Path(path).expanduser().resolve()
    sys.path.insert(0, str(p.parent))
    s = importlib.util.spec_from_file_location(p.stem, p)
    mod = importlib.util.module_from_spec(s)
    sys.modules[p.stem] = mod
    s.loader.exec_module(mod)
    return getattr(mod, name)


def main(argv=None):
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--runtime", choices=("isaac", "python"), default="python")
    pre, _ = ap.parse_known_args(argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--runtime", choices=("isaac", "python"), default="python")
    ap.add_argument("--task", required=True)
    ap.add_argument("--num_envs", type=int, default=16)
    ap.add_argument("--socket", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--sealed", default="", help="sealed start partitions (default: <sim dir>/sealed/<partition_kind>_partitions.json "
                                                 "as named by the backend)")
    ap.add_argument("--token", default="", help="coordinator token file (default: <sim dir>/sealed/heldout_token)")
    ap.add_argument("--check_act", default="", help="validation: partition name to run with the backend's own controller, then exit")
    ap.add_argument("--check_ids", default="")
    ap.add_argument("--idle_exit", type=float, default=1800.0, help="exit after this many seconds without a client (frees the GPU)")
    app = None
    if pre.runtime == "isaac":
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(ap)
    cls_holder = {}
    if pre.runtime != "isaac":
        cls_holder["cls"] = load_class(pre.backend)
        cls_holder["cls"].add_args(ap)
        args = ap.parse_args(argv)
    else:
        # Isaac: backend options must be known before the app starts, the backend module may import isaaclab only after
        args, rest = ap.parse_known_args(argv)
        app = AppLauncher(args).app
        cls_holder["cls"] = load_class(args.backend)
        sub = argparse.ArgumentParser(allow_abbrev=False)
        cls_holder["cls"].add_args(sub)
        extra = sub.parse_args(rest)
        for k, v in vars(extra).items():
            setattr(args, k, v)
    cls = cls_holder["cls"]
    from server.core import StationServer
    backend = cls(args, args.num_envs)
    sealed_dir = Path(os.environ.get("STATION_SEALED_DIR", HERE.parent / "sealed"))      # <station root>/sim/sealed
    sealed = Path(args.sealed) if args.sealed else sealed_dir / f"{backend.partition_kind}_partitions.json"
    token = Path(args.token) if args.token else sealed_dir / "heldout_token"
    srv = StationServer(backend, args.run, sealed, token)
    if args.check_act:
        try:
            print("[CHECK]", json.dumps({"task": args.task, "partition": args.check_act,
                                         **srv.check(args.check_act, [int(i) for i in args.check_ids.split(",")])}), flush=True)
        except BaseException:  # noqa: BLE001  (Isaac does not exit on an exception: exit explicitly)
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
            os._exit(1)
        os._exit(0)
    srv.serve(args.socket, args.idle_exit)
    if app is not None:
        app.close()


if __name__ == "__main__":
    main()
