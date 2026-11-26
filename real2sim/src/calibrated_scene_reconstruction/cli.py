"""Command line for calibrated scene reconstruction and its experience records."""
import argparse
import sys

PROGRAM = "calibrated-scene-reconstruction"
_RUNTIME_MODULES = {"torch", "torchvision", "warp", "newton", "pytorch3d", "kaolin", "nvdiffrast", "utils3d"}


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"--help", "-h"}:
        parser = argparse.ArgumentParser(prog=PROGRAM, description=__doc__)
        parser.add_argument("tool", help="schemas, a scene tool (for example stage_s0), or evolve")
        parser.add_argument("arguments", nargs=argparse.REMAINDER, help="arguments passed to the selected tool")
        parser.print_help()
        return 0 if args else 2

    is_evolution = args[0] == "evolve"
    forwarded = args[1:] if is_evolution else args
    program = PROGRAM + " evolve" if is_evolution else PROGRAM
    try:
        if is_evolution:
            from .agent.evolution import main as run
        else:
            from .agent.tool import main as run
        return int(run(forwarded) or 0)
    except ModuleNotFoundError as exc:
        missing = (exc.name or str(exc)).split(".")[0]
        if missing == __package__:
            raise
        how = ("Use envs/recon/bin/python from setup/setup_env.sh." if missing in _RUNTIME_MODULES
               else "Install the scene-pipeline and objects extras or use envs/recon/bin/python.")
        print(f"{program} requires {exc.name}. {how}", file=sys.stderr)
        return 2
