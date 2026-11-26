"""Compatibility for harness clients that still dispatch pipeline and evolve subcommands."""
import argparse

from calibrated_scene_reconstruction.cli import main as reconstruction_main

FORWARDED = {
    "pipeline": ("calibrated_scene_reconstruction.agent.tool", "scene reconstruction tools"),
    "evolve": ("calibrated_scene_reconstruction.agent.evolution", "capability and experience records"),
}


def forward(command: str, arguments: list[str]) -> int:
    if command not in FORWARDED:
        raise KeyError(command)
    return reconstruction_main(arguments if command == "pipeline" else ["evolve", *arguments])


def add_parser(commands) -> None:
    for name, (_, description) in FORWARDED.items():
        sub = commands.add_parser(name, help=description, add_help=False)
        sub.add_argument("arguments", nargs=argparse.REMAINDER)
        sub.set_defaults(func=lambda args: forward(args.command, args.arguments))
