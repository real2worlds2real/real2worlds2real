"""Operator-selected MPM subprocess, shared by object and scene reviews."""

import json
from pathlib import Path


def physics_command(runner, configuration, output, frames, fps):
    config = json.loads(Path(runner).read_text())
    if config.get("schema_version") != 1 or not {"mpm", "finite_static_boxes"} <= set(
        config.get("capabilities", [])
    ):
        raise ValueError("Configure an MPM/finite-static-box physics runner")
    argv = config.get("argv")
    if not isinstance(argv, list) or not argv or not all(isinstance(x, str) and x for x in argv):
        raise ValueError("Physics runner requires a nonempty argv list")
    replacements = {
        "{configuration}": str(Path(configuration).resolve()),
        "{output}": str(Path(output).resolve()),
        "{frames}": str(frames),
        "{fps}": str(fps),
    }
    if any(token not in " ".join(argv) for token in replacements):
        raise ValueError("Physics runner must forward configuration, output, frames and fps")
    command = []
    for value in argv:
        for token, replacement in replacements.items():
            value = value.replace(token, replacement)
        command.append(value)
    return command
