"""Compatibility entry point for apps/datagen/command_tcp.py."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "datagen/command_tcp.py"), run_name="__main__")
