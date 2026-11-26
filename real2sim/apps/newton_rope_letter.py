"""Compatibility entry point for apps/tasks/rope_letter.py."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "tasks/rope_letter.py"), run_name="__main__")
