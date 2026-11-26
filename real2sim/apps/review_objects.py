"""Compatibility entry point for apps/scene/review_objects.py."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "scene" / "review_objects.py"), run_name="__main__")
