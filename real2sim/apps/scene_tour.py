"""Compatibility entry point for apps/scene/scene_tour.py."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "scene" / "scene_tour.py"), run_name="__main__")
