"""Compatibility entry point for apps/scene/wall_views.py."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "scene" / "wall_views.py"), run_name="__main__")
