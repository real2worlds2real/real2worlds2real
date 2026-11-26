"""Compatibility entry point for apps/physics/video.py."""
from pathlib import Path
import runpy
if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "physics/video.py"), run_name="__main__")
