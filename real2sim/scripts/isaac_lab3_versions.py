"""Report the simulator packages available inside the Lab 3 container."""

import platform
from importlib import metadata
from pathlib import Path


def main() -> None:
    print(f"python={platform.python_version()}")
    sim_version_file = Path("/isaac-sim/VERSION")
    if sim_version_file.is_file():
        print(f"isaac-sim-build={sim_version_file.read_text().strip()}")
    for package in ("isaaclab", "isaaclab-newton", "isaaclab-physx", "isaacsim", "newton", "warp-lang"):
        try:
            version = metadata.version(package)
        except metadata.PackageNotFoundError:
            version = "not installed as a Python distribution"
        print(f"{package}={version}")


if __name__ == "__main__":
    main()
