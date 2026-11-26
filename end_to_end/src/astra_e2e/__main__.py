"""Run the end-to-end workflow from a checkout or an installed distribution."""
from .chain import main

if __name__ == "__main__":
    raise SystemExit(main())
