"""Build portable cloth render-surface bindings from an exported asset."""

import sys

from real2sim_harness.usd_surface_binding import build

if __name__ == "__main__":
    build(sys.argv[1])
