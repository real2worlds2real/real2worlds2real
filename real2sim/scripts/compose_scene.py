"""Compose or verify a portable scene; no simulator startup is needed."""
import argparse
import json

from real2sim_harness.scene import compose_scene, verify_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--output")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if not args.verify and not args.output:
        parser.error("--output is required for composition")
    result = verify_scene(args.manifest) if args.verify else compose_scene(args.manifest, args.output)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
