"""Create the sealed start partitions of a new task kind (coordinator only; the file never enters a repository).

    python partitions.py <sim dir>/sealed/<kind>_partitions.json [--dev 51-65] [--unseen 66-80] [--heldout 1-50] [--train-rows N]

Each start id maps to a secret random seed (a backend that draws its start from a seed, e.g. cup, nut, twins); peg-like
backends that draw from a start bank need row ids instead and keep their own file.  The ids are the site's protocol
(the same for every task); only the seeds are secret.
"""
import argparse
import json
import secrets
from pathlib import Path


def rng(spec):
    a, _, b = spec.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--dev", default="51-65"); ap.add_argument("--unseen", default="66-80")
    ap.add_argument("--heldout", default="1-50")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; sealed partitions are never regenerated")
    ids = {"dev": rng(a.dev), "unseen": rng(a.unseen), "heldout": rng(a.heldout)}
    starts = {k: {str(i): secrets.randbelow(2 ** 31) for i in v} for k, v in ids.items()}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as fh:
        json.dump({"ids": ids, "starts": starts}, fh)
    out.chmod(0o600)
    print(json.dumps({"written": str(out), "partitions": {k: len(v) for k, v in ids.items()}}))


if __name__ == "__main__":
    main()
