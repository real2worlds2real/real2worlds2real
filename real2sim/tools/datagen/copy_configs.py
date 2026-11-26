"""Configurations copied under a new letter: the same randomized scene (dr/<tag>.json) and its obstacles
(grasps/_env/<tag>.json; the grasp selection is keyed by their content, so it carries over) as <scene>__<letter><k>.
A DAgger collection on a training configuration records its episodes under the copy: the recorded episodes and their
result.json stay as they were (a later run on a tag can rewrite its first episode's result.json), and the copies'
episodes are told apart from the expert's.

    python3 tools/copy_configs.py sessions/<name> --letter a --sources twin__x03 cousin_00__y02 ...

The copies' sources go to <session>/dr/_copies.json ({copy: source}).
"""
import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--letter", required=True)
    ap.add_argument("--sources", nargs="+", required=True)
    a = ap.parse_args()
    s = Path(a.session)
    reg = s / "dr" / "_copies.json"
    copies = json.loads(reg.read_text()) if reg.exists() else {}
    for src in a.sources:
        if src in copies.values() and any(c.split("__")[1][0] == a.letter for c, v in copies.items() if v == src):
            continue  # already copied under this letter
        scene = src.split("__")[0]
        k = len(list((s / "dr").glob(f"{scene}__{a.letter}[0-9]*.json")))
        dst = f"{scene}__{a.letter}{k:02d}"
        shutil.copyfile(s / "dr" / f"{src}.json", s / "dr" / f"{dst}.json")
        env = s / "grasps" / "_env" / f"{src}.json"
        if env.exists():
            shutil.copyfile(env, s / "grasps" / "_env" / f"{dst}.json")
        copies[dst] = src
        print(f"{dst} <- {src}")
    reg.write_text(json.dumps(copies, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
