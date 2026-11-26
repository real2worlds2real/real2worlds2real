"""S6b cousin rooms: what the photo did not see, completed again for every cousin in its own style and seed.

    python -m calibrated_scene_reconstruction.recon.cousin_rooms sessions/<name> [--only cousin_00,cousin_01] [--model sdxl|flux] [--force]

The twin's unseen walls come from one panorama completion (recon.panorama -> scene/pano, walls already moved). Here
each cousin gets its own: the twin's completion.json prompts with the cousin theme's style phrase (configs/themes/
<theme>.json "prompt") in place of the twin's, seed 1000 + cousin id (or the review's completion.json "tags"), the
twin's room geometry (no --apply). The completion is never a texture: the agent reviews its views
(recon.completion_review) and its objects are reconstructed in 3D (recon.room_objects); the cousin's walls keep the
theme's PBR. A completion already there with the same seed, style, model and per-tag review settings is reused.
"""
import argparse
import json
import subprocess
import sys

from calibrated_scene_reconstruction.assets.styles import themes
from calibrated_scene_reconstruction.scene.cousins.variants import ceiling_look, surface
from calibrated_scene_reconstruction.scene.graph import cousin_parts
from calibrated_scene_reconstruction.scene.session import Session


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--only", default="", help="comma-separated cousin ids (default: all)")
    ap.add_argument("--model", default="sdxl", choices=["sdxl", "flux"])
    ap.add_argument("--force", action="store_true", help="complete again even if this cousin's completion exists")
    args = ap.parse_args()
    s = Session(args.session)
    styles = themes()
    tags_cfg = s.config("completion.json", {}).get("tags", {})
    cp = {**cousin_parts(s.scene_graph, s.manifest),
          **{k: v for k, v in s.config("cousins.json", {}).items() if k.endswith("_parts")}}
    only = set(filter(None, args.only.split(",")))
    for path in sorted((s / "cousins").glob("cousin_*.json")):
        if only and path.stem not in only:
            continue
        c = json.loads(path.read_text())
        theme = c["theme"]["theme"]
        style = styles.get(theme, {}).get("prompt", f"{theme} style interior")
        tag, seed = path.stem, 1000 + int(c["id"])
        review = tags_cfg.get(tag, {})
        seed = int(review.get("seed", seed))
        done = s / "scene" / "pano" / tag / "room.json"
        prev = json.loads(done.read_text()) if done.exists() else {}
        key = (seed, style, args.model, json.dumps(review, sort_keys=True))
        if args.force or (prev.get("seed"), prev.get("style"), prev.get("model"), prev.get("review", "{}")) != key:
            subprocess.run([sys.executable, "-m", "calibrated_scene_reconstruction.scene.background.panorama", str(s.dir), "--model", args.model,
                            "--seed", str(seed), "--tag", tag, "--style", style], check=True)
            room = json.loads(done.read_text())
            room["review"] = key[3]
            done.write_text(json.dumps(room, indent=1))
        ov = c.setdefault("room_overrides", {})
        for k in cp["observed_wall_parts"] + cp["unobserved_wall_parts"]:  # the theme's clean wall everywhere (an
            if k not in ov or "tex" in ov[k]:  # earlier version baked the completion onto the unseen walls)
                ov[k] = surface(c["theme"]["wall_pbr"], 2.0, c["theme"]["wall_tint"])
        ceil = next((k for k, v in s.scene_graph["structure"].items() if v.get("role", k) == "ceiling"), "ceiling")
        if "pbr" not in ov.get(ceil, {}):
            ov[ceil] = ceiling_look(c["theme"])
        c.pop("room_pano", None)
        c["room_completion"] = {"tag": tag, "seed": seed, "style": style}
        path.write_text(json.dumps(c, indent=1))
        print(f"[cousin_rooms] {tag}: {theme}, seed {seed}, completion in scene/pano/{tag}")


if __name__ == "__main__":
    main()
