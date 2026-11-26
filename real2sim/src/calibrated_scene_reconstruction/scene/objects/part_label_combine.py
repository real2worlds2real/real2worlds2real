"""Combine source-only semantic queries by explicitly reviewed frame identity.

A second query can repair a merged stem/leaf label while missing other views.
Keep the original acquisitions and every current view. This tool copies a whole
source proposal per frame; it never sees geometry, render errors or fit scores.
The resulting component review is still pending until reviewed against RGB.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .blender.store import digest


def select_frames(names, choices, proposals):
    """Pure selection with complete, unique source frame coverage."""
    if len(set(names)) != len(names) or set(choices) != set(names):
        raise ValueError("Explicit identity choices must cover every source frame exactly once")
    output = []
    shape = None
    for name in names:
        choice = choices[name]
        if (set(choice) != {"proposal", "basis"} or not isinstance(choice["basis"], str)
                or not choice["basis"].strip() or choice["proposal"] not in proposals):
            raise ValueError("Each frame needs one source proposal and an explicit identity basis")
        proposal = proposals[choice["proposal"]]
        source_names, masks = proposal["names"], proposal["masks"]
        if len(set(source_names)) != len(source_names) or name not in source_names:
            raise ValueError("Chosen source proposal lacks unique frame identity")
        mask = np.asarray(masks[source_names.index(name)])
        if mask.ndim != 2 or mask.dtype != np.bool_ or not mask.any():
            raise ValueError("Choose a nonempty source label; no silent view removal")
        if shape is not None and mask.shape != shape:
            raise ValueError("Source proposals must retain the same native image dimensions")
        shape = mask.shape
        output.append(mask.copy())
    return np.stack(output)


def combine(session, obj, prompt, choices_path):
    from calibrated_scene_reconstruction.core.session_code import source_path
    from .blender.evidence import load
    from .part_label_review import prepare

    source = source_path(session.config_dir, choices_path)
    decision = json.loads(source.read_text())
    if (set(decision) != {"version", "object", "prompt", "frames"}
            or decision["version"] != 1 or decision["object"] != obj or decision["prompt"] != prompt):
        raise ValueError("Source choices need matching object, prompt and version 1")
    cameras, frames, _, _ = load(session, obj)
    current_names = [str(cameras["names"][frame]) for frame in frames]
    if set(decision["frames"]) != set(current_names):
        raise ValueError("Review every currently used frame without dropping views")
    root = (session / "qa/source_parts" / obj).resolve()
    loaded, hashes, template = {}, {}, None
    expected = {
        "raw_labels_sha256": digest(session / "splat/object_fit" / f"{obj}.components.acquired.npz"),
        "acquisition_sha256": digest(session / "splat/object_fit" / f"{obj}.acquired.npz"),
        "whole_observations_sha256": digest(session / "splat/object_fit" / f"{obj}.npz"),
        "cameras_sha256": digest(session / "parsing/posed_frames/cameras.npz"),
    }
    for label in sorted({row["proposal"] for row in decision["frames"].values()}):
        path = (session.dir / label).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Source proposal escaped the object's review scope")
        manifest = json.loads((path / "manifest.json").read_text())
        if (manifest["object"] != obj or manifest["prompt"] != prompt
                or any(manifest.get(k) != v for k, v in expected.items())
                or manifest["proposed_labels_sha256"] != digest(path / "labels.npz")):
            raise ValueError("Source proposal inputs changed or belong to another acquisition")
        for key, sha in {**manifest.get("parent_inputs", {}),
                         **manifest.get("source_domain_inputs", {})}.items():
            evidence = (session.dir / key).resolve()
            if not evidence.is_relative_to(session.dir.resolve()) or digest(evidence) != sha:
                raise ValueError("Source proposal evidence changed")
            hashes[key] = sha
        for file in [path / "manifest.json", path / "labels.npz"]:
            hashes[str(file.relative_to(session.dir.resolve()))] = digest(file)
        with np.load(path / "labels.npz") as data:
            loaded[label] = {"names": list(map(str, data["names"])), "masks": data["masks"]}
        template = manifest
    # Retain the acquisition's complete ordering, not a model-selected subset.
    names = next(iter(loaded.values()))["names"]
    if set(names) != set(current_names):
        raise ValueError("Acquisition/proposal frame sets differ; refresh source queries")
    selected = select_frames(names, decision["frames"], loaded)
    number = max((int(p.name[6:]) for p in root.glob("round_*") if p.name[6:].isdigit()), default=0) + 1
    archive = root / f"round_{number}"
    archive.mkdir()
    (archive / "source_choices.json").write_bytes(source.read_bytes())
    (archive / "source.py").write_bytes(Path(__file__).read_bytes())
    hashes[str((archive / "source_choices.json").relative_to(session.dir.resolve()))] = digest(archive / "source_choices.json")
    hashes[str((archive / "source.py").relative_to(session.dir.resolve()))] = digest(archive / "source.py")
    np.savez_compressed(archive / "labels.npz", names=names, masks=selected)
    template = dict(template, proposed_labels_sha256=digest(archive / "labels.npz"),
                    alternative_prompt="Explicit per-frame source query selection",
                    source_annotations_sha256=None, parent_inputs=hashes)
    (archive / "manifest.json").write_text(json.dumps(template, indent=2) + "\n")
    output = prepare(session, obj, prompt, label_proposal=str(archive.relative_to(session.dir.resolve())))
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["parent_inputs"].update(hashes)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    review_path = output / "review.json"
    review = json.loads(review_path.read_text())
    review["manifest_sha256"] = digest(manifest_path)
    review_path.write_text(json.dumps(review, indent=2) + "\n")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session")
    parser.add_argument("object")
    parser.add_argument("prompt")
    parser.add_argument("choices", help="Config-relative JSON source identity choices")
    args = parser.parse_args()
    from calibrated_scene_reconstruction.scene.session import Session
    print(combine(Session(args.session), args.object, args.prompt, args.choices))
