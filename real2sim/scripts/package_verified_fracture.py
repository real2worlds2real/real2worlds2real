"""Package an accepted, hash-matched native fracture asset and its reviewed movie."""

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from real2sim_harness.artifacts import atomic_json, sha256_file  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--name", default="fracture")
    args = parser.parse_args()
    if Path(args.name).name != args.name:
        raise ValueError("Use a filename stem")
    run = args.run.resolve()
    out = args.output.resolve()
    asset = args.asset.resolve()
    final = json.loads((run / "final.json").read_text())
    if final["status"] != "accepted" or final["delivery_status"] != "simulation_verified":
        raise ValueError("An accepted native asset is required")
    attempt = run / "attempts" / final["attempt_id"]
    report = final["report"]
    native = report["fracture"]
    review = report["motion_review"]
    folder = (attempt / native["directory"]).resolve()
    movie = (attempt / review["directory"]).resolve()
    if not folder.is_relative_to(attempt) or not movie.is_relative_to(attempt):
        raise ValueError("Invalid evidence path")
    if (
        not native["ok"]
        or native["source_scene_sha256"] != sha256_file(attempt / "scene.blend")
        or review["source_scene_sha256"] != native["source_scene_sha256"]
    ):
        raise ValueError("Source or native evidence mismatch")
    if sha256_file(folder / "fracture_motion.npz") != review["motion_sha256"]:
        raise ValueError("Movie uses another native cache")
    for name, digest in native["cache_manifest"].items():
        path = (attempt / name).resolve()
        if not path.is_relative_to(attempt) or sha256_file(path) != digest:
            raise ValueError("Native evidence changed")
    encoding = json.loads((movie / "encoding_report.json").read_text())
    if not encoding["ok"]:
        raise ValueError("Incomplete encoded movie")
    events = [json.loads(s) for s in (run / "events.jsonl").read_text().splitlines()]
    video_id = f"candidate-{final['attempt_id']}-{review['directory']}"
    inspections = [
        r for r in events if r.get("type") == "video_inspected" and r.get("video_id") == video_id
    ]
    if not inspections:
        raise ValueError("Final movie has no agent viewing evidence")
    if not (run / "video-review" / sha256_file(movie / "candidate_motion.mp4")).is_dir():
        raise ValueError("Movie hash does not match retained agent viewing evidence")
    out.mkdir(parents=True, exist_ok=True)
    asset.mkdir(parents=True, exist_ok=True)
    for suffix in ("mp4", "gif"):
        shutil.copy2(movie / f"candidate_motion.{suffix}", out / f"{args.name}.{suffix}")
    shutil.copy2(movie / "frames" / f"frame_{native['frames'] - 1:04d}.png", out / "preview.png")
    shutil.copy2(attempt / "hero_front_right.png", out / "intact.png")
    for name in ("scene.blend", "agent_scene.py", "object.glb"):
        shutil.copy2(attempt / name, asset / name)
    for name in (
        "fractured.blend",
        "fracture_configuration.json",
        "fracture_validation.json",
        "fracture_motion.npz",
        "initial_physics.usdc",
        "preview_report.json",
    ):
        shutil.copy2(folder / name, asset / name)
    runtime = asset / "runtime"
    runtime.mkdir(exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    for path in (
        root / "scripts/simulate_physx_fracture.py",
        root / "src/real2sim_harness/fracture_math.py",
        root / "src/real2sim_harness/blender_fracture_harness.py",
    ):
        shutil.copy2(path, runtime / path.name)
    reviewed = sorted({i for r in inspections for i in r["frames"]})
    evidence = {
        "ok": True,
        "status": "accepted",
        "asset_kind": "fracture",
        "source_run": str(run),
        "attempt": final["attempt_id"],
        "native_directory": native["directory"],
        "source_scene_sha256": native["source_scene_sha256"],
        "video": f"{args.name}.mp4",
        "video_sha256": sha256_file(out / f"{args.name}.mp4"),
        "motion_sha256": review["motion_sha256"],
        "backend": native["backend"],
        "actual_physx_fragment_motion": True,
        "frames": native["frames"],
        "fps": native["fps"],
        "duration_seconds": native["frames"] / native["fps"],
        "pieces": native["pieces"],
        "coverage": native["coverage"],
        "break_event": native["break_event"],
        "retained_state_audit": native["retained_state_audit"],
        "final_max_speed_m_s": native["final_max_speed_m_s"],
        "astra_review": {
            "accepted": True,
            "unique_frames": len(reviewed),
            "frames": reviewed,
            "complete_frame_by_frame_review": len(reviewed) == native["frames"],
            "calls": [{k: r.get(k) for k in ("mode", "frames", "crop")} for r in inspections],
            "reason": final["reason"],
        },
        "scope": "Accepted source-bound prefractured asset. Predefined fracture surfaces and global contact-triggered release; no claim of calibrated material failure, crack-growth prediction or robot policy.",
    }
    atomic_json(out / "evidence.json", evidence)
    demo = asset / "demo"
    demo.mkdir(exist_ok=True)
    for name in (f"{args.name}.mp4", "evidence.json", "preview.png", "intact.png"):
        shutil.copy2(out / name, demo / name)
    readme = """real2sim — contact-triggered prefractured asset

scene.blend and agent_scene.py retain the intact editable source.
object.glb is a static portable appearance, not a fracture runtime.
fractured.blend retains the intact source and closed prefractured pieces.
fracture_configuration.json retains editable geometry, mass, contact thresholds,
friction and control assumptions. fracture_motion.npz records native PhysX motion.
fracture_validation.json records activation, state transfer, energy and contact checks.
initial_physics.usdc contains the intact body and inactive dynamic fragments.
Loading the USD alone does not execute the threshold and activation controller.

Run scripts/simulate_physx_fracture.py from a compatible real2sim checkout.
See docs/ISAAC_SIM_51.md and docs/ENVIRONMENTS.md for setup. The script accepts
the directory containing fracture_configuration.json and options such as
--frames 240 --viz none --device cpu. runtime/ retains relevant source snapshots;
their framework dependencies and Blender replay tools belong to the checkout.

Fracture surfaces are predefined; contact triggers release of dynamic pieces.
This does not establish real-time crack growth or measured failure parameters.
PhysX computes fragment contact and settling without an added radial impulse.
demo/ contains the reviewed movie and evidence, including numerical tolerances
and the actual inspection coverage. Review does not imply every frame was viewed.
"""
    (asset / "README.txt").write_text(readme)
    bundle = {
        "schema_version": 1,
        "ok": True,
        "status": "simulation_verified_and_astra_reviewed",
        "asset_kind": "fracture",
        "editable_scene": "scene.blend",
        "fractured_scene": "fractured.blend",
        "native_configuration": "fracture_configuration.json",
        "native_initial_scene": "initial_physics.usdc",
        "demonstration_cache": "fracture_motion.npz",
        "source_scene_sha256": native["source_scene_sha256"],
        "files": {
            str(p.relative_to(asset)): sha256_file(p)
            for p in sorted(asset.rglob("*"))
            if p.is_file() and p.name != "asset_bundle.json"
        },
    }
    atomic_json(asset / "asset_bundle.json", bundle)
    archive = out / f"{args.name}_asset.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(asset.rglob("*")):
            if path.is_file():
                z.write(path, f"{args.name}/" + str(path.relative_to(asset)))
    print(
        json.dumps(
            {
                "ok": True,
                "video": str(out / f"{args.name}.mp4"),
                "asset_zip": str(archive),
                "asset_zip_bytes": archive.stat().st_size,
                "reviewed_frames": len(reviewed),
            }
        )
    )


if __name__ == "__main__":
    main()
