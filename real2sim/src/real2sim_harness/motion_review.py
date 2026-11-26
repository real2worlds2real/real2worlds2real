"""Optionally render source-matched simulator motion for the common video reader."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from .artifacts import atomic_json, sha256_file
from .blender import FinalRenderResult


def matching_review(attempt, hybrid, review):
    """A review becomes stale when either the source, trajectory or movie changes."""
    if not review.get("complete_frame_coverage") or review.get(
        "source_scene_sha256"
    ) != sha256_file(attempt / "scene.blend"):
        return False
    cache = (attempt / hybrid["directory"] / "hybrid_motion.npz").resolve()
    if not cache.is_relative_to(attempt.resolve()) or review.get("motion_sha256") != sha256_file(
        cache
    ):
        return False
    files = review.get("file_manifest", {})
    if not files:
        return False
    for name, digest in files.items():
        path = (attempt / name).resolve()
        if (
            not path.is_relative_to(attempt.resolve())
            or not path.is_file()
            or sha256_file(path) != digest
        ):
            return False
    return True


def prepare_candidate_motion(executor, store, attempt_id):
    """Called only after candidate numeric, appearance and simulator checks pass."""
    attempt = store.attempts_dir / attempt_id
    report = json.loads((attempt / "report.json").read_text())
    if report.get("selected_simulation_review") == "native_deformable":
        from .source_deformable import prepare_motion
        return prepare_motion(store, attempt_id, report)
    if report.get("selected_simulation_review") == "rigid_contact":
        from .contact_asset import prepare_motion
        return prepare_motion(store, attempt_id, report)
    if report.get("solid"):
        from .solids import prepare_solid_motion
        return prepare_solid_motion(executor, store, attempt_id, report)
    if report.get("fracture"):
        from .fracture import prepare_fracture_motion
        return prepare_fracture_motion(executor, store, attempt_id, report)
    if report.get("strand"):
        return prepare_strand_motion(executor, store, attempt_id, report)
    if report.get("fluid"):
        return prepare_fluid_motion(executor, store, attempt_id, report)
    if "hybrid" not in report:
        return prepare_simulation_motion(executor, store, attempt_id, report)
    hybrid = report["hybrid"]
    output = (
        attempt
        / f"motion-review-{len(list(attempt.glob('motion-review-[0-9][0-9][0-9]'))) + 1:03d}"
    )
    output.mkdir()
    root = Path(__file__).resolve().parents[2]
    source = attempt / "scene.blend"
    folder = (attempt / hybrid["directory"]).resolve()
    if not folder.is_relative_to(attempt.resolve()):
        raise ValueError("Hybrid cache directory is outside the saved attempt")
    source_digest, motion_digest = sha256_file(source), sha256_file(folder / "hybrid_motion.npz")
    commands = [
        (
            "render",
            [
                str(executor.blender),
                "--background",
                str(source),
                "--python",
                str(Path(__file__).with_name("blender_hybrid_harness.py")),
                "--",
                str(folder),
                "--mode",
                "render",
                "--output",
                str(output),
                "--resolution",
                "768",
                "--aspect",
                str(4 / 3),
                "--samples",
                "32",
            ],
        ),
        (
            "encode",
            [
                sys.executable,
                str(root / "scripts/encode_object_demo.py"),
                str(output),
                "--name",
                "candidate_motion",
            ],
        ),
        (
            "index",
            [
                sys.executable,
                str(root / "scripts/prepare_motion_review.py"),
                str(output / "frames"),
                "--output",
                str(output / "timeline"),
                "--fps",
                str(hybrid["fps"]),
                "--video",
                str(output / "candidate_motion.mp4"),
            ],
        ),
    ]
    for name, command in commands:
        with (output / f"{name}.log").open("w") as stream:
            result = subprocess.run(
                command,
                cwd=root,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=executor.timeout_seconds,
            )
        if result.returncode:
            raise RuntimeError(
                f"Candidate motion {name} failed: "
                + (output / f"{name}.log").read_text(errors="replace")[-4000:]
            )
    index = json.loads((output / "timeline/index.json").read_text())
    encoding = json.loads((output / "encoding_report.json").read_text())
    if (
        sha256_file(source) != source_digest
        or sha256_file(folder / "hybrid_motion.npz") != motion_digest
        or not encoding["ok"]
        or index["frames"] != hybrid["frames"]
    ):
        raise ValueError("Candidate source/cache changed or final motion evidence is incomplete")
    review = {
        "status": "awaiting_agent_review",
        "complete_frame_coverage": True,
        "source_scene_sha256": source_digest,
        "motion_sha256": motion_digest,
        "frames": index["frames"],
        "directory": str(output.relative_to(attempt)),
        "representation": index["representation"],
        "file_manifest": {
            str(path.relative_to(attempt)): sha256_file(path)
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.suffix != ".log"
        },
    }
    report["motion_review"] = review
    atomic_json(attempt / "report.json", report)
    store.append_event(
        {
            "type": "candidate_motion_returned_for_review",
            "attempt_id": attempt_id,
            "frames": index["frames"],
        }
    )
    summary = {key: value for key, value in review.items() if key != "file_manifest"}
    summary.update(
        acceptance_pending_review=True,
        next_action="Use inspect_video to review the rendered operation, both sides and suspicious intervals at full resolution. Repair and resimulate any demonstrated defect. When satisfied, call accept_result with your motion-review findings. Rendering a movie does not accept delivery.",
    )
    return FinalRenderResult(
        attempt_id, True, summary, [output / "timeline" / name for name in index["sheets"]]
    )


def prepare_strand_motion(executor, store, attempt_id, report):
    """Expose actual native rod trajectories through the shared movie reader."""
    attempt = store.attempts_dir / attempt_id
    native = report['strand']
    source = attempt / 'scene.blend'
    folder = (attempt / native['directory']).resolve()
    if not native.get('ok') or not folder.is_relative_to(attempt.resolve()) or native.get('source_scene_sha256') != sha256_file(source):
        raise ValueError('Use a source-matched successful native strand cache')
    for name, digest in native.get('cache_manifest', {}).items():
        path = (attempt / name).resolve()
        if not path.is_relative_to(attempt.resolve()) or sha256_file(path) != digest:
            raise ValueError('Native strand evidence changed after verification')
    output = attempt / f"motion-review-{len(list(attempt.glob('motion-review-[0-9][0-9][0-9]'))) + 1:03d}"
    output.mkdir()
    root = Path(__file__).resolve().parents[2]
    commands = [('render', [str(executor.blender), '--background', str(source), '--python',
                  str(Path(__file__).with_name('blender_strand_harness.py')), '--', str(folder),
                  '--mode', 'render', '--output', str(output), '--resolution', '1024', '--samples', '64']),
                ('encode', [sys.executable, str(root/'scripts/encode_object_demo.py'), str(output), '--name', 'candidate_motion'])]
    for name, command in commands:
        with (output/f'{name}.log').open('w') as log:
            p = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=executor.timeout_seconds, check=False)
        if p.returncode:
            raise RuntimeError('Strand movie failed: '+(output/f'{name}.log').read_text(errors='replace')[-4000:])
    if sha256_file(source) != native['source_scene_sha256']:
        raise ValueError('Strand source changed during movie rendering')
    review = {'status':'awaiting_agent_review', 'directory':str(output.relative_to(attempt)),
              'backend':native['backend'], 'source_scene_sha256':sha256_file(source),
              'motion_sha256':sha256_file(folder/'strand_motion.npz'), 'frames':native['frames'],
              'video':str((output/'candidate_motion.mp4').relative_to(attempt)), 'complete_video_encoded':True,
              'review_policy':'Encoding does not accept an asset; agent chooses viewing coverage and judges the motion.'}
    report['motion_review'] = review
    atomic_json(attempt/'report.json', report)
    return FinalRenderResult(attempt_id, True, review, [])


def read_behavior_plan(store, attempt_id):
    """Keep the chosen operation explicit and tied to its modeled candidate."""
    path = store.workspace_dir / "motion_plan.json"
    if not path.is_file():
        raise ValueError(
            "For a rigid/articulated movie, write workspace motion_plan.json as "
            '{"attempt_id":"' + attempt_id + '","plan":{"duration":6,"fps":24,'
            '"controls":[{"joint":"EXACT_SIMULATOR_JOINT_NAME",'
            '"keyframes":[[0,"lower"],[3,"upper"],[6,"lower"]]}]}}. '
            "Choose the intended operation and sequencing yourself. Rigid drops use "
            "controls=[] and a positive root_height. This file is not needed for hybrid replay."
        )
    record = json.loads(path.read_text())
    if record.get("attempt_id") != attempt_id or not isinstance(record.get("plan"), dict):
        raise ValueError(
            "motion_plan.json must name the selected attempt and contain a plan object"
        )
    return record["plan"]


def prepare_simulation_motion(executor, store, attempt_id, report):
    """Expose target-backend trajectories through the common movie tool."""
    from .sim_bundle import verify_bundle

    attempt = store.attempts_dir / attempt_id
    simulation = report.get("simulation", {})
    if not simulation.get("ok") or not simulation.get("visual_render_verified"):
        raise ValueError("Finish target simulation with rendered review before requesting a movie")
    folder = (attempt / simulation["directory"]).resolve()
    if not folder.is_relative_to(attempt.resolve()):
        raise ValueError("Simulator directory is outside the saved attempt")
    if not verify_bundle(folder)["ok"]:
        raise ValueError("Simulator bundle changed after validation")
    bundle = json.loads((folder / "asset_bundle.json").read_text())
    source = attempt / "scene.blend"
    source_digest = sha256_file(source)
    deformable = bundle["asset_kind"] == "deformable"
    if bundle["asset_kind"] not in {"rigid", "articulated", "deformable"}:
        raise ValueError(
            "This movie route supports rigid/articulated PhysX, Newton cloth and hybrid replay"
        )
    if bundle["source_blend_sha256"] != source_digest:
        raise ValueError("Simulator bundle does not match the editable source")
    plan = (
        {"replay_bundle_trajectory": True} if deformable else read_behavior_plan(store, attempt_id)
    )
    output = (
        attempt
        / f"motion-review-{len(list(attempt.glob('motion-review-[0-9][0-9][0-9]'))) + 1:03d}"
    )
    output.mkdir()
    behavior = output / "behavior"
    plan_path = output / "control_plan.json"
    atomic_json(plan_path, plan)
    root = Path(__file__).resolve().parents[2]
    local_runtime = os.environ.get("REAL2SIM_SIM_RUNTIME", "docker") == "local"

    def runtime(path):
        # The local launcher takes host paths; task directories may live outside the checkout.
        if local_runtime:
            return str(path.resolve())
        return str(Path("/workspace/real2sim_harness") / path.resolve().relative_to(root))

    commands = [
        (
            "simulate",
            [
                str(root / "scripts/run_isaac_lab3.sh"),
                runtime(root / "scripts/simulate_physx_behavior.py"),
                runtime(folder),
                "--plan",
                runtime(plan_path),
                "--output",
                runtime(behavior),
                "--viz",
                "none",
                "--device",
                "cuda:0",
            ],
        ),
        (
            "render",
            [
                str(executor.blender),
                "--background",
                str(source),
                "--python",
                str(root / "scripts/render_physics_behavior.py"),
                "--",
                str(behavior),
                "--output",
                str(output),
                "--resolution",
                "768",
                "--samples",
                "32",
            ],
        ),
        (
            "encode",
            [
                sys.executable,
                str(root / "scripts/encode_object_demo.py"),
                str(output),
                "--name",
                "candidate_motion",
            ],
        ),
    ]
    if deformable:
        commands[0] = (
            "package",
            [
                sys.executable,
                str(root / "scripts/package_cloth_behavior.py"),
                str(folder),
                str(folder / "simulation_motion.npz"),
                str(folder / bundle["validation"]),
                "--output",
                str(behavior),
            ],
        )
    for name, command in commands:
        with (output / f"{name}.log").open("w") as stream:
            result = subprocess.run(
                command,
                cwd=root,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=executor.timeout_seconds,
            )
        if result.returncode:
            raise RuntimeError(
                f"Candidate motion {name} failed: "
                + (output / f"{name}.log").read_text(errors="replace")[-4000:]
            )
    evidence = json.loads((behavior / "behavior_report.json").read_text())
    encoding = json.loads((output / "encoding_report.json").read_text())
    if (
        not evidence["ok"]
        or not encoding["ok"]
        or sha256_file(source) != source_digest
        or evidence["source_scene_sha256"] != source_digest
        or not verify_bundle(folder)["ok"]
    ):
        raise ValueError("Source/bundle changed or movie evidence did not pass")
    cache = behavior / "behavior_motion.npz"
    review = {
        "status": "awaiting_agent_review",
        "directory": str(output.relative_to(attempt)),
        "source_scene_sha256": source_digest,
        "motion_sha256": sha256_file(cache),
        "motion_cache": str(cache.relative_to(attempt)),
        "frames": evidence["frames"],
        "backend": bundle["backend"],
        "actual_simulator_motion": True,
        "video": str((output / "candidate_motion.mp4").relative_to(attempt)),
        "complete_video_encoded": True,
        "review_policy": "Encoding is not acceptance; agent judgment and viewing coverage are recorded separately.",
        "scope": "Caller-chosen physical test operation; not a robot policy or measured physics calibration.",
    }
    report["motion_review"] = review
    atomic_json(attempt / "report.json", report)
    store.append_event(
        {
            "type": "candidate_motion_returned_for_review",
            "attempt_id": attempt_id,
            "backend": bundle["backend"],
            "frames": evidence["frames"],
        }
    )
    return FinalRenderResult(attempt_id, True, review, [])


def prepare_fluid_motion(executor, store, attempt_id, report):
    """Render the verified MPM surfaces and their retained material-phase state."""
    attempt = store.attempts_dir / attempt_id
    fluid = report["fluid"]
    folder = (attempt / fluid["directory"]).resolve()
    source = attempt / "scene.blend"
    if not folder.is_relative_to(attempt.resolve()) or not fluid.get("ok"):
        raise ValueError("Use a successful fluid cache inside the selected attempt")
    if fluid.get("source_scene_sha256") != sha256_file(source):
        raise ValueError("Fluid source changed after simulation")
    for name, digest in fluid.get("cache_manifest", {}).items():
        if sha256_file(attempt / name) != digest:
            raise ValueError("Fluid motion or material state changed after verification")
    output = (
        attempt
        / f"motion-review-{len(list(attempt.glob('motion-review-[0-9][0-9][0-9]'))) + 1:03d}"
    )
    output.mkdir()
    root = Path(__file__).resolve().parents[2]
    commands = [
        (
            "render",
            [
                str(executor.blender),
                "--background",
                str(source),
                "--python",
                str(Path(__file__).with_name("blender_fluid_harness.py")),
                "--",
                str(folder),
                "--mode",
                "render",
                "--output",
                str(output),
                "--resolution",
                "768",
                "--aspect",
                str(4 / 3),
                "--samples",
                "64",
            ],
        ),
        (
            "encode",
            [
                sys.executable,
                str(root / "scripts/encode_object_demo.py"),
                str(output),
                "--name",
                "candidate_motion",
            ],
        ),
    ]
    for name, command in commands:
        with (output / f"{name}.log").open("w") as log:
            p = subprocess.run(
                command,
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=executor.timeout_seconds,
                check=False,
            )
        if p.returncode:
            raise RuntimeError(
                f"Fluid movie {name} failed: "
                + (output / f"{name}.log").read_text(errors="replace")[-4000:]
            )
    encoding = json.loads((output / "encoding_report.json").read_text())
    if not encoding["ok"]:
        raise ValueError("Fluid movie encoding did not pass")
    review = {
        "status": "awaiting_agent_review",
        "backend": "Newton implicit MPM",
        "source_scene_sha256": sha256_file(source),
        "frames": fluid["frames"],
        "directory": str(output.relative_to(attempt)),
        "motion_cache": str((folder / "particle_motion.npz").relative_to(attempt)),
        "motion_sha256": sha256_file(folder / "particle_motion.npz"),
        "video": str((output / "candidate_motion.mp4").relative_to(attempt)),
        "actual_simulator_motion": True,
        "complete_video_encoded": True,
    }
    report["motion_review"] = review
    atomic_json(attempt / "report.json", report)
    store.append_event(
        {
            "type": "candidate_motion_returned_for_review",
            "attempt_id": attempt_id,
            "backend": review["backend"],
            "frames": fluid["frames"],
        }
    )
    return FinalRenderResult(attempt_id, True, review, [])
