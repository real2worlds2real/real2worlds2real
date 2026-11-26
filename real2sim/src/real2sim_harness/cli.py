from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from .agent import TOOLS, run_agent
from .artifacts import RunStore, atomic_json
from .blender import BlenderExecutor
from .capabilities import TASK_TOOLS as CODEX_TOOLS
from .providers import chatgpt_plan_client, openai_client
from .video import extract_keyframes, prepare_reference_inputs
from .video_review import retain_input_videos


def run_codex_agent(*args, **kwargs):
    """Load the optional managed transport only when that entrypoint is selected."""
    from .codex_backend import run_codex_agent as run
    return run(*args, **kwargs)


def blender_path(value: str | None) -> Path | None:
    found = value or shutil.which("blender")
    return Path(found).resolve() if found else None


def doctor(args) -> int:
    blender = blender_path(args.blender)
    report = {
        "python": sys.version.split()[0],
        "blender": str(blender) if blender else None,
        "blender_ready": bool(blender and blender.is_file()),
        "task_execution_ready": sys.platform == "linux" and bool(shutil.which("bwrap")),
        "openai_api_key_set": bool(os.environ.get("OPENAI_API_KEY")),
        "chatgpt_plan_auth_dir": str(args.auth_dir.resolve()) if args.auth_dir else None,
        "chatgpt_plan_auth_present": bool(
            args.auth_dir and (args.auth_dir / "credentials.json").is_file()
        ),
    }
    print(json.dumps(report, indent=2))
    return 0 if report["blender_ready"] and report["task_execution_ready"] else 1


def generate(args) -> int:
    if getattr(args, 'scene', False) and args.backend != 'codex-app-server':
        raise SystemExit('Scene orchestration requires codex-app-server')
    if args.cousins is not None and args.cousins < 0:
        raise SystemExit("Cousin quantity must be nonnegative; 0 disables the optional stage")
    if args.backend != "codex-app-server" and (args.cousins or args.cousin_prompt):
        raise SystemExit("The optional cousin stage requires codex-app-server")
    if getattr(args, 'measurements', None) and args.backend != 'codex-app-server':
        raise SystemExit('Measurement tools require codex-app-server')
    blender = blender_path(args.blender)
    if blender is None or not blender.is_file():
        raise SystemExit("Blender not found; pass --blender or run `real2sim-harness doctor`")
    if args.asset_kind == "auto" and args.backend != "codex-app-server":
        raise SystemExit("Automatic asset analysis currently requires --backend codex-app-server")
    if args.asset_kind in {"deformable", "hybrid", "fluid", "strand", "fracture", "solid"} and args.backend != "codex-app-server":
        raise SystemExit("Native deformation tools currently require --backend codex-app-server")
    schemas = CODEX_TOOLS if args.backend == "codex-app-server" else TOOLS
    with tempfile.TemporaryDirectory(prefix="real2sim-harness-inputs-") as temporary:
        references, inputs = prepare_reference_inputs(
            [Path(path) for path in args.reference], Path(temporary), video_frames=args.video_frames
        )
        store = RunStore.create(
            args.runs,
            references=references,
            objective=args.objective,
            model=args.model,
            max_turns=args.max_turns,
            tool_schemas=schemas,
            asset_kind=args.asset_kind,
            front_axis=args.front_axis,
            reference_mask_dir=args.reference_mask_dir,
            required_parts=args.required_part,
        )
    retain_input_videos(store.root, inputs)
    if getattr(args, 'scene', False):
        record = json.loads((store.root / 'run.json').read_text())
        record['generation_mode'] = 'scene'
        record['contract_inference_required'] = False
        if args.scene_runner:
            runner = args.scene_runner.resolve()
            if not runner.is_file():
                raise SystemExit('--scene-runner must name an existing adapter configuration')
            record['scene_runner'] = str(runner)
        if getattr(args, 'gaussian_runner', None):
            runner = args.gaussian_runner.resolve()
            if not runner.is_file():
                raise SystemExit('--gaussian-runner must name an existing backend configuration')
            record['gaussian_runner'] = str(runner)
        atomic_json(store.root / 'run.json', record)
    if getattr(args, 'measurements', None):
        from .measurement import retain_inputs
        retain_inputs(store, args.measurements)
    atomic_json(store.root / "input_manifest.json", {"inputs": inputs})
    executor = BlenderExecutor(blender, timeout_seconds=args.blender_timeout)
    if args.backend == "codex-app-server":
        if args.auth == "chatgpt-plan" and args.auth_dir is None:
            raise SystemExit("--auth-dir is required for chatgpt-plan")
        result = run_codex_agent(
            executor,
            store,
            model=args.model,
            objective=args.objective,
            auth_mode=args.auth.replace("-", "_"),
            auth_dir=args.auth_dir,
            codex=args.codex,
            seed_script=args.seed_script,
            web_search=args.web_search,
            max_turns=args.max_turns,
            max_calls=args.max_calls,
            reasoning_effort=args.reasoning_effort,
            timeout_seconds=args.agent_timeout,
        )
    else:
        if args.auth == "api-key":
            client = openai_client()
        else:
            if args.auth_dir is None:
                raise SystemExit("--auth-dir is required for chatgpt-plan")
            client = chatgpt_plan_client(args.auth_dir)
        try:
            result = run_agent(
                client,
                executor,
                store,
                model=args.model,
                objective=args.objective,
                max_turns=args.max_turns,
                max_output_tokens=args.max_output_tokens,
                reasoning_effort=args.reasoning_effort,
                auth_dir=args.auth_dir,
            )
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                close()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] == "accepted" and args.backend == "codex-app-server":
        run_record = json.loads((store.root / "run.json").read_text())
        if args.cousins != 0 and (args.cousins or args.cousin_prompt or run_record.get("cousin_request")):
            family = _start_cousins(args, store.root,
                                    prompt=args.cousin_prompt or run_record.get("cousin_request", ""), count=args.cousins)
            print(json.dumps(family, ensure_ascii=False, indent=2))
            return 0 if family["status"] == "complete" or (family["status"] == "planned" and family.get("execute") is False) else 2
    return 0 if result["status"] == "accepted" else 2


def extract_video(args) -> int:
    manifest = extract_keyframes(args.video, args.output, count=args.frames)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _cousin_options(args):
    return {"auth_mode": args.auth.replace("-", "_"), "auth_dir": args.auth_dir,
            "codex": args.codex,
            "web_search": args.web_search, "max_turns": args.max_turns,
            "max_calls": args.max_calls, "reasoning_effort": args.reasoning_effort,
            "timeout_seconds": args.agent_timeout}


def _start_cousins(args, parent, *, prompt, count=None):
    from .cousins import create_planning_run, run_family
    blender = blender_path(args.blender)
    if not blender or not blender.is_file():
        raise SystemExit("Blender not found; pass --blender")
    executor = BlenderExecutor(blender, timeout_seconds=args.blender_timeout)
    options = _cousin_options(args)
    family, planner = create_planning_run(parent, executor, prompt=prompt, count=count,
                                         max_turns=args.max_turns, model=getattr(args, "model", None),
                                         scene_runner=getattr(args,'scene_runner',None),
                                         gaussian_runner=getattr(args,'gaussian_runner',None))
    planning_run = json.loads((planner.root / "run.json").read_text())
    result = run_codex_agent(executor, planner, model=planning_run["model"],
                             objective=planning_run["objective"], **options)
    if result["status"] != "planned":
        return {**json.loads((family / "family.json").read_text()), "planner_status": result["status"]}
    return run_family(family, executor, agent_options=options)


def generate_cousins(args):
    from .cousins import run_family
    if args.resume:
        blender = blender_path(args.blender)
        if not blender or not blender.is_file():
            raise SystemExit("Blender not found; pass --blender")
        executor = BlenderExecutor(blender, timeout_seconds=args.blender_timeout)
        options = _cousin_options(args)
        batch = json.loads((args.parent / "family.json").read_text())
        if batch.get('family_kind')=='scene':
            for key in ('scene_runner','gaussian_runner'):
                value=getattr(args,key,None)
                if value is not None:
                    path=value.resolve()
                    if not path.is_file():
                        raise SystemExit('Scene cousin operator configuration does not exist: '+key)
                    batch.setdefault('runners',{})[key]=str(path)
                    for row in batch['variants']:
                        if row.get('run') and row['status']!='accepted':
                            run_path=args.parent/row['run']/'run.json'
                            record=json.loads(run_path.read_text())
                            record[key]=str(path)
                            atomic_json(run_path,record)
            atomic_json(args.parent/'family.json',batch)
        if batch["status"] == "awaiting_plan":
            planner = RunStore(args.parent / batch["planner"])
            record = json.loads((planner.root / "run.json").read_text())
            planned = run_codex_agent(executor, planner, model=record["model"], objective=record["objective"],
                                      resume_existing=(planner.root / "codex/thread.json").is_file(), **options)
            if planned["status"] != "planned":
                print(json.dumps(planned, ensure_ascii=False, indent=2))
                return 2
        result = run_family(args.parent, executor, agent_options=options, execute_plan=True)
    else:
        result = _start_cousins(args, args.parent, prompt=args.prompt, count=args.count)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "complete" or (result["status"] == "planned" and result.get("execute") is False) else 2


def resume_generation(args) -> int:
    store = RunStore(args.run.resolve())
    run = json.loads((store.root / "run.json").read_text())
    turn_limit = getattr(args, 'max_turns', None)
    if turn_limit is not None:
        if not 1 <= turn_limit <= 30:
            raise SystemExit('--max-turns must be from 1 to 30 for this invocation')
        run['max_turns'] = turn_limit
    blender = blender_path(args.blender)
    if not blender or not blender.is_file():
        raise SystemExit("Blender not found; pass --blender")
    if turn_limit is not None:
        atomic_json(store.root / 'run.json', run)
    result = run_codex_agent(
        BlenderExecutor(blender, timeout_seconds=args.blender_timeout), store,
        model=run["model"], objective=run["objective"], auth_mode=args.auth.replace("-", "_"),
        auth_dir=args.auth_dir, codex=args.codex,
        web_search=args.web_search, max_turns=run["max_turns"], max_calls=args.max_calls,
        reasoning_effort=args.reasoning_effort, timeout_seconds=args.agent_timeout,
        resume_existing=True,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] == "accepted" and run.get("generation_mode", "twin") == "twin":
        latest = json.loads((store.root / "run.json").read_text())
        if latest.get("cousin_request"):
            args.max_turns = run["max_turns"]
            family = _start_cousins(args, store.root, prompt=latest["cousin_request"])
            print(json.dumps(family, ensure_ascii=False, indent=2))
            return 0 if family["status"] == "complete" or (family["status"] == "planned" and family.get("execute") is False) else 2
    return 0 if result["status"] == "accepted" else 2


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="real2sim-harness")
    commands = root.add_subparsers(dest="command", required=True)
    from .external_agent import add_parser
    add_parser(commands)
    from . import delivery
    delivery.add_parser(commands)
    check = commands.add_parser("doctor")
    check.add_argument("--blender")
    check.add_argument("--auth-dir", type=Path)
    check.set_defaults(func=doctor)

    video = commands.add_parser("extract-video")
    video.add_argument("video", type=Path)
    video.add_argument("--output", type=Path, required=True)
    video.add_argument("--frames", type=int, default=12)
    video.set_defaults(func=extract_video)

    scene_review = commands.add_parser('reopen-scene', help='retain an accepted scene and reopen Astra review for new evidence or reported defects')
    scene_review.add_argument('run', type=Path)
    scene_review.add_argument('--reason', required=True)
    def reopen_scene_command(args):
        from .scene_review import reopen_scene_review
        print(json.dumps(reopen_scene_review(RunStore(args.run.resolve()), reason=args.reason), indent=2))
        return 0
    scene_review.set_defaults(func=reopen_scene_command)

    run = commands.add_parser("generate")
    run.add_argument("--reference", action="append", required=True, help="input image or video; repeat for multiple inputs")
    run.add_argument("--video-frames", type=int, default=12)
    run.add_argument('--measurements', type=Path, help='optional metric evidence JSON: dimensions, board/camera, or observed surface points')
    run.add_argument('--scene', action='store_true', help='Astra orchestrates a complete scene instead of one object')
    run.add_argument('--scene-runner', type=Path, help='operator-owned simulator adapter configuration; independent of simulator version')
    run.add_argument('--gaussian-runner', type=Path, help='optional operator-owned Gaussian training/transcoding backend; CPU preparation is built in')
    run.add_argument("--cousins", type=int, help="optional count override; 0 disables the post-twin cousin stage")
    run.add_argument("--cousin-prompt", help="optional free-form cousin request; quantity may be stated in the prompt")
    run.add_argument(
        "--objective",
        default="Generate an editable, visually faithful 3D asset of the main object in the supplied references. Infer its essential parts and motion, review real render evidence, and document assumptions.",
    )
    run.add_argument(
        "--asset-kind",
        choices=("auto", "rigid", "articulated", "deformable", "hybrid", "fluid", "strand", "fracture", "solid"),
        default="auto",
    )
    run.add_argument("--front-axis", choices=("+X", "-X", "+Y", "-Y"), default="+Y")
    run.add_argument("--reference-mask-dir", type=Path)
    run.add_argument("--required-part", action="append", default=[])
    run.add_argument("--model", default="gpt-6-astra")
    run.add_argument(
        "--backend", choices=("codex-app-server", "responses"), default="codex-app-server"
    )
    run.add_argument("--auth", choices=("api-key", "chatgpt-plan"), default="chatgpt-plan")
    run.add_argument("--auth-dir", type=Path)
    run.add_argument("--codex", default="codex")
    run.add_argument("--web-search", choices=("live", "cached", "disabled"), default="disabled")
    run.add_argument(
        "--seed-script",
        type=Path,
        help="optional prior Blender Python source copied into the agent workspace",
    )
    run.add_argument("--blender")
    run.add_argument("--runs", type=Path, default=Path("runs"))
    run.add_argument("--max-turns", type=int, default=12)
    run.add_argument(
        "--max-calls",
        type=int,
        default=36,
        help="maximum Blender execution attempts for the Codex backend",
    )
    run.add_argument("--max-output-tokens", type=int, default=24_000)
    run.add_argument(
        "--reasoning-effort", choices=("low", "medium", "high", "xhigh"), default="high"
    )
    run.add_argument("--blender-timeout", type=int, default=600)
    run.add_argument("--agent-timeout", type=int, default=2700)
    run.set_defaults(func=generate)
    resume = commands.add_parser("resume", help="resume a retained Codex generation run without resetting evidence or call counters")
    resume.add_argument("run", type=Path)
    resume.add_argument("--blender", required=True)
    resume.add_argument("--auth", choices=("api-key", "chatgpt-plan"), default="chatgpt-plan")
    resume.add_argument("--auth-dir", type=Path)
    resume.add_argument("--codex", default="codex")
    resume.add_argument("--web-search", choices=("live", "cached", "disabled"), default="disabled")
    resume.add_argument("--max-calls", type=int, default=36)
    resume.add_argument("--max-turns", type=int, help="turn limit for this explicit resume; defaults to the retained setting; cumulative counters are preserved")
    resume.add_argument("--reasoning-effort", choices=("low", "medium", "high", "xhigh"), default="high")
    resume.add_argument("--blender-timeout", type=int, default=600)
    resume.add_argument("--agent-timeout", type=int, default=1800)
    resume.set_defaults(func=resume_generation)
    cousins = commands.add_parser("cousins", help="plan and generate object or whole-scene cousins after an accepted twin")
    cousins.add_argument("parent", type=Path, help="accepted twin run, or family directory with --resume")
    cousins.add_argument("--prompt", default="", help="natural-language quantity, shared intent and per-variant wishes")
    cousins.add_argument("--count", type=int, help="optional explicit quantity override; otherwise Astra interprets --prompt")
    cousins.add_argument("--resume", action="store_true", help="continue an existing family without rebuilding accepted siblings")
    cousins.add_argument("--model")
    cousins.add_argument('--scene-runner',type=Path,help='optional scene-family simulator adapter override')
    cousins.add_argument('--gaussian-runner',type=Path,help='optional scene-family Gaussian backend override')
    cousins.add_argument("--blender", required=True)
    cousins.add_argument("--auth", choices=("api-key", "chatgpt-plan"), default="chatgpt-plan")
    cousins.add_argument("--auth-dir", type=Path)
    cousins.add_argument("--codex", default="codex")
    cousins.add_argument("--web-search", choices=("live", "cached", "disabled"), default="disabled")
    cousins.add_argument("--max-turns", type=int, default=12)
    cousins.add_argument("--max-calls", type=int, default=36)
    cousins.add_argument("--reasoning-effort", choices=("low", "medium", "high", "xhigh"), default="high")
    cousins.add_argument("--blender-timeout", type=int, default=1200)
    cousins.add_argument("--agent-timeout", type=int, default=2700)
    cousins.set_defaults(func=generate_cousins)
    return root


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in {"pipeline", "evolve"}:
        from .scene_pipeline import forward
        return forward(argv[0], argv[1:])
    args = parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
