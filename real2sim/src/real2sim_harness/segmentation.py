"""Model-independent mask evidence and explicitly heuristic temporal association."""
from __future__ import annotations

import numpy as np


def prompt_box_inputs(view, prompt, image_size):
    """Caller-reviewed spatial hints in displayed-image pixels, not geometry."""
    hints = view.get('prompt_boxes', {})
    if not isinstance(hints, dict):
        raise ValueError('prompt_boxes must map prompt names to spatial hints')
    if prompt not in hints:
        return None
    hint = hints[prompt]
    if not isinstance(hint, dict) or set(hint)-{'boxes', 'labels', 'box_only'}:
        raise ValueError('Spatial hints use boxes, optional labels and box_only')
    boxes = np.asarray(hint.get('boxes'), dtype=float)
    width, height = image_size
    if (boxes.ndim != 2 or boxes.shape[1] != 4 or not 1 <= len(boxes) <= 32
            or not np.isfinite(boxes).all() or np.any(boxes[:, :2] < 0)
            or np.any(boxes[:, 2:] > [width, height])
            or np.any(boxes[:, 2:] <= boxes[:, :2])):
        raise ValueError('Use1..32 finite positive-area xyxy boxes inside the displayed image')
    labels = hint.get('labels', [1]*len(boxes))
    if (not isinstance(labels, list) or len(labels) != len(boxes)
            or any(type(label) is not int or label not in (0, 1) for label in labels)):
        raise ValueError('Box labels must be matching integer0/1 exclusion/inclusion hints')
    box_only = hint.get('box_only', False)
    if type(box_only) is not bool or not any(labels):
        raise ValueError('Use a boolean box_only and at least one inclusion hint')
    return {'text': None if box_only else prompt, 'original_sizes': [[height, width]],
            'input_boxes': [boxes.tolist()], 'input_boxes_labels': [list(labels)]}


def backend_status():
    """Inspect the configured task runner; another Python owns model imports."""
    import json
    import os
    import shutil
    from pathlib import Path

    configured = os.environ.get("REAL2SIM_SEGMENTATION_RUNNER")
    result = {"available": False, "configured": bool(configured),
              "execution_verified": False, "status": "not_configured",
              "role": "instance masks; adjacent-mask association is heuristic"}
    if not configured:
        return result
    try:
        config = json.loads(Path(configured).read_text())
        argv = config.get("argv")
        if (config.get("schema_version") != 1 or not isinstance(argv, list) or not argv
                or not all(isinstance(arg, str) and arg for arg in argv)):
            raise ValueError("Invalid operator segmentation runner")
        if shutil.which(argv[0]) is None:
            raise ValueError("Configured segmentation executable is unavailable")
    except (OSError, ValueError, TypeError, AttributeError) as error:
        return {**result, "status": "invalid_configuration", "error": str(error)}
    return {**result, "available": True, "status": "configured_not_executed"}


def run_backend(request_path, output, *, run=None, timeout_seconds=300):
    """Invoke only an operator-configured local model backend, never task-supplied argv."""
    import json
    import os
    import subprocess
    from pathlib import Path

    from PIL import Image

    from .artifacts import atomic_json, sha256_file
    configured = os.environ.get("REAL2SIM_SEGMENTATION_RUNNER")
    if not configured:
        raise RuntimeError("Configure operator-owned REAL2SIM_SEGMENTATION_RUNNER or use the native SAM3 command")
    config_path = Path(configured).resolve()
    config = json.loads(config_path.read_text())
    argv = config.get("argv")
    if config.get("schema_version") != 1 or not isinstance(argv, list) or not argv or not all(isinstance(a, str) and a for a in argv):
        raise RuntimeError("Invalid operator segmentation runner")
    config_hash = sha256_file(config_path)
    output, request_path = Path(output).resolve(), Path(request_path).resolve()
    model_output = output / "model"
    argv = [a.replace("{request}", str(request_path)).replace("{output}", str(model_output)) for a in argv]
    environment = None
    if run is not None:
        from .task_execution import policy_for, sandbox_command
        policy = policy_for(run, executable=argv[0], writable=[output])
        paths = config.get("read_paths", [])
        if not isinstance(paths, list) or len(paths) > 32 or not all(isinstance(p, str) for p in paths):
            raise RuntimeError("Operator read_paths must list specific model/runtime paths")
        for value in paths:
            path = Path(value).resolve()
            if not path.exists() or path == Path('/') or path == Path.home() or Path.home().is_relative_to(path):
                raise RuntimeError("Use specific existing model/runtime paths, not home or filesystem root")
            policy["readable"].append(str(path))
        environment = policy["environment"]
        argv = sandbox_command(argv, policy, cwd=output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "execution.log").open("w") as log:
        process = subprocess.run(argv, env=environment, cwd=output, stdout=log, stderr=subprocess.STDOUT,
                                 timeout=timeout_seconds, check=False)
    report = model_output / "segmentation_report.json"
    if process.returncode or not report.is_file() or sha256_file(config_path) != config_hash:
        raise RuntimeError("Segmentation backend failed or configuration changed; inspect execution.log")
    result = json.loads(report.read_text())
    request = json.loads(request_path.read_text())
    if (result.get("ok") is not True or not result.get("backend", {}).get("name")
            or not result.get("backend", {}).get("version") or len(result.get("views", [])) != len(request["views"])):
        raise RuntimeError("Segmentation backend omitted current view/implementation evidence")
    previews = []
    for original, view in zip(request["views"], result["views"], strict=True):
        if view.get("source_sha256") != sha256_file(request_path.parent / original["image"]):
            raise RuntimeError("Segmentation output does not match retained image source")
        for instance in view.get("instances", []):
            path = (model_output / instance["mask"]).resolve()
            if Path(instance["mask"]).is_absolute() or not path.is_relative_to(model_output) or not path.is_file():
                raise RuntimeError("Segmentation mask escaped backend output")
            with Image.open(path) as mask:
                if list(mask.size) != view["image_size"]:
                    raise RuntimeError("Segmentation mask changed the displayed pixel domain")
            instance["mask"] = str(path.relative_to(output))
            if len(previews) < 6:
                previews.append(instance["mask"])
    result.update(operation="segmentation", runner_config_sha256=config_hash, preview_images=previews)
    atomic_json(output / "report.json", result)
    return result


def mask_records(masks, scores):
    masks = np.asarray(masks, bool)
    scores = np.asarray(scores, float)
    if masks.size == 0 and scores.shape == (0,):
        return []
    if masks.ndim != 3 or scores.shape != (len(masks),) or not np.isfinite(scores).all():
        raise ValueError("Expected NxHxW masks and finite per-instance scores")
    records = []
    for mask, score in zip(masks, scores, strict=True):
        y, x = np.nonzero(mask)
        if not len(x):
            continue
        records.append({"mask": mask, "score": float(score), "area_px": len(x),
                        "centroid_px": [float(x.mean()), float(y.mean())],
                        "bbox_xywh": [int(x.min()), int(y.min()), int(x.max() - x.min() + 1), int(y.max() - y.min() + 1)]})
    return records


def associate(previous, current, next_id, *, min_iou=.1):
    """Conservative mask-IoU proposals; a missing/ambiguous match is not invented."""
    if not 0 < min_iou <= 1:
        raise ValueError("Temporal association IoU must be in (0,1]")
    candidates = []
    for i, old in enumerate(previous):
        for j, new in enumerate(current):
            if old["mask"].shape != new["mask"].shape:
                continue
            union = (old["mask"] | new["mask"]).sum()
            iou = float((old["mask"] & new["mask"]).sum() / max(union, 1))
            if iou >= min_iou:
                candidates.append((iou, i, j))
    used_old, used_new = set(), set()
    for iou, i, j in sorted(candidates, reverse=True):
        if i in used_old or j in used_new:
            continue
        alternatives = [q for q, a, b in candidates if (a == i or b == j) and (a, b) != (i, j)]
        if alternatives and max(alternatives) >= iou - .05:
            current[j]["association_ambiguous"] = True
            continue
        current[j].update(track_id=previous[i]["track_id"], association_iou=iou)
        used_old.add(i)
        used_new.add(j)
    for row in current:
        if "track_id" not in row:
            row["track_id"] = next_id
            next_id += 1
    return next_id
