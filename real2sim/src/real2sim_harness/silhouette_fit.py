"""Bounded fitting of an agent-authored parametric mesh, without fixed object kits."""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .artifacts import atomic_json, sha256_file
from .posed_observations import (
    file_path,
    load_mesh,
    load_views,
    mesh_arrays,
    render_depth,
    write_report,
)
from .shape_evidence import feature_evidence, silhouette_metrics


def fit(request, base, output):
    from types import SimpleNamespace

    from scipy.optimize import minimize

    evaluate_only = request.get('evaluate_only', False)
    if type(evaluate_only) is not bool:
        raise ValueError('evaluate_only must be a boolean')
    views, hashes = load_views(request, base, required=("mask",))
    distinct_poses = []
    for view in views:
        if not any(np.allclose(view["pose"], pose, atol=1e-7, rtol=0) for pose in distinct_poses):
            distinct_poses.append(view["pose"])
    fit_scope = "single_camera_projection" if len(distinct_poses) == 1 else "multiple_camera_projections"
    static_mesh = 'mesh' in request
    if static_mesh and (not evaluate_only or request.get('model_script') or request.get('parameters')):
        raise ValueError('A static mesh is only for evaluate_only inspection without model/parameters')
    model_path = file_path(base, request['mesh'] if static_mesh else request["model_script"])
    hashes[str(model_path)] = sha256_file(model_path)
    blender_model = not static_mesh and request.get("model_format", "mesh") == "blender"
    if request.get("model_format", "mesh") not in {"mesh", "blender"}:
        raise ValueError("model_format must be mesh or blender")
    if static_mesh and request.get('model_format', 'mesh') != 'mesh':
        raise ValueError('Static mesh inspection uses model_format=mesh')
    if not blender_model and (request.get('selected_objects') is not None or request.get('selected_roots') is not None):
        raise ValueError('Object/root selection requires a Blender source')
    if model_path.suffix.lower() == '.blend' and (not blender_model or not evaluate_only or request.get('parameters')):
        raise ValueError('Saved Blender source requires model_format=blender and evaluate_only without parameters')
    if static_mesh:
        module = SimpleNamespace(build_mesh=lambda _: load_mesh(model_path))
    elif blender_model:
        from .blender_fit import BlenderMesh
        if not request.get("blender_executable"):
            raise ValueError("Blender fitting needs the operator's configured executable")
        module = BlenderMesh(request["blender_executable"], model_path, Path(output) / "model_evaluation",
                             request.get("selected_objects"), request.get('selected_roots'))
    else:
        module_spec = importlib.util.spec_from_file_location("agent_parametric_mesh", model_path)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    if not callable(getattr(module, "build_mesh", None)):
        raise ValueError(
            "Agent model script must define build_mesh(parameters) returning vertices/triangles"
        )
    parameters = request.get("parameters", {})
    if not isinstance(parameters, dict) or not (0 if evaluate_only else 1) <= len(parameters) <= 12:
        raise ValueError("Fit 1..12 named scalar geometry/pose parameters")
    names = list(parameters)
    initial = np.array([parameters[n]["initial"] for n in names], dtype=float)
    bounds = np.array([parameters[n]["bounds"] for n in names], dtype=float).reshape((-1, 2))
    if (
        bounds.shape != (len(names), 2)
        or not np.isfinite(bounds).all()
        or not np.isfinite(initial).all()
        or not (bounds[:, 1] > bounds[:, 0]).all()
        or not ((initial >= bounds[:, 0]) & (initial <= bounds[:, 1])).all()
    ):
        raise ValueError("Each parameter needs finite initial and ordered bounds containing it")
    max_evaluations = 1 if evaluate_only else request.get("max_evaluations", 100)
    if not evaluate_only and (type(max_evaluations) is not int or not 10 <= max_evaluations <= 1000):
        raise ValueError("max_evaluations must be 10..1000")
    time_budget = request.get('time_budget_seconds')
    if time_budget is not None and (isinstance(time_budget, bool)
            or not isinstance(time_budget, (int, float))
            or not np.isfinite(time_budget) or time_budget <= 0):
        raise ValueError('time_budget_seconds must be finite and positive')
    masks, valid_masks = [], []
    for view in views:
        mask = view["mask"] > 0
        valid = view.get("ignore", np.zeros_like(mask)) == 0
        if not (mask & valid).any():
            raise ValueError("Every fitting view needs observed, non-ignored foreground pixels")
        masks.append(mask)
        valid_masks.append(valid)
    landmark_rows = None
    landmark_weight = float(request.get('landmark_weight', 1.0))
    if not np.isfinite(landmark_weight) or landmark_weight <= 0:
        raise ValueError('landmark_weight must be finite and positive')
    if request.get('landmarks'):
        from .landmark_fit import load_landmarks, score_landmarks
        landmark_rows = load_landmarks(request['landmarks'], base, views, hashes)
    optimizer = request.get('optimizer', 'Nelder-Mead')
    if optimizer not in {'Nelder-Mead', 'Powell'}:
        raise ValueError('optimizer must be Nelder-Mead or Powell')
    gates = request.get('quality_targets', {})
    fraction_gates = {'min_mean_iou', 'max_mean_local_difference'}
    if not isinstance(gates, dict) or set(gates) - (fraction_gates | {'max_landmark_rms_sigma'}):
        raise ValueError('Unknown fitting quality targets')
    for name, value in gates.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError('Quality targets must be finite nonnegative numbers')
        if name in fraction_gates and value > 1:
            raise ValueError('IoU and local difference targets must be fractions')
    if 'max_landmark_rms_sigma' in gates and landmark_rows is None:
        raise ValueError('Landmark quality target requires observed landmarks')

    def passes_targets(iou, local, landmark):
        return (iou >= gates.get('min_mean_iou', 0)
                and local <= gates.get('max_mean_local_difference', 1)
                and ('max_landmark_rms_sigma' not in gates
                     or landmark['rms_sigma'] <= gates['max_landmark_rms_sigma']))

    trace, best, initial_metrics = [], None, None
    local_weight = float(request.get("local_weight", 0))
    if not np.isfinite(local_weight) or not 0 <= local_weight <= 10:
        raise ValueError("local_weight must be from 0 to 10")
    local_options = request.get("local_residual", {})
    span = bounds[:, 1] - bounds[:, 0]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'fit_request.json', request)
    hashes[str(output / 'fit_request.json')] = sha256_file(output / 'fit_request.json')
    started = time.monotonic()
    longest_evaluation = 0.
    checkpoint = None

    class SearchTimeExpired(Exception):
        pass

    def evaluate(normalized):
        nonlocal best, initial_metrics, longest_evaluation, checkpoint
        evaluation_started = time.monotonic()
        if (trace and time_budget is not None
                and evaluation_started - started + longest_evaluation >= time_budget):
            raise SearchTimeExpired
        values = bounds[:, 0] + np.clip(normalized, 0, 1) * span
        params = dict(zip(names, values.tolist(), strict=True))
        raw = module.build_mesh(params)
        mesh = mesh_arrays(raw["vertices"], raw["triangles"], raw.get("uv"))
        scores, silhouettes, local = [], [], []
        for view, observed, valid in zip(views, masks, valid_masks, strict=True):
            predicted = np.isfinite(render_depth(mesh, view))
            detail = silhouette_metrics(predicted, observed, valid, **local_options)
            scores.append(detail["iou"])
            local.append(detail)
            silhouettes.append(predicted)
        # Every registered view contributes. Do not silently drop a difficult view as an outlier.
        loss = 1 - float(np.mean(scores)) + local_weight * float(np.mean([m["local_difference"] for m in local]))
        landmark_result = score_landmarks(raw.get('features', []), landmark_rows, views) if landmark_rows else None
        if landmark_result:
            loss += landmark_weight * landmark_result['loss']
        feasible = passes_targets(float(np.mean(scores)),
                                  float(np.mean([m['local_difference'] for m in local])), landmark_result)
        trace.append(
            {
                "evaluation": len(trace) + 1,
                "parameters": params,
                "mean_iou": float(np.mean(scores)),
                "loss": loss,
                "landmark_rms_sigma": landmark_result['rms_sigma'] if landmark_result else None,
                "quality_targets_passed": feasible if gates else None,
                "mean_local_difference": float(np.mean([m["local_difference"] for m in local])),
                "per_view_iou": scores,
            }
        )
        if initial_metrics is None:
            initial_metrics = trace[-1]
        if (best is None or (feasible and not best['feasible'])
                or (feasible == best['feasible'] and loss < best['loss'])):
            best = {
                "loss": loss,
                "mesh": mesh,
                "parameters": params,
                "silhouettes": silhouettes,
                "per_view_iou": scores,
                "local": local,
                "features": raw.get("features", []),
                "landmarks": landmark_result,
                "feasible": feasible,
            }
            # Publish a complete immutable candidate before advancing its pointer.
            # A killed model/evaluation must not erase previously evaluated work.
            previous_checkpoint = checkpoint
            checkpoint = output / f'candidate-{len(trace):04d}.npz'
            temporary = checkpoint.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, **best['mesh'])
            temporary.replace(checkpoint)
            atomic_json(output / 'search_progress.json', {
                'schema_version': 1, 'completed': False,
                'scope': 'Best fully evaluated candidate so far; search may be interrupted. Not accepted or verified 3D.',
                'mesh': checkpoint.name, 'mesh_sha256': sha256_file(checkpoint),
                'evaluation': len(trace), 'parameters': params,
                'mean_iou': float(np.mean(scores)), 'per_view_iou': scores,
                'quality_targets_passed': feasible if gates else None,
                'input_hashes': hashes,
            })
            if previous_checkpoint is not None:
                previous_checkpoint.unlink(missing_ok=True)
        longest_evaluation = max(longest_evaluation, time.monotonic() - evaluation_started)
        return loss

    normalized = (initial - bounds[:, 0]) / span
    # A meaningful initial simplex avoids vanishing steps for zero/near-zero parameters.
    simplex = np.tile(normalized, (len(names) + 1, 1))
    for i in range(len(names)):
        direction = 0.08 if normalized[i] <= 0.9 else -0.08
        simplex[i + 1, i] = np.clip(normalized[i] + direction, 0, 1)
    time_budget_exhausted = False
    try:
        options = {"maxfev": max_evaluations}
        if optimizer == 'Nelder-Mead':
            options.update(xatol=.0005, fatol=.0001, initial_simplex=simplex)
        else:
            options.update(xtol=.0005, ftol=.0001)
        if evaluate_only:
            evaluate(normalized)
            optimized = SimpleNamespace(success=False, message='Read-only inspection; no parameter search')
        else:
            optimized = minimize(evaluate, normalized, method=optimizer,
                                 bounds=[(0, 1)] * len(names), options=options)
    except SearchTimeExpired:
        time_budget_exhausted = True
        optimized = SimpleNamespace(success=False, message='Search time budget reached; returning best evaluated candidate')
    finally:
        if blender_model:
            module.close()
    mesh_name = 'evaluated_mesh.npz' if evaluate_only else 'fitted_mesh.npz'
    np.savez_compressed(output / mesh_name, **best["mesh"])
    atomic_json(output / "parameters.json", best["parameters"])
    atomic_json(output / "optimization_trace.json", trace)
    previews = []
    features = [feature_evidence(best["features"], view, i) for i, view in enumerate(views)]
    for i, (observed, predicted, valid) in enumerate(
        zip(masks, best["silhouettes"], valid_masks, strict=True)
    ):
        image = np.zeros((*observed.shape, 3), dtype=np.uint8)
        image[observed & valid] = [225, 80, 60]
        image[predicted & valid] = [60, 130, 240]
        image[observed & predicted & valid] = [60, 200, 110]
        image[~valid] = [100, 100, 100]
        name = f"fit-{i:03d}.png"
        Image.fromarray(image).save(output / name)
        previews.append(name)
        if evaluate_only and 'image' in views[i]:
            import cv2
            overlay = views[i]['image'].copy()
            for mask, color in [(observed & valid, (60, 220, 100)),
                                (predicted & valid, (220, 60, 220))]:
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, color, 1)
            name = f'projection-rgb-{i:03d}.png'
            Image.fromarray(overlay).save(output / name)
            previews.append(name)
        if features[i]:
            import cv2
            overlay = views[i].get("image", image).copy()
            for feature in features[i]:
                for pixel, visible in zip(feature["pixels"], feature["in_frame"], strict=True):
                    if visible:
                        cv2.circle(overlay, tuple(np.rint(pixel).astype(int)), 2, (255, 0, 255), -1)
            name = f"features-{i:03d}.png"
            Image.fromarray(overlay).save(output / name)
            previews.append(name)
    mean_iou = float(np.mean(best["per_view_iou"]))
    local_mean = float(np.mean([row["local_difference"] for row in best["local"]]))
    quality_pass = best['feasible']
    return write_report(
        output,
        {
            "schema_version": 1,
            "ok": True,
            "operation": "silhouette_inspection" if evaluate_only else "silhouette_fit",
            "evaluate_only": evaluate_only,
            "geometry_optimized": not evaluate_only and len(trace) > 1,
            "source_selection": {'objects': request.get('selected_objects'), 'roots': request.get('selected_roots')},
            "model_format": "blender" if blender_model else "mesh",
            "fit_scope": fit_scope,
            "view_count": len(views),
            "distinct_camera_pose_count": len(distinct_poses),
            "camera_sources": [view["source"] for view in views],
            "camera_optimized": False,
            "three_d_reconstruction_verified": False,
            "frame": request["frame"],
            "input_hashes": hashes,
            "parameters": best["parameters"],
            "initial_mean_iou": initial_metrics["mean_iou"],
            "mean_iou": mean_iou,
            "mean_local_difference": local_mean,
            "local_per_view": best["local"],
            "feature_projections": features,
            "landmark_evidence": best['landmarks'],
            "landmark_weight": landmark_weight if landmark_rows else None,
            "optimizer": None if evaluate_only else optimizer,
            "quality_targets": gates,
            "quality_targets_passed": quality_pass if gates else None,
            "per_view_iou": best["per_view_iou"],
            "evaluations": len(trace),
            "search_seconds": time.monotonic() - started,
            "time_budget_seconds": time_budget,
            "time_budget_exhausted": time_budget_exhausted,
            "optimizer_converged": bool(optimized.success),
            "optimizer_message": str(optimized.message),
            "mesh": mesh_name,
            "preview_images": previews,
            "limits": [
                "Read-only projection inspection; no geometry or camera optimization." if evaluate_only else
                "Local bounded search; the best observation score is not a global optimum.",
                "A single camera constrains visible projections only. Hidden shape, depth and absolute scale are not recovered; duplicate views do not add viewing directions.",
                "Only the supplied parameterization can change. Hidden shape, materials and articulation are not certified.",
                "Input cameras remain fixed. Estimated camera/scale assumptions must be identified in provenance; fit scores are conditional on those assumptions, not camera calibration.",
                "Inspection does not repair or accept an asset; resolve discrepancies with current source evidence." if evaluate_only else
                "Import parameters into editable source, inspect RGB and revalidate simulation before delivery.",
            ],
        },
    )
