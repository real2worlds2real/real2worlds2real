"""The twin's check report: every automatic check of every stage that has run, in one place, for the agent's loop
(docs/twin_agent.md). Reads what the stages wrote; builds nothing.

    envs/recon/bin/python -m calibrated_scene_reconstruction.twin_report sessions/<name>

Each item: the stage, what is measured, its value, the threshold it is held to (only quantities the capture itself
provides -- never a true size), ok / flag, the evidence to look at, and when flagged the actions that usually fix
it (modules to run, configs to change). Writes qa/report.json and qa/report.md; prints the flagged items.
"""
import argparse
import json
import re

import numpy as np

from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.twin.gates import options, review


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def item(out, stage, what, value, ok, evidence=None, actions=None, threshold=None, *, diagnostic=False):
    out.append({"stage": stage, "what": what, "value": value, "threshold": threshold, "ok": bool(ok),
                "evidence": evidence or [], "actions": [] if ok else (actions or [])})
    if diagnostic:
        out[-1]['diagnostic'] = True


def blender_object_report(s, name, spec, out):
    """Read current source acceptance; closed-solid rules do not apply to cloth."""
    directory = s / 'objects' / name / f"round_{spec.get('fix', {}).get('round')}"
    evidence = [str((directory / 'fit.json').relative_to(s.dir))]
    try:
        from calibrated_scene_reconstruction.scene.objects.blender.export import verify_round
        result = verify_round(s, name, directory, exported=True, physical_review=False)
    except (OSError, ValueError, KeyError, TypeError) as error:
        item(out, 'S2', name + ': current Blender source/export', str(error), False, evidence,
             ['Regenerate the current fit/export; do not use an older cached fit.'])
        return
    metrics = result['metrics']
    acceptance = result.get('source_acceptance', {})
    accepted_scope = acceptance.get('pass') is True and acceptance.get('relaxed') is True
    item(out, 'S2', name + ': full captured arrangement (raw source check)',
         {k: metrics.get(k) for k in ['iou_median', 'local_difference', 'frames', 'ok']},
         metrics.get('ok') is True, evidence, ['Inspect the source geometry and independently labelled parts.'],
         metrics.get('thresholds'), diagnostic=accepted_scope)
    if acceptance:
        controls = acceptance.get('input_acceptance', {})
        retained = controls.get('defaults', {}).get('source_geometry', {}).get('raw', {}).get('measurements', {})
        item(out, 'S2', name + ': current scoped source acceptance',
             {'default_pass': acceptance.get('default_pass'), 'relaxed': acceptance.get('relaxed'),
              'unrelaxed_geometry': {k: retained.get(k) for k in ['iou_median', 'local_difference', 'ok']},
              'criteria': {k: v.get('pass') for k, v in controls.get('criteria', {}).items()},
              'reason': acceptance.get('reason')}, acceptance.get('pass') is True,
             [*evidence, 'qa/task_check.json'],
             ['Complete current input criteria and all unrelaxed checks; retain the raw default verdict.'])
    try:
        from calibrated_scene_reconstruction.assets.metric_mesh import asset_geometry
        with np.load(directory / 'mesh.npz') as mesh:
            geometry = asset_geometry(mesh['V'], mesh['F'],
                                      json.loads(str(mesh['links_json'])) if 'links_json' in mesh else None)
        item(out, 'S2', name + ': current mesh and source SDF',
             {**geometry, 'sdf': spec['fix']['sdf']}, True,
             [str((directory / 'export.json').relative_to(s.dir))],
             threshold='Closed positive rigid solids; finite triangular flexible surfaces; verified declared SDF')
    except (OSError, ValueError, KeyError, TypeError) as error:
        item(out, 'S2', name + ': current mesh and source SDF', str(error), False, evidence,
             ['Repair the physical representation without applying rigid-solid rules to flexible sheets.'])


def build(s):
    out = []
    sp = s / "splat"
    opts = options(s) or {}
    # --- S0 capture
    log = s / "logs" / "splat_capture.log"
    if log.exists():
        m = re.search(r"\[colmap\] registered (\d+) / (\d+)", log.read_text())
        if m:
            share = int(m[1]) / max(int(m[2]), 1)
            mm = re.search(r"(\d+) model\(s\)", log.read_text())
            if mm:
                item(out, "S0", "COLMAP's models (one walk: one model)", int(mm[1]), int(mm[1]) == 1,
                     ["logs/splat_capture.log"], ["denser frames where the walk broke (splat.json motion lower, "
                                                  "max_gap, or step)"], "1")
            item(out, "S0", "frames registered by COLMAP", f"{m[1]} / {m[2]}", share >= 0.8, ["logs/splat_capture.log"],
                 ["fewer frames or a sharper selection (splat.json step / keep), the video's camera recalibrated "
                  "(calib), a second model joined (the log's models)"], ">= 80 %")
    cov = load(sp / "coverage.json")
    if cov:
        item(out, "S0", "the walk's coverage (columns of 0.25 m)", cov.get("share"), True, ["splat/coverage.png"])
    # --- S1 scale
    al = load(sp / "align.json")
    if al:
        item(out, "S1", "the scale's source", al.get("scale_from"), True, ["splat/align.json"])
        for v, b in (al.get("board") or {}).items():
            if "residual_mm_median" in b:
                ok = b["residual_mm_median"] <= 3 and b["reprojection_px_median"] <= 2.5
                item(out, "S1", f"the board's corners against the board (video {v})",
                     f"{b['residual_mm_median']} mm median, {b['reprojection_px_median']} px reprojected, "
                     f"{b['corners']} corners", ok, ["qa/splat/board_fit.jpg"],
                     ["configs/<name>/charuco_board.json against the board in the frames (squares, sizes, "
                      "dictionary)", "the camera's calibration (calib)"], "<= 3 mm, <= 2.5 px")
        ra = al.get("robot_anchor")
        if ra:
            item(out, "S1", "the robot at its readout fitted to the frames", f"{1000 * ra['residual_median_m']:.1f} mm "
                 "median", ra["residual_median_m"] <= 0.02, ["qa/splat/robot_fit.png", "logs/robot_anchor.log"],
                 ["the joint readout (robot_state.json) of this capture", "the robot spec (--robots)"], "<= 20 mm")
        c = al.get("checks") or {}
        if "floor_z" in c:
            item(out, "S1", "the floor plane at z 0", f"{1000 * c['floor_z']:.1f} mm", abs(c["floor_z"]) <= 0.01,
                 ["splat/align.json"], ["the placement again (poses)"], "within 10 mm")
    robot_check = load(sp / "robot_fit.json")
    if robot_check and 'free_scale' in robot_check:
        item(out, "S1", "final aligned robot scale consistency (reported)",
             {k: robot_check[k] for k in ('free_scale', 'residual_median_m', 'within_2cm_share',
                                          'height_offset_m', 'lowest_point_m') if k in robot_check},
             abs(float(robot_check['free_scale'])-1.) <= .05,
             ["splat/robot_fit.json", "qa/splat/robot_fit.png"],
             ['Refine uncertain commanded joints against current video; inspect floor/base visibility'],
             'free-scale consistency within 5%; diagnostic, not ground-truth size error')
    joint = load(sp/'robot_joint_fit.json')
    if joint:
        item(out, 'S1', 'robot CAD scale with uncertain commanded joints',
             {key: joint.get(key) for key in ('metres_per_map_unit','silhouette_iou_median',
               'silhouette_local_difference','residual_median_m','plausible_scale_spread',
               'joint_prior_audit','fixed_command_baseline','reasons')},
             joint.get('ok') is True and joint.get('applied') is True,
             ['splat/robot_joint_fit.json'],
             ['Inspect independent views, commanded/refined pose differences and scale ambiguity; no task dimensions'])
    visual = load(sp/'robot_visual_fit.json')
    if visual:
        item(out, 'S1', 'robot CAD scale without supplied joints',
             {key: visual.get(key) for key in ('metres_per_map_unit', 'residual_median_m',
                                               'surface_coverage', 'plausible_scale_spread',
                                               'unestimated_joints', 'reasons')},
             visual.get('ok') is True and visual.get('applied') is True,
             ['splat/robot_visual_fit.json', 'qa/splat/robot_visual_frames.jpg'],
             ['inspect robot-mask tracks, CAD identity and articulated hypotheses; abstain if scale is ambiguous'])
    mv = s / "logs" / "multiview.log"
    if mv.exists():
        for line in mv.read_text().splitlines():
            if line.startswith("[scale]") and "against it" in line:
                item(out, "S1", "initial key-frame scale comparison (before final alignment; diagnostic)", line[8:], True,
                     ["logs/multiview.log"])
    pb = (s.scene_graph.get("photo_robot") or {}).get("base")
    for pf in sorted((s / "placements").glob("*.json")) if pb else []:
        from calibrated_scene_reconstruction.robots.viewpoints import is_photo_base

        real = [c for c in json.loads(pf.read_text()).get("candidates", []) if is_photo_base(c, pb)]
        if real:
            c = real[0]
            item(out, "S1", f"rendered robot pose follows this capture ({pf.stem})",
                 {'base':c['base'],'task_validation':'object_model','robot_execution_required':False}, True,
                 [f"placements/{pf.stem}.json", "qa/robot_views/twin_sheet.jpg"])
    rcv = load(sp / "robot_carve.json")
    if rcv and rcv.get("base"):  # no readout: the robot carved from its silhouettes -- it stands at the support
        sb = s.scene_graph["furniture"][s.scene_graph["task"]["support"]]["box"]
        bx, by = rcv["base"]["xy"]
        gap = float(np.hypot(max(sb[0][0] - bx, 0, bx - sb[1][0]), max(sb[0][1] - by, 0, by - sb[1][1])))
        item(out, "S1", "the robot's carved base off the support's edge", f"{gap:.2f} m", 0.05 <= gap <= 0.6,
             ["qa/splat/robot_carve_frames.jpg", "qa/splat/robot_carve.png"],
             ["the robot's masks (splat_masks _robot: the torso's prompts; another robot in the room)",
              "the column's heights (robot_carve --z0 --z1)"], "0.05-0.6 m")
    # --- S2 task zone
    sc = load(sp / "support_carve.json")
    if sc:
        item(out, "S2", "the support's top carved from the frames",
             f"{sc['size'][0]:.3f} x {sc['size'][1]:.3f} m at z {sc['top_z']:.3f}", not sc.get("at_search_border"),
             ["splat/support_carve.png", "qa/splat/support_box.jpg"], ["support_carve --margin wider"])
        if "top_yaw_deg" in sc:
            oriented = opts.get('support_route') == 'agent_blender'
            item(out, "S2", "observed support yaw (retained by Blender support)" if oriented else
                 "the support's edges against the world's axes", f"{sc['top_yaw_deg']:+.1f} deg",
                 oriented or abs(sc["top_yaw_deg"]) <= 3, ["splat/support_carve.png", "qa/object_fit/frame.jpg"],
                 ["background zone: support_carve --apply turns the world to the support (then from graph)",
                  "other backgrounds: the support's box cannot hold a turn yet -- note it"],
                 "diagnostic; current Blender support has an independent geometry/pose gate" if oriented else "within 3 deg")
        item(out, "S2", "the support's footprint seen by too few frames", f"{100 * sc['unseen_share_of_footprint']:.0f} %",
             sc["unseen_share_of_footprint"] <= 0.25, ["splat/support_carve.png"],
             ["the side the walk did not pass: noted in the report"], "<= 25 %")
    fit = load(sp / "object_fit.json")
    active_blender = {k: o for k, o in (s.scene_graph.get('dynamic_objects') or {}).items()
                      if (o.get('fit') or {}).get('model') or (o.get('fix') or {}).get('model')}
    for k, o in active_blender.items():
        blender_object_report(s, k, o, out)
    for k, o in (s.scene_graph.get("dynamic_objects") or {}).items():  # a task object asked for and not fitted
        if o.get("fit") and k not in (fit or {}) and k not in active_blender:
            item(out, "S2", f"{k}: fitted to its silhouettes", "not fitted", False, ["logs/object_fit.log"],
                 ["too few frames found it whole (a part prompt 'a|b' missing in most: the prompt, --frames)",
                  "another task object took its instance (object_fit's relocation: the log)"], "fitted")
    for k, v in (fit or {}).items():
        if k in active_blender:
            continue
        rig = v.get("rig")
        thr = 0.5 if rig else 0.85
        n = len(v.get("frames", []))
        label = v.get('model') if v.get('route') == 'blender' else v.get('kit')
        fit_sheet = f"qa/objects/{k}/round_{v['round']}.jpg" if v.get('route') == 'blender' else f"qa/object_fit/{k}.jpg"
        item(out, "S2", f"{k}: its fitted shape against its silhouettes ({label})",
             f"IoU median {v.get('iou_median')} over {n} frames", (v.get("iou_median") or 0) >= thr and n >= 8,
             [fit_sheet, f"qa/object_closeups/{k}.jpg"],
             ["the kit (scene graph dynamic_objects.<obj>.fit {kit, prompt}): a shape the kit cannot take",
              "the prompt: SAM 3 took another thing or a part (fit.prompt 'a|b' for parts)",
              "rigs: the start (posts, height) and the yaw range (object_fit)"], f">= {thr}, >= 8 frames")
        if "unexplained_median" in v:
            item(out, "S2", f"{k}: its silhouettes' share the kit leaves out (a missing piece)",
                 f"{v['unexplained_median']}" + (f"; bands {v['params'].get('bands')} from {v['band_cover']}"
                                                  if v.get("band_cover") else ""),
                 v["unexplained_median"] <= 0.25, [f"qa/object_fit/{k}.jpg"],
                 ["a piece the kit lacks (a band, a handle, a plate): the kit's parameters or another kit",
                  "the prompt took more than the thing (the robot's arm with the frame)"], "<= 0.25")
    for k, v in (fit or {}).items():
        if k in active_blender:
            continue
        if v.get("route") != "blender":
            continue
        ex = v.get("export_checks", {})
        from calibrated_scene_reconstruction.assets.metric_mesh import sdf_settings
        request = (s.scene_graph.get('dynamic_objects', {}).get(k, {}).get('fit') or {}).get('sdf')
        try:
            expected_sdf = sdf_settings(request)
        except (ValueError,TypeError):
            expected_sdf = None
        item(out, "S2", f"{k}: Blender mesh export and SDF", ex,
             ex.get("watertight") and ex.get("winding_consistent") and ex.get("volume_m3", 0) > 0
             and expected_sdf is not None and ex.get("sdf") == expected_sdf,
             [f"objects/{k}/round_{v.get('round')}/export.json"],
             ["repair the Blender solid; retain the object's own surface for SDF collision"],
             "watertight, positive volume, own-surface SDF matching validated declared precision")
    for k, v in (load(sp / "object_refine.json") or {}).items():
        if k in active_blender:
            continue  # Current Blender metrics above supersede an older refinement cache.
        lk = v["local_kit"]["pieces_share_median"]
        item(out, "S2", f"{k}: the shape's local differences from its silhouettes (cut corners, handles, holes)",
             f"{100 * lk:.1f} % (carved: {100 * v['local_carved']['pieces_share_median']:.1f} %)", lk <= 0.02,
             [v["sheet"]], ["a feature the kit lacks: its parameters (dynamic_objects.<obj>.fit.params; "
                            "recon.object_refine tune), another kit, or the carved mesh (object_refine run --apply)"],
             "<= 2 %")
    for k, v in (load(sp / "structures.json") or {}).items():
        item(out, "S2", f"{k}: the structure carved from its silhouettes (precision: nothing added)",
             f"{v['precision_median']} over {v['frames']} frames", (v["precision_median"] or 0) >= 0.6,
             [f"qa/structures/{k}.jpg"], ["the prompt took more than the structure (structures.<name>.prompt)",
                                         "too few frames from some side (--frames)", "--agree higher"], ">= 0.6")
        item(out, "S2", f"{k}: the structure against its silhouettes (recall: nothing missing)",
             f"{v['recall_median']}; seen up to {v['top_seen_over_support_m']} m over the support, "
             f"{v['meets_unseen_cells']} cells where it meets what no frame saw", (v["recall_median"] or 0) >= 0.8,
             [f"qa/structures/{k}.jpg"], ["a pose off: --slack", "a piece seen by few frames: --min-seen, --agree "
                                          "lower", "the region: structures.<name>.margin / height"], ">= 0.8")
    cu = load(s / "qa" / "object_closeups" / "closeups.json")
    for k, v in (cu or {}).items():
        look = v.get("look") or {}
        if look:
            ok = abs(look.get("contrast", 0)) <= 0.5 and look.get("chroma", 0) <= 0.15
            item(out, "S2", f"{k}: the twin's close-ups against the frames' (look)", look, ok,
                 [f"qa/object_closeups/{k}.jpg"],
                 ["the texture (task_look frames: object_texture), the colour (fit colour)",
                  "the render's light (the close-ups light the photo's room)"], "|contrast| <= 0.5, chroma <= 0.15")
    marks = load(sp / "support_marks.json")
    if marks is not None:
        n = len([k for k in marks if not k.startswith("_")])
        item(out, "S2", "marks carved on the support (tape, labels)", n, True, ["qa/support_marks"])
    # --- S5 whole scene
    cl = load(s / "qa" / "clearance_twin.json")
    if cl:
        moved = [g.get("item") for g in cl.get("geometry", []) if g.get("moved_m")]
        bad = [p["item"] for p in cl.get("physics", []) if not p.get("ok")]
        item(out, "S5", "clearance and physics settle", {"pass": cl.get("pass"), "moved_clear": moved,
                                                          "physics_failed": bad},
             cl.get("pass") and not any("Objects" in str(m) for m in moved), ["qa/clearance_twin.json"],
             ["what a task object was pushed out of: that piece's fit (a rig band through a cup)",
              "an object floating or sunk: its support's height"], "pass, no task object pushed")
    tc = load(s / "qa" / "task_check.json")
    if tc and tc.get("pass") is not None:
        item(out, "S5", "the task's end state holds under physics (its objects support the task)",
             tc.get("what") or tc.get("why"), tc["pass"], ["qa/task_check.jpg", "qa/task_check.json"],
             ["the objects' shapes and sizes (S2: a goal's opening narrower than the target, a size off by the scale)",
              "the kit's colliders (a hollow thing's collider a solid hull)", "the task's relation (task.json "
              "relation: into / on)"], "pass")
    # --- reviews
    from calibrated_scene_reconstruction.scene.twin.gates import status
    states = status(s)
    for st in ("S0", "S1", "S2", "S4", "S5", "S6"):
        r = review(s, st)
        if r:
            state = states['stages'][st]
            item(out, st, "the agent's latest review", {"round": r.get("round"), "pass": r.get("pass"),
                                                        'current_state': state['state'], 'reason': state['reason'],
                                                        "open_findings": sum(not isinstance(f, dict) or f.get("status") != "resolved"
                                                                             for f in r.get("findings", []))}, state['passed'],
                 [f"qa/review/{st}_r{r.get('round')}.json"])
    for prefix, group in [('OBJ', states.get('objects', {})), ('SUP', states.get('supports', {}))]:
        for name, state in group.items():
            stage = prefix + '.' + name
            r = review(s, stage)
            evidence = [f'qa/object_understanding/{name}.json']
            if r:
                evidence.append(f"qa/review/{stage}_r{r.get('round')}.json")
            item(out, stage, 'current modeling and semantic gate',
                 {'state': state['state'], 'reason': state['reason']}, state['passed'], evidence,
                 ['Complete the current source, semantic, structural and physical part reviews.'])
    return {"session": s.dir.name, "options": opts, "items": out,
            "flagged": [f"{i['stage']} {i['what']}" for i in out if not i["ok"] and not i.get('diagnostic')]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    a = ap.parse_args()
    s = Session(a.session)
    rep = build(s)
    (s / "qa").mkdir(exist_ok=True)
    (s / "qa" / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
    lines = [f"# {s.dir.name}: check report", "", f"options: {json.dumps(rep['options'])}", "",
             "| stage | check | value | threshold | ok | evidence | if flagged |", "|---|---|---|---|---|---|---|"]
    for i in rep["items"]:
        verdict = 'raw diagnostic' if i.get('diagnostic') else ('ok' if i['ok'] else '**FLAG**')
        lines.append(f"| {i['stage']} | {i['what']} | {json.dumps(i['value'], ensure_ascii=False)} | {i['threshold'] or ''} | "
                     f"{verdict} | {', '.join(i['evidence'])} | {'; '.join(i['actions'])} |")
    (s / "qa" / "report.md").write_text("\n".join(lines) + "\n")
    for i in rep["items"]:
        if not i["ok"] and not i.get('diagnostic'):
            print(f"[report] FLAG {i['stage']} {i['what']}: {json.dumps(i['value'], ensure_ascii=False)} "
                  f"({i['threshold']}) -> {i['evidence']}")
    print(f"done: {len(rep['items'])} checks, {len(rep['flagged'])} flagged -> {s / 'qa' / 'report.md'}")


if __name__ == "__main__":
    main()
