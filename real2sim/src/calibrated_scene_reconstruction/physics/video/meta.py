"""meta.json, the agent's reading of a video, checked before anything is run on it (CPU, no model).

    python -m calibrated_scene_reconstruction.physics.video.meta real/<name> [...]        # what is wrong with each, or "ok"; exit 1 if any is
    python -m calibrated_scene_reconstruction.physics.video.meta --motions                # the kinds of motion: what each must show and what it gives
    python -m calibrated_scene_reconstruction.physics.video.meta --after real/<name>      # after a run: what was made of what the agent wrote

A key the pipeline does not know is kept and not read (a source line, a note). A key that is one letter off a known
one, a motion that is no kind of motion, a prompt that is no text: those would be read as if they were not there, the
default taken in their place and a number given for a video that was described otherwise. They are said here.

After a run the same agent is told what its words found (`after`): in how many frames the object's prompt found
something and the manipulators' did, whether a support and a camera that stands were seen, and what the fit made
of it, each with what to change in meta.json if that is not what the video shows. The loop is: watch, write, run,
read this and the summary video, write again. No number is changed by it; only what the pipeline is told.
"""
from calibrated_scene_reconstruction.physics.common.schema import read_document
import argparse
import difflib
import json
import sys
from pathlib import Path

from .readings import readings

# kind of motion -> (what the video must show for it to be that kind, what is read off it, what else meta.json says)
MOTIONS = {
    "slide": ("an object pushed along a level support and let go, coming to rest in view",
              "kinetic friction, from how it slows", "support"),
    "roll": ("an object set rolling on a level support and let go",
             "rolling resistance, from how it slows", "support"),
    "drop": ("an object let fall onto a support that bounces at least once, seen from the side",
             "restitution, and the scene's scale by gravity", "support"),
    "incline": ("an object on a support that is not level, or that is tilted while the video runs",
                "kinetic friction if it slides; a bound on static friction if it stays, the friction itself if the "
                "support tilts until it goes", "support; level: a surface that is level; support_moves"),
    "pendulum": ("a body that swings on a string or rod from a pivot, let go, seen from beside its plane",
                 "the length to its middle, and how the swing decays",
                 "level, if a level surface is in view; no support; object: the bob"),
    "turntable": ("an object on a plate that turns, riding it or sliding off",
                  "a bound on static friction while it rides; kinetic friction once it slides", "support"),
    "carried": ("an object on a support that is pulled along under it",
                "a bound on static friction, from how hard it is carried", "support"),
    "stickslip": ("a long body stood on a support and drawn along it by its other end, going in jerks",
                  "kinetic over static friction", "support"),
    "beam": ("a strip held at one end that sags and, let go, swings", "its modulus over its density",
             "level; parts; section"),
    "strain": ("a body pulled, squeezed, bent or twisted by what holds it",
               "the strain, Poisson's ratio, the twist or the bow", "section; parts"),
    "sheet": ("a sheet hung by an edge that is carried and swings", "its mass per area", "level; prior.size"),
}
TEXT, NUMBER = "text", "a number"
KEYS = {"object": "prompt", "support": TEXT, "manipulator": "prompts", "motion": TEXT, "level": TEXT,
        "support_moves": "true or false", "support_round": "true or false", "shape": TEXT, "prior": "an object",
        "fps": NUMBER, "frames": "[first, last]", "every": NUMBER, "max_width": NUMBER, "slow_motion": NUMBER,
        "K": "3 x 3", "object_at": "[x, y]", "parts": "an object", "section": "an object"}
INSIDE = {"prior": ("size", "size_rel_std", "mass", "mass_range", "restitution", "static_ratio"),
          "parts": ("soft", "soft_density", "stiff_density"), "section": ("width_over_thickness", "shape")}
SHAPES = ("blob", "cylinder", "box")


def near(key, known):
    m = difflib.get_close_matches(key, known, 1, 0.75)
    return m[0] if m and m[0] != key else None


def check(meta):
    """(what is wrong, what is only worth saying): two lists of sentences for one meta.json as a dict."""
    bad, note = [], []
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)  # noqa: E731
    text = lambda v: isinstance(v, str) and v.strip()  # noqa: E731
    pair = lambda v: isinstance(v, (list, tuple)) and len(v) == 2 and all(num(x) for x in v)  # noqa: E731
    for k in meta:
        # ("slow_motion_from": a note beside a key that is read is a note)
        if k not in KEYS and not any(k.startswith(x + "_") for x in KEYS) and near(k, list(KEYS)):
            bad.append(f'"{k}" is not read: "{near(k, list(KEYS))}" is')
    for k, known in INSIDE.items():
        if k in meta and not isinstance(meta[k], dict):
            bad.append(f'"{k}" is {KEYS[k]} with {", ".join(known)}')
            continue
        for j in meta.get(k, {}):
            if j not in known:
                bad.append(f'"{k}": "{j}" is not read' + (f': "{near(j, known)}" is' if near(j, known) else
                                                             f' (it has {", ".join(known)})'))
    obj = meta.get("object")
    if not (text(obj) or (isinstance(obj, list) and obj and all(text(p) for p in obj))):
        bad.append('"object": the prompt of the thing that moves, as it looks ("red cone"), is not there')
    motion = meta.get("motion")
    if motion is None:
        note.append('no "motion": read as a slide')
    elif motion not in MOTIONS:
        guess = near(str(motion), list(MOTIONS))
        bad.append(f'"motion": "{motion}" is no kind of motion that is read' + (f' ("{guess}"?)' if guess else "")
                   + ": " + ", ".join(MOTIONS))
    man = meta.get("manipulator", [])
    if not (isinstance(man, list) and all(text(p) for p in man)):
        bad.append('"manipulator": a list of prompts of whatever touches the object (["hand"]), or none')
    elif not man and motion not in ("beam", "sheet", "strain"):
        note.append("nothing is said to touch the object: every frame is taken for free motion")
    for k in ("support", "level"):
        if k in meta and not text(meta[k]):
            bad.append(f'"{k}": a prompt, as text')
    if meta.get("shape", "blob") not in SHAPES:
        bad.append(f'"shape": one of {", ".join(SHAPES)}')
    for k in ("fps", "every", "max_width", "slow_motion"):
        if k in meta and not (num(meta[k]) and meta[k] > 0) and (k, meta[k]) != ("slow_motion", "unknown"):
            bad.append(f'"{k}": a number above 0' + (', or "unknown"' if k == "slow_motion" else ""))
    if meta.get("slow_motion") == "unknown":
        note.append("footage slowed by a factor nobody states: a drop and a pendulum say it, a strain and a "
                    "stick-slip ratio do not need it, an incline keeps its bounds, anything else is not read")
    if "frames" in meta and not (pair(meta["frames"]) and 0 <= meta["frames"][0] < meta["frames"][1]):
        bad.append('"frames": [first, last], the first before the last')
    if "object_at" in meta and not (pair(meta["object_at"]) and all(0 <= x <= 1 for x in meta["object_at"])):
        bad.append('"object_at": [x, y] as fractions of the picture, each from 0 to 1')
    pr = meta.get("prior") if isinstance(meta.get("prior"), dict) else {}
    for k in ("size", "size_rel_std", "mass", "restitution", "static_ratio"):
        if k in pr and not (num(pr[k]) and pr[k] > 0):
            bad.append(f'"prior": "{k}" is a number above 0 (metres, kilograms)')
    if "mass_range" in pr and not (pair(pr["mass_range"]) and 0 < pr["mass_range"][0] <= pr["mass_range"][1]):
        bad.append('"prior": "mass_range" is [least, most] in kilograms')
    if num(pr.get("size")) and not 1e-3 <= pr["size"] <= 20:
        note.append(f'"prior": a size of {pr["size"]} m: it is in metres')
    if motion in ("beam", "sheet") and "level" not in meta:
        note.append(f'a {motion} with no "level": a surface named "table" is looked for')
    if motion == "sheet" and "size" not in pr:
        note.append('a sheet with no "prior": {"size"}: its longer side is taken as the depth model sees it')
    if "section" in meta and isinstance(meta["section"], dict) and meta["section"].get("shape", "square") != "square":
        bad.append('"section": "shape" is "square" or left out')
    return bad, note


def after(path):
    """Sentences for the agent that wrote meta.json, from what perception and the fit made of it."""
    import numpy as np

    path = Path(path)
    meta = read_document(path / "meta.json")
    out = []
    if not (path / "perception.npz").exists():
        return ["perception has not run, or failed: its own message says why"]
    per = np.load(path / "perception.npz", allow_pickle=True)
    obj = per["object"]  # (read once: every access to the file unpacks it again)
    seen = obj.any((1, 2))
    F = len(seen)
    area = obj.reshape(F, -1).sum(1)
    part = float(area[seen].mean() / obj[0].size) if seen.any() else 0.0
    out.append(f'object "{meta.get("object")}": found in {int(seen.sum())} of {F} frames'
               + (f", {100 * part:.1f} % of the picture" if seen.any() else ""))
    if seen.mean() < 0.5:
        out.append("  -> in under half of them: a prompt of how it looks (its colour, its shape), or \"object_at\"")
    elif part > 0.5:
        out.append("  -> over half the picture: the prompt may have taken the background; \"object_at\" picks one")
    man = meta.get("manipulator") or []
    held = per["manipulator"].any((1, 2)) if "manipulator" in per.files else np.zeros(F, bool)
    if man:
        out.append(f"manipulator {man}: seen in {int(held.sum())} of {F} frames")
        if not held.any() and meta.get("motion") not in ("strain", "beam", "sheet"):
            out.append("  -> never: every frame counts as free motion; is the prompt what touches the object?")
        elif held.all() and meta.get("motion") not in ("strain", "beam", "sheet", "carried", "stickslip"):
            out.append("  -> in every frame: if it is in the picture and away from the object that is as it should "
                       "be; if the prompt names the object or the support too, nothing is ever free")
    # where the mask is, frame by frame: one that leaps is on several things in turn, one that stays is no motion
    at = np.full((F, 2), np.nan)
    for i in np.nonzero(seen)[0]:
        ys, xs = np.nonzero(obj[i])
        at[i] = xs.mean(), ys.mean()
    k = np.nonzero(seen)[0]
    if len(k) > 2:
        step = np.linalg.norm(np.diff(at[k], axis=0), axis=1) / np.maximum(np.diff(k), 1)
        size = float(np.sqrt(np.median(area[k])))
        # (a leap: there in one frame and not on the way of the frames beside it; a small thing that is fast is on it)
        # (of the frames the segmenter itself saw: one filled in from the motion is where the motion put it. A
        # bounce is a leap too, and a few are no sign of anything)
        own = at if "score" not in per.files else np.where((per["score"] > 0.05)[:, None], at, np.nan)
        turn = np.linalg.norm(own[2:] - 2 * own[1:-1] + own[:-2], axis=1)
        leaps = int((turn[np.isfinite(turn)] > 3 * size).sum())
        if leaps > max(3, 0.02 * len(k)):
            out.append(f"  -> the mask leaps {leaps} times by more than three times its own size: the prompt names "
                       "several things in turn; a prompt of how this one looks, or \"object_at\"")
        still, run, best = step < 0.3, 0, (0, 0)
        for j, s in enumerate(still):
            run = run + 1 if s else 0
            if run > best[0]:
                best = (run, j)
        a, b = int(k[best[1] - best[0] + 1]), int(k[best[1] + 1])
        # (at rest in a hand before it is let go, or at the end where it came to rest, is the scene)
        if best[0] > max(len(k) // 3, 10) and b < k[-1] and not held[a:b + 1].any():
            out.append(f"  it does not move from frame {a} to {b} of those used ({best[0]} frames): if that is a "
                       "freeze frame, a title or a stepped passage, \"frames\" picks the part that is the scene")
    cam = str(per["camera"]) if "camera" in per.files else "static"
    out.append(f"camera: {cam}" + {"moving": "  -> every fit refuses it; \"frames\" may pick a part where it stands",
                                   "unknown": " (too little background to follow: readings are given with that said)"
                                   }.get(cam, ""))
    if "support" in per.files and meta.get("motion") not in ("pendulum", "strain", "beam", "sheet"):
        band = bool(per["support_band"]) if "support_band" in per.files else False
        ok = bool(per["plane_ok"]) if "plane_ok" in per.files else True
        out.append(f'support "{meta.get("support", "(none named)")}": '
                   + ("no surface of that name; the pixels about the object\'s path stand in" if band else
                      f"{100 * float(per['support'].mean()):.0f} % of the picture")
                   + ("" if ok else "; not one plane in every frame looked at"))
    got = readings(path)
    if got is None:
        out.append("fit: has not run")
        return out
    _, got, why = got
    word = {"value": "", "lower_bound": "at least ", "upper_bound": "at most "}
    out.append("fit: " + ("; ".join(
        f"{g['parameter']} {word[g['kind']]}{g['value']:.4g}" + (f" +- {g['stderr']:.2g}" if g["stderr"] else "")
        for g in got) if got else f"nothing measured ({why})"))
    phys = read_document(path / "physics.json")
    for k, m in (phys.get("parameters") or {}).items():  # what is not given, why, and what would give it
        src = str(m.get("source", "")) if isinstance(m, dict) else ""
        if src.startswith("not taken"):
            out.append(f"  {k} {src}")
        elif src.startswith(("default", "weakly")) and k != "object.mass":
            out.append(f"  {k} not read ({src}" + (f": {m['how']}" if m.get("how") and src == "default" else "") + ")")
    for ph in phys.get("free_phases") or []:  # the stretches of free motion that were looked at and left
        if isinstance(ph, dict) and ph.get("rejected"):
            out.append(f"  frames {ph.get('first_frame')}-{ph.get('last_frame')} not read: {ph['rejected']}")
    for k in ("camera_note", "warning"):
        if phys.get(k):
            out.append(f"  {k}: {phys[k]}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recording", nargs="*")
    ap.add_argument("--after", action="store_true", help="after a run: what was made of what meta.json says")
    ap.add_argument("--motions", action="store_true", help="the kinds of motion and what each must show")
    args = ap.parse_args()
    if args.motions:
        for k, (shows, gives, says) in MOTIONS.items():
            print(f"{k:10s} shows: {shows}\n{'':10s} gives: {gives}\n"
                  f"{'':10s} meta.json besides object, manipulator: {says}")
    if args.after:
        for r in args.recording:
            print(f"{r}:\n  " + "\n  ".join(after(r)))
        return
    wrong = 0
    for r in args.recording:
        f = Path(r) / "meta.json"
        try:
            bad, note = check(read_document(f))
        except (OSError, json.JSONDecodeError) as e:
            bad, note = [f"meta.json cannot be read ({e})"], []
        wrong += bool(bad)
        print(f"{r}: " + ("; ".join(bad) if bad else "ok") + ("".join(f"\n  ({n})" for n in note)))
    sys.exit(1 if wrong else 0)


if __name__ == "__main__":
    main()
