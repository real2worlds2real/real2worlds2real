"""Compose the legacy demo video: photo -> twin -> cousins -> robot data.

This presentation expects the historical SAM 3D/template outputs. Use current
scene tours and checked native trace renderers for agent-model delivery; its
legacy step captions are not the current reconstruction contract.

    python -m calibrated_scene_reconstruction.video sessions/<name> out.mp4
"""
import json
import subprocess as subprocess  # noqa: F401 -- compatibility for callers patching the encoder
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from calibrated_scene_reconstruction.core.paths import CONFIGS

W, H, FPS = 1280, 720, 30
FONT = next((f for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",  # fonts-noto-cjk-extra
                          "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc") if Path(f).exists()),
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
BG = (18, 20, 24)


def font(size, bold=False):
    return ImageFont.truetype(FONT_B if bold else FONT, size, index=2)  # index 2 = SC


def fit(img, w, h):
    img = img.convert("RGB")
    s = min(w / img.width, h / img.height)
    return img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.LANCZOS)


def canvas():
    return Image.new("RGB", (W, H), BG)


def caption(c, title, sub=None):
    d = ImageDraw.Draw(c)
    d.rectangle([0, 0, W, 64], fill=(10, 12, 16))
    d.text((24, 12), title, font=font(30, True), fill=(240, 240, 240))
    if sub:
        d.text((W - 24, 22), sub, font=font(18), fill=(170, 180, 190), anchor="ra")
    return c


def label(c, xy, text, size=18, fill=(255, 255, 255), bg=(0, 0, 0)):
    d = ImageDraw.Draw(c)
    f = font(size)
    bb = d.textbbox(xy, text, font=f)
    d.rectangle([bb[0] - 6, bb[1] - 4, bb[2] + 6, bb[3] + 4], fill=bg)
    d.text(xy, text, font=f, fill=fill)


class Writer:
    def __init__(self, path):
        from calibrated_scene_reconstruction.core.video import open_rgb_pipe

        self.p = open_rgb_pipe(path, W, H, FPS, ["-c:v", "libx264", "-pix_fmt", "yuv420p",
                                                 "-crf", "19", "-preset", "medium"])
        self.n = 0

    def put(self, img, repeat=1):
        b = np.asarray(img.convert("RGB"), np.uint8).tobytes()
        for _ in range(repeat):
            self.p.stdin.write(b)
            self.n += 1

    def hold(self, img, seconds):
        self.put(img, int(seconds * FPS))

    def fade(self, a, b, seconds):
        n = int(seconds * FPS)
        for i in range(n):
            self.put(Image.blend(a, b, (i + 1) / n))

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def qa_crop(S, name, margin=120, qa="qa_align.png", ids=None):
    """QA overlay cropped around the instance masks."""
    ids = ids or json.loads((S / "objects" / name / "pose.json").read_text())["ids"]
    masks = np.load(S / "parsing" / "masks.npz")["masks"]
    ys, xs = np.where(np.any(masks[ids], axis=0))
    im = Image.open(S / "objects" / name / qa)
    return im.crop((max(xs.min() - margin, 0), max(ys.min() - margin, 0), min(xs.max() + margin, im.width),
                    min(ys.max() + margin, im.height)))


def furniture_slide(w, S, sg):
    """Movable furniture: shared reconstruction of identical pieces, retrieval fallback, library."""
    from ..assets.library import catalog

    tiles = []
    for group, spec in sg.get("movable_furniture", {}).items():
        rep = S / "objects" / group / "furniture.json"
        if rep.exists() and json.loads(rep.read_text()).get("ok"):
            r = json.loads(rep.read_text())
            tiles.append((qa_crop(S, group, 60, "qa_furniture.png", spec["ids"]),
                          f"shared model {len(spec['ids'])} instances: reconstruct once, fit placement, then instance (IoU {r['iou_ref']:.2f})"))
            side = S / "renders" / "views" / "twin_left_side.jpg"
            if side.exists():
                tiles.append((Image.open(side), "Complete simulated 3D stool (SAM 3D leg completion)"))
    for name in list(sg.get("static_objects", {})) + list(sg.get("movable_furniture", {})):
        f = S / "objects" / name / "retrieved.json"
        if f.exists():
            r = json.loads(f.read_text())
            ids = (sg.get("static_objects", {}).get(name) or sg["movable_furniture"][name])["ids"]
            qa = "qa_align.png" if (S / "objects" / name / "qa_align.png").exists() else "qa_furniture.png"  # group
            tiles.append((qa_crop(S, name, 60, qa, ids), f"{name}: low SAM 3D IoU -> explicit retrieval fallback"))
            tiles.append((qa_crop(S, name, 60, "qa_retrieve.png", ids), f"DINOv2 ranking + silhouette reranking → {r['model']}"))
    if not tiles:
        return
    c = caption(canvas(), "2b Furniture: shared reconstruction and retrieval (Poly Haven + ABO)")
    tw, th = 600, 210
    for i, (im, t) in enumerate(tiles[:4]):
        x, y = 30 + (i % 2) * (tw + 20), 80 + (i // 2) * (th + 42)
        im = fit(im, tw, th)
        c.paste(im, (x + (tw - im.width) // 2, y))
        label(c, (x + 6, y + th + 6), t, 15)
    cat = catalog()
    thumbs = [n for n, v in cat.items() if v["category"] in ("stool", "bench") and Path(v["file"]).exists()]
    for k, n in enumerate(thumbs[:14]):
        im = fit(Image.open(cat[n]["thumb"]).convert("RGB"), 80, 80)
        c.paste(im, (30 + k * 88 + (80 - im.width) // 2, 590))
    label(c, (30, 680), f"Cousin furniture library: {len(thumbs)} downloaded stools / benches (ABO CC BY 4.0 · Poly Haven CC0)", 15)
    w.hold(c, 5.5)


def clearance_slide(w, S, tags):
    """Penetration QA of every scene (apps/scene/check_clearance.py)."""
    rows = [(t, json.loads((S / "qa" / f"clearance_{t}.json").read_text())) for t in tags
            if (S / "qa" / f"clearance_{t}.json").exists()]
    if not rows:
        return
    c = caption(canvas(), "7b Geometry, semantic review and 2 s settling",
                f"passed {sum(r['pass'] for _, r in rows)}/{len(rows)}")
    d = ImageDraw.Draw(c)
    cols = [(30, "Scene"), (150, "Deepest stool/counter penetration -> displacement"), (480, "Other corrections"), (690, "Semantics"), (930, "Settling"),
            (1150, "Result")]
    for x, h in cols:
        d.text((x, 92), h, font=font(18, True), fill=(200, 210, 220))
    for k, (t, r) in enumerate(rows):
        y = 130 + k * 64
        geo = r["geometry"]
        stools = [g for g in geo if "/Support/" in g["item"] and g["depth_m"]]
        deepest = max(stools, key=lambda g: g["depth_m"]) if stools else None
        other = len({g["item"] for g in geo if "/Support/" not in g["item"] and (g["moved_m"] or g["action"])})
        still = all(p["ok"] for p in r["physics"])
        sem = r.get("semantics", [])
        vals = ["Twin" if t == "twin" else t.replace("cousin_", "Cousin "),
                f"{len({g['item'] for g in stools})} stools, maximum {deepest['depth_m'] * 100:.1f} cm → moved clear" if deepest else "none",
                f"{other} items (centimetre scale)", f"{sum(x['ok'] for x in sem)}/{len(sem)} checks passed",
                f"{len(r['physics'])} objects stationary" if still else "objects moved",
                "PASS" if r["pass"] else "FAIL"]
        for (x, _), v in zip(cols, vals):
            d.text((x, y), v, font=font(18), fill=(120, 220, 140) if v == "PASS" else (240, 120, 110) if v == "FAIL"
                   else (235, 235, 235))
    w.hold(c, 4.0)


def frames_of(d):
    return sorted(Path(d).glob("*.jpg"))


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session")
    parser.add_argument("output")
    args = parser.parse_args()
    S = Path(args.session)
    sg = json.loads((CONFIGS / S.name / "scene_graph.json").read_text())
    target = sg["task"]["target"]
    wf = json.loads((S / "parsing" / "world_frame.json").read_text())
    n_inst = len(json.loads((S / "parsing" / "instances.json").read_text()))
    w = Writer(args.output)
    photo = Image.open(S / "capture" / "rgb.png")

    # 0 title -------------------------------------------------------------------------------------
    c = canvas()
    d = ImageDraw.Draw(c)
    d.text((W // 2, 250), "Photo -> digital twin -> digital cousins", font=font(54, True), fill=(245, 245, 245), anchor="mm")
    d.text((W // 2, 330), "Instanced, relightable cousins | Isaac Sim 5.1 | Dexmate Vega + Sharpa", font=font(26),
           fill=(180, 190, 200), anchor="mm")
    d.text((W // 2, 400), "SAM 3 · MoGe-2 · SAM 3D Objects · LaMa · projected texture baking · PhysX", font=font(20),
           fill=(130, 140, 150), anchor="mm")
    w.hold(c, 3.0)

    # 1 input + perception -------------------------------------------------------------------------
    c = caption(canvas(), "1 Input: one photograph", S.name)
    c.paste(fit(photo, W - 80, H - 100), (40, 84))
    w.hold(c, 2.5)
    geo = np.load(S / "parsing" / "geometry.npz")
    dep = geo["depth"].copy()
    dep[~geo["mask"]] = np.nan
    dn = (dep - np.nanmin(dep)) / (np.nanmax(dep) - np.nanmin(dep))
    import matplotlib.cm as cm

    dimg = Image.fromarray((cm.turbo(np.nan_to_num(1 - dn))[..., :3] * 255).astype(np.uint8))
    tiles = [(Image.open(S / "parsing" / "overlay.png"), f"SAM 3 text segmentation ({n_inst} instances)"),
             (dimg, f"MoGe-2 metric depth + intrinsics (camera height {wf['camera_height_m']:.2f} m)"),
             (Image.open(S / "scene" / "plate_albedo.png"), "LaMa object removal -> albedo (photo / Isaac irradiance)"),
             (qa_crop(S, target), "SAM 3D reconstruction + gravity alignment (red: projection)")]
    c = caption(canvas(), "2 Perception and reconstruction (agent-orchestrated pipeline)")
    tw, th = 600, 300
    for i, (im, t) in enumerate(tiles):
        x, y = 30 + (i % 2) * (tw + 20), 84 + (i // 2) * (th + 18)
        im = fit(im, tw, th)
        c.paste(im, (x + (tw - im.width) // 2, y))
        label(c, (x + 10, y + th - 34), t, 17)
    w.hold(c, 5.0)
    furniture_slide(w, S, sg)

    # 2 photo vs twin -----------------------------------------------------------------------------
    twin = Image.open(S / "renders" / "views" / "twin_photo.jpg")
    a = caption(canvas(), "3 Digital twin: captured camera (Isaac Sim RTX)")
    a.paste(fit(photo, W - 80, H - 100), (40, 84))
    b = caption(canvas(), "3 Digital twin: captured camera (Isaac Sim RTX)")
    b.paste(fit(twin, W - 80, H - 100), (40, 84))
    label(a, (60, 104), "Source photo", 22)
    label(b, (60, 104), "Isaac Sim render", 22, bg=(30, 60, 120))
    w.hold(a, 1.2)
    w.fade(a, b, 1.2)
    w.hold(b, 1.2)
    w.fade(b, a, 0.8)
    w.fade(a, b, 0.8)
    c = caption(canvas(), "3 Digital twin: photo left, simulation right")
    c.paste(fit(photo, 610, 600), (24, 90))
    c.paste(fit(twin, 610, 600), (646, 90))
    w.hold(c, 2.5)

    # 3 fly-through -------------------------------------------------------------------------------
    for f in frames_of(S / "renders" / "flythrough" / "frames"):
        c = Image.open(f).convert("RGB").resize((W, H))
        label(c, (20, 18), "4 Novel-view tour: 3D scene, interactive objects and Vega", 22)
        label(c, (20, H - 44), "3D instances: structure, furniture, SAM 3D background and faucet template; relit albedo + PBR", 16,
              fill=(220, 230, 240))
        w.put(c)

    # 4 placements --------------------------------------------------------------------------------
    pl = json.loads((S / "placements" / "placements.json").read_text())["candidates"]
    show = [p for p in pl if p["accepted"]][:3] + [p for p in pl if not p["accepted"]][:3]
    for p in show:
        i = p["id"]
        c = caption(canvas(), "5 Viewpoint sampling: collision, IK and head-camera coverage", f"Candidate #{i}")
        ims = [Image.open(S / "placements" / f"cand{i:02d}_{k}.jpg") for k in ["over", "head", "cov"]]
        for k, (im, t) in enumerate(zip(ims, ["Top view", "Head ZED camera", "Coverage: observed white, completed black, robot purple"])):
            im = fit(im, 410, 300)
            c.paste(im, (20 + k * 420, 110))
            label(c, (26 + k * 420, 420), t, 15)
        ok = p["accepted"]
        why = [] if ok else [n for n, flag in [("unreachable", not p["reachable"]), ("low coverage", not p["coverage_ok"]),
                                               ("collision", not p["collision_free"])] if flag]
        d = ImageDraw.Draw(c)
        txt = (f"base=({p['base'][0]:.2f}, {p['base'][1]:.2f}, {p['base'][2]:.0f}°)   coverage {p['coverage']:.2f}   "
               f"{'[accepted]' if ok else '[rejected] ' + ', '.join(why)}")
        d.text((40, 480), txt, font=font(26, True), fill=(120, 220, 140) if ok else (240, 120, 110))
        if p["stools_moved"]:
            d.text((40, 530), f"Stools displaced from the robot swept volume (home -> pregrasp -> grasp -> lift): {len(p['stools_moved'])} instances",
                   font=font(20), fill=(180, 190, 200))
        w.hold(c, 1.6)

    # 5 twin grasp --------------------------------------------------------------------------------
    def episode(dirname, title, sub=None):
        res = json.loads((S / "episodes" / dirname / "result.json").read_text())
        fs = frames_of(S / "episodes" / dirname / "frames")
        for f in fs:
            im = Image.open(f)
            c = caption(canvas(), title, sub)
            third = fit(im.crop((0, 0, im.width // 2, im.height)), 620, 520)
            head = fit(im.crop((im.width // 2, 0, im.width, im.height)), 620, 520)
            c.paste(third, (16, 120))
            c.paste(head, (644, 120))
            label(c, (24, 128), "External camera", 16)
            label(c, (652, 128), "Head ZED X Mini (training view)", 16)
            ok = res.get("success")
            d = ImageDraw.Draw(c)
            rc = res.get("robot_clearance", {})
            moved = [m for m in rc.get("resolved", {}).get("moved", []) if m["action"] == "slid"]
            d.text((24, 80), f"Lift {res.get('lift_m', 0) * 100:.1f} cm · native validation {res.get('n_trials', 1)} grasp candidates · "
                             f"{'success' if ok else 'failure'} · robot penetration check {'passed' if rc.get('ok') else 'failed'}"
                             + (f" (stools displaced for swept volume: {len(moved)} instances)" if moved else ""),
                   font=font(20), fill=(120, 220, 140) if ok and rc.get("ok") else (240, 120, 110))
            w.put(c)
        return res

    episode("twin", "6 Twin execution: torso/right-arm IK -> side grasp -> lift", "PhysX contact")

    # 6 cousins grid ------------------------------------------------------------------------------
    cm_ = json.loads((S / "cousins" / "manifest.json").read_text())
    c = caption(canvas(), "7 Cousins: PBR surfaces, albedo, library/scanned assets and lighting variation")
    tags = ["twin"] + [f"cousin_{x['id']:02d}" for x in cm_]
    for k, t in enumerate(tags[:6]):
        im = fit(Image.open(S / "renders" / "views" / f"{t}_oblique.jpg"), 404, 270)
        x, y = 22 + (k % 3) * 416, 84 + (k // 3) * 316
        c.paste(im, (x, y))
        label(c, (x + 6, y + 276), "Twin" if t == "twin" else t.replace("cousin_", "Cousin "), 16)
    w.hold(c, 3.0)
    c = caption(canvas(), "7 Cousins: captured-camera comparison")
    for k, t in enumerate(tags[:6]):
        im = fit(Image.open(S / "renders" / "views" / f"{t}_photo.jpg"), 404, 270)
        x, y = 22 + (k % 3) * 416, 84 + (k // 3) * 316
        c.paste(im, (x, y))
        summ = "Twin" if t == "twin" else cm_[int(t[-2:])]["summary"].split(", ")[0:3]
        label(c, (x + 6, y + 276), summ if isinstance(summ, str) else " / ".join(summ), 13)
    w.hold(c, 3.5)
    clearance_slide(w, S, ["twin"] + [t for t in tags[1:]])

    # 7 cousin episodes tiled ---------------------------------------------------------------------
    eps = [t for t in tags[1:] if (S / "episodes" / t / "result.json").exists() and frames_of(S / "episodes" / t / "frames")]
    seqs = {t: frames_of(S / "episodes" / t / "frames") for t in eps}
    res = {t: json.loads((S / "episodes" / t / "result.json").read_text()) for t in eps}
    n = max(len(v) for v in seqs.values()) if seqs else 0
    for i in range(n):
        c = caption(canvas(), "8 Batch collection in cousins (head camera)",
                    f"success {sum(r.get('success', False) for r in res.values())}/{len(res)}")
        for k, t in enumerate(eps[:6]):
            f = seqs[t][min(i, len(seqs[t]) - 1)]
            im = Image.open(f)
            head = fit(im.crop((im.width // 2, 0, im.width, im.height)), 404, 290)
            x, y = 22 + (k % 3) * 416, 80 + (k // 3) * 316
            c.paste(head, (x, y))
            ok = res[t].get("success")
            label(c, (x + 6, y + 296), f"{t.replace('cousin_', 'Cousin ')} · {'success' if ok else 'failure'} · "
                                      f"Lift {res[t].get('lift_m', 0) * 100:.0f}cm", 14,
                  bg=(20, 90, 40) if ok else (110, 30, 30))
        w.put(c)
    for t in eps[:2]:
        episode(t, f"⑧ {t.replace('cousin_', 'Cousin ')}: " + ' / '.join(json.loads((S / 'cousins' / (t + '.json')).read_text())['summary'].split(', ')[:3]))

    # 8 end ---------------------------------------------------------------------------------------
    c = canvas()
    d = ImageDraw.Draw(c)
    lines = ["Recorded pipeline", "Photo -> SAM 3 masks + MoGe-2 geometry -> SAM 3D objects + ICP",
             "-> Scene graph and 3D instances -> relit albedo + PBR -> Isaac Sim USD",
             "-> Collision/IK/coverage sampling -> cousin variation -> grasp validation -> recording",
             f"This run: {len(res)} cousins, successful grasps {sum(r.get('success', False) for r in res.values())}/{len(res)}"]
    for k, t in enumerate(lines):
        d.text((W // 2, 200 + k * 70), t, font=font(34 if k == 0 else 22, k == 0), fill=(235, 235, 235), anchor="mm")
    w.hold(c, 4.0)
    w.close()
    print("frames", w.n, "->", args.output)


if __name__ == "__main__":
    main()
