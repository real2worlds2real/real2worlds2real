"""Splat background, step 1b: the training masks -- the task zone's things out of every frame.

    python -m calibrated_scene_reconstruction.recon.splat_masks sessions/<name>        (envs/recon: SAM 3)

What the meshes and the simulated robot stand for must not be learnt by the background: the real robot (its arms
differ between walks), the support and the manipulated objects on it (the cups; a calibration board lying on its top).
The rest of the zone -- a frame over the support, cameras, cables -- is the splat's. SAM 3 finds them in each frame by
the prompts the scene's perception used (the robot's, the support's, the dynamic objects'); an instance counts when
most of it lies inside the zone's projection (<image>_zone.png from recon.splat_capture: a desk across the room
answers "table" too, and stays). The union, dilated 6 px, is 0 in <image>_mask.png, which 3DGRUT drops from the loss
(rgb and prediction both times the mask). Writes splat/masks.json and qa/splat/masks.jpg (four frames overlaid).
"""
import argparse
import json

import cv2
import numpy as np
import torch
from PIL import Image

from calibrated_scene_reconstruction.core.gpu import device, share
from calibrated_scene_reconstruction.scene.session import Session
from calibrated_scene_reconstruction.scene.twin.graph import ROBOT, SUPPORT
from calibrated_scene_reconstruction.scene.background.splat.config import folder


def reusable(metadata, prompts, threshold, inside, dilation_pixels=6):
    """A cached mask must describe the same segmentation problem."""
    expected={'prompts':prompts,'threshold':threshold,'inside':inside,
              'dilation_pixels':dilation_pixels}
    return all(metadata.get(key)==value for key,value in expected.items())


def zone_prompts(s, role="all"):
    """The perception prompts of the zone's things: robot, support, dynamic objects, clutter, a board. `role`
    "support": the support's alone (recon.support_carve's evidence for the top); "occluders": all but the support (what
    stands between a frame and the top: no evidence either way)."""
    if role in {'support','occluders'}:
        from calibrated_scene_reconstruction.scene.support.mask_inputs import settings
        prompts=settings(s).get('support_prompts' if role=='support' else 'occluder_prompts')
        if prompts is not None:return list(prompts)
    sg = s.scene_graph
    inst = json.loads((s / "parsing" / "instances.json").read_text())
    instance_labels={i["id"]:i["label"] for i in inst}
    labels = set(instance_labels.values())
    per = s.config("perception.json", {})
    support = {lab for lab in labels if SUPPORT.search(lab)}
    if role == "support":
        return sorted(support)
    clutter=set((sg.get('clutter') or {}).get('labels',[]))
    clutter_ids=set((sg.get('clutter') or {}).get('ids',[]))
    clutter.update(i['label'] for i in inst if i['id'] in clutter_ids)
    if role == 'calibration_occluders':
        # Only observed scene classes, never task sizes or a robot joint state.
        declared=s.config('task.json',{}).get('objects',{})
        objects=set()
        for name,obj in sg.get('dynamic_objects',{}).items():
            prompt=(declared.get(name) or {}).get('prompt') or (obj.get('fit') or {}).get('prompt')
            if prompt:
                # Object fitting uses | to union separately described parts.
                # SAM must receive those phrases as independent queries here
                # too, rather than interpreting the separator as prose.
                objects.update(part.strip() for part in prompt.split('|') if part.strip())
            else:objects.update(i['label'] for i in inst if i['id'] in obj.get('ids',[]))
        return sorted(support|objects|clutter|{'black metal frame','camera rig','cable','robot power adapter'})
    prompts = {p for p in per.get("prompts", []) if ROBOT.search(p)}
    prompts |= support
    prompts |= {instance_labels[i] for o in sg.get("dynamic_objects", {}).values() for i in o.get("ids", [])}
    # the training's masks: only what the meshes stand for -- the clutter (a frame, cameras, cables) is the splat's to
    # learn (the user, 2026-10-27: only what the task touches is a mesh); the support's carve still counts it as what
    # stands between a frame and the top
    if role == "occluders":
        prompts |= clutter
    if (s.config_dir / "charuco_board.json").exists():
        prompts.add("checkerboard")
    return sorted(prompts - support) if role == "occluders" else sorted(prompts)


def rep(x, n):
    """The vision encoder's output for one image, repeated for a batch of n prompts."""
    if torch.is_tensor(x):
        return x.expand(n, *x.shape[1:])
    if isinstance(x, (list, tuple)):
        return type(x)(rep(y, n) for y in x)
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument('--all-capture-frames',action='store_true',
                    help='Independent whole-image labels, including frames not yet registered; requires --inside0')
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--inside", type=float, default=0.5, help="an instance counts with this share of it in the zone")
    ap.add_argument("--prompts", default=None, help="comma-separated prompts instead of the zone's (recon.support_carve)")
    ap.add_argument("--every", type=int, default=1, help="every n-th frame")
    ap.add_argument('--prompt-batch', type=int, default=4,
                    help='Bound GPU memory for native-resolution mask decoding; reuse image features')
    ap.add_argument("--suffix", default="_mask", help="the mask file's suffix")
    ap.add_argument("--frames", default=None, help="a json list of the frames to mask (recon.splat_anchored: its anchors)")
    ap.add_argument("--reuse", action="store_true", help="keep a frame's mask already made with the same prompts "
                    "(the masks come from the images and the zone, not the scale: a rerun of S1 need not redo them)")
    ap.add_argument("--role", default="all", choices=["all", "support", "occluders", "calibration_occluders"],
                    help="the zone's prompts of one role (recon.support_carve: _carve the support, _occ the occluders)")
    a = ap.parse_args()
    if a.prompt_batch < 1:
        ap.error('--prompt-batch must be positive')
    if a.all_capture_frames and a.inside != 0:
        ap.error('--all-capture-frames requires --inside0; a task-zone query needs registered projection')
    s = Session(a.session)
    if (s/'qa/independent_freeze.json').exists():
        ap.error('Frozen reconstruction: create a new mask session')
    out = folder(s)
    img_dir = out / "data" / "images"
    prompts = a.prompts.split(",") if a.prompts else zone_prompts(s, a.role)
    print(f"[masks] prompts {prompts}", flush=True)
    from transformers import Sam3Model, Sam3Processor

    share()
    dev = device()
    model = Sam3Model.from_pretrained("facebook/sam3").to(dev).eval()
    proc = Sam3Processor.from_pretrained("facebook/sam3")
    # the registered frames (the ones COLMAP left out have no zone mask, and the training no pose for them)
    frames = sorted(p for p in img_dir.glob("*.jpg") if a.all_capture_frames or
                    p.with_name(p.stem + "_zone.png").exists())[::a.every]
    if a.frames:  # a list of frame names (json) alone
        only = set(json.loads(open(a.frames).read()))
        frames = [p for p in frames if p.name in only]
    shares, qa = [], []
    kernel = np.ones((13, 13), np.uint8)
    tin = proc(text=prompts, return_tensors="pt").to(dev)
    done = out / f"masks{a.suffix if a.suffix != '_mask' else ''}.json"
    metadata=json.loads(done.read_text()) if done.exists() else {}
    same = a.reuse and done.exists() and reusable(metadata,prompts,a.threshold,a.inside)
    if a.reuse and done.exists() and not same:
        raise SystemExit('Mask settings changed; preserve existing evidence and use a new dataset or --suffix')
    for k, f in enumerate(frames):
        dst = f.with_name(f.stem + a.suffix + ".png")
        if same and dst.exists():
            shares.append(float((cv2.imread(str(dst), cv2.IMREAD_GRAYSCALE) < 128).mean()))
            continue
        pil = Image.open(f).convert("RGB")
        zone = (np.ones((pil.height,pil.width),bool) if a.all_capture_frames else
                np.asarray(Image.open(f.with_name(f.stem + "_zone.png"))) == 0)
        mask = np.zeros(zone.shape, bool)
        # Encode once, then bound both decoder and native-resolution output
        # memory. A long inventory must not allocate all 4K masks at once.
        iin = proc(images=pil, return_tensors="pt").to(dev)
        with torch.no_grad():
            ve = model.get_vision_features(pixel_values=iin["pixel_values"])
            for first in range(0, len(prompts), a.prompt_batch):
                stop = min(first+a.prompt_batch, len(prompts))
                batch = type(ve)(**{key: rep(value, stop-first) for key, value in ve.items()})
                out_ = model(vision_embeds=batch, input_ids=tin['input_ids'][first:stop],
                             attention_mask=tin['attention_mask'][first:stop])
                results = proc.post_process_instance_segmentation(
                    out_, threshold=a.threshold, mask_threshold=0.5,
                    target_sizes=iin['original_sizes'].tolist() * (stop-first))
                for res in results:
                    for m in res['masks']:
                        m = m.cpu().numpy().astype(bool)
                        if m.sum() and (m & zone).sum() / m.sum() >= a.inside:
                            mask |= m
                del results, out_, batch
        mask = cv2.dilate(mask.astype(np.uint8), kernel) > 0
        cv2.imwrite(str(f.with_name(f.stem + a.suffix + ".png")), np.where(mask, 0, 255).astype(np.uint8))
        shares.append(float(mask.mean()))
        if k % max(1, len(frames) // 4) == 0 and len(qa) < 4:
            vis = np.asarray(pil).astype(np.float32)
            vis[mask] = 0.35 * vis[mask] + np.array([160, 0, 0]) * 0.65
            qa.append(Image.fromarray(vis.astype(np.uint8)).resize((640, 360)))
        if k % 50 == 0:
            print(f"[masks] {k}/{len(frames)}: {100 * np.mean(shares):.1f} % masked so far", flush=True)
    sheet = Image.new("RGB", (1280, 720))
    for i, im in enumerate(qa):
        sheet.paste(im, ((i % 2) * 640, (i // 2) * 360))
    (s / "qa" / "splat").mkdir(parents=True, exist_ok=True)
    if qa:  # (all reused: the earlier sheet stays)
        sheet.save(s / "qa" / "splat" / f"masks{a.suffix if a.suffix != '_mask' else ''}.jpg", quality=85)
    (out / f"masks{a.suffix if a.suffix != '_mask' else ''}.json").write_text(json.dumps({"prompts": prompts, "threshold": a.threshold, "inside": a.inside, "dilation_pixels": 6,
                                                "masked_pixel_share": float(np.mean(shares))}) + "\n")
    print(f"done: {len(frames)} frames, {100 * np.mean(shares):.1f} % of the pixels masked")


if __name__ == "__main__":
    main()
