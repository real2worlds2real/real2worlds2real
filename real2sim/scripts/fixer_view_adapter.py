"""Optional local official Fixer image-restoration backend; no weights or authorization bundled."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import transformer_engine.pytorch  # noqa: F401
    import transformer_engine.pytorch.attention as attention

    compatibility = []
    if not hasattr(attention, "apply_rotary_pos_emb"):
        from transformer_engine.pytorch.attention.rope import apply_rotary_pos_emb

        attention.apply_rotary_pos_emb = apply_rotary_pos_emb
        compatibility.append(
            "TE rotary embedding public function moved to attention.rope; original implementation retained"
        )
    sys.path.insert(0, str(args.checkout.resolve() / "src"))
    import pix2pix_turbo_nocond_cosmos_base_faster_tokenizer as implementation

    from real2sim_harness.artifacts import atomic_json, sha256_file

    request = json.loads(args.request.read_text())
    images = request.get("images", [])
    if not isinstance(images, list) or not 1 <= len(images) <= 8:
        raise ValueError("Restore 1..8 anchor images per call")
    checkpoint = args.weights / "pretrained/pretrained_fixer.pkl"
    for path in (
        checkpoint,
        args.weights / "base/model_fast_tokenizer.pt",
        args.weights / "base/tokenizer_fast.pth",
    ):
        if not path.is_file():
            raise ValueError("Missing official Fixer base/pretrained weights")
    implementation.config.dit_path = str((args.weights / "base/model_fast_tokenizer.pt").resolve())
    implementation.config.tokenizer["vae_pth"] = str(
        (args.weights / "base/tokenizer_fast.pth").resolve()
    )
    model = implementation.Pix2Pix_Turbo(
        pretrained_path=str(checkpoint.resolve()),
        timestep=250,
        vae_skip_connection=False,
        batch_size=1,
    ).to("cuda", dtype=torch.bfloat16)
    model.set_eval()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for i, name in enumerate(images):
        image = Image.open(name).convert("RGB")
        if max(image.size) > 2048 or min(image.size) < 32:
            raise ValueError("Use restoration images from 32 to 2048 pixels per axis")
        # Preserve the input aspect and pixel domain; pad instead of stretching.
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1
        tensor = torch.tensor(array.transpose(2, 0, 1), device="cuda", dtype=torch.bfloat16)[None]
        h, w = tensor.shape[-2:]
        padded = torch.nn.functional.pad(tensor, (0, (-w) % 16, 0, (-h) % 16), mode="replicate")
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            restored = model(padded)[:, :, :h, :w]
        result = (restored[0].float().cpu().numpy().transpose(1, 2, 0) + 1) * 127.5
        if not np.isfinite(result).all():
            raise ValueError("Fixer produced nonfinite pixels")
        target = args.output / f"anchor-{i:03d}.png"
        Image.fromarray(np.rint(np.clip(result, 0, 255)).astype(np.uint8)).save(target)
        records.append(
            {
                "image": target.name,
                "input_sha256": sha256_file(Path(name)),
                "output_sha256": sha256_file(target),
            }
        )
    revision = subprocess.check_output(
        ["git", "-C", str(args.checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    atomic_json(
        args.output / "restoration_report.json",
        {
            "ok": True,
            "images": records,
            "backend": {"name": "nv-tlabs/Fixer", "version": revision},
            "checkpoint_sha256": sha256_file(checkpoint),
            "compatibility": compatibility,
            "scope": "Generated restoration proposals; not observed pixels or accepted 3D geometry",
        },
    )


if __name__ == "__main__":
    main()
