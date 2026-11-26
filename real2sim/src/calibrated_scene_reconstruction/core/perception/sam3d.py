"""SAM 3D model lifecycle and file input/output adapter, without scene selection or fitting."""
import argparse
import json
import os
import subprocess
import sys
import numpy as np
from calibrated_scene_reconstruction.core.paths import ROOT, SAM3D_DIR
from calibrated_scene_reconstruction.core.session import Session

def gpu_only_during(pipe, method, model):
    """Keep one generator on the GPU only while its stage runs. Both generators resident (~11 GB of checkpoints)
    leave too little room for stage 2's sparse convolutions on a 16 GB GPU."""
    import torch

    run = getattr(pipe, method)
    pipe.models[model].to("cpu")
    torch.cuda.empty_cache()

    def wrapped(*args, **kwargs):
        pipe.models[model].to(pipe.device)
        try:
            return run(*args, **kwargs)
        finally:
            pipe.models[model].to("cpu")
            torch.cuda.empty_cache()

    setattr(pipe, method, wrapped)

def lean_decode(pipe):
    """decode_slat with only the decoder at work on the GPU: every other model is parked on the CPU and the mesh /
    Gaussian decoders take turns (a large object's mesh decoder upsampling ran out of memory next to them)."""
    import torch

    decoders = {"mesh": "slat_decoder_mesh", "gaussian": "slat_decoder_gs", "gaussian_4": "slat_decoder_gs_4"}

    def decode_slat(slat, formats=("mesh", "gaussian")):
        resident = [k for k, m in pipe.models.items() if m is not None and next(m.parameters(), None) is not None
                    and next(m.parameters()).is_cuda]
        for k in resident:
            pipe.models[k].to("cpu")
        torch.cuda.empty_cache()
        ret = {}
        try:
            with torch.no_grad():
                for fmt in formats:
                    if fmt in decoders:
                        m = pipe.models[decoders[fmt]].to(pipe.device)
                        ret[fmt] = m(slat)
                        m.to("cpu")
                        torch.cuda.empty_cache()
        finally:
            for k in resident:
                pipe.models[k].to(pipe.device)
        return ret

    pipe.decode_slat = decode_slat

def cap_sparse_structure(max_coords=None):
    """Lower SAM 3D's voxel cap before stage 2 on a small card: its 42k default only guards int32 indexing, and the
    mesh decoder's 4x upsampling of a dense structure exceeds 16 GB. Structures above
    the cap are halved in resolution (SAM 3D rescales the pose by the factor)."""
    import torch
    from sam3d_objects.pipeline import inference_pipeline as ip

    if max_coords is None:
        gb = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        max_coords = 20000 if gb < 20 else 42000
    orig = ip.downsample_sparse_structure
    ip.downsample_sparse_structure = lambda coords, **kw: orig(coords, **{**kw, "max_coords": max_coords})

def reconstruct(session, name, ids, inputs=None):
    import torch
    import trimesh
    from PIL import Image

    sys.path.insert(0, str(SAM3D_DIR / "notebook"))
    sys.path.insert(0, str(SAM3D_DIR))
    os.environ.setdefault("CUDA_HOME", str(ROOT / "envs" / "cuda128"))
    from inference import Inference

    s = Session(session)
    out_dir = s / "objects" / name
    if inputs:
        d = np.load(inputs)
        img, mask, pts = d["rgb"], d["mask"].astype(bool), d["points"].astype(np.float32)
        if "out_dir" in d:  # another stage's piece (recon.room_objects): its own folder
            out_dir = s.dir / str(d["out_dir"])
    else:
        img = np.asarray(Image.open(s / "capture" / "rgb.png").convert("RGB"))
        geo, masks = s.geometry(), s.masks()
        pts = geo["points"].astype(np.float32)
        pts[~geo["mask"]] = np.nan
        mask = np.zeros(masks.shape[1:], bool)
        for i in ids:
            mask |= masks[i]
    pointmap_p3d = torch.from_numpy(pts * np.array([-1, -1, 1], dtype=np.float32))  # OpenCV -> PyTorch3D camera

    inference = Inference(str(SAM3D_DIR / "checkpoints" / "hf" / "pipeline.yaml"), compile=False)
    pipe = inference._pipeline
    pipe.rendering_engine = "pytorch3d"
    for method, model in (("sample_sparse_structure", "ss_generator"), ("sample_slat", "slat_generator")):
        gpu_only_during(pipe, method, model)
    lean_decode(pipe)
    cap_sparse_structure()
    out = pipe.run(inference.merge_mask_to_rgba(img, mask), None, seed=42, stage1_only=False,
                   with_mesh_postprocess=True, with_texture_baking=True, with_layout_postprocess=True,
                   use_vertex_color=False, pointmap=pointmap_p3d)
    d = out_dir
    d.mkdir(parents=True, exist_ok=True)
    glb: trimesh.Trimesh = out["glb"]
    glb.export(d / "raw.glb")
    out["gs"].save_ply(str(d / "splat.ply"))
    (d / "pose.json").write_text(json.dumps({
        "ids": list(ids),
        "inputs": str(inputs) if inputs else None,
        "scale": out["scale"][0].tolist(),
        "rotation_wxyz_p3d": out["rotation"][0].tolist(),
        "translation_p3d": out["translation"][0].tolist(),
    }, indent=1))
    print(f"[{name}] reconstructed: {len(glb.vertices)} vertices")

def run_worker(session, name, ids, inputs=None):
    """SAM 3D in a fresh process (it does not release host RAM between objects); returns the exit code."""
    cmd = [sys.executable, "-m", "calibrated_scene_reconstruction.core.perception.sam3d", str(session), "--worker", name, "+".join(map(str, ids))]
    return subprocess.run(cmd + (["--inputs", str(inputs)] if inputs else [])).returncode


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("session")
    parser.add_argument("--worker", nargs=2, required=True, metavar=("NAME", "IDS"))
    parser.add_argument("--inputs")
    args = parser.parse_args()
    reconstruct(args.session, args.worker[0], [int(i) for i in args.worker[1].split("+") if i], args.inputs)


if __name__ == "__main__":
    main()
