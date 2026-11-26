"""MoGe model loading and inference input/output adaptation."""
from calibrated_scene_reconstruction.core.gpu import share

def run_moge(img, device, fov_x=None):
    """`fov_x`: the horizontal field of view in degrees when it is known (configs/<name>/camera.json, calibrated_scene_reconstruction.calib);
    else MoGe-2 guesses it from the image."""
    import torch
    from moge.model.v2 import MoGeModel

    share()
    model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device).eval()
    x = torch.tensor(img / 255.0, dtype=torch.float32, device=device).permute(2, 0, 1)
    with torch.no_grad():
        out = model.infer(x, fov_x=fov_x, use_fp16=device == "cuda")  # half precision: CUDA only
    res = {k: v.cpu().numpy() for k, v in out.items() if torch.is_tensor(v)}
    del model
    torch.cuda.empty_cache()
    return res


def load_moge(device="cuda"):
    from moge.model.v2 import MoGeModel

    return MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device).eval()
