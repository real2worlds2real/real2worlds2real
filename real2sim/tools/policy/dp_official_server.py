"""An official Diffusion Policy checkpoint (third_party/diffusion_policy, configs/dp_official) as a local policy server
for the Isaac evaluation (calibrated_scene_reconstruction.policy.runners.ServedDPRunner, apps/episode.py --dp-server): one model on the GPU for
every evaluation process. Python's multiprocessing.connection (stdlib: pickled observations in, plain lists out, as
the Isaac python has numpy 1); one thread per client, the model behind a lock.

    PYTHONPATH=src envs/dp_official/bin/python tools/dp_official_server.py <run>/checkpoints/latest.ckpt \
        [--port 5581] [--inference-steps 16]

Loaded as the official eval_real_robot.py loads it: the workspace from the checkpoint's own config, its EMA model,
DDIM with `--inference-steps` (16 there). A request {"layout": True} answers the export's layout (the replay
buffer's r2s2r_layout: camera, image size, frames per sample, hand joints) and the policy's n_obs_steps / horizon;
{"obs": {key: (n_obs_steps, ...) array}} (images H, W, 3 uint8; low-dim float32) answers {"action": (T, A) nested lists}: the
predicted actions from the current observation's step on (the official n_action_steps of them and the rest of the
horizon, for the caller's interpolation), or {"error": message}.
"""
import argparse
import threading
import traceback
from multiprocessing.connection import Listener

AUTH = b"calibrated_scene_reconstruction-dp"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="a .ckpt of a configs/dp_official run")
    ap.add_argument("--port", type=int, default=5581)
    ap.add_argument("--inference-steps", type=int, default=16, help="DDIM steps (the official real-robot eval: 16)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    import dill
    import hydra
    import numpy as np
    import torch
    import zarr
    from omegaconf import OmegaConf

    from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    register_codecs()
    payload = torch.load(open(a.checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    workspace = hydra.utils.get_class(cfg._target_)(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.eval().to(torch.device(a.device))
    policy.num_inference_steps = a.inference_steps
    with zarr.ZipStore(cfg.task.dataset_path, mode="r") as st:
        layout = dict(zarr.open(st, "r").attrs["r2s2r_layout"])
    layout.update(n_obs_steps=int(policy.n_obs_steps), horizon=int(policy.horizon),
                  n_action_steps=int(policy.n_action_steps), checkpoint=a.checkpoint)
    rgb = [k for k, v in cfg.shape_meta.obs.items() if v.get("type", "low_dim") == "rgb"]
    print(f"{a.checkpoint}: {cfg.name} on {cfg.task.name}, EMA {cfg.training.use_ema}, DDIM {a.inference_steps} steps, "
          f"n_obs_steps {policy.n_obs_steps}, horizon {policy.horizon}; listening on {a.port}", flush=True)
    lock = threading.Lock()

    def predict(obs):
        x = {}
        for k, v in obs.items():
            if k not in cfg.shape_meta.obs:  # the runner sends every key; an image-only policy reads the image
                continue
            v = torch.as_tensor(np.asarray(v))
            if k in rgb:  # (T, H, W, 3) uint8 -> (1, T, 3, H, W) in [0, 1]
                v = v.permute(0, 3, 1, 2).float() / 255.0
            x[k] = v.float()[None].to(a.device)
        with lock, torch.no_grad():
            out = policy.predict_action(x)
        # plain lists: the Isaac python has numpy 1, which cannot unpickle numpy 2 arrays
        return out["action_pred"][0, policy.n_obs_steps - 1:].cpu().numpy().tolist()

    def serve(conn):
        with conn:
            while True:
                try:
                    req = conn.recv()
                except (EOFError, OSError):
                    return
                try:
                    conn.send({"layout": layout} if req.get("layout") else {"action": predict(req["obs"])})
                except Exception as e:  # noqa: BLE001 -- the client reports it
                    conn.send({"error": f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"})

    with Listener(("127.0.0.1", a.port), authkey=AUTH) as listener:
        while True:
            threading.Thread(target=serve, args=(listener.accept(),), daemon=True).start()


if __name__ == "__main__":
    main()
