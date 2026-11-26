"""A fine-tuned pi0.5 checkpoint (calibrated_scene_reconstruction.policy.openpi) as the Isaac evaluation's policy server
(calibrated_scene_reconstruction.policy.runners.ServedOpenpiRunner, apps/episode.py --openpi-server): openpi's own serve_policy.py -- the policy
from create_trained_policy (the checkpoint's params in bfloat16 and its norm stats), its websocket server -- bound to
this machine only, its metadata the dataset's joint layout (tools/export_openpi.py's meta/r2s2r_openpi.json: the
state's joints, their index in the full state, the rate) and its task text, the prompt.

    HF_LEROBOT_HOME=outputs/openpi OPENPI_DATA_HOME=envs/openpi_data XLA_PYTHON_CLIENT_PREALLOCATE=false \
        envs/openpi/bin/python tools/openpi_serve.py <run>/<step> [--config pi05_pick_place] [--port 8000]
"""
import argparse
import json
import logging
import os
from pathlib import Path

from calibrated_scene_reconstruction.core.paths import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="a training step's directory (<checkpoint dir>/<config>/<run>/<step>)")
    ap.add_argument("--config", default="pi05_pick_place")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    import calibrated_scene_reconstruction.policy.openpi  # noqa: E402,F401  (registers the configs)
    logging.basicConfig(level=logging.INFO, force=True)
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as _config

    cfg = _config.get_config(a.config)
    policy = policy_config.create_trained_policy(cfg, a.checkpoint)
    home = Path(os.environ.get("HF_LEROBOT_HOME", Path.home() / ".cache" / "huggingface" / "lerobot"))
    meta = home / cfg.data.repo_id / "meta"
    layout = json.loads((meta / "r2s2r_openpi.json").read_text())
    tasks = [json.loads(s)["task"] for s in (meta / "tasks.jsonl").read_text().splitlines() if s.strip()]
    info = json.loads((meta / "info.json").read_text())
    cams = [k.removeprefix("observation.images.") for k, f in info["features"].items() if f["dtype"] == "video"]
    layout.update(prompt=tasks[0], camera=cams[0], action_horizon=int(cfg.model.action_horizon), checkpoint=a.checkpoint,
                  config=a.config)
    print(f"{a.checkpoint}: {a.config}, {len(layout['joints'])} joints at {layout['fps']} Hz, chunks of "
          f"{layout['action_horizon']}; listening on {a.host}:{a.port}", flush=True)
    websocket_policy_server.WebsocketPolicyServer(policy, host=a.host, port=a.port, metadata=layout).serve_forever()


if __name__ == "__main__":
    main()
