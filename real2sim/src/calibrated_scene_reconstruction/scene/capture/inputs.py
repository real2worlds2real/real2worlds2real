"""Deployment input policy for scene reconstruction; not a general session-storage concern."""
from calibrated_scene_reconstruction.scene.session import Session


def reject_measured_sizes(config, name):
    """Reject legacy measured-size inputs, including empty or nested references fields."""
    def visit(value):
        if isinstance(value, dict):
            if "references" in value:
                raise ValueError(f"{name}: references is forbidden in deployment; use board or robot calibration")
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(config)
    if name == "task.json" and "sizes" in config:
        raise ValueError("task.json: measured sizes are forbidden in deployment")
    if name == "twin.json" and any(config.get(k) == "references" for k in ("scale", "scale_check")):
        raise ValueError("twin.json: references is not a deployment scale or scale check")


def validate_session(s):
    """Reject measured-size configuration before any reconstruction writes."""
    for name in ("multiview.json", "task.json", "twin.json", "environment.json"):
        reject_measured_sizes(s.config(name, {}), name)
    from .supplementary import load
    plan = load(s)
    scale = s.config('twin.json', {}).get('scale')
    if plan is not None and scale is not None and scale != plan['metric_scale']['method']:
        raise ValueError('twin.json scale differs from the reviewed media scale plan')


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate deployment inputs without running any stage.")
    parser.add_argument("--check-inputs", required=True, metavar="SESSION")
    args = parser.parse_args()
    try:
        validate_session(Session(args.check_inputs))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
