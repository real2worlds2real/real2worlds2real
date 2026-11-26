"""Read Blender asset metadata across releases without rewriting the source asset."""


def asset_key(owner, key):
    """Return the key to read. An explicit current value, including False, takes precedence."""
    if key in owner:
        return key
    if key.startswith("real2sim_"):
        previous = "remo_" + key[len("real2sim_"):]
    elif key.startswith("Real2Sim_"):
        previous = "REMO_" + key[len("Real2Sim_"):]
    else:
        return key
    return previous if previous in owner else key


def asset_value(owner, key, default=None):
    return owner.get(asset_key(owner, key), default)


def is_harness_datablock(name):
    return name.startswith(("Real2Sim_", "REMO_"))


def matches_datablock(name, current):
    return name == current or name == "REMO_" + current.removeprefix("Real2Sim_")
