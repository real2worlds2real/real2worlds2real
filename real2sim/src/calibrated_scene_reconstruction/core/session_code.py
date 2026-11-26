"""Versioned per-session source paths, with fallback for unmigrated configurations."""
from pathlib import Path


def source_path(config_dir, relative):
    """Prefer configs/<session>/agent/; preserve explicit agent and absolute paths."""
    relative = Path(relative)
    if relative.is_absolute():
        return relative
    config_dir = Path(config_dir)
    if relative.parts[:1] == ('agent',):
        return config_dir / relative
    preferred = config_dir / 'agent' / relative
    legacy = config_dir / relative
    return preferred if preferred.exists() or not legacy.exists() else legacy
