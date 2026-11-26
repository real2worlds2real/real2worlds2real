"""File lookup shared by consumers of tagged JSON artifacts."""
from pathlib import Path


def tagged_json_path(root, tag, directories, *, identity_tag=None, label='file'):
    """Return the first stored JSON path, or None for an explicit identity tag."""
    if tag == identity_tag:
        return None
    for directory in directories:
        path = Path(root / directory / f'{tag}.json')
        if path.exists():
            return path
    raise FileNotFoundError(f'no {label} for tag {tag}')
