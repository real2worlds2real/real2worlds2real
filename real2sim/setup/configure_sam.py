"""Configure the operator's local SAM 3D Objects checkout at revision f91db411."""

import hashlib
import sys
from pathlib import Path


_BACKBONE = "sam3d_objects/model/backbone/tdfy_dit/"
# Digests cover each complete file before and after the local configuration.
_FILES = (
    ("notebook/inference.py",
     "a3e20615885d2b5eaf1a8af8573d271072de4e27639212317edffcddf1d39cf3",
     "8d74c93cb3b5c7308431c380d112f74969fc2fe9f2ea94755b958924ccc2c52c", (5,)),
    (_BACKBONE + "renderers/gaussian_render.py",
     "7ab7e1343393ca9b792c18636d1a654fd6d1f9a5b8f1147e6b235a9d064371d4",
     "7a699bb2af53ecc40af13ee74a7a3a072d544ced81f47e0e105201ee2df3a2ec", (72, 240)),
    (_BACKBONE + "utils/render_utils.py",
     "4d72bb35dafee631d639dceefce0624eabf99e5874556fbd3077ca88d7a645ee",
     "b1f677e4dc8244a779d550367f51fd254488411d1f27eb56ac586f2101bdac01", (77, 140, 162)),
)


def configure(root: Path) -> None:
    updates = {}
    for name, original, configured, line_numbers in _FILES:
        path = root / name
        content = path.read_bytes().replace(b"\r\n", b"\n")
        digest = hashlib.sha256(content).hexdigest()
        if digest == configured:
            continue
        if digest != original:
            raise SystemExit(f"Unexpected SAM source at {path}. Check the pinned revision and local edits.")
        lines = content.splitlines(keepends=True)
        for number in line_numbers:
            if name == "notebook/inference.py":
                lines[number - 1] = b'os.environ.setdefault("CUDA_HOME", os.environ.get("CONDA_PREFIX", ""))\n'
            else:
                lines[number - 1] = lines[number - 1].replace(b'"inria"', b'"gsplat"')
        result = b"".join(lines)
        if hashlib.sha256(result).hexdigest() != configured:
            raise SystemExit(f"SAM configuration failed for {path}.")
        updates[path] = result
    for path, result in updates.items():
        path.write_bytes(result)


if __name__ == "__main__":
    configure(Path(sys.argv[1]))
