"""Persistent Blender mesh evaluation for the shared bounded geometry fitter."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


class BlenderMesh:
    def __init__(self, executable, source, output, selected_objects=None, selected_roots=None):
        self.executable, self.source, self.output = Path(executable), Path(source), Path(output)
        self.selected = selected_objects
        self.selected_roots = selected_roots
        self.process = None

    def build_mesh(self, parameters):
        if self.process is None:
            self.output.mkdir(parents=True, exist_ok=True)
            worker = Path(__file__).with_name("blender_fit_worker.py")
            self.process = subprocess.Popen([str(self.executable), "--background", "--factory-startup",
                "--python", str(worker), "--", str(self.source)], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.process.stdin.write(json.dumps({"parameters": parameters, "selected_objects": self.selected,
                                            "selected_roots": self.selected_roots}) + "\n")
        self.process.stdin.flush()
        with (self.output / "blender.log").open("a") as log:
            while line := self.process.stdout.readline():
                if line.startswith("REAL2SIM_FIT_RESULT="):
                    result = json.loads(line.split("=", 1)[1])
                    if not result["ok"]:
                        raise ValueError("Blender parameter evaluation failed: " + result["error"])
                    return result["mesh"]
                log.write(line)
        raise RuntimeError("Blender fitting worker exited; inspect blender.log")

    def close(self):
        if self.process is None:
            return
        try:
            self.process.stdin.close()
            self.process.wait(timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            self.process.kill()
            self.process.wait()
        finally:
            self.process.stdout.close()
