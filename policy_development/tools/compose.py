"""Tool `compose`: chain two candidates by stage with an observation switch (see compose_policy.py).  Free; evaluate the
result afterwards."""
import json
import shutil

from _adapter import RUNTIME, main

import materialize


def run(job):
    a = job.args
    r = job.request["resolved"]["candidates"]
    d = job.ws / "candidates" / a["name"]
    d.mkdir(parents=True)
    materialize.as_policy(job, r[a["stage_a"]], d / "stage_a")
    materialize.as_policy(job, r[a["stage_b"]], d / "stage_b")
    shutil.copy2(RUNTIME / "compose_policy.py", d / "policy.py")
    (d / "switch.json").write_text(json.dumps(a["switch"], indent=1))
    note = f"{a['stage_a']} until {a['switch']} then {a['stage_b']}"
    job.finish(facts={"composed": note}, created={"candidates": [{"name": a["name"], "kind": "policy", "path": str(d), "entry": "policy.py",
                                                                  "method": "composition", "parent": a["stage_a"], "note": note, "params": a}]})


if __name__ == "__main__":
    main(run)
