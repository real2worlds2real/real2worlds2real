from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any


class SkillRegistry:
    def __init__(self, root: Path | None = None):
        self.root = root or Path(__file__).resolve().parents[2] / "skills"

    def _records(self) -> dict[str, dict[str, Any]]:
        records = {}
        for path in sorted(self.root.glob("*/SKILL.md")):
            text = path.read_text(encoding="utf-8")
            match = re.match(r"\A---\n(.*?)\n---\n", text, re.DOTALL)
            if not match:
                continue
            metadata = {}
            for line in match.group(1).splitlines():
                key, separator, value = line.partition(":")
                if separator:
                    metadata[key.strip()] = value.strip()
            name = metadata.get("name")
            description = metadata.get("description")
            if not name or not description or name != path.parent.name:
                continue
            records[name] = {
                "name": name,
                "description": description,
                "path": path,
                "content": text,
            }
        return records

    def catalog(self) -> list[dict[str, str]]:
        return [
            {"name": record["name"], "description": record["description"]}
            for record in self._records().values()
        ]

    def startup(self, mode: str = "twin") -> dict[str, Any]:
        """Load common methods and the selected task workflow, never every skill."""
        task = (
            "generate-scene-cousins" if mode in {"scene_cousin", "scene_cousin_planning"}
            else "plan-scene-generation" if mode == "scene"
            else "generate-functional-cousins" if mode in {"cousin", "cousin_planning"}
            else "generate-editable-object"
        )
        layers, content = [], []
        steps = [("common", "work-with-task-evidence")]
        if mode in {"scene", "twin"}:
            domain = "environment" if mode == "scene" else "object"
            steps.extend([("task", "reconstruct-" + domain), ("native", task)])
        else:
            steps.append(("task", task))
        for level, name in steps:
            record = self.read(name)
            layers.append({"layer":level,"name":name,"sha256":record["sha256"]})
            content.append(f'<startup-skill layer="{level}" name="{name}">\n{record["content"]}\n</startup-skill>')
        return {"schema_version":1,"generation_mode":mode,"layers":layers,
                "instructions":"\n\n".join(content)}

    def search(self, query: str, limit: int = 5) -> dict[str, Any]:
        if not query.strip() or not 1 <= limit <= 10:
            raise ValueError("Supply a query and a limit from 1 to 10")
        terms = set(re.findall(r"[a-z0-9_-]+", query.lower()))
        scored = []
        for record in self._records().values():
            haystack = f"{record['name']} {record['description']} {record['content']}".lower()
            score = sum(term in haystack for term in terms)
            if score:
                scored.append((score, record["name"], record["description"]))
        scored.sort(key=lambda value: (-value[0], value[1]))
        return {
            "query": query,
            "results": [
                {"name": name, "description": description, "score": score}
                for score, name, description in scored[:limit]
            ],
        }

    def read(self, name: str, resource: str | None = None) -> dict[str, Any]:
        record = self._records().get(name)
        if record is None:
            raise ValueError(f"Unknown skill: {name}")
        directory = record["path"].parent.resolve()
        resources = sorted(
            str(path.relative_to(directory))
            for path in directory.rglob("*.md")
            if path.name != "SKILL.md"
            and not path.is_symlink()
            and path.resolve().is_relative_to(directory)
        )
        documents = (self.root.parent / "docs").resolve()
        for target in re.findall(r"\]\(([^)]+)\)", record["content"]):
            target = target.split("#", 1)[0]
            if not target or Path(target).is_absolute() or ":" in target:
                continue
            path = directory / target
            if (
                path.suffix == ".md"
                and path.is_file()
                and not path.is_symlink()
                and path.resolve().is_relative_to(documents)
            ):
                resources.append(target)
        resources = sorted(set(resources))
        if resource is not None:
            if resource not in resources:
                raise ValueError("Choose an exact Markdown resource listed by read_skill")
            content = (directory / resource).read_text(encoding="utf-8")
        else:
            content = record["content"]
        return {
            "name": name,
            "resource": resource,
            "resources": resources,
            "content": content,
            "sha256": hashlib.sha256(content.encode()).hexdigest(),
        }
