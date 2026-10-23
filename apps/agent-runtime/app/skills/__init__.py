"""Discover skill metadata and load instructions on demand.

Follows Pi's core/skills.ts loadSkillFromFile/formatSkillsForPrompt: expose
name/description first, then let the model read the matching SKILL.md. This
adapter uses a named read tool because workspace read_file cannot read package
assets. Skill instructions do not add a separate execution loop.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

import yaml

from ..contracts import ToolResult
from ..tools import FunctionTool


def load_skills(directory: Path = Path(__file__).parent) -> dict[str, dict]:
    skills = {}
    for path in sorted(directory.glob("*/SKILL.md")):
        text = path.read_text(encoding="utf-8")
        if not text.startswith("---\n"):
            raise ValueError(f"skill requires YAML frontmatter: {path}")
        metadata = yaml.safe_load(text.split("---", 2)[1])
        name, description = metadata["name"], metadata["description"]
        if not isinstance(name, str) or not isinstance(description, str) or not description.strip():
            raise ValueError(f"skill requires a name and description: {path}")
        if name in skills:
            raise ValueError(f"duplicate skill: {name}")
        skills[name] = {"description": description, "path": path}
    return skills


def skills_prompt(skills: dict[str, dict]) -> str:
    if not skills:
        return ""
    rows = [
        "## Available skills",
        "When a task matches a skill description, load it with read_skill(name). "
        "Use its guidance with the existing tools; choose steps from actual findings. "
        "Unrelated tasks continue normally.",
        "<available_skills>",
    ]
    for name, skill in skills.items():
        rows.append(f'<skill name="{escape(name)}">{escape(skill["description"])}</skill>')
    return "\n".join([*rows, "</available_skills>"])


def read_skill_tool(skills: dict[str, dict]) -> FunctionTool:
    def read(args, context):
        path = skills[args["name"]]["path"]
        return ToolResult(content=f"Skill directory: {path.parent}\n\n"
                          + path.read_text(encoding="utf-8"))

    return FunctionTool(
        "read_skill", "Load a named skill's full instructions when its description matches.",
        {"type": "object", "properties": {"name": {"type": "string", "enum": list(skills)}},
         "required": ["name"], "additionalProperties": False},
        read, idempotent=True, execution_mode="parallel",
    )
