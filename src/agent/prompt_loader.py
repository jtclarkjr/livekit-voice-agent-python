"""Load and render model-facing prompts from Markdown resources."""

from __future__ import annotations

import re
from functools import cache
from pathlib import Path
from typing import Literal

type PromptName = Literal[
    "runtime_policy.md",
    "search_already_completed.md",
    "search_auto.md",
    "search_forbidden.md",
    "search_query_description.md",
    "search_query_hint.md",
    "search_required.md",
    "search_required_hint.md",
    "search_tool_description.md",
    "voice_instructions.md",
]

_PROMPT_DIRECTORY = Path(__file__).with_name("prompts")
_PLACEHOLDER_PATTERN = re.compile(r"\{\{([a-z][a-z0-9_]*)\}\}")


@cache
def load_prompt(name: PromptName) -> str:
    """Load one non-empty Markdown prompt and cache it for the process lifetime."""

    if Path(name).name != name or not name.endswith(".md"):
        raise ValueError(f"invalid prompt resource name: {name}")
    path = _PROMPT_DIRECTORY / name
    try:
        prompt = path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise RuntimeError(f"failed to load prompt resource: {name}") from error
    if not prompt:
        raise ValueError(f"prompt resource is empty: {name}")
    return prompt


def render_prompt(name: PromptName, /, **variables: str) -> str:
    """Render a prompt while requiring an exact set of named placeholders."""

    template = load_prompt(name)
    expected = set(_PLACEHOLDER_PATTERN.findall(template))
    provided = set(variables)
    if expected != provided:
        missing = sorted(expected - provided)
        unexpected = sorted(provided - expected)
        raise ValueError(
            f"prompt variables do not match {name}: missing={missing}, unexpected={unexpected}"
        )
    return _PLACEHOLDER_PATTERN.sub(
        lambda match: variables[match.group(1)],
        template,
    )


__all__ = ["PromptName", "load_prompt", "render_prompt"]
