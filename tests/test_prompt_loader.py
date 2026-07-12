"""Tests for Markdown-backed prompt resources."""

from __future__ import annotations

import pytest

from agent.prompt_loader import load_prompt, render_prompt


def test_loads_and_caches_markdown_prompt() -> None:
    first = load_prompt("voice_instructions.md")
    second = load_prompt("voice_instructions.md")

    assert first is second
    assert first.startswith("You are a concise, friendly voice assistant")
    assert "Never open the call with an automatic greeting" in first


def test_renders_exact_named_placeholders() -> None:
    rendered = render_prompt(
        "runtime_policy.md",
        current_utc="2026-07-12T00:00:00.000Z",
        search_instruction="Per-turn search instruction: AUTO.",
    )

    assert "Current UTC time: 2026-07-12T00:00:00.000Z" in rendered
    assert rendered.endswith("Per-turn search instruction: AUTO.")


def test_rejects_missing_or_unexpected_template_variables() -> None:
    with pytest.raises(ValueError, match=r"missing=\['search_instruction'\]"):
        render_prompt(
            "runtime_policy.md",
            current_utc="2026-07-12T00:00:00.000Z",
        )

    with pytest.raises(ValueError, match=r"unexpected=\['extra'\]"):
        render_prompt(
            "search_query_hint.md",
            query_hint_json='"latest gpt"',
            extra="not allowed",
        )
