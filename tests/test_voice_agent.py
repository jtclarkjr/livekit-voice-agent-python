"""Tests for model-facing tool-choice enforcement and agent wiring."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from livekit.agents import NOT_GIVEN, ModelSettings, llm

from agent.recency import RecencyDecision, SearchMode
from agent.voice_agent import (
    WEB_SEARCH_TOOL_NAME,
    VoiceFastAgent,
    create_voice_agent,
    resolve_voice_tool_choice,
)
from agent.web_search import (
    WebSearchResult,
    WebSearchTool,
    create_web_search_tool,
)


class EmptySearchClient:
    async def search(self, query: str) -> list[WebSearchResult]:
        del query
        return []


def search_tool() -> WebSearchTool:
    return create_web_search_tool(EmptySearchClient())


def decision(mode: SearchMode) -> RecencyDecision:
    return RecencyDecision(mode=mode, reason="test")


def test_respects_sdk_none_before_policy_signals() -> None:
    assert (
        resolve_voice_tool_choice(
            decision=decision("required"),
            has_current_turn_search=False,
            requested_tool_choice="none",
            tools=[search_tool()],
        )
        == "none"
    )


def test_disables_tools_after_search_or_explicit_opt_out() -> None:
    assert (
        resolve_voice_tool_choice(
            decision=decision("required"),
            has_current_turn_search=True,
            requested_tool_choice="auto",
            tools=[search_tool()],
        )
        == "none"
    )
    assert (
        resolve_voice_tool_choice(
            decision=decision("forbidden"),
            has_current_turn_search=False,
            requested_tool_choice="required",
            tools=[search_tool()],
        )
        == "none"
    )


def test_forces_exact_search_tool_when_required_and_available() -> None:
    assert resolve_voice_tool_choice(
        decision=decision("required"),
        has_current_turn_search=False,
        requested_tool_choice="auto",
        tools=[search_tool()],
    ) == {"type": "function", "function": {"name": WEB_SEARCH_TOOL_NAME}}


def test_does_not_force_missing_tool_and_preserves_auto_choice() -> None:
    assert (
        resolve_voice_tool_choice(
            decision=decision("required"),
            has_current_turn_search=False,
            requested_tool_choice="auto",
            tools=[],
        )
        == "auto"
    )
    assert (
        resolve_voice_tool_choice(
            decision=decision("auto"),
            has_current_turn_search=False,
            requested_tool_choice=NOT_GIVEN,
            tools=[search_tool()],
        )
        is NOT_GIVEN
    )


def test_agent_registers_search_and_fixed_model_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeInferenceLlm:
        def __init__(self, model: str, **kwargs: object) -> None:
            captured["model"] = model
            captured.update(kwargs)

    import agent.voice_agent as module

    monkeypatch.setattr(module.inference, "LLM", FakeInferenceLlm)  # type: ignore[attr-defined]
    tool = search_tool()
    agent = create_voice_agent(llm.ChatContext.empty(), tool)

    assert any(registered.id == "searchWeb" for registered in agent.tools)
    assert captured == {
        "model": "google/gemma-4-31b-it",
        "extra_kwargs": {"parallel_tool_calls": False},
    }


def test_llm_node_rewrites_context_and_mutates_effective_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeInferenceLlm:
        def __init__(self, model: str, **_kwargs: object) -> None:
            self.model = model

    import agent.voice_agent as module

    monkeypatch.setattr(module.inference, "LLM", FakeInferenceLlm)  # type: ignore[attr-defined]

    def fake_default(
        _agent: VoiceFastAgent,
        chat_context: llm.ChatContext,
        tools: list[llm.Tool],
        settings: ModelSettings,
    ) -> str:
        captured["chat_context"] = chat_context
        captured["tools"] = tools
        captured["tool_choice"] = settings.tool_choice
        return "stream"

    monkeypatch.setattr(module.Agent.default, "llm_node", fake_default)  # type: ignore[attr-defined]
    tool = search_tool()
    canonical = llm.ChatContext(
        [llm.ChatMessage(id="user", role="user", content=["Search the web for today's update"])]
    )
    agent = VoiceFastAgent(
        llm.ChatContext.empty(),
        tool,
        clock=lambda: datetime(2026, 7, 12, tzinfo=UTC),
    )
    settings = ModelSettings(tool_choice="auto")

    assert agent.llm_node(canonical, [tool], settings) == "stream"
    assert settings.tool_choice == {
        "type": "function",
        "function": {"name": "searchWeb"},
    }
    prepared = captured["chat_context"]
    assert isinstance(prepared, llm.ChatContext)
    assert prepared is not canonical
    assert any(
        isinstance(item, llm.ChatMessage)
        and "Per-turn search instruction: REQUIRED" in (item.text_content or "")
        for item in prepared.items
    )


def test_llm_node_propagates_none_after_current_turn_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeInferenceLlm:
        def __init__(self, model: str, **_kwargs: object) -> None:
            self.model = model

    import agent.voice_agent as module

    monkeypatch.setattr(module.inference, "LLM", FakeInferenceLlm)  # type: ignore[attr-defined]

    def fake_default(
        _agent: VoiceFastAgent,
        _context: llm.ChatContext,
        _tools: list[llm.Tool],
        settings: ModelSettings,
    ) -> str:
        assert settings.tool_choice == "none"
        return "stream"

    monkeypatch.setattr(module.Agent.default, "llm_node", fake_default)  # type: ignore[attr-defined]
    tool = search_tool()
    context = llm.ChatContext(
        [
            llm.ChatMessage(id="user", role="user", content=["latest update?"]),
            llm.FunctionCall(
                id="call",
                call_id="call",
                name="searchWeb",
                arguments='{"query":"latest update"}',
            ),
            llm.FunctionCallOutput(
                id="output",
                call_id="call",
                name="searchWeb",
                output='{"results":[]}',
                is_error=False,
            ),
        ]
    )
    settings = ModelSettings(tool_choice="auto")
    agent = VoiceFastAgent(llm.ChatContext.empty(), tool)
    assert agent.llm_node(context, [tool], settings) == "stream"
    assert settings.tool_choice == "none"
